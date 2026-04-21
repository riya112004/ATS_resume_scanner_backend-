import os
import uuid
import logging
import time
import re
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from recruiter.core.config import settings
from recruiter.utils.extractor import extract_text_from_file
from recruiter.core.database import db
from seeker.services.analysis_manager import analysis_manager

router = APIRouter()

# Configure Seeker Specific Logger
logger = logging.getLogger("seeker_api")
logger.setLevel(logging.INFO)

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

def validate_jd(jd_text: str):
    """
    Robust validation for Job Description.
    Checks for length, word count, alphabetic ratio, and repetition.
    """
    if not jd_text or len(jd_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Job description is too short. Please provide at least 100 characters.")

    # 1. Word Count Check
    words = re.findall(r'\w+', jd_text.lower())
    if len(words) < 20:
        raise HTTPException(status_code=400, detail="Job description is too vague. Please provide more details about the role.")

    # 2. Alphabetic Ratio Check (Ensure it's not just numbers or symbols)
    alpha_chars = sum(c.isalpha() for c in jd_text)
    if alpha_chars / len(jd_text) < 0.6:
        raise HTTPException(status_code=400, detail="Job description contains too many non-alphabetic characters.")

    # 3. Repeated Text Detection
    from collections import Counter
    word_counts = Counter(words)
    if word_counts:
        most_common_word, count = word_counts.most_common(1)[0]
        if count / len(words) > 0.3 and len(words) > 30:
            raise HTTPException(status_code=400, detail="Job description contains repetitive text.")

@router.post("/analyze")
async def analyze_seeker_resume(
    job_description: str = Form(...),
    job_title: Optional[str] = Form("Job Role"),
    candidate_experience: Optional[float] = Form(None), 
    resume_file: UploadFile = File(...)
):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    logger.info(f"[{request_id}] START - Analysis request for file: {resume_file.filename}")
    
    # 1. Robust JD Validation
    try:
        validate_jd(job_description)
    except HTTPException as e:
        logger.warning(f"[{request_id}] VALIDATION FAILED - JD Validation: {e.detail}")
        raise

    # Candidate Experience Validation
    if candidate_experience is not None:
        if candidate_experience < 0 or candidate_experience > 60:
            logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid experience: {candidate_experience}")
            raise HTTPException(status_code=400, detail="Invalid candidate_experience. Must be between 0 and 60.")

    # 2. File Check
    _, ext = os.path.splitext(resume_file.filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid extension: {ext}")
        raise HTTPException(status_code=415, detail="Only PDF/DOCX allowed.")

    if resume_file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"[{request_id}] VALIDATION FAILED - Invalid MIME type: {resume_file.content_type}")
        raise HTTPException(status_code=415, detail="Invalid file type. Only PDF and DOCX are allowed.")

    file_name = f"seeker_{request_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)
    
    try:
        # 3. Save File
        content = await resume_file.read()
        logger.info(f"[{request_id}] STEP 1 - File read. Size: {len(content)} bytes")
        
        # File Size Validation
        if len(content) > MAX_FILE_SIZE:
            logger.warning(f"[{request_id}] VALIDATION FAILED - File too large: {len(content)} bytes")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE // (1024 * 1024)}MB."
            )

        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # 4. Text Extraction
        logger.info(f"[{request_id}] STEP 2 - Extracting text...")
        raw_text = await extract_text_from_file(file_path)
        if not raw_text or len(raw_text.strip()) < 50:
             logger.error(f"[{request_id}] EXTRACTION FAILED - Text too short or empty.")
             raise ValueError("Could not extract text.")
        logger.info(f"[{request_id}] STEP 3 - Text extracted. Length: {len(raw_text)} chars")

        # 5. Orchestrate Analysis
        logger.info(f"[{request_id}] STEP 4 - Starting AI Analysis Pipeline...")
        try:
            analysis_data = await analysis_manager.analyze(
                raw_text, 
                job_title, 
                job_description, 
                candidate_experience=candidate_experience
            )
            logger.info(f"[{request_id}] STEP 5 - AI Analysis Completed.")
        except ValueError as ve:
            logger.error(f"[{request_id}] ANALYSIS FAILED - {str(ve)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail=f"Analysis failed: {str(ve)}. Please ensure the resume file is readable and follows a standard format."
            )

        # 6. Database Storage
        parsed_resume = analysis_data.get("parsed_resume", {})
        contact_info = parsed_resume.get("contact", {})
        
        name = contact_info.get("name", "Unknown")
        email = contact_info.get("email", "Not Found")
        overall_score = analysis_data.get("overall_ats_score", 0)
        relative_url = f"/uploads/{file_name}"

        analysis_record = {
            "analysis_id": request_id,
            "candidate_name": name,
            "candidate_email": email,
            "job_title": job_title,
            "resume_filename": resume_file.filename,
            "resume_url": f"{settings.BASE_URL}{relative_url}",
            "job_description": job_description,
            "ats_score": overall_score,
            "status": "completed",
            "created_at": datetime.utcnow()
        }
        await db.db["seeker_analysis_history"].insert_one(analysis_record)
        logger.info(f"[{request_id}] STEP 6 - Record saved to MongoDB for {name}")

        duration = time.time() - start_time
        logger.info(f"[{request_id}] SUCCESS - Analysis finished in {duration:.2f}s")

        # 7. Simplified Response (Include ATS score, exclude breakdown)
        return {
            "success": True,
            "message": "Resume analyzed and candidate details saved successfully.",
            "data": {
                "candidate_name": name,
                "candidate_email": email,
                "overall_ats_score": overall_score,
                "improvements": analysis_data.get("improvement_points", []),
                "matched_skills": analysis_data.get("matched_skills", []),
                "missing_critical_skills": analysis_data.get("missing_critical_skills", []),
                "warnings": analysis_data.get("warnings", []),
                "verdict": analysis_data.get("verdict", ""),
                "resume_url": f"{settings.BASE_URL}{relative_url}"
            }
        }

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions so they reach the client as intended
        raise
    except Exception as e:
        logger.error(f"[{request_id}] CRITICAL SYSTEM ERROR - {str(e)}")
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail="Internal server error occurred.")
