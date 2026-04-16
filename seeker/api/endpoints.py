import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
from recruiter.core.config import settings
from recruiter.utils.extractor import extract_text_from_file
from recruiter.services.parser import parser
from recruiter.services.matching import calculate_match_score
from seeker.services.feedback import seeker_service

from recruiter.services.embeddings import embedding_service
from recruiter.core.database import db

router = APIRouter()

@router.post("/resumes/upload")
async def upload_and_parse_resume(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None)
):
    """
    1. Resume Upload + Parsing
    Upload resume + parse + generate embedding + store in DB + return structured JSON
    """
    # 1. Save file locally
    file_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    file_name = f"seeker_{file_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    
    try:
        raw_text = await extract_text_from_file(file_path)
        parsed_data = await parser.parse_resume_text(raw_text)
        
        text_for_embedding = f"Name: {parsed_data.name} Title: {parsed_data.job_title} Skills: {', '.join(parsed_data.skills)} Experience: {parsed_data.experience} years"
        embedding = await embedding_service.generate_embedding(text_for_embedding.strip())
        
        match_score = None
        feedback = None
        if job_description:
            match_score = await calculate_match_score(raw_text, job_description)
            feedback = await seeker_service.get_resume_feedback(raw_text, job_description)

        # 6. Save to MongoDB (Use relative URL)
        relative_resume_url = f"/uploads/{file_name}"
        resume_doc = {
            "resumeURL": relative_resume_url,
            "extracted_data": parsed_data.dict(),
            "embedding": embedding,
            "status": "seeker_upload",
            "updated_at": uuid.uuid4().hex
        }
        
        # --- DUPLICATE CHECK (Email or Phone) ---
        email = str(parsed_data.email).strip().lower() if parsed_data.email else None
        phone = re.sub(r"\D", "", str(parsed_data.phone_number)) if parsed_data.phone_number else None
        
        query_parts = []
        if email:
            query_parts.append({"extracted_data.email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
        if phone:
            phone_regex = "".join([f"{c}[^0-9]*" for c in phone])
            query_parts.append({"extracted_data.phone_number": {"$regex": phone_regex}})
        
        existing = None
        if query_parts:
            existing = await db.db["seeker_resumes"].find_one({"$or": query_parts})
        
        if existing:
            # Delete the newly uploaded file as it is a duplicate
            if os.path.exists(file_path): os.remove(file_path)
            raise HTTPException(status_code=400, detail="Resume already exists.")
        
        result = await db.db["seeker_resumes"].insert_one(resume_doc)
        db_id = str(result.inserted_id)
        
        # Return structured JSON (Full URL for client)
        return {
            "_id": db_id,
            "filename": file.filename,
            "is_update": True if existing else False,
            "resume_url": f"{settings.BASE_URL}{relative_resume_url}",
            "structured_data": parsed_data.dict(),
            "analysis": {
                "match_score": match_score,
                "feedback": feedback.dict() if feedback else None
            }
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None)
):
    """
    Analyzes a seeker's resume and provides feedback or matching score.
    """
    # 1. Save file temporarily
    file_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    file_name = f"seeker_{file_id}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    
    # 2. Extract text from file
    try:
        raw_text = await extract_text_from_file(file_path)
        if not raw_text or len(raw_text.strip()) < 10:
            raise ValueError("Extracted text is too short or empty.")
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Text extraction failed: {str(e)}")
    
    # 3. Parse resume with AI
    try:
        parsed_data = await parser.parse_resume_text(raw_text)
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"AI Parsing failed: {str(e)}")
    
    # 4. Calculate match score and get feedback if JD is provided
    match_score = None
    feedback = None
    if job_description:
        match_score = await calculate_match_score(raw_text, job_description)
        feedback = await seeker_service.get_resume_feedback(raw_text, job_description)
    
    return {
        "status": "success",
        "filename": file.filename,
        "parsed_data": parsed_data.dict(),
        "match_score": match_score,
        "feedback": feedback.dict() if feedback else None,
        "resumeURL": f"{settings.BASE_URL}/uploads/{file_name}"
    }
