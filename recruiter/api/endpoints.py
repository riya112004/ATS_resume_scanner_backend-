import os
import uuid
import numpy as np
import re
import logging
import asyncio
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import List, Optional, Dict
from recruiter.core.config import settings
from recruiter.core.database import db
from recruiter.utils.extractor import extract_text_from_file
from recruiter.services.parser import parser
from recruiter.services.embeddings import embedding_service
from recruiter.services.matching import calculate_match_score

# Configure Logging
log_file = os.path.join(os.getcwd(), "activity.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recruiter_api")

router = APIRouter()

# --- JOB TITLE SYNONYMS ---
JOB_SYNONYMS = {
    "dev": ["developer", "engineer", "software"],
    "developer": ["dev", "engineer", "software"],
    "engineer": ["dev", "developer", "software"],
    "frontend": ["front-end", "ui", "ux", "client side"],
    "backend": ["back-end", "server side", "api", "distributed"],
    "fullstack": ["full stack", "full-stack", "mern", "mean"],
    "data": ["analyst", "scientist", "ai", "ml"],
    "hr": ["human resources", "recruiter", "talent"],
    "qa": ["quality assurance", "tester", "testing", "sdet"]
}

# --- HELPERS ---

def get_strict_skill_regex(skill: str):
    escaped = re.escape(skill.lower())
    return f"(^|[ /]){escaped}($|[ /])"

def normalize_val(val: str) -> str:
    if not val: return ""
    return str(val).strip().lower()

def is_valid_location_query(query: str) -> bool:
    if not query: return False
    clean_q = normalize_val(query)
    junk = ["unknown", "n/a", "none", "undefined", "null", "na", "n / a"]
    if len(clean_q) < 3 or clean_q in junk:
        return False
    return True

def tokenize_and_expand_job(query: str) -> List[List[str]]:
    words = re.findall(r'\w+', query.lower())
    token_groups = []
    for word in words:
        group = [word]
        if word in JOB_SYNONYMS:
            group.extend(JOB_SYNONYMS[word])
        token_groups.append(list(set(group)))
    return token_groups

def rank_job_results(results: List[Dict], original_query: str) -> List[Dict]:
    if not original_query: return results
    q_lower = original_query.lower().strip()
    
    for res in results:
        title = normalize_val(res.get("extracted_data", {}).get("job_title", ""))
        score = 0
        if title == q_lower: score = 100
        elif q_lower in title: score = 80
        else:
            tokens = q_lower.split()
            if any(t in title for t in tokens): score = 50
            else: score = 20
        res["job_rank_score"] = score
        
    results.sort(key=lambda x: (x.get("job_rank_score", 0), x.get("match_score", 0)), reverse=True)
    return results

# --- SEMAPHORE (Limit concurrency) ---
MAX_CONCURRENT_TASKS = 30
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

@router.post("/upload")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = Form(None)
):
    upload_start = time.time()
    logger.info(f"Upload request received for {len(files)} files. Limit: {MAX_CONCURRENT_TASKS}.")
    
    # Pre-calculate JD Embedding once to save time
    jd_embedding = None
    if job_description:
        logger.info("Pre-calculating Job Description Embedding...")
        jd_embedding = await embedding_service.generate_embedding(job_description)

    async def process_single_file(file: UploadFile):
        async with semaphore:
            start_time = time.time()
            file_id = str(uuid.uuid4())
            _, ext = os.path.splitext(file.filename)
            file_name = f"{file_id}{ext}"
            file_path = os.path.join(settings.UPLOAD_DIR, file_name)
            
            try:
                content = await file.read()
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
                
                # 1. Extraction & AI Parsing
                raw_text = await extract_text_from_file(file_path)
                parsed_data = await parser.parse_resume_text(raw_text)
                
                # 2. Duplicate Check
                email = str(parsed_data.email).strip().lower() if parsed_data.email else None
                phone = re.sub(r"\D", "", str(parsed_data.phone_number)) if parsed_data.phone_number else None
                
                query_parts = []
                if email:
                    query_parts.append({"extracted_data.email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
                if phone:
                    phone_regex = "".join([f"{c}[^0-9]*" for c in phone])
                    query_parts.append({"extracted_data.phone_number": {"$regex": phone_regex}})
                
                if query_parts and await db.db["resumes"].find_one({"$or": query_parts}):
                    if os.path.exists(file_path): os.remove(file_path)
                    return {"filename": file.filename, "status": "duplicate", "message": "Candidate exists."}

                # 3. AI Tasks (Parallel)
                parsed_text = f"Name: {parsed_data.name} Title: {parsed_data.job_title} Skills: {', '.join(parsed_data.skills)}"
                embedding_task = embedding_service.generate_embedding(parsed_text.strip())
                match_task = calculate_match_score(raw_text, job_description, jd_embedding=jd_embedding) if job_description else None
                
                if match_task:
                    embedding, match_score = await asyncio.gather(embedding_task, match_task)
                else:
                    embedding, match_score = await embedding_task, 0.0

                # 4. Save to MongoDB
                relative_url = f"/uploads/{file_name}"
                await db.db["resumes"].insert_one({
                    "resumeURL": relative_url,
                    "extracted_data": parsed_data.dict(),
                    "embedding": embedding,
                    "updated_at": uuid.uuid4().hex
                })
                
                logger.info(f"PERF: {file.filename} - Total {round(time.time() - start_time, 2)}s")
                return {
                    "filename": file.filename,
                    "status": "success",
                    "match_score": match_score,
                    "resumeURL": f"{settings.BASE_URL}{relative_url}"
                }
            except Exception as e:
                logger.error(f"Error {file.filename}: {str(e)}")
                if os.path.exists(file_path): os.remove(file_path)
                return {"filename": file.filename, "error": str(e)}

    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Batch completed in {round(time.time() - upload_start, 2)}s")
    return results

@router.get("/search")
async def search_resumes(
    min_experience: Optional[float] = None,
    max_experience: Optional[float] = None,
    location: Optional[str] = None,
    skills: Optional[str] = None,
    education: Optional[str] = None,
    job_title: Optional[str] = None,
    match_all: bool = Query(False),
    limit: int = 10
):
    mongo_filter = {}
    combined_filters = []
    
    if min_experience is not None or max_experience is not None:
        exp_filter = {}
        if min_experience is not None: exp_filter["$gte"] = min_experience
        if max_experience is not None: exp_filter["$lte"] = max_experience
        combined_filters.append({"extracted_data.experience": exp_filter})

    if skills:
        skill_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
        if skill_list:
            conditions = [{"extracted_data.skills": {"$regex": get_strict_skill_regex(s), "$options": "i"}} for s in skill_list]
            combined_filters.append({"$and" if match_all else "$or": conditions})

    if location and is_valid_location_query(location):
        clean_loc = normalize_val(location)
        words = [re.escape(w) for w in clean_loc.split() if len(w) >= 2]
        if words:
            loc_pattern = "|".join(words)
            combined_filters.append({"extracted_data.location": {"$regex": loc_pattern, "$options": "i"}})

    if job_title:
        token_groups = tokenize_and_expand_job(job_title)
        if token_groups:
            job_conditions = [{"extracted_data.job_title": {"$regex": "|".join([re.escape(t) for t in g]), "$options": "i"}} for g in token_groups]
            combined_filters.append({"$and": job_conditions})

    if combined_filters:
        mongo_filter["$and" if len(combined_filters) > 1 else None] = combined_filters if len(combined_filters) > 1 else combined_filters[0]
        if None in mongo_filter: mongo_filter = combined_filters[0]

    all_resumes = await db.db["resumes"].find(mongo_filter).to_list(length=100)
    
    scored_results = []
    if job_title or skills or location:
        search_query = f"{job_title or ''} {skills or ''} {location or ''}".strip()
        query_embedding = await embedding_service.generate_embedding(search_query)
        for res in all_resumes:
            if "embedding" in res:
                a, b = np.array(query_embedding), np.array(res["embedding"])
                similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                res["match_score"] = float(round(similarity * 100, 2))
                res["_id"] = str(res["_id"])
                if res["resumeURL"].startswith("/uploads/"): res["resumeURL"] = f"{settings.BASE_URL}{res['resumeURL']}"
                res.pop("embedding", None)
                scored_results.append(res)
        return rank_job_results(scored_results, job_title)[:limit]

    for res in all_resumes:
        res["_id"] = str(res["_id"])
        if res["resumeURL"].startswith("/uploads/"): res["resumeURL"] = f"{settings.BASE_URL}{res['resumeURL']}"
        res.pop("embedding", None)
        res["match_score"] = 0.0
    return all_resumes[:limit]
