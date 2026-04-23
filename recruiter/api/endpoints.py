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
    "qa": ["quality assurance", "tester", "testing", "sdet"],
    "cna": ["nursing assistant", "certified nursing assistant", "patient care", "nurse"],
    "nursing": ["cna", "nurse", "assistant", "medical"],
    "assistant": ["helper", "support", "aide", "assistant"]
}

# --- HELPERS ---

def get_strict_skill_regex(skill: str):
    # Flexible substring match: Removed all strict boundaries (^, $, /)
    # This ensures MongoDB acts as a 'Wide Net' to fetch anything remotely relevant
    return re.escape(skill.lower())

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

def rank_job_results(results: List[Dict], original_query: str, skill_query: Optional[str] = None) -> List[Dict]:
    # 1. Clean Skill Query
    target_skills = []
    if skill_query:
        target_skills = [s.strip().lower() for s in skill_query.split(",") if s.strip()]

    for res in results:
        # 2. Count Matched Skills
        matched_count = 0
        resume_skills = [s.lower() for s in res.get("extracted_data", {}).get("skills", [])]
        for ts in target_skills:
            if any(ts in rs for rs in resume_skills):
                matched_count += 1
        res["skill_match_count"] = matched_count

        # 3. Job Title Score
        title_score = 0
        if original_query:
            q_lower = original_query.lower().strip()
            title = normalize_val(res.get("extracted_data", {}).get("job_title", ""))
            if title == q_lower: title_score = 100
            elif q_lower in title: title_score = 80
            else:
                tokens = q_lower.split()
                if any(t in title for t in tokens): title_score = 50
                else: title_score = 20
        res["job_rank_score"] = title_score
        
    # 4. Final MULTI-LEVEL Sort:
    # Priority 1: Most Skills Matched (Descending: 4, 3, 2, 1...)
    # Priority 2: Job Title Match (Highest first)
    # Priority 3: Experience (Ascending: 0, 1, 2... - AS REQUESTED)
    results.sort(key=lambda x: (
        -x.get("skill_match_count", 0),
        -x.get("job_rank_score", 0), 
        x.get("extracted_data", {}).get("experience", 0),
        -x.get("match_score", 0)
    ))
    return results

from recruiter.utils.hashing import generate_identity_hash

# --- SEMAPHORE (Limit concurrency) ---
MAX_CONCURRENT_TASKS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

@router.post("/upload")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = Form(None)
):
    upload_start = time.time()
    logger.info(f"Upload request received for {len(files)} files. Limit: {MAX_CONCURRENT_TASKS}.")
    
    # Pre-calculate JD Embedding once
    jd_embedding = None
    if job_description:
        jd_embedding = await embedding_service.generate_embedding(job_description)

    async def process_single_file(file: UploadFile):
        async with semaphore:
            start_time = time.time()
            try:
                # 1. SAVE FILE TEMPORARILY
                file_id = str(uuid.uuid4())
                _, ext = os.path.splitext(file.filename)
                file_name = f"{file_id}{ext}"
                file_path = os.path.join(settings.UPLOAD_DIR, file_name)
                
                content = await file.read()
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
                
                # 2. EXTRACTION & AI PARSING
                logger.info(f"[{file_id}] STEP 2 - Extracting text and AI parsing...")
                raw_text = await extract_text_from_file(file_path)
                parsed_data = await parser.parse_resume_text(raw_text)
                logger.info(f"[{file_id}] STEP 2 DONE - AI parsed for {parsed_data.name}")
                
                # 3. GENERATE IDENTITY HASH
                identity_hash = generate_identity_hash(parsed_data.name, parsed_data.email)
                
                # 4. ULTRA-STRICT DUPLICATE CHECK
                logger.info(f"[{file_id}] STEP 4 - Checking duplicates in MongoDB...")
                duplicate_query = {
                    "$or": [
                        {"identity_hash": identity_hash},
                        {"extracted_data.email": {"$regex": f"^{re.escape(str(parsed_data.email).strip())}$", "$options": "i"}}
                    ]
                }
                
                existing = await db.db["recruiter's resume"].find_one(duplicate_query)
                if existing:
                    logger.info(f"[{file_id}] DUPLICATE - Found for {parsed_data.name}")
                    if os.path.exists(file_path): os.remove(file_path)
                    return {
                        "filename": file.filename,
                        "status": "duplicate_resume",
                        "message": f"Candidate {parsed_data.name} already exists.",
                        "identity_hash": identity_hash,
                        "resumeURL": f"{settings.BASE_URL}{existing.get('resumeURL')}"
                    }

                # 5. AI TASKS (Parallel)
                logger.info(f"[{file_id}] STEP 5 - Generating local embeddings and matching...")
                parsed_text = f"Name: {parsed_data.name} Title: {parsed_data.job_title} Skills: {', '.join(parsed_data.skills)}"
                
                # We do this step-by-step for debugging
                embedding = await embedding_service.generate_embedding(parsed_text.strip())
                logger.info(f"[{file_id}] Embedding generated.")
                
                match_score = 0.0
                if job_description:
                    match_score = await calculate_match_score(raw_text, job_description, jd_embedding=jd_embedding)
                    logger.info(f"[{file_id}] Match score calculated: {match_score}")

                # 6. SAVE TO MONGODB
                logger.info(f"[{file_id}] STEP 6 - Saving to MongoDB...")
                relative_url = f"/uploads/{file_name}"
                await db.db["recruiter's resume"].insert_one({
                    "identity_hash": identity_hash,
                    "filename": file.filename,
                    "resumeURL": relative_url,
                    "extracted_data": parsed_data.dict(),
                    "embedding": embedding,
                    "updated_at": uuid.uuid4().hex
                })
                logger.info(f"[{file_id}] STEP 6 DONE - Saved successfully.")
                
                return {
                    "filename": file.filename,
                    "status": "success",
                    "match_score": match_score,
                    "identity_hash": identity_hash,
                    "resumeURL": f"{settings.BASE_URL}{relative_url}"
                }
            except Exception as e:
                logger.error(f"Error {file.filename}: {str(e)}")
                if os.path.exists(file_path): os.remove(file_path)
                return {"filename": file.filename, "error": str(e)}

    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return results

from math import ceil

@router.get("/search")
async def search_resumes(
    min_experience: Optional[float] = None,
    max_experience: Optional[float] = None,
    location: Optional[str] = None,
    skills: Optional[str] = None,
    education: Optional[str] = None,
    job_title: Optional[str] = None,
    match_all: bool = Query(False),
    current_page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    mongo_filter = {}
    combined_filters = []
    
    if min_experience is not None or max_experience is not None:
        exp_filter = {}
        if min_experience is not None: exp_filter["$gte"] = min_experience
        if max_experience is not None: exp_filter["$lte"] = max_experience
        combined_filters.append({"extracted_data.experience": exp_filter})

    if skills:
        # Split skills by comma first
        skill_queries = [s.strip().lower() for s in skills.split(",") if s.strip()]
        all_conditions = []
        
        for s_query in skill_queries:
            # For each skill query, split by space to handle multi-word skills like "health care"
            sub_parts = s_query.split()
            if not sub_parts: continue
            
            # Create a regex that matches ANY of the words in the multi-word skill
            # e.g. "health care" -> matches anything with "health" OR "care"
            sub_conditions = [{"extracted_data.skills": {"$regex": get_strict_skill_regex(part), "$options": "i"}} for part in sub_parts]
            
            if len(sub_conditions) > 1:
                all_conditions.append({"$or": sub_conditions})
            else:
                all_conditions.append(sub_conditions[0])

        if all_conditions:
            combined_filters.append({"$and" if match_all else "$or": all_conditions})

    # --- Dynamic AI-Driven Location Search ---
    search_loc_parts = []
    if location and is_valid_location_query(location):
        # AI will have stored these fields. We match the query against any of them.
        search_loc_parts = [p.strip() for p in location.split(",")]
        
        loc_conditions = []
        for part in search_loc_parts:
            p_esc = re.escape(part)
            # Relaxed: Removed ^ and $ to allow partial matching (e.g., "New York" matches "New York City")
            loc_conditions.extend([
                {"extracted_data.city": {"$regex": p_esc, "$options": "i"}},
                {"extracted_data.state": {"$regex": p_esc, "$options": "i"}},
                {"extracted_data.country": {"$regex": p_esc, "$options": "i"}}
            ])
        
        if loc_conditions:
            combined_filters.append({"$or": loc_conditions})

    if job_title:
        token_groups = tokenize_and_expand_job(job_title)
        if token_groups:
            # Change to $or to get more results
            job_conditions = [{"extracted_data.job_title": {"$regex": "|".join([re.escape(t) for t in g]), "$options": "i"}} for g in token_groups]
            combined_filters.append({"$or": job_conditions})

    if combined_filters:
        if len(combined_filters) > 1:
            mongo_filter["$and"] = combined_filters
        else:
            mongo_filter = combined_filters[0]

    # Fetch matching resumes with database-level sorting for experience
    # 1 is for Ascending (4, 5, 6...)
    all_resumes = await db.db["recruiter's resume"].find(mongo_filter).sort("extracted_data.experience", 1).to_list(length=10000)
    
    scored_results = []
    search_query = f"{job_title or ''} {skills or ''} {location or ''}".strip()
    query_embedding = await embedding_service.generate_embedding(search_query) if (job_title or skills or location) else None

    for res in all_resumes:
        res["_id"] = str(res["_id"])
        if res["resumeURL"].startswith("/uploads/"): res["resumeURL"] = f"{settings.BASE_URL}{res['resumeURL']}"
        
        # 1. AI Vector Score
        vector_score = 0.0
        if query_embedding and "embedding" in res:
            a, b = np.array(query_embedding), np.array(res["embedding"])
            vector_score = float(round((np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 100, 2))
        
        # 2. Location Match Score (Multiplier)
        loc_boost = 1.0
        if search_loc_parts:
            res_data = res.get("extracted_data", {})
            # If any part matches city, state, or country, give full boost
            res_loc_values = [
                res_data.get("city", "").lower(), 
                res_data.get("state", "").lower(), 
                res_data.get("country", "").lower()
            ]
            if any(p.lower() in res_loc_values for p in search_loc_parts):
                loc_boost = 1.2 # Give 20% boost for location relevance
        
        res["match_score"] = vector_score * loc_boost
        res.pop("embedding", None)
        scored_results.append(res)

    # Sort results if searching (to maintain ranking)
    if job_title or skills or location:
        final_list = rank_job_results(scored_results, job_title, skill_query=skills)
    else:
        final_list = scored_results

    # --- Page-based Pagination Logic ---
    total_count = len(final_list)
    total_pages = ceil(total_count / limit) if total_count > 0 else 1
    
    # Calculate slice indices
    start_idx = (current_page - 1) * limit
    end_idx = start_idx + limit
    
    paginated_results = final_list[start_idx : end_idx]

    return {
        "metadata": {
            "total_records": total_count,
            "total_pages": total_pages,
            "current_page": current_page,
            "limit": limit,
            "has_next": current_page < total_pages,
            "has_previous": current_page > 1
        },
        "results": paginated_results
    }

