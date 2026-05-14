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
from recruiter.services.scoring_engine import recruiter_scoring

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

        # 3. Job Title Score (Universal Semantic Matching)
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
    # Priority 1: Semantic Match Score (Highest first)
    # Priority 2: Most Skills Matched (Descending)
    # Priority 3: Job Title Match (Highest first)
    # Priority 4: Experience (Descending)
    results.sort(key=lambda x: (
        -x.get("match_score", 0),
        -x.get("skill_match_count", 0),
        -x.get("job_rank_score", 0), 
        -x.get("extracted_data", {}).get("experience", 0)
    ))
    return results

from recruiter.utils.hashing import generate_identity_hash

# --- SEMAPHORE (Limit concurrency) ---
MAX_CONCURRENT_TASKS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

from fastapi.responses import StreamingResponse
import io

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
                # 1. READ FILE CONTENT
                file_content = await file.read()
                
                # 2. EXTRACTION & AI PARSING
                logger.info(f"STEP 2 - Extracting text and AI parsing for {file.filename}...")
                raw_text = await extract_text_from_file(file_content=file_content, filename=file.filename)
                parsed_data = await parser.parse_resume_text(raw_text)
                logger.info(f"STEP 2 DONE - AI parsed for {parsed_data.name}")
                
                # 3. GENERATE IDENTITY HASH
                identity_hash = generate_identity_hash(parsed_data.name, parsed_data.email)
                
                # 4. ULTRA-STRICT DUPLICATE CHECK
                logger.info(f"STEP 4 - Checking duplicates in MongoDB...")
                duplicate_query = {
                    "$or": [
                        {"identity_hash": identity_hash},
                        {"extracted_data.email": {"$regex": f"^{re.escape(str(parsed_data.email).strip())}$", "$options": "i"}}
                    ]
                }
                
                existing = await db.db["recruiter's resume"].find_one(duplicate_query)
                if existing:
                    logger.info(f"DUPLICATE - Found for {parsed_data.name}")
                    return {
                        "filename": file.filename,
                        "status": "duplicate_resume",
                        "message": f"Candidate {parsed_data.name} already exists.",
                        "identity_hash": identity_hash,
                        "resumeURL": existing.get("resumeURL")
                    }

                # 5. SAVE FILE LOCALLY
                file_uuid = str(uuid.uuid4())
                extension = os.path.splitext(file.filename)[1]
                new_filename = f"{file_uuid}{extension}"
                file_path = os.path.join(settings.UPLOAD_DIR, new_filename)
                
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                relative_url = f"/uploads/{new_filename}"

                # 6. AI TASKS (Parallel)
                logger.info(f"STEP 6 - Generating embeddings and matching...")
                parsed_text = f"Name: {parsed_data.name} Title: {parsed_data.job_title} Skills: {', '.join(parsed_data.skills)}"
                
                embedding = await embedding_service.generate_embedding(parsed_text.strip())
                
                match_score = 0.0
                if job_description:
                    match_score = await calculate_match_score(raw_text, job_description, jd_embedding=jd_embedding)

                # 7. SAVE TO MONGODB
                logger.info(f"STEP 7 - Saving to MongoDB...")
                
                await db.db["recruiter's resume"].insert_one({
                    "identity_hash": identity_hash,
                    "filename": file.filename,
                    "resumeURL": relative_url,
                    "extracted_data": parsed_data.dict(),
                    "embedding": embedding,
                    "raw_content": raw_text,
                    "updated_at": uuid.uuid4().hex
                })
                logger.info(f"STEP 7 DONE - Saved successfully.")
                
                return {
                    "filename": file.filename,
                    "status": "success",
                    "match_score": match_score,
                    "identity_hash": identity_hash,
                    "resumeURL": f"{settings.BASE_URL}{relative_url}"
                }
            except Exception as e:
                logger.error(f"Error {file.filename}: {str(e)}")
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
        # Universal search: Look for the job title tokens directly in the stored title
        title_tokens = re.findall(r'\w+', job_title.lower())
        if title_tokens:
            # Match if ANY word from the search query exists in the job title
            job_conditions = [{"extracted_data.job_title": {"$regex": re.escape(t), "$options": "i"}} for t in title_tokens]
            combined_filters.append({"$or": job_conditions})

    if combined_filters:
        if len(combined_filters) > 1:
            mongo_filter["$and"] = combined_filters
        else:
            mongo_filter = combined_filters[0]

    # Fetch matching resumes with database-level sorting for experience
    # -1 is for Descending (10, 9, 8...)
    all_resumes = await db.db["recruiter's resume"].find(mongo_filter).sort("extracted_data.experience", -1).to_list(length=10000)
    
    scored_results = []
    search_query = f"{job_title or ''} {skills or ''} {location or ''}".strip()
    query_embedding = await embedding_service.generate_embedding(search_query) if (job_title or skills or location) else None

    for res in all_resumes:
        res["_id"] = str(res["_id"])
        if res["resumeURL"].startswith("/uploads/"): res["resumeURL"] = f"{settings.BASE_URL}{res['resumeURL']}"
        
        # 1. AI Vector Score
        vector_score = 0.0
        if query_embedding and "embedding" in res:
            vector_score = await recruiter_scoring.calculate_vector_score(query_embedding, res["embedding"])
        
        # 2. Location Match Score (Multiplier)
        res["match_score"] = recruiter_scoring.apply_location_boost(vector_score, search_loc_parts, res.get("extracted_data", {}))
        
        # 3. Keyword Frequency Boost (Smart Filtering)
        raw_content = res.get("raw_content", "")
        res["keyword_occurrence_count"] = 0
        
        if raw_content and job_title:
            raw_lower = raw_content.lower()
            # 1. User ke search query ko tokens mein split karein
            query_tokens = [t.strip().lower() for t in re.split(r'[,\s/]+', job_title) if len(t.strip()) > 1]
            
            # 2. Noise words (Developer, Engineer, etc.) ki list
            noise_words = {"developer", "engineer", "manager", "lead", "senior", "junior", "staff", "associate", "role", "position", "analyst", "specialist"}
            
            # 3. Sirf core keywords rakhein (jo noise list mein nahi hain)
            # Example: "React Developer" -> ["react"]
            core_keywords = [t for t in query_tokens if t not in noise_words]
            
            # Agar sab delete ho jaye (e.g. user ne sirf "Developer" search kiya), toh original hi use karein
            search_terms = core_keywords if core_keywords else query_tokens

            total_occ = 0
            for kw in set(search_terms):
                matches = re.findall(r'\b' + re.escape(kw) + r'\b', raw_lower)
                total_occ += len(matches)
            
            res["keyword_occurrence_count"] = total_occ
            
            # Boost logic: Reduced from 2.0 to 0.5 per hit as requested
            freq_boost = float(total_occ * 0.5)
            res["match_score"] += freq_boost
            logger.info(f"Applied Smart Job Title boost of {freq_boost} ({total_occ} hits) to {res.get('filename')}")
        
        res.pop("embedding", None)
        res.pop("raw_content", None)
        res.pop("updated_at", None)
        scored_results.append(res)

    # Sort results if searching (to maintain ranking)
    if job_title or skills or location:
        final_list = recruiter_scoring.rank_results(scored_results, job_title, skill_query=skills)
    else:
        final_list = scored_results

    # --- Page-based Pagination Logic ---
    total_count = len(final_list)
    total_pages = ceil(total_count / limit) if total_count > 0 else 1
    
    # Calculate slice indices
    start_idx = (current_page - 1) * limit
    end_idx = start_idx + limit
    
    paginated_results = final_list[start_idx : end_idx]

    # Remove internal sorting fields before response
    for item in paginated_results:
        item.pop("job_rank_score", None)
        # item.pop("skill_match_count", None)  # Restored as requested

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
