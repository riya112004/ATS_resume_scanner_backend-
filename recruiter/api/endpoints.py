import os
import uuid
import re
import logging
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import List, Optional, Dict
from recruiter.core.config import settings
from recruiter.core.database import db
from recruiter.utils.extractor import extract_text_from_file
from recruiter.utils.compressor import compress_file
from recruiter.services.parser import parser
from recruiter.services.embeddings import embedding_service
from recruiter.services.matching import calculate_match_score
from recruiter.services.scoring_engine import recruiter_scoring
from recruiter.services.boolean_search import boolean_engine

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

import shutil
from recruiter.utils.hashing import generate_identity_hash

# --- In-memory upload progress store ---
upload_progress: Dict[str, dict] = {}

# --- DISK-BASED QUEUE + WORKER POOL (temp_uploads/) ---
upload_queue: asyncio.Queue = asyncio.Queue()
WORKER_COUNT = 5
_workers_started = False

async def _ensure_workers():
    global _workers_started
    if _workers_started:
        return
    _workers_started = True
    for i in range(WORKER_COUNT):
        asyncio.create_task(_worker_loop(i))
    logger.info(f"Started {WORKER_COUNT} upload workers")

async def _worker_loop(worker_id: int):
    while True:
        item = await upload_queue.get()
        try:
            await _process_one(item, worker_id)
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
        finally:
            upload_queue.task_done()

async def _process_one(item: dict, worker_id: int):
    temp_path = item["temp_path"]
    orig_filename = item["orig_filename"]
    batch_id = item["batch_id"]
    new_filename = item["new_filename"]
    jd_embedding = item.get("jd_embedding")
    job_description = item.get("job_description")

    try:
        raw_text = await extract_text_from_file(file_path=temp_path, filename=orig_filename)
        parsed_data = await parser.parse_resume_text(raw_text)

        invalid_name = (
            not parsed_data.name
            or str(parsed_data.name).strip().lower() in ["unknown", "null", "none"]
        )
        invalid_email = (
            not str(parsed_data.email)
            or str(parsed_data.email).strip().lower() in ["none", "null", "unknown", ""]
        )

        if invalid_name or invalid_email:
            logger.warning(f"Worker {worker_id} | INVALID - {orig_filename} (missing name/email)")
            os.remove(temp_path)
            _record_result(batch_id, {"filename": orig_filename, "status": "invalid_resume", "message": "Invalid resume content. Please upload a valid resume PDF with candidate details."})
            return

        identity_hash = generate_identity_hash(parsed_data.name, parsed_data.email)

        duplicate_query = {
            "$or": [
                {"identity_hash": identity_hash},
                {"extracted_data.email": {"$regex": f"^{re.escape(str(parsed_data.email).strip())}$", "$options": "i"}}
            ]
        }

        existing = await db.db["recruiter's resume"].find_one(duplicate_query)
        if existing:
            logger.info(f"Worker {worker_id} | DUPLICATE - {parsed_data.name}")
            os.remove(temp_path)
            _record_result(batch_id, {"filename": orig_filename, "status": "duplicate_resume", "message": f"Candidate {parsed_data.name} already exists.", "identity_hash": identity_hash, "resumeURL": existing.get("resumeURL")})
            return

        # Move from temp_uploads/ → uploads/
        perm_path = os.path.join(settings.UPLOAD_DIR, new_filename)
        shutil.move(temp_path, perm_path)

        relative_url = f"/uploads/{new_filename}"

        embedding = await embedding_service.generate_embedding(raw_text[:2000].strip())

        match_score = 0.0
        if job_description:
            match_score = await calculate_match_score(raw_text, job_description, jd_embedding=jd_embedding)

        await db.db["recruiter's resume"].insert_one({
            "identity_hash": identity_hash,
            "filename": orig_filename,
            "resumeURL": relative_url,
            "extracted_data": parsed_data.dict(),
            "embedding": embedding,
            "raw_content": raw_text,
            "updated_at": uuid.uuid4().hex
        })

        _record_result(batch_id, {"filename": orig_filename, "status": "success", "match_score": match_score, "identity_hash": identity_hash, "resumeURL": f"{settings.BASE_URL}{relative_url}"})
    except Exception as e:
        logger.error(f"Worker {worker_id} | Error {orig_filename}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        _record_result(batch_id, {"filename": orig_filename, "error": str(e)})

def _record_result(batch_id: str, result: dict):
    progress = upload_progress.get(batch_id)
    if progress:
        progress["done"] += 1
        progress["results"].append(result)
        if progress["done"] >= progress["total"]:
            progress["status"] = "completed"

@router.post("/upload")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = Form(None)
):
    await _ensure_workers()

    batch_id = str(uuid.uuid4())
    logger.info(f"Upload request received for {len(files)} files. Batch: {batch_id}")

    upload_progress[batch_id] = {
        "total": len(files),
        "done": 0,
        "status": "processing",
        "results": []
    }

    # JD embedding — compute once
    jd_embedding = None
    if job_description:
        jd_embedding = await embedding_service.generate_embedding(job_description)

    for f in files:
        content = await f.read()
        ext = os.path.splitext(f.filename)[1]
        file_uuid = str(uuid.uuid4())
        new_filename = f"{file_uuid}{ext}"

        # Save to disk-based temp folder immediately (not held in RAM)
        temp_path = os.path.join(settings.TEMP_UPLOAD_DIR, new_filename)
        with open(temp_path, "wb") as out:
            out.write(content)

        # Compress immediately — kam space lega chahe temp ho ya permanent
        await compress_file(temp_path, f.filename)

        await upload_queue.put({
            "temp_path": temp_path,
            "orig_filename": f.filename,
            "new_filename": new_filename,
            "batch_id": batch_id,
            "jd_embedding": jd_embedding,
            "job_description": job_description,
        })

    return {"batch_id": batch_id, "total": len(files)}

@router.get("/upload-status/{batch_id}")
async def get_upload_status(batch_id: str):
    progress = upload_progress.get(batch_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {
        "batch_id": batch_id,
        "total": progress["total"],
        "done": progress["done"],
        "status": progress["status"],
        "results": progress["results"] if progress["status"] == "completed" else []
    }

from math import ceil

@router.get("/search")
async def search_resumes(
    query: Optional[str] = Query(None),
    min_experience: Optional[float] = None,
    max_experience: Optional[float] = None,
    location: Optional[str] = None,
    skills: Optional[str] = None,
    job_title: Optional[str] = None,
    is_boolean: bool = Query(False),
    match_all: bool = Query(False),
    current_page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    # Auto-detect Boolean syntax in job_title (only if boolean operators present, not just parentheses)
    if job_title and re.search(r'\b(?:AND|OR|NOT|NOR|XOR|NAND|XNOR)\b', job_title, re.IGNORECASE):
        logger.info(f"Boolean syntax detected in job_title. Auto-switching to boolean mode.")
        is_boolean = True
        query = job_title
        job_title = None

    mongo_filter = {}
    combined_filters = []
    
    # --- 1. Boolean Search Logic ---
    if is_boolean:
        # Use ONLY the 'query' parameter for the Boolean logic
        if query:
            # Initial 'Wide Net' fetch: Find resumes containing ANY of the literal keywords
            keywords = boolean_engine.extract_keywords(query)
            if keywords:
                or_conditions = []
                for kw in keywords:
                    if kw.endswith("*"):
                        root = kw[:-1]
                        # Match any word starting with the root
                        or_conditions.append({"raw_content": {"$regex": rf"\b{re.escape(root)}\w*", "$options": "i"}})
                    else:
                        or_conditions.append({"raw_content": {"$regex": re.escape(kw), "$options": "i"}})
                mongo_filter = {"$or": or_conditions}
            logger.info(f"Boolean Mode activated. Query: {query}")
        
        # Apply strict experience/location filters if provided along with boolean query
        extra_filters = []
        if min_experience is not None or max_experience is not None:
            exp_filter = {}
            if min_experience is not None: exp_filter["$gte"] = min_experience
            if max_experience is not None: exp_filter["$lte"] = max_experience
            extra_filters.append({"extracted_data.experience": exp_filter})
        
        if location and is_valid_location_query(location):
            search_loc_parts = [p.strip() for p in location.split(",")]
            loc_conditions = []
            for part in search_loc_parts:
                p_esc = re.escape(part)
                loc_conditions.extend([
                    {"extracted_data.city": {"$regex": p_esc, "$options": "i"}},
                    {"extracted_data.state": {"$regex": p_esc, "$options": "i"}},
                    {"extracted_data.country": {"$regex": p_esc, "$options": "i"}}
                ])
            if loc_conditions: extra_filters.append({"$or": loc_conditions})
        
        if extra_filters:
            if mongo_filter: mongo_filter = {"$and": [mongo_filter] + extra_filters}
            else: mongo_filter = {"$and": extra_filters} if len(extra_filters) > 1 else extra_filters[0]

    # --- 2. Standard Search Logic (is_boolean=False) ---
    else:
        if min_experience is not None or max_experience is not None:
            exp_filter = {}
            if min_experience is not None: exp_filter["$gte"] = min_experience
            if max_experience is not None: exp_filter["$lte"] = max_experience
            combined_filters.append({"extracted_data.experience": exp_filter})

        if skills:
            skill_queries = [s.strip().lower() for s in skills.split(",") if s.strip()]
            all_conditions = []
            for s_query in skill_queries:
                sub_parts = s_query.split()
                if not sub_parts: continue
                sub_conditions = [{"extracted_data.skills": {"$regex": get_strict_skill_regex(part), "$options": "i"}} for part in sub_parts]
                if len(sub_conditions) > 1: all_conditions.append({"$or": sub_conditions})
                else: all_conditions.append(sub_conditions[0])
            if all_conditions: combined_filters.append({"$and" if match_all else "$or": all_conditions})

        if location and is_valid_location_query(location):
            search_loc_parts = [p.strip() for p in location.split(",")]
            loc_conditions = []
            for part in search_loc_parts:
                p_esc = re.escape(part)
                loc_conditions.extend([
                    {"extracted_data.city": {"$regex": p_esc, "$options": "i"}},
                    {"extracted_data.state": {"$regex": p_esc, "$options": "i"}},
                    {"extracted_data.country": {"$regex": p_esc, "$options": "i"}}
                ])
            if loc_conditions: combined_filters.append({"$or": loc_conditions})

        if job_title:
            title_tokens = re.findall(r'\w+', job_title.lower())
            if title_tokens:
                job_conditions = [{"extracted_data.job_title": {"$regex": re.escape(t), "$options": "i"}} for t in title_tokens]
                combined_filters.append({"$or": job_conditions})

        if combined_filters:
            mongo_filter = {"$and": combined_filters} if len(combined_filters) > 1 else combined_filters[0]

    # --- 3. EXECUTION & ACCURACY PRUNING ---
    all_resumes = await db.db["recruiter's resume"].find(mongo_filter).sort("extracted_data.experience", -1).to_list(length=10000)
    
    scored_results = []
    # Generation text depends on mode
    search_emb_text = query if (is_boolean and query) else f"{job_title or ''} {skills or ''} {location or ''}".strip()
    query_embedding = await embedding_service.generate_embedding(search_emb_text) if search_emb_text else None
    search_loc_parts = [p.strip() for p in location.split(",")] if location else []

    for res in all_resumes:
        raw_text = res.get("raw_content", "")
        
        # Mandatory Strict Validation for Boolean Mode
        if is_boolean and query:
            eval_result = boolean_engine.evaluate_query(query, raw_text)
            if not eval_result.get("matched", False):
                logger.info(f"PRUNED - {res.get('filename')} failed strict boolean evaluation.")
                continue
            
            # Add Debug Fields for Boolean mode
            res["boolean_matched"] = True
            all_keywords = boolean_engine.extract_keywords(query)
            res["matched_terms"] = [kw for kw in all_keywords if boolean_engine.phrase_exists(kw, boolean_engine.preprocess_text(raw_text))]
            res["wildcard_terms"] = [kw for kw in all_keywords if kw.endswith("*")]
            res["boolean_expression"] = eval_result.get("expression")

        res["_id"] = str(res["_id"])
        if res["resumeURL"].startswith("/uploads/"): res["resumeURL"] = f"{settings.BASE_URL}{res['resumeURL']}"
        
        vector_score = 0.0
        if query_embedding and "embedding" in res:
            vector_score = await recruiter_scoring.calculate_vector_score(query_embedding, res["embedding"])
        
        res["semantic_score"] = round(vector_score, 2)
        base_score = recruiter_scoring.apply_location_boost(vector_score, search_loc_parts, res.get("extracted_data", {}))
        res["match_score"] = base_score
        
        # Keyword Frequency Boost
        res["keyword_occurrence_count"] = 0
        if raw_text and (query if is_boolean else job_title):
            raw_lower = raw_text.lower()
            if is_boolean:
                search_terms = boolean_engine.extract_keywords(query)
            else:
                # Use ONLY job_title for frequency boost in Standard Mode
                query_tokens = [t.strip().lower() for t in re.split(r'[,\s/]+', job_title) if len(t.strip()) > 1]
                search_terms = [t for t in query_tokens if t.upper() not in {"DEVELOPER", "ENGINEER", "MANAGER"}]

            total_occ = 0
            for kw in set(search_terms):
                if not kw: continue
                matches = re.findall(rf"\b{re.escape(kw)}\b" if ' ' not in kw else re.escape(kw), raw_lower)
                total_occ += len(matches)
            res["keyword_occurrence_count"] = total_occ
            keyword_boost = float(total_occ * 0.5)
            res["keyword_boost"] = keyword_boost
            res["match_score"] = round(base_score + keyword_boost, 2)
        res["match_score"] = min(res.get("match_score", 0), 99.0)
        
        res.pop("embedding", None)
        res.pop("updated_at", None)
        scored_results.append(res)

    # Sort results
    if is_boolean and query:
        # Use the sorting logic defined specifically in the Boolean search file
        final_list = boolean_engine.sort_results(scored_results)
    elif job_title or skills or location:
        final_list = recruiter_scoring.rank_results(scored_results, job_title, skill_query=skills)
    else:
        final_list = scored_results

    # --- Page-based Pagination Logic ---
    total_count = len(final_list)
    total_pages = ceil(total_count / limit) if total_count > 0 else 1
    start_idx = (current_page - 1) * limit
    paginated_results = final_list[start_idx : start_idx + limit]

    for item in paginated_results:
        item.pop("job_rank_score", None)

    return {
        "metadata": {"total_records": total_count, "total_pages": total_pages, "current_page": current_page, "limit": limit},
        "results": paginated_results
    }
