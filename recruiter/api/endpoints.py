import os
import uuid
import numpy as np
import re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import List, Optional
from recruiter.core.config import settings
from recruiter.core.database import db
from recruiter.utils.extractor import extract_text_from_file
from recruiter.services.parser import parser
from recruiter.services.embeddings import embedding_service
from recruiter.services.matching import calculate_match_score

router = APIRouter()

# --- HELPERS ---

def get_strict_skill_regex(skill: str):
    """
    Creates a strict regex that matches the word ONLY if it is:
    - At the start or preceded by space/slash
    - At the end or followed by space/slash
    This handles "C" vs "C++" and "MongoDb/Sanity" correctly.
    """
    escaped = re.escape(skill.lower())
    return f"(^|[ /]){escaped}($|[ /])"

def normalize_val(val: str) -> str:
    """Trims and lowercases a string."""
    if not val: return ""
    return str(val).strip().lower()

def is_valid_location_query(query: str) -> bool:
    """
    Validates if the location query is searchable:
    - MUST be at least 3 letters.
    - MUST NOT be a junk word (unknown, n/a, etc).
    """
    if not query: return False
    clean_q = normalize_val(query)
    junk = ["unknown", "n/a", "none", "undefined", "null", "na", "n / a"]
    
    # STRICT RULE: Mandatory 3 letters for search to trigger
    if len(clean_q) < 3 or clean_q in junk:
        return False
    return True

@router.post("/upload")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = Form(None)
):
    results = []
    for file in files:
        # Reset variables for each file to prevent data carry-over
        raw_text = ""
        parsed_data = None
        embedding = []
        
        # 1. Save file locally
        file_id = str(uuid.uuid4())
        _, ext = os.path.splitext(file.filename)
        file_name = f"{file_id}{ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)
        
        print(f"Processing file: {file.filename}")
        
        try:
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            print(f"File saved to: {file_path}")
        except Exception as e:
            print(f"Save error: {e}")
            results.append({"filename": file.filename, "error": f"Could not save file: {str(e)}"})
            continue
        
        # 2. Extract text from file
        try:
            raw_text = await extract_text_from_file(file_path)
            print(f"Extracted text length: {len(raw_text) if raw_text else 0}")
            if not raw_text or len(raw_text.strip()) < 10:
                raise ValueError("Extracted text is too short or empty. Is this a scanned image?")
        except Exception as e:
            print(f"Extraction error: {e}")
            if os.path.exists(file_path): os.remove(file_path)
            results.append({"filename": file.filename, "error": f"Text extraction failed: {str(e)}"})
            continue
        
        # 3. Parse resume with AI
        try:
            parsed_data = await parser.parse_resume_text(raw_text)
            print(f"AI Parsed successfully: {parsed_data.name}")
        except Exception as e:
            print(f"AI Parsing error: {e}")
            if os.path.exists(file_path): os.remove(file_path)
            results.append({"filename": file.filename, "error": f"AI Parsing failed: {str(e)}"})
            continue
        
        # 4. Generate embeddings from the parsed data
        try:
            parsed_text_for_embedding = f"""
            Candidate Name: {parsed_data.name}
            Job Title: {parsed_data.job_title}
            Skills: {', '.join(parsed_data.skills)}
            Experience: {parsed_data.experience} years
            Location: {parsed_data.location}
            """
            embedding = await embedding_service.generate_embedding(parsed_text_for_embedding.strip())
        except Exception as e:
            if os.path.exists(file_path): os.remove(file_path)
            results.append({"filename": file.filename, "error": f"Embedding generation failed: {str(e)}"})
            continue
        
        # 5. Calculate match score if job description is provided
        match_score = 0.0
        if job_description:
            match_score = await calculate_match_score(raw_text, job_description)
        
        # 6. Prepare document for MongoDB
        full_resume_url = f"{settings.BASE_URL}/uploads/{file_name}"
        resume_document = {
            "resumeURL": full_resume_url,
            "extracted_data": parsed_data.dict(),
            "embedding": embedding
        }
        
        try:
            # Insert into DB
            await db.db["resumes"].insert_one(resume_document)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "match_score": match_score,
                "resumeURL": full_resume_url
            })
        except Exception as e:
            if os.path.exists(file_path): os.remove(file_path)
            results.append({"filename": file.filename, "error": f"Database insertion failed: {str(e)}"})
            
    return results

@router.get("/search")
async def search_resumes(
    min_experience: Optional[float] = None,
    max_experience: Optional[float] = None,
    location: Optional[str] = None,
    skills: Optional[str] = None,
    education: Optional[str] = None,
    job_title: Optional[str] = None,
    match_all: bool = Query(False, description="If true, resume must contain ALL searched skills"),
    limit: int = 10
):
    # 1. Construct MongoDB Filter
    mongo_filter = {}
    
    # Strict Experience Range Filter
    if min_experience is not None or max_experience is not None:
        mongo_filter["extracted_data.experience"] = {}
        if min_experience is not None:
            mongo_filter["extracted_data.experience"]["$gte"] = min_experience
        if max_experience is not None:
            mongo_filter["extracted_data.experience"]["$lte"] = max_experience

    # STRICT Skills Filter (Sanitized & Regex Boundary)
    skill_condition = None
    if skills:
        # Split, trim, lowercase, and remove empty tokens
        skill_list = [s.strip().lower() for s in skills.split(",") if s.strip()]
        
        if skill_list:
            # Create regex conditions for each skill using strict boundaries
            conditions = [
                {"extracted_data.skills": {"$regex": get_strict_skill_regex(s), "$options": "i"}} 
                for s in skill_list
            ]
            
            # Logic: Match ALL ($and) vs Match ANY ($or)
            skill_condition = {"$and" if match_all else "$or": conditions}

    # INTELLIGENT Location Filter (Strict 3-letter rule)
    location_condition = None
    if location and is_valid_location_query(location):
        clean_loc = normalize_val(location)
        # Handle partial words for better matching (e.g. "west ben" -> "west|ben")
        words = [re.escape(w) for w in clean_loc.split() if len(w) >= 2]
        if words:
            loc_pattern = "|".join(words)
            location_condition = {
                "extracted_data.location": {
                    "$regex": loc_pattern, 
                    "$options": "i",
                    # STRICT: Exclude junk values from results
                    "$nin": ["unknown", "n/a", "", None, "undefined", "N/A", "Unknown", "n / a"]
                }
            }

    # Combine Filters into MongoDB Query
    combined_filters = []
    if skill_condition: combined_filters.append(skill_condition)
    if location_condition: combined_filters.append(location_condition)
    
    if combined_filters:
        if len(combined_filters) > 1:
            mongo_filter["$and"] = combined_filters
        else:
            # Only one filter is active
            mongo_filter.update(combined_filters[0])

    # 2. Fetch Filtered Resumes
    all_resumes = await db.db["resumes"].find(mongo_filter).to_list(length=100)

    # 3. Construct Query for AI Ranking
    rank_parts = []
    if job_title: rank_parts.append(f"Job Title: {job_title}")
    if skills: rank_parts.append(f"Skills: {skills}")
    if location: rank_parts.append(f"Location: {location}")
    
    search_query = " ".join(rank_parts)
    
    if not search_query or not all_resumes:
        # Fallback if no ranking criteria or no results
        results = []
        for res in all_resumes[:limit]:
            res["_id"] = str(res["_id"])
            res.pop("embedding", None)
            res["match_score"] = 0.0
            results.append(res)
        return results

    # 4. Perform AI Semantic Ranking on Filtered Results
    try:
        query_embedding = await embedding_service.generate_embedding(search_query)
        scored_results = []
        
        for res in all_resumes:
            if "embedding" in res:
                # Calculate overall similarity between search profile and resume profile
                a = np.array(query_embedding)
                b = np.array(res["embedding"])
                similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                
                res["_id"] = str(res["_id"])
                # Handle absolute resumeURL
                if res["resumeURL"].startswith("/uploads/"):
                    res["resumeURL"] = f"{settings.BASE_URL}{res['resumeURL']}"
                
                res.pop("embedding", None)
                # Convert to percentage
                res["match_score"] = float(round(similarity * 100, 2))
                scored_results.append(res)
        
        # Sort by overall match score descending
        scored_results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return scored_results[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")
