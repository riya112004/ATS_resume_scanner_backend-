import numpy as np
from typing import List, Dict, Optional

class RecruiterScoringEngine:
    def __init__(self):
        self.LOCATION_BOOST = 1.2
        self.SEMANTIC_WEIGHT = 1.0

    async def calculate_vector_score(self, query_embedding: List[float], resume_embedding: List[float]) -> float:
        if not query_embedding or not resume_embedding:
            return 0.0
        
        a, b = np.array(query_embedding), np.array(resume_embedding)
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(round(similarity * 100, 2))

    def apply_location_boost(self, base_score: float, search_loc_parts: List[str], res_data: Dict) -> float:
        if not search_loc_parts:
            return base_score
            
        res_loc_values = [
            str(res_data.get("city", "")).lower(), 
            str(res_data.get("state", "").lower()), 
            str(res_data.get("country", "").lower())
        ]
        
        if any(p.lower() in res_loc_values for p in search_loc_parts):
            return round(base_score * self.LOCATION_BOOST, 2)
            
        return base_score

    def rank_results(self, results: List[Dict], original_query: Optional[str] = None, skill_query: Optional[str] = None) -> List[Dict]:
        """
        Final MULTI-LEVEL Sort for recruiters:
        1. Most Skills Matched (Descending)
        2. Job Title Match (Highest first)
        3. Experience (Ascending)
        4. Semantic Match Score (Descending)
        """
        target_skills = []
        if skill_query:
            target_skills = [s.strip().lower() for s in skill_query.split(",") if s.strip()]

        for res in results:
            # 1. Count Matched Skills
            matched_count = 0
            resume_skills = [s.lower() for s in res.get("extracted_data", {}).get("skills", [])]
            for ts in target_skills:
                if any(ts in rs for rs in resume_skills):
                    matched_count += 1
            res["skill_match_count"] = matched_count

            # 2. Job Title Score
            title_score = 0
            if original_query:
                q_lower = original_query.lower().strip()
                title = str(res.get("extracted_data", {}).get("job_title", "")).strip().lower()
                if title == q_lower: title_score = 100
                elif q_lower in title: title_score = 80
                else:
                    tokens = q_lower.split()
                    if any(t in title for t in tokens): title_score = 50
                    else: title_score = 20
            res["job_rank_score"] = title_score

        # Multi-level Sort
        results.sort(key=lambda x: (
            -x.get("match_score", 0),
            -x.get("skill_match_count", 0),
            -x.get("job_rank_score", 0), 
            -x.get("extracted_data", {}).get("experience", 0)
        ))
        
        return results

recruiter_scoring = RecruiterScoringEngine()
