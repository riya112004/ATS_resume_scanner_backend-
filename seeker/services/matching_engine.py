from seeker.models.analysis_schema import ParsedResume, ParsedJD
from seeker.services.normalization import normalizer
from recruiter.services.embeddings import embedding_service
import numpy as np
import re
from typing import List, Dict

class MatchingEngine:
    def __init__(self):
        self.SEMANTIC_THRESHOLD = 0.75 # Lowered from 0.85 for better flexibility (0.72-0.80 range)

    async def calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b: return 0.0
        try:
            emb_a = await embedding_service.generate_embedding(text_a)
            emb_b = await embedding_service.generate_embedding(text_b)
            
            a, b = np.array(emb_a), np.array(emb_b)
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return float(round(similarity * 100, 2))
        except:
            return 0.0

    async def match_skills_hybrid(self, resume_skills: List[str], jd_skills: List[str]) -> Dict:
        """
        Optimized Multi-Stage Matching:
        1. Exact Match (Raw)
        2. Normalized Exact Match (Rule-based synonyms)
        3. Token Overlap Match (Fuzzy)
        4. Semantic Match (Embeddings) - ONLY for remaining
        """
        if not jd_skills: return {"matched": [], "missing": [], "coverage": 100.0}

        # Stage 1 & 2: Exact and Normalized Matching
        matched_set = set()
        
        # Pre-normalize resume skills for faster lookup
        norm_resume_map = {normalizer.normalize_skill(s): s for s in resume_skills if s}
        norm_jd_skills = [normalizer.normalize_skill(s) for s in jd_skills if s]
        
        final_matched_names = []
        remaining_jd_norm = []
        
        for i, j_norm in enumerate(norm_jd_skills):
            original_jd_skill = jd_skills[i]
            if j_norm in norm_resume_map:
                final_matched_names.append(original_jd_skill)
                matched_set.add(j_norm)
            else:
                remaining_jd_norm.append((j_norm, original_jd_skill))

        # Stage 3: Token Overlap (Fuzzy)
        still_remaining = []
        resume_tokens = set()
        for nr in norm_resume_map.keys():
            resume_tokens.update(nr.split())

        for j_norm, original in remaining_jd_norm:
            j_tokens = set(j_norm.split())
            # If a multi-word skill has high overlap (e.g., "Python Programming" vs "Python")
            if j_tokens.intersection(resume_tokens) and len(j_tokens.intersection(resume_tokens)) / len(j_tokens) >= 0.7:
                final_matched_names.append(original)
                matched_set.add(j_norm)
            else:
                still_remaining.append((j_norm, original))

        # Stage 4: Semantic Match (Only if necessary)
        if not still_remaining or not norm_resume_map:
            missing = [orig for _, orig in still_remaining]
            coverage = (len(final_matched_names) / len(jd_skills)) * 100
            return {"matched": list(set(final_matched_names)), "missing": missing, "coverage": round(coverage, 2)}

        # Only get embeddings for items that reached this stage
        final_missing = []
        
        # Optimization: Batch calculate resume embeddings (only those not already matched)
        resume_embs = []
        unmatched_resume_norms = [nr for nr in norm_resume_map.keys() if nr not in matched_set]
        
        for nr in unmatched_resume_norms:
            try:
                emb = await embedding_service.generate_embedding(nr)
                resume_embs.append(np.array(emb))
            except: continue

        for j_norm, original in still_remaining:
            try:
                j_emb = np.array(await embedding_service.generate_embedding(j_norm))
                is_match = False
                for r_emb in resume_embs:
                    sim = np.dot(j_emb, r_emb) / (np.linalg.norm(j_emb) * np.linalg.norm(r_emb))
                    if sim >= self.SEMANTIC_THRESHOLD:
                        final_matched_names.append(original)
                        is_match = True
                        break
                if not is_match:
                    final_missing.append(original)
            except:
                final_missing.append(original)

        coverage = (len(final_matched_names) / len(jd_skills)) * 100
        
        return {
            "matched": list(set(final_matched_names)),
            "missing": final_missing,
            "coverage": round(coverage, 2)
        }

    async def match_experience_enhanced(self, parsed_resume: ParsedResume, parsed_jd: ParsedJD) -> float:
        qty_score = 0.0
        if parsed_jd.min_experience == 0:
            qty_score = 100.0
        else:
            qty_score = min(100.0, (parsed_resume.experience_years / parsed_jd.min_experience) * 100)

        if not parsed_resume.work_history:
            return round(qty_score * 0.5, 2)

        all_work_text = " ".join([job.description for job in parsed_resume.work_history if job.description])
        relevance_sim = await self.calculate_semantic_similarity(all_work_text, parsed_jd.raw_text)
        
        final_exp_score = (qty_score * 0.4) + (relevance_sim * 0.6)
        return round(final_exp_score, 2)

    def detect_keyword_stuffing(self, resume_text: str, jd_skills: List[str]) -> List[str]:
        warnings = []
        text_lower = resume_text.lower()
        for skill in jd_skills:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 8:
                warnings.append(f"Potential keyword stuffing detected for '{skill}' ({count} occurrences).")
        return warnings

    def match_education(self, resume_edu: List[Dict], jd_edu_req: List[str]) -> float:
        if not jd_edu_req: return 100.0
        if not resume_edu: return 0.0
        hierarchy = {"phd": 4, "masters": 3, "bachelors": 2, "diploma": 1, "high school": 0}
        
        highest_resume = 0
        for edu in resume_edu:
            deg = str(edu.get("degree", "")).lower()
            for k, v in hierarchy.items():
                if k in deg: highest_resume = max(highest_resume, v)
        
        required_level = 0
        for req in jd_edu_req:
            req_l = req.lower()
            for k, v in hierarchy.items():
                if k in req_l: required_level = max(required_level, v)
                    
        if highest_resume >= required_level: return 100.0
        return round((highest_resume / required_level) * 100, 2) if required_level > 0 else 100.0

    def calculate_keyword_coverage(self, resume_text: str, domain_keywords: List[str]) -> float:
        if not domain_keywords: return 100.0
        text = resume_text.lower()
        found = 0
        for kw in domain_keywords:
            if kw.lower() in text: found += 1
        return round((found / len(domain_keywords)) * 100, 2)

    async def match_projects(self, resume_projects: List[str], jd_text: str) -> float:
        if not resume_projects: return 0.0
        combined_projects = " ".join(resume_projects)
        return await self.calculate_semantic_similarity(combined_projects, jd_text)

matcher = MatchingEngine()
