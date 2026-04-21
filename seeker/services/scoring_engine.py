from seeker.models.analysis_schema import ScoreBreakdown
from typing import List, Dict

class ScoringEngine:
    # --- Weight Profiles based on Role Context ---
    PROFILES = {
        "STANDARD": {
            "skills": 0.30, "experience": 0.25, "role": 0.10, "projects": 0.15, 
            "education": 0.05, "keywords": 0.10, "formatting": 0.05
        },
        "HEALTHCARE_OPS": {
            "skills": 0.40, "experience": 0.35, "role": 0.10, "projects": 0.0, # Projects removed
            "education": 0.05, "keywords": 0.05, "formatting": 0.05
        },
        "TECH_SENIOR": {
            "skills": 0.25, "experience": 0.30, "role": 0.10, "projects": 0.25, # High project weight
            "education": 0.02, "keywords": 0.05, "formatting": 0.03
        },
        "ENTRY_LEVEL": {
            "skills": 0.40, "experience": 0.10, "role": 0.05, "projects": 0.20, 
            "education": 0.10, "keywords": 0.10, "formatting": 0.05
        }
    }

    # --- Penalty Config ---
    CRITICAL_SKILL_PENALTY = 2.0

    def _detect_profile(self, job_title: str, min_exp: float) -> str:
        title = job_title.lower()
        
        # 1. Healthcare / Support / Ops
        healthcare_keywords = ["nurse", "cna", "care", "medical", "attendant", "support", "ops", "operation"]
        if any(k in title for k in healthcare_keywords):
            return "HEALTHCARE_OPS"
        
        # 2. Senior Tech Roles
        senior_keywords = ["senior", "lead", "architect", "manager", "principal", "sr."]
        if any(k in title for k in senior_keywords) and min_exp >= 5:
            return "TECH_SENIOR"
        
        # 3. Entry Level
        entry_keywords = ["junior", "intern", "trainee", "entry", "fresher", "jr."]
        if any(k in title for k in entry_keywords) or min_exp <= 1:
            return "ENTRY_LEVEL"
        
        return "STANDARD"

    def calculate_overall(self, breakdown: ScoreBreakdown, missing_critical_count: int, job_title: str, min_exp: float) -> float:
        """
        Calculates weighted sum using a context-aware profile.
        Dynamic Handling: If projects are None, their weight is set to 0.
        """
        profile_name = self._detect_profile(job_title, min_exp)
        w = self.PROFILES[profile_name].copy()
        
        # --- DYNAMIC PROJECT LOGIC ---
        project_relevance = 0.0
        project_bonus = 0.0

        if breakdown.project_relevance is None:
            # Case 1: No projects in resume
            w["projects"] = 0.0
            project_relevance = 0.0
        else:
            # Case 2: Projects exist
            project_relevance = breakdown.project_relevance
            project_bonus = 3.0 # Small incentive boost for having projects
        
        # Calculate Weighted Score
        base_score = (
            (breakdown.skills_match * w.get("skills", 0)) +
            (breakdown.experience_relevance * w.get("experience", 0)) +
            (breakdown.role_alignment * w.get("role", 0)) +
            (project_relevance * w.get("projects", 0)) +
            (breakdown.education_certifications * w.get("education", 0)) +
            (breakdown.keyword_coverage * w.get("keywords", 0)) +
            (breakdown.formatting_readability * w.get("formatting", 0))
        )
        
        # Apply Explicit Penalty for missing Must-Have skills
        penalty = missing_critical_count * self.CRITICAL_SKILL_PENALTY
        
        # Final Score with Bonus
        final_score = base_score - penalty + project_bonus
        
        return float(max(0.0, min(100.0, round(final_score, 1))))

    def identify_weak_areas(self, breakdown: ScoreBreakdown) -> List[str]:
        weak = []
        if breakdown.skills_match < 65: weak.append("Technical Skills Alignment")
        if breakdown.experience_relevance < 70: weak.append("Relevant Work History")
        
        # FIX: Only check if project_relevance is not None
        if breakdown.project_relevance is not None and breakdown.project_relevance < 50:
            weak.append("Project/Portfolio Highlights")
            
        if breakdown.keyword_coverage < 50: weak.append("Industry Keyword Density")
        if breakdown.formatting_readability < 80: weak.append("ATS-Friendly Formatting")
        return weak

    def generate_verdict(self, score: float) -> str:
        if score >= 85: return "Excellent match! Your resume is highly optimized for this role."
        if score >= 70: return "Good match. Consider adding missing keywords to reach the top tier."
        if score >= 50: return "Average match. Significant gaps detected in skills or experience."
        return "Weak match. Major alignment needed between your resume and the job requirements."

scoring_engine = ScoringEngine()
