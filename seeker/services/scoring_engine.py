from seeker.models.analysis_schema import ScoreBreakdown
from typing import List

class ScoringEngine:
    # --- Universal Weight Profile (Fair for all industries) ---
    UNIVERSAL_WEIGHTS = {
        "skills": 0.35,        # Technical & Soft skills
        "experience": 0.30,    # Relevant work history
        "role": 0.10,          # Job title alignment
        "projects": 0.10,      # Portfolio & projects
        "education": 0.05,     # Degrees & Certifications
        "keywords": 0.05,      # Domain-specific terms
        "formatting": 0.05     # ATS readability
    }

    # --- Penalty Config ---
    CRITICAL_SKILL_PENALTY = 2.0

    def calculate_overall(self, breakdown: ScoreBreakdown, missing_critical_count: int, job_title: str, min_exp: float) -> float:
        """
        Calculates weighted sum using a Universal profile for all jobs.
        """
        w = self.UNIVERSAL_WEIGHTS.copy()
        
        # --- DYNAMIC PROJECT LOGIC ---
        project_relevance = 0.0
        project_bonus = 0.0

        if breakdown.project_relevance is None:
            # If no projects section exists, distribute its weight to Skills and Experience
            w["skills"] += 0.05
            w["experience"] += 0.05
            w["projects"] = 0.0
            project_relevance = 0.0
        else:
            project_relevance = breakdown.project_relevance
            project_bonus = 2.0 # Standard incentive for having projects
        
        # Calculate Weighted Score
        base_score = (
            (breakdown.skills_match * w["skills"]) +
            (breakdown.experience_relevance * w["experience"]) +
            (breakdown.role_alignment * w["role"]) +
            (project_relevance * w["projects"]) +
            (breakdown.education_certifications * w["education"]) +
            (breakdown.keyword_coverage * w["keywords"]) +
            (breakdown.formatting_readability * w["formatting"])
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
