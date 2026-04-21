from typing import List, Dict
from seeker.models.analysis_schema import ParsedResume, ParsedJD, ScoreBreakdown

class FeedbackService:
    def generate_improvements(
        self, 
        parsed_resume: ParsedResume, 
        parsed_jd: ParsedJD, 
        breakdown: ScoreBreakdown,
        missing_skills: List[str],
        role_alignment: float
    ) -> List[str]:
        """
        Generates actionable improvement points based on the analysis results.
        Single source of truth for all seeker feedback.
        """
        improvements = []
        
        # 1. Skill Gaps
        if missing_skills:
            top_missing = [s.title() for s in missing_skills[:5]]
            improvements.append(f"Missing Keywords: Add {', '.join(top_missing)} to your skills section.")
        
        # 2. Experience Optimization
        if parsed_resume.experience_years < parsed_jd.min_experience:
            gap = round(parsed_jd.min_experience - parsed_resume.experience_years, 1)
            improvements.append(f"Experience Gap: The JD asks for {parsed_jd.min_experience} years, but you have {parsed_resume.experience_years} years. Highlight transferable skills to bridge this {gap} year gap.")
        
        # 3. Role & Project Alignment
        if role_alignment < 70:
            improvements.append(f"Role Alignment: Your recent experience as a '{parsed_resume.work_history[0].role if parsed_resume.work_history else 'Unknown'}' doesn't strongly reflect a '{parsed_jd.role}' role. Update your professional summary to use this title.")
        
        if parsed_resume.projects and breakdown.project_relevance is not None and breakdown.project_relevance < 60:
            improvements.append("Optimization Tip: Use more bullet points in your project descriptions that quantify achievements (e.g., 'Reduced costs by 20%').")

        # 4. Domain Keywords
        if breakdown.keyword_coverage < 50:
             improvements.append("Domain Optimization: Incorporate more industry-specific terminology found in the Job Description.")

        # 5. ATS Readability
        if breakdown.formatting_readability < 80:
             improvements.append("Formatting Tip: Ensure your resume uses a single-column layout and standard font to avoid ATS parsing errors.")

        return improvements

feedback_service = FeedbackService()
