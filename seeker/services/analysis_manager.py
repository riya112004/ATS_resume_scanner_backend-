import asyncio
from typing import List
from seeker.services.resume_parser import resume_parser
from seeker.services.jd_parser import jd_parser
from seeker.services.matching_engine import matcher
from seeker.services.scoring_engine import scoring_engine
from seeker.services.feedback import feedback_service
from seeker.models.analysis_schema import AnalysisResult, ScoreBreakdown

from dateutil import parser as date_parser
from datetime import datetime

class AnalysisManager:
    def _get_recent_role(self, work_history) -> str:
        if not work_history:
            return "Unknown"
        
        def parse_end_date(job):
            end_str = str(job.end_date).lower() if job.end_date else ""
            if any(word in end_str for word in ["present", "current", "now", "today"]) or not end_str:
                return datetime.now()
            try:
                return date_parser.parse(end_str)
            except:
                return datetime(1900, 1, 1)

        # Sort by end date descending
        sorted_history = sorted(work_history, key=parse_end_date, reverse=True)
        return sorted_history[0].role

    async def analyze(self, raw_resume_text: str, job_title: str, job_description: str, candidate_experience: float = None) -> dict:
        # Step 1: Parallel Parsing (AI)
        resume_task = resume_parser.parse(raw_resume_text)
        jd_task = jd_parser.parse(job_title, job_description)
        
        parsed_resume, parsed_jd = await asyncio.gather(resume_task, jd_task)

        # --- COMBINED SKILLS LIST ---
        combined_resume_content = parsed_resume.skills + [raw_resume_text]

        if candidate_experience is not None:
            parsed_resume.experience_years = candidate_experience

        # Step 2: Matching Logic
        # 2.1 Hybrid Skill Matching
        must_have_match = await matcher.match_skills_hybrid(combined_resume_content, parsed_jd.must_have_skills)
        pref_have_match = await matcher.match_skills_hybrid(combined_resume_content, parsed_jd.preferred_skills)
        
        # 2.2 Experience Matching
        exp_score = await matcher.match_experience_enhanced(parsed_resume, parsed_jd)
        
        # 2.3 Universal Role Alignment (Using AI Semantic Similarity for ALL roles)
        recent_role = self._get_recent_role(parsed_resume.work_history)
        role_alignment = await matcher.calculate_semantic_similarity(recent_role, parsed_jd.role)
        
        # Boost if the role title is mentioned in the resume (common for all industries)
        if parsed_jd.role.lower() in raw_resume_text.lower():
            role_alignment = max(role_alignment, 85.0)
        
        # 2.4 Project Relevance
        if parsed_resume.projects:
            project_texts = [f"{p.title} {p.description}" for p in parsed_resume.projects]
            project_relevance = await matcher.match_projects(project_texts, job_description)
        else:
            project_relevance = None
        
        # 2.5 Education Match (Actual Logic)
        edu_score = matcher.match_education(parsed_resume.education, parsed_jd.education_requirements)

        # 2.6 Keyword Coverage (Domain terms)
        keyword_cov_score = matcher.calculate_keyword_coverage(raw_resume_text, parsed_jd.domain_keywords)

        # 2.7 Keyword Stuffing Detection
        stuffing_warnings = matcher.detect_keyword_stuffing(raw_resume_text, parsed_jd.must_have_skills)

        # 2.8 Formatting Score
        formatting_score = 100.0 - (len(parsed_resume.parsing_warnings) * 10)
        formatting_score = max(0.0, formatting_score)

        # Step 3: Breakdown & Scoring
        breakdown = ScoreBreakdown(
            skills_match=must_have_match["coverage"],
            experience_relevance=exp_score,
            role_alignment=role_alignment,
            project_relevance=project_relevance,
            education_certifications=edu_score,
            keyword_coverage=keyword_cov_score,
            formatting_readability=formatting_score
        )

        overall = scoring_engine.calculate_overall(
            breakdown, 
            len(must_have_match["missing"]),
            job_title,
            parsed_jd.min_experience
        )
        weak_areas = scoring_engine.identify_weak_areas(breakdown)
        
        # --- Actionable Improvement Points (Single Source of Truth) ---
        improvement_points = feedback_service.generate_improvements(
            parsed_resume, 
            parsed_jd, 
            breakdown, 
            must_have_match["missing"],
            role_alignment
        )

        # Step 4: Warnings Aggregation
        final_warnings = list(parsed_resume.parsing_warnings) + stuffing_warnings
        if not parsed_resume.contact.email: final_warnings.append("Critical: Contact email not detected.")
        if not parsed_resume.contact.phone: final_warnings.append("Phone number missing or unreadable.")
        if not parsed_resume.skills: final_warnings.append("No skills section detected; this significantly hurts ATS rank.")

        # Step 5: Final Construction
        return {
            "overall_ats_score": overall,
            "confidence_score": 0.95,
            "breakdown": breakdown.dict(),
            "matched_skills": must_have_match["matched"],
            "missing_critical_skills": must_have_match["missing"],
            "missing_preferred_skills": pref_have_match["missing"],
            "weak_areas": weak_areas,
            "improvement_points": improvement_points,
            "warnings": final_warnings,
            "verdict": scoring_engine.generate_verdict(overall),
            "parsed_resume": parsed_resume.dict(),
            "parsed_jd": parsed_jd.dict()
        }

analysis_manager = AnalysisManager()
