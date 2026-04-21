import json
from datetime import datetime
from openai import AsyncOpenAI
from recruiter.core.config import settings
from seeker.models.analysis_schema import ParsedResume, ContactInfo, WorkExperience
from seeker.utils.experience_manager import seeker_exp_manager

class SeekerResumeParser:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def parse(self, text: str) -> ParsedResume:
        today_str = datetime.now().strftime("%B %Y")
        
        # Increase limit slightly but remain context-efficient
        # Most resumes fit in 8000-10000 chars unless extremely verbose
        resume_content = text[:8000]

        prompt = f"""
        You are an expert ATS (Applicant Tracking System) parser specialized in HEALTHCARE. TODAY'S DATE IS {today_str}.
        Extract all structured data from the resume text provided.
        
        IMPORTANT - IMPLIED SKILLS EXTRACTION:
        Also extract IMPLIED SKILLS from work experience descriptions.
        Examples:
        - "Took vital signs, blood pressure" → include "monitoring vital signs" in skills.
        - "Assisted patients with bathing, dressing, grooming" → include "activities of daily living" in skills.
        - "Transferred patients using Hoyer lift" → include "patient transfer equipment" in skills.
        - "Maintained patient charts" → include "patient documentation" in skills.
        - "Followed strict PPE and sanitization protocols" → include "safety & hygiene" in skills.

        Return ONLY a JSON object with these keys:
        - name
        - email
        - phone
        - links (list of URLs)
        - skills (comprehensive list of technical, clinical, and IMPLIED skills)
        - work_history (list of objects: company, role, start_date, end_date, description)
        - education (list of objects: degree, field, institution)
        - projects (list of project titles and descriptions)
        - certifications (list of professional certifications)
        - formatting_issues (list of issues detected)
        - parse_confidence (float 0.0 to 1.0)

        Resume text:
        {resume_content}
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            raw = json.loads(response.choices[0].message.content)
            
            # 1. Check Confidence
            confidence = raw.get("parse_confidence", 0.0)
            if confidence < 0.4:
                raise ValueError(f"Low parsing confidence ({confidence}). The resume might be unreadable or improperly formatted.")

            # 2. Experience Calculation
            exp_entries = []
            for job in raw.get("work_history", []):
                exp_entries.append({
                    "company": job.get("company"),
                    "start_date": job.get("start_date"),
                    "end_date": job.get("end_date")
                })
            
            exp_result = seeker_exp_manager.calculate_total_experience(exp_entries)

            # 3. Map to Model
            return ParsedResume(
                contact=ContactInfo(
                    name=raw.get("name", "Unknown"),
                    email=raw.get("email"),
                    phone=raw.get("phone"),
                    links=raw.get("links", [])
                ),
                skills=[s.lower().strip() for s in raw.get("skills", []) if s],
                experience_years=exp_result["decimal"],
                work_history=[WorkExperience(**job) for job in raw.get("work_history", [])],
                education=raw.get("education", []),
                projects=raw.get("projects", []),
                certifications=raw.get("certifications", []),
                raw_text=text,
                parsing_warnings=raw.get("formatting_issues", [])
            )
        except Exception as e:
            # Re-raise the exception to be caught by the analysis manager
            raise ValueError(f"Resume Parsing Failed: {str(e)}")

resume_parser = SeekerResumeParser()
