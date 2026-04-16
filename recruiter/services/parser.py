import json
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Optional
from recruiter.core.config import settings
from recruiter.utils.experience_manager import exp_manager

class ResumeData(BaseModel):
    name: Optional[str] = "Unknown"
    email: Optional[str] = None
    phone_number: Optional[str] = None
    skills: List[str] = []
    education: List[str] = []
    experience: float = 0.0
    location: Optional[str] = "Unknown"
    job_title: Optional[str] = "Unknown"

class AIParser:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def parse_resume_text(self, text: str) -> ResumeData:
        """
        Parses resume text. AI extracts raw data, then Python logic calculates 
        total unique experience as a single decimal value.
        """
        prompt = f"""
        You are an expert resume parsing engine.
        Extract information from the resume text provided below.
        
        STRICT EXPERIENCE EXTRACTION:
        1. Identify all work experience entries.
        2. For each, extract: company, start date, and end date.
        3. Normalize dates (e.g., 'Oct 2022').
        
        Return ONLY a JSON object with these EXACT keys:
        - name
        - email
        - phone_number
        - skills (list of strings)
        - education (list of strings)
        - companies (list of objects with: company, start_date, end_date)
        - location
        - job_title

        Resume text:
        {text[:4000]}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            raw_data = json.loads(response.choices[0].message.content)
            
            # 1. Normalize skills
            clean_skills = [s.strip().lower() for s in raw_data.get("skills", []) if s and str(s).strip()]
            
            # 2. Internal calculation logic
            exp_result = exp_manager.calculate_total_experience(raw_data.get("companies", []))
            final_experience = exp_result["decimal"]
            
            # 3. Return clean data
            return ResumeData(
                name=raw_data.get("name") or "Unknown",
                email=raw_data.get("email"),
                phone_number=raw_data.get("phone_number"),
                skills=clean_skills, 
                education=raw_data.get("education", []),
                experience=final_experience,
                location=raw_data.get("location") or "Unknown",
                job_title=raw_data.get("job_title") or "Unknown"
            )
            
        except Exception as e:
            print(f"Error parsing resume: {e}")
            raise ValueError(f"Failed to parse resume with AI: {str(e)}")

parser = AIParser()
