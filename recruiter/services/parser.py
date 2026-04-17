import json
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Optional
from recruiter.core.config import settings
from recruiter.utils.experience_manager import exp_manager
from recruiter.utils.location_manager import loc_manager

class ResumeData(BaseModel):
    name: Optional[str] = "Unknown"
    email: Optional[str] = None
    phone_number: Optional[str] = None
    skills: List[str] = []
    education: List[str] = []
    experience: float = 0.0
    location_raw: Optional[str] = "Unknown"
    city: Optional[str] = ""
    state: Optional[str] = ""
    country: Optional[str] = ""
    postal_code: Optional[str] = ""
    job_title: Optional[str] = "Unknown"

class AIParser:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def parse_resume_text(self, text: str) -> ResumeData:
        """
        Parses resume text. Uses AI for intelligent location classification 
        (City, State, Country) and numeric experience extraction.
        """
        prompt = f"""
        You are an expert resume parsing engine.
        Extract information from the resume text provided below.
        
        STRICT LOCATION DETECTION RULES:
        1. Identify the location mentioned in the resume.
        2. Break it down into:
           - city: The specific city (e.g., "Savannah", "Paris").
           - state: The state or province (e.g., "GA", "Ontario"). Normalize to 2-letter codes for US/Canada if possible.
           - country: The full country name (e.g., "United States", "India").
           - postal_code: Zip or Pin code if found.
        3. If a field is missing, return an empty string.

        STRICT EXPERIENCE EXTRACTION RULES:
        1. Extract total years of experience as a numeric FLOAT value.
        2. "10+ yrs" -> 10.0, "Senior level" -> 10.0, "Junior" -> 2.0.
        
        Return ONLY a JSON object with these EXACT keys:
        - name
        - email
        - phone_number
        - skills (list of strings)
        - education (list of strings)
        - companies (list of objects with: company, start_date, end_date)
        - location_raw (the full original location string)
        - city
        - state
        - country
        - postal_code
        - job_title
        - experience (numeric float)

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
            
            # 2. Experience logic: Prioritize AI numeric extraction
            ai_exp = raw_data.get("experience")
            final_experience = 0.0
            if ai_exp is not None:
                try:
                    final_experience = float(ai_exp)
                except (ValueError, TypeError):
                    final_experience = 0.0
            
            if final_experience <= 0:
                exp_result = exp_manager.calculate_total_experience(raw_data.get("companies", []))
                final_experience = exp_result["decimal"]
            
            if ai_exp and float(ai_exp) >= 10.0:
                final_experience = max(final_experience, float(ai_exp))
            
            # 3. Return clean data with AI-detected Location
            return ResumeData(
                name=raw_data.get("name") or "Unknown",
                email=raw_data.get("email"),
                phone_number=raw_data.get("phone_number"),
                skills=clean_skills, 
                education=raw_data.get("education", []),
                experience=final_experience,
                location_raw=raw_data.get("location_raw") or "Unknown",
                city=raw_data.get("city", ""),
                state=raw_data.get("state", ""),
                country=raw_data.get("country", ""),
                postal_code=raw_data.get("postal_code", ""),
                job_title=raw_data.get("job_title") or "Unknown"
            )
            
        except Exception as e:
            print(f"Error parsing resume: {e}")
            raise ValueError(f"Failed to parse resume with AI: {str(e)}")

parser = AIParser()
