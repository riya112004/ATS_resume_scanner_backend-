import json
from datetime import datetime
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
        Parses resume text. Uses AI for extraction with dynamic date context, 
        and Python logic for strict mathematical experience calculation.
        """
        # Dynamically get today's month and year
        today_str = datetime.now().strftime("%B %Y")
        
        prompt = f"""
        You are an expert resume parsing engine. TODAY'S DATE IS {today_str}.
        Extract information from the resume text provided below.
        
        STRICT EXPERIENCE RULES:
        1. Identify ALL work experience entries.
        2. For each, extract: company name, start date, and end date.
        3. If an end date is 'Present', 'Current', or 'Now', use '{today_str}'.
        4. Also provide a direct "ai_suggested_exp" value IF explicitly stated (e.g. "10+ years").
        
        LOCATION DETECTION:
        - Break location into city, state, and country fields.

        Return ONLY a JSON object with these EXACT keys:
        - name
        - email
        - phone_number
        - skills (list of strings)
        - education (list of strings)
        - companies (list of objects with: company, start_date, end_date)
        - location_raw
        - city
        - state
        - country
        - job_title
        - ai_suggested_exp (numeric float or null)

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
            
            # 2. ADVANCED EXPERIENCE LOGIC (Double Layer)
            # A. Calculate strictly from dates using Python (Reliable for 1yr 11mo cases)
            exp_result = exp_manager.calculate_total_experience(raw_data.get("companies", []))
            calculated_exp = exp_result["decimal"]
            
            # B. Check for AI suggested "10+" style values
            ai_val = raw_data.get("ai_suggested_exp")
            final_experience = calculated_exp
            
            if ai_val is not None:
                try:
                    # If AI explicitly found a high number (like 10+), trust it over date math
                    if float(ai_val) > calculated_exp:
                        final_experience = float(ai_val)
                except (ValueError, TypeError):
                    pass
            
            # 3. Return clean data
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
                job_title=raw_data.get("job_title") or "Unknown"
            )
            
        except Exception as e:
            print(f"Error parsing resume: {e}")
            raise ValueError(f"Failed to parse resume with AI: {str(e)}")

parser = AIParser()
