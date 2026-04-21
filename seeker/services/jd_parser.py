import json
from openai import AsyncOpenAI
from recruiter.core.config import settings
from seeker.models.analysis_schema import ParsedJD

class JDParser:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def parse(self, title: str, description: str) -> ParsedJD:
        prompt = f"""
        Analyze the following Job Title and Job Description.
        Extract hiring signals and structure them. 
        CLEANING RULE: Remove redundant phrases like "passionate about", "competitive salary", or company history. Focus ONLY on candidate requirements.

        Return ONLY a JSON object with these keys:
        - role (standardized job title)
        - must_have_skills (critical hard skills/languages/frameworks)
        - preferred_skills (plus points or nice-to-have)
        - tools_and_tech (list of specific softwares/tools: e.g. 'Docker', 'AWS', 'Jira')
        - soft_skills (e.g. 'Leadership', 'Communication')
        - responsibilities (list of core job duties)
        - min_experience (numeric years, 0 if not specified)
        - education_requirements (list of degrees: e.g., 'Bachelors', 'Masters')
        - domain_keywords (core industry terms: e.g., 'Fintech', 'SaaS', 'Healthcare')

        Job Title: {title}
        Job Description: {description}
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            raw = json.loads(response.choices[0].message.content)
            
            # Combine hard skills and tools for a broader comparison base if needed
            must_haves = [s.lower().strip() for s in raw.get("must_have_skills", []) if s]
            
            return ParsedJD(
                role=raw.get("role") or title,
                must_have_skills=must_haves,
                preferred_skills=[s.lower().strip() for s in raw.get("preferred_skills", []) if s],
                min_experience=float(raw.get("min_experience", 0)),
                education_requirements=raw.get("education_requirements", []),
                domain_keywords=[k.lower().strip() for k in raw.get("domain_keywords", []) if k],
                raw_text=description
            )
        except Exception as e:
            # We raise a ValueError with a specific prefix so endpoint can catch and map to 422
            raise ValueError(f"JD_STRUCTURE_ERROR: {str(e)}")

jd_parser = JDParser()
