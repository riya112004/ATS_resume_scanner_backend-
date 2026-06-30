import json
from typing import List
from openai import AsyncOpenAI
from seeker.models.analysis_schema import ParsedJD
from recruiter.core.config import settings

class AIDrivenJDParser:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def parse(self, title: str, description: str) -> ParsedJD:
        prompt = f"""You are an expert job description parser. Extract structured information from the job description below.

Job Title: {title}
Job Description: {description}

Return ONLY a JSON object with these EXACT keys:
- role: the job title
- must_have_skills: list of critical/required technical and soft skills (max 15)
- preferred_skills: list of nice-to-have skills (max 10)
- min_experience: minimum years of experience required (as a number, 0 if not specified)
- education_requirements: list of required education qualifications (empty list if none specified)
- domain_keywords: list of 5-10 industry-specific keywords/terms from the JD

Requirements:
- must_have_skills should contain skills explicitly mentioned as required
- preferred_skills should contain skills mentioned as "nice to have" or "preferred" or "plus"
- Extract real technical skills (languages, frameworks, tools, platforms) not generic terms
- Include both hard skills and relevant soft skills
- Do NOT include generic words like "teamwork" or "communication" unless they are the ONLY skills mentioned
"""

        response = await self.client.chat.completions.create(
            model=settings.AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw = json.loads(response.choices[0].message.content)

        return ParsedJD(
            role=raw.get("role", title),
            must_have_skills=raw.get("must_have_skills", []),
            preferred_skills=raw.get("preferred_skills", []),
            min_experience=float(raw.get("min_experience", 0)),
            education_requirements=raw.get("education_requirements", []),
            domain_keywords=raw.get("domain_keywords", ["General"]),
            raw_text=description
        )

jd_parser = AIDrivenJDParser()
