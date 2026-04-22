import re
from typing import List
from seeker.models.analysis_schema import ParsedJD
from seeker.services.normalization import normalizer

class LocalJDParser:
    """
    100% FREE Local JD Parser. 
    Uses Regex and Healthcare Mappings instead of OpenAI.
    """
    
    def extract_experience(self, text: str) -> float:
        # Regex to find patterns like "3+ years", "5 years", "2-4 years"
        patterns = [
            r'(\d+)\+?\s*(?:year|yr)s?',
            r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:year|yr)s?'
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        return 0.0

    def extract_skills(self, text: str) -> List[str]:
        # Scan JD text against our healthcare skill dictionary
        found_skills = []
        text_lower = text.lower()
        
        # Use existing healthcare mappings from normalizer
        for skill_key in normalizer.SKILL_MAPPINGS.keys():
            if f" {skill_key} " in f" {text_lower} " or text_lower.startswith(skill_key):
                found_skills.append(normalizer.SKILL_MAPPINGS[skill_key])
        
        return list(set(found_skills))

    async def parse(self, title: str, description: str) -> ParsedJD:
        # Extract basic info locally
        skills = self.extract_skills(description)
        exp = self.extract_experience(description)
        
        # Build ParsedJD object
        return ParsedJD(
            role=title,
            must_have_skills=skills[:8], # Top 8 skills
            preferred_skills=skills[8:12],
            min_experience=exp,
            education_requirements=["Associate" if "associate" in description.lower() else "Bachelors"],
            domain_keywords=["Healthcare"],
            raw_text=description
        )

jd_parser = LocalJDParser()
