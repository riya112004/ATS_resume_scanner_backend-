import re
from typing import List
from seeker.models.analysis_schema import ParsedJD
from seeker.services.normalization import normalizer

class LocalJDParser:
    """
    Hybrid Local JD Parser. 
    Handles both Healthcare and Tech skills using local logic.
    """
    
    def extract_experience(self, text: str) -> float:
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
        found_skills = []
        text_lower = f" {text.lower()} "
        
        # Check against common standardized mappings
        for skill_key in normalizer.COMMON_MAPPINGS.keys():
            if re.search(rf'\b{re.escape(skill_key)}\b', text_lower):
                found_skills.append(normalizer.COMMON_MAPPINGS[skill_key])
        
        # In general mode, we rely more on the AI parser for JD skills, 
        # but for local parsing, we'll extract capitalized terms as potential skills
        # if they aren't common words.
        return list(set(found_skills))

    async def parse(self, title: str, description: str) -> ParsedJD:
        skills = self.extract_skills(description)
        exp = self.extract_experience(description)
        
        return ParsedJD(
            role=title,
            must_have_skills=skills[:10],
            preferred_skills=skills[10:15],
            min_experience=exp,
            education_requirements=["As specified in JD"],
            domain_keywords=["General"],
            raw_text=description
        )

jd_parser = LocalJDParser()
