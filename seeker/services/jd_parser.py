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
        
        # Check against ALL mappings (Tech + Healthcare)
        for skill_key in normalizer.SKILL_MAPPINGS.keys():
            # Match word with boundaries to avoid sub-word matching
            if re.search(rf'\b{re.escape(skill_key)}\b', text_lower):
                found_skills.append(normalizer.SKILL_MAPPINGS[skill_key])
        
        # Add some direct tech keywords that might not have synonyms
        tech_keywords = ["html", "css", "java", "python", "c++", "c#", "php", "angular", "vue", "docker", "kubernetes"]
        for tech in tech_keywords:
            if re.search(rf'\b{re.escape(tech)}\b', text_lower):
                found_skills.append(tech)
                
        return list(set(found_skills))

    async def parse(self, title: str, description: str) -> ParsedJD:
        skills = self.extract_skills(description)
        exp = self.extract_experience(description)
        
        return ParsedJD(
            role=title,
            must_have_skills=skills[:10],
            preferred_skills=skills[10:15],
            min_experience=exp,
            education_requirements=["Computer Science" if "computer" in description.lower() else "Any"],
            domain_keywords=["Tech" if any(s in description.lower() for s in ["react", "node", "java"]) else "Healthcare"],
            raw_text=description
        )

jd_parser = LocalJDParser()
