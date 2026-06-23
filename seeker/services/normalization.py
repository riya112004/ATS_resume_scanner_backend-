import re

class SeekerNormalizer:
    """
    Universal Normalizer.
    Focuses on text cleaning and standardizing common variations.
    Relies on AI for complex semantic mapping.
    """
    
    # Common variations that should always be standardized (Industry Neutral)
    COMMON_MAPPINGS = {
        "mgt": "management",
        "mgmt": "management",
        "comm": "communication",
        "dept": "department",
        "exp": "experience",
        "hr": "human resources",
        "pr": "public relations",
        "admin": "administration"
    }

    def clean_string(self, text: str) -> str:
        if not text: return ""
        s = text.lower().strip()
        # Keep . and # for tech skills like .js or C#
        s = re.sub(r'[^a-z0-9\.\/\+\#]', ' ', s)
        return " ".join(s.split())

    def normalize_skill(self, skill: str) -> str:
        s = self.clean_string(skill)
        return self.COMMON_MAPPINGS.get(s, s)

normalizer = SeekerNormalizer()
