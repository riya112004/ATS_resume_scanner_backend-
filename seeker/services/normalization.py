import re

class SeekerNormalizer:
    # --- Healthcare-Specific Skill Mappings ---
    SKILL_MAPPINGS = {
        # Activities of Daily Living
        "adl": "activities of daily living",
        "adls": "activities of daily living",
        "assisting with daily living activities": "activities of daily living",
        "personal care": "activities of daily living",
        "daily functions": "activities of daily living",
        "assisted patients with daily functions": "activities of daily living",
        "bathing dressing feeding": "activities of daily living",
        
        # Vital Signs
        "monitoring vital signs": "vital signs monitoring",
        "recording vital signs": "vital signs monitoring",
        "vitals": "vital signs monitoring",
        "blood pressure": "vital signs monitoring",
        "temperature check": "vital signs monitoring",
        "temperature pulse blood pressure": "vital signs monitoring",
        "took vital signs": "vital signs monitoring",
        
        # Patient Care & Support
        "patient care": "patient care support",
        "bedside care": "patient care support",
        "patient support": "patient care support",
        "patient assistance": "patient care support",
        "comfort care": "patient care support",
        "patient focused care": "patient care support",
        
        # Emergency & Certifications
        "cpr": "cpr & bls",
        "bls": "cpr & bls",
        "basic life support": "cpr & bls",
        "cardiopulmonary resuscitation": "cpr & bls",
        "aed": "cpr & bls",
        
        # Compliance & Safety
        "hipaa": "hipaa compliance",
        "patient privacy": "hipaa compliance",
        "infection control": "safety & hygiene",
        "sanitization": "safety & hygiene",
        "ppe": "safety & hygiene",
        
        # Clinical Tasks
        "blood sugar testing": "glucose monitoring",
        "blood glucose": "glucose monitoring",
        "mechanical lift": "patient transfer equipment",
        "hoyer lift": "patient transfer equipment",
        "patient transfer": "patient transfer equipment",
        "ambulation": "patient transfer equipment",
        
        # Personal Care
        "hygiene care": "personal hygiene assistance",
        "bathing": "personal hygiene assistance",
        "grooming": "personal hygiene assistance",
        "dressing": "personal hygiene assistance",
        "toileting": "personal hygiene assistance",
        
        # Documentation
        "charting": "patient documentation",
        "electronic health records": "patient documentation",
        "ehr": "patient documentation",
        "emr": "patient documentation",
        "incident reporting": "patient documentation"
    }

    # --- Healthcare Role Normalization ---
    ROLE_MAPPINGS = {
        "cna": "certified nursing assistant",
        "nurse assistant": "certified nursing assistant",
        "nursing assistant": "certified nursing assistant",
        "nursing aide": "certified nursing assistant",
        "patient care assistant": "certified nursing assistant",
        "pca": "certified nursing assistant",
        "pct": "patient care technician",
        "patient care tech": "patient care technician",
        "caregiver": "patient care support",
        "house attendant": "patient care support",
        "ward attendant": "patient care support",
        "home health aide": "patient care support",
        "hha": "patient care support",
        "medical assistant": "medical assistant",
        "ma": "medical assistant"
    }

    def clean_string(self, text: str) -> str:
        if not text: return ""
        s = text.lower().strip()
        s = re.sub(r'[^a-z0-9\.\/\+\#]', ' ', s)
        return " ".join(s.split())

    def normalize_skill(self, skill: str) -> str:
        s = self.clean_string(skill)
        return self.SKILL_MAPPINGS.get(s, s)

    def normalize_role(self, role: str) -> str:
        r = self.clean_string(role)
        for key, standard in self.ROLE_MAPPINGS.items():
            if key in r:
                return standard
        return r

    def normalize_skills_list(self, skills: list) -> list:
        return list(set(self.normalize_skill(s) for s in skills if s))

    def normalize_text(self, text: str) -> str:
        if not text: return ""
        return self.clean_string(text)

normalizer = SeekerNormalizer()
