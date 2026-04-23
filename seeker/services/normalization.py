import re

class SeekerNormalizer:
    # --- HYBRID SKILL MAPPINGS (Healthcare + Tech) ---
    SKILL_MAPPINGS = {
        # --- Tech Skills ---
        "react": "react.js",
        "reactjs": "react.js",
        "react.js": "react.js",
        "nextjs": "next.js",
        "next.js": "next.js",
        "node": "node.js",
        "nodejs": "node.js",
        "node.js": "node.js",
        "express": "express.js",
        "expressjs": "express.js",
        "express.js": "express.js",
        "mongo": "mongodb",
        "mongodb": "mongodb",
        "js": "javascript",
        "javascript": "javascript",
        "ts": "typescript",
        "typescript": "typescript",
        "redux": "redux",
        "tailwind": "tailwind css",
        "rest": "restful api",
        "restful api": "restful api",
        "rest api": "restful api",
        "aws": "amazon web services",
        "git": "git",
        "github": "git",
        "mysql": "mysql",
        "postgres": "postgresql",
        "postgresql": "postgresql",

        # --- Healthcare Skills ---
        "adl": "activities of daily living",
        "adls": "activities of daily living",
        "assisting with daily living activities": "activities of daily living",
        "personal care": "activities of daily living",
        "daily functions": "activities of daily living",
        "monitoring vital signs": "vital signs monitoring",
        "recording vital signs": "vital signs monitoring",
        "vitals": "vital signs monitoring",
        "blood pressure": "vital signs monitoring",
        "patient care": "patient care support",
        "bedside care": "patient care support",
        "hipaa": "hipaa compliance",
        "cpr": "cpr & bls",
        "bls": "cpr & bls"
    }

    # --- HYBRID ROLE MAPPINGS ---
    ROLE_MAPPINGS = {
        # Tech
        "fullstack": "full stack developer",
        "frontend": "frontend developer",
        "backend": "backend developer",
        "sde": "software engineer",
        "mern": "full stack developer",
        "web developer": "software engineer",
        # Healthcare
        "cna": "certified nursing assistant",
        "nurse assistant": "certified nursing assistant",
        "pca": "certified nursing assistant",
        "ma": "medical assistant"
    }

    def clean_string(self, text: str) -> str:
        if not text: return ""
        s = text.lower().strip()
        # Keep . and # for tech skills like .js or C#
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
