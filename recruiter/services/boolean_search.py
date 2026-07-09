import re
from typing import List, Dict

class BooleanSearchEngine:
    def __init__(self):
        self.operators = {"AND", "OR", "NOT", "NOR", "XOR", "NAND", "XNOR"}

    def preprocess_text(self, text: str) -> str:
        """Standardizes text for strict phrase matching."""
        return " ".join(text.lower().split())

    def tokenize(self, query: str) -> List[str]:
        """Splits query into operators, parentheses, and literal phrases."""
        pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\bNOR\b|\bXOR\b|\bNAND\b|\bXNOR\b|\(|\))'
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p and p.strip()]

    def extract_keywords(self, query: str) -> List[str]:
        """Extracts literal terms for MongoDB fetch."""
        tokens = self.tokenize(query)
        keywords = []
        for t in tokens:
            if t.upper() not in self.operators and t not in ["(", ")"]:
                keywords.append(t.lower())
                if ' ' in t:
                    keywords.append(t.replace(' ', ''))
        return keywords

    def extract_positive_keywords(self, query: str) -> List[str]:
        """Extracts only non-negated literal terms (skips terms after NOT/NOR)."""
        tokens = self.tokenize(query)
        keywords = []
        negated = False
        for t in tokens:
            upper_t = t.upper()
            if upper_t in ("NOT", "NOR"):
                negated = True
                continue
            if upper_t in self.operators or t in ["(", ")"]:
                negated = False
                continue
            if not negated:
                keywords.append(t.lower())
                if ' ' in t:
                    keywords.append(t.replace(' ', ''))
        return keywords

    def phrase_exists(self, phrase: str, text: str) -> bool:
        """Checks for the EXACT phrase or word in the text (with Wildcard support)."""
        phrase = phrase.lower().strip()
        if not phrase: return False
        
        if phrase.endswith("*"):
            root = phrase[:-1]
            if not root: return False
            # Wildcard logic: Match any word starting with the root
            return bool(re.search(rf"\b{re.escape(root)}\w*", text))

        if phrase in text:
            return True

        if ' ' in phrase:
            return phrase.replace(' ', '') in text

        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))

    def evaluate_query(self, query_str: str, resume_text: str) -> Dict:
        """
        Evaluates query with 100% strictness, handling complex parentheses and NOT logic.
        """
        text = self.preprocess_text(resume_text)
        tokens = self.tokenize(query_str)
        if not tokens: return {"matched": False}

        # 1. Evaluate phrases to True/False
        processed_tokens = []
        for i, t in enumerate(tokens):
            upper_t = t.upper()
            if upper_t in self.operators:
                if upper_t == "AND": processed_tokens.append("and")
                elif upper_t == "OR": processed_tokens.append("or")
                elif upper_t == "XOR": processed_tokens.append("^")
                elif upper_t == "NOT":
                    # Smart NOT: Add 'and' if it follows a value or ')'
                    if i > 0 and (processed_tokens[-1] == ")" or processed_tokens[-1] in ["True", "False"]):
                        processed_tokens.append("and")
                    processed_tokens.append("not")
                else:
                    processed_tokens.append(upper_t) # NAND, NOR, XNOR handled by regex
            elif upper_t in ["(", ")"]:
                processed_tokens.append(t)
            else:
                exists = self.phrase_exists(t, text)
                processed_tokens.append(str(exists))

        # 2. Final string formatting
        expr = " ".join(processed_tokens)

        # Advanced Logic Fixes (NAND, NOR, XNOR)
        expr = re.sub(r'(\bTrue\b|\bFalse\b)\s+NAND\s+(\bTrue\b|\bFalse\b)', r'not (\1 and \2)', expr)
        expr = re.sub(r'(\bTrue\b|\bFalse\b)\s+NOR\s+(\bTrue\b|\bFalse\b)', r'not (\1 or \2)', expr)
        expr = re.sub(r'(\bTrue\b|\bFalse\b)\s+XNOR\s+(\bTrue\b|\bFalse\b)', r'(\1 == \2)', expr)

        # 3. Group OR clauses so AND terms remain globally required.
        #    "True and False and True or True" → "True and False and (True or True)"
        #    Skips if the term before OR is negated by NOT.
        parts = expr.split()
        i = 0
        while i < len(parts) - 2:
            if parts[i] in ("True", "False") and parts[i+1] == "or" and parts[i+2] in ("True", "False"):
                if i >= 1 and parts[i-1] == "not":
                    i += 1
                    continue
                parts[i] = f"({parts[i]}"
                parts[i+2] = f"{parts[i+2]})"
                i += 3
            else:
                i += 1
        expr = " ".join(parts)

        try:
            result = eval(expr)
            return {"matched": bool(result), "expression": expr}
        except Exception as e:
            print(f"Boolean Eval Error: {e} for expression: {expr}")
            return {"matched": False, "error": str(e)}

    def sort_results(self, results: List[Dict]) -> List[Dict]:
        """Sorts a list of candidate results by match_score in descending order."""
        return sorted(results, key=lambda x: x.get("match_score", 0), reverse=True)

boolean_engine = BooleanSearchEngine()
