import hashlib
import re

def generate_identity_hash(name: str, email: str) -> str:
    """
    Generates an ultra-strict deterministic hash.
    Removes ALL special characters, dots, spaces and symbols.
    """
    # 1. Clean Name: Only alphanumeric characters (a-z, 0-9)
    name_clean = re.sub(r'[^a-zA-Z0-9]', '', str(name).lower())
    
    # 2. Clean Email: Standardize lowercase and trim
    email_clean = str(email).strip().lower()
    
    # 3. Combine with a strict separator
    combined = f"{name_clean}|{email_clean}"
    
    return hashlib.sha256(combined.encode()).hexdigest()
