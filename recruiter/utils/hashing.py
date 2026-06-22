import hashlib
import re

def generate_identity_hash(name: str, email: str) -> str:
    """
    Generates an ultra-strict deterministic hash only for valid resume identity data.
    Prevents invalid resumes from creating hashes like Unknown|None.
    """

    invalid_values = ["unknown", "none", "null", "n/a", "na", ""]

    name_value = str(name).strip().lower() if name else ""
    email_value = str(email).strip().lower() if email else ""

    # Safety guard
    if name_value in invalid_values or email_value in invalid_values:
        raise ValueError("Cannot generate identity hash for invalid resume data")

    # 1. Clean Name: Only alphanumeric characters
    name_clean = re.sub(r'[^a-zA-Z0-9]', '', name_value)

    # 2. Clean Email
    email_clean = email_value

    # 3. Combine with a strict separator
    combined = f"{name_clean}|{email_clean}"

    return hashlib.sha256(combined.encode()).hexdigest()