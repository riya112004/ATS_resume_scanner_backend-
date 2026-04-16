import numpy as np
from recruiter.services.embeddings import embedding_service

async def calculate_match_score(resume_text: str, job_description: str, jd_embedding: list = None) -> float:
    """Calculates cosine similarity between resume text and job description."""
    try:
        resume_embedding = await embedding_service.generate_embedding(resume_text)
        
        if jd_embedding is None:
            jd_embedding = await embedding_service.generate_embedding(job_description)
        
        # Calculate cosine similarity
        a = np.array(resume_embedding)
        b = np.array(jd_embedding)
        
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Convert to percentage (0.0 to 100) and round to 2 decimal places
        match_score = float(round(similarity * 100, 2))
        return match_score
    except Exception as e:
        print(f"Error calculating match score: {e}")
        return 0.0
