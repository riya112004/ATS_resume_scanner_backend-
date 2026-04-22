import asyncio
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self):
        # 'all-MiniLM-L6-v2' is a fast, accurate, and lightweight local model.
        # It runs entirely on your server for FREE.
        # Note: On first run, it will automatically download the model (~80MB).
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding locally using sentence-transformers."""
        if not text:
            return []
            
        try:
            # Running the blocking 'encode' function in a thread to keep the API responsive
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, lambda: self.model.encode(text))
            return embedding.tolist()
        except Exception as e:
            print(f"Local Embedding Generation Error: {e}")
            return []

embedding_service = EmbeddingService()
