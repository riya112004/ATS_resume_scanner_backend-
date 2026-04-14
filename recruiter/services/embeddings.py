from openai import AsyncOpenAI
from typing import List
from recruiter.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding for a given text."""
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=settings.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise ValueError(f"Failed to generate embedding: {str(e)}")

embedding_service = EmbeddingService()
