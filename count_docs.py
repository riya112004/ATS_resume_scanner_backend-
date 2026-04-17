import asyncio
from recruiter.core.database import db
from recruiter.core.config import settings
import os

async def count():
    # Simulate startup_db_client
    db.connect_to_mongo()
    try:
        n = await db.db["recruiter's resume"].count_documents({})
        print(f"TOTAL_DOCS_IN_COLLECTION: {n}")
    finally:
        db.close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(count())
