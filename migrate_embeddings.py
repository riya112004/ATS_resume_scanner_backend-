import asyncio
import logging
from recruiter.core.database import db
from recruiter.services.embeddings import embedding_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("migration")

async def migrate():
    db.connect_to_mongo()
    try:
        total = await db.db["recruiter's resume"].count_documents({})
        logger.info(f"Total documents: {total}")

        cursor = db.db["recruiter's resume"].find({})
        updated = 0
        async for doc in cursor:
            raw = doc.get("raw_content", "")
            if not raw or len(raw.strip()) < 50:
                logger.warning(f"Skipping {doc.get('filename')} — no raw_content")
                continue

            embedding = await embedding_service.generate_embedding(raw[:2000].strip())
            if embedding:
                await db.db["recruiter's resume"].update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": embedding}}
                )
                updated += 1
                logger.info(f"Updated {updated}/{total} — {doc.get('filename')}")

        logger.info(f"Done! {updated} embeddings regenerated.")
    finally:
        db.close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(migrate())
