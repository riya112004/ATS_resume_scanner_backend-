from motor.motor_asyncio import AsyncIOMotorClient
from recruiter.core.config import settings

class Database:
    client: AsyncIOMotorClient = None
    db = None

    def connect_to_mongo(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.DATABASE_NAME]
        print(f"Connected to MongoDB: {settings.DATABASE_NAME}")

    def close_mongo_connection(self):
        self.client.close()
        print("MongoDB connection closed.")

db = Database()
