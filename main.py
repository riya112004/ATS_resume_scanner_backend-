from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from recruiter.api.endpoints import router as api_router
from seeker.api.endpoints import router as seeker_router
from recruiter.core.database import db
from recruiter.core.config import settings
from contextlib import asynccontextmanager
import os
from fastapi.responses import Response, FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to DB
    db.connect_to_mongo()
    yield
    # Shutdown: Close DB
    db.close_mongo_connection()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(seeker_router, prefix="/seeker-api")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/")
async def root():
    return FileResponse(os.path.join("recruiter/static", "index.html"))

@app.get("/seeker")
async def seeker_root():
    return FileResponse(os.path.join("seeker/static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3333, reload=False)
