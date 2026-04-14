from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from recruiter.api.endpoints import router as api_router
from seeker.api.endpoints import router as seeker_router
from recruiter.core.database import db
from recruiter.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Mount uploads folder as static to serve files
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(seeker_router, prefix="/seeker-api")

@app.on_event("startup")
async def startup_db_client():
    db.connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    db.close_mongo_connection()

from fastapi.responses import FileResponse, Response
import os

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
    uvicorn.run("main:app", host="0.0.0.0", port=2263, reload=False)
