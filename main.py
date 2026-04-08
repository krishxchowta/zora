from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.board import router as board_router
from routes.ops import router as ops_router
from routes.run import router as run_router
from routes.stream import router as stream_router
from routes.agent import router as agent_router
from utils.config import settings
import os

app = FastAPI(title="Zora API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

app.include_router(run_router, prefix="/api")
app.include_router(board_router, prefix="/api")
app.include_router(ops_router, prefix="/api")
app.include_router(agent_router, prefix="/api/agent")
app.include_router(stream_router)


@app.get("/health")
def health():
    return {"status": "ok", "app": "Zora", "stage": "S1-S5+Narrator"}
