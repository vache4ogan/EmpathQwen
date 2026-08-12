from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from front_back.backend.model_upp import get_model, load_model
from front_back.backend.routers.chat import router as chat_router
from front_back.backend.services.chat import ChatService


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    app.state.chat_service = ChatService(get_model())
    yield


app = FastAPI(
    title="EmpathLLaMa API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "mock"}


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
