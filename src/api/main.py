"""
FastAPI application — Research Intelligence Assistant API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Research Intelligence Assistant",
    description="Hybrid RAG + GraphRAG for scientific literature",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "service": "Research Intelligence Assistant",
    }
