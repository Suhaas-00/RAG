"""FastAPI application factory for the RAG platform."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rag_system.service import AnswerRequest, RAGService
from rag_system.utils.config import Settings


def create_app(settings: Settings | None = None):
    """Create and configure the HTTP API application."""

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "The API server requires fastapi, pydantic, and uvicorn. "
            "Install project dependencies from requirements.txt."
        ) from exc

    service = RAGService.from_settings(settings or Settings.from_env())
    app = FastAPI(title="Medical RAG Platform", version="1.0.0")

    class QueryRequest(BaseModel):
        question: str = Field(min_length=1)
        model: str = "llama-3.1-8b-instant"
        top_k: int = Field(default=5, ge=1, le=50)
        alpha: float = Field(default=0.55, ge=0.0, le=1.0)
        allow_global_search: bool | None = None
        use_cache: bool = True

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        status = service.health()
        if status["status"] != "ok":
            raise HTTPException(status_code=503, detail=status)
        return status

    @app.get("/papers")
    def papers() -> dict[str, Any]:
        return {"papers": [{"label": label, "source": source} for label, source in service.list_papers()]}

    @app.post("/query")
    def query(payload: QueryRequest) -> dict[str, Any]:
        response = service.answer(AnswerRequest(**payload.model_dump()))
        return asdict(response)

    return app


app = create_app()

