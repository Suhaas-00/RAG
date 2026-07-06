"""Run the RAG HTTP API with environment-driven settings."""

from __future__ import annotations

import uvicorn

from rag_system.logging_config import configure_logging
from rag_system.utils.config import Settings


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level, json_logs=settings.json_logs)
    uvicorn.run(
        "rag_system.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        factory=False,
    )


if __name__ == "__main__":
    main()

