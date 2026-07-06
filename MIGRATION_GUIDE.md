# Migration Guide

## Existing CLI Users

No command migration is required. Existing commands continue to work:

```powershell
python DataIngestion.py
python -m rag_system.qa_cli "question"
python -m rag_system.qa_bulkload
```

## Environment Configuration

The CLI now honors `Settings.from_env()` for defaults. Existing explicit command-line arguments still take precedence.

## API Users

Start the HTTP service with:

```powershell
python scripts/run_api.py
```

Send questions to `POST /query` with a JSON body containing `question`, and optionally `model`, `top_k`, `alpha`, `allow_global_search`, and `use_cache`.

## Index Artifacts

No index migration is required. The service reads the existing `vectors.index` and `metadata.pkl` artifacts.

