# LLM Inference

## Table of Contents
- [Module](#module)
- [API Key](#api-key)
- [Model Parameters](#model-parameters)
- [System Prompt](#system-prompt)
- [User Prompt](#user-prompt)
- [Fallbacks](#fallbacks)

## Module

`rag_system.llm` defines the Groq inference path.

## API Key

`load_groq_api_key` searches from the current directory upward for `.env`, loads the first file with `override=False`, and returns `GROQ_API_KEY`. The repository `.env` contains this variable; the value is not documented.

## Model Parameters

The default model is `llama-3.1-8b-instant`. The Groq call sets `temperature=0.1`. The code does not set `top_p`, `max_tokens`, streaming, retry, or timeout options.

## System Prompt

The system prompt enforces precision-oncology grounding, citation requirements, no external knowledge, fallback on insufficient context, no patient-specific treatment speculation, and a fixed `ANSWER`, `EVIDENCE`, `CONFIDENCE SUMMARY` response structure.

## User Prompt

The user prompt wraps retrieved context and the question, then instructs the model to synthesize only from retrieved chunks and cite used chunks.

## Fallbacks

If no API key exists, `answer_with_groq` returns `context or FALLBACK`. `FALLBACK` is `Not found in the document`.
