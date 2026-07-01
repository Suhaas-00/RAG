# Environment

## Table of Contents
- [Environment File](#environment-file)
- [Variables](#variables)
- [Loading Order](#loading-order)
- [Precedence](#precedence)
- [Fallbacks](#fallbacks)
- [Not Configured](#not-configured)

## Environment File

The checkout contains `.env` with `GROQ_API_KEY`. The secret value is intentionally omitted.

## Variables

Only `GROQ_API_KEY` is consumed by active code. `RAG_*` variables are supported only through explicit use of `Settings.from_env`.

## Loading Order

`load_groq_api_key` checks current directory `.env`, then parent directories, loads the first existing file, and returns `os.getenv("GROQ_API_KEY")`.

## Precedence

`load_dotenv(..., override=False)` means existing process environment values win over `.env` values.

## Fallbacks

Without a Groq key, diagnostics are printed and the LLM wrapper returns retrieved context or `Not found in the document`.

## Not Configured

The current code does not read environment variables for device, GPU/CPU, thread count, cache, timeout, top-p, max tokens, or retry count.
