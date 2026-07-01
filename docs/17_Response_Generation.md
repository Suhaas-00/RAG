# Response Generation

## Table of Contents
- [Answer Paths](#answer-paths)
- [LLM Structure](#llm-structure)
- [Citations](#citations)
- [Grounding](#grounding)
- [Batch Responses](#batch-responses)

## Answer Paths

`answer_question` can return list-papers output, metadata output, direct retrieved section text, Groq answer text, or fallback.

## LLM Structure

The prompt requires `ANSWER`, `EVIDENCE`, and `CONFIDENCE SUMMARY` sections.

## Citations

The required citation format is `(Source: <filename>; Section: <section>; Page: <n>; Confidence: <0.00-1.00>)`. The code prompts this format but does not post-validate it.

## Grounding

Grounding is achieved through retrieved context, strict prompt rules, and low temperature. There is no external verifier.

## Batch Responses

`qa_bulkload` writes one row per retrieved chunk and repeats the final response for every chunk row belonging to the question. The summary sheet collapses chunk IDs per question.
