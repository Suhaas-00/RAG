# Context Building

## Table of Contents
- [Function](#function)
- [Format](#format)
- [Fields](#fields)
- [Token Budget](#token-budget)
- [Direct Lookup Format](#direct-lookup-format)

## Function

The active function is `rag_system.hybrid_retriever.format_context`.

## Format

```text
[Source: <source> | Paper: <paper_id> | Section: <section> | Page: <page> | Confidence: <confidence>]
<chunk text>
```

Blocks are separated by blank lines.

## Fields

The context builder reads `source`, `paper_id`, `section`, `page` or `page_number`, `confidence`, and `text`.

## Token Budget

`Settings.max_context_tokens` exists but is not enforced by the active context builder.

## Direct Lookup Format

`qa_cli._format_direct_section` uses a plain header without brackets when returning retrieved section text directly.
