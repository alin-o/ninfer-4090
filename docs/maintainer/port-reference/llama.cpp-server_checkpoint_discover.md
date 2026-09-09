# Sanitized llama.cpp checkpoint-discovery reference

Provenance: `llama.cpp` commit `983f0aeb7c1b33dd234f16c086a44466e7da1b76`,
`tools/server/server-common.cpp`, function `server_checkpoint_discover` and
its helpers (`checkpoint_date_at`, `checkpoint_trim_horizontal`,
`checkpoint_volatile_line`, `checkpoint_fence_run`). The source SHA-256 is
recorded in `ninfer-boundary-port-reference-manifest.json`; the accompanying
MIT license is `llama.cpp-LICENSE`.

This repository-local sanitized reference preserves the recognition contract:

- Reconstruct token byte offsets, then inspect only the initial system span.
  If reconstruction fails, report a reconstruction skip rather than inventing
  depths.
- Trim horizontal space only. Ignore lines in fenced blocks (backtick or tilde,
  opener of at least three; a closer is the same character, at least opener
  length, and otherwise horizontal whitespace only).
- Recognize the final whole-line `<|im_end|>`, complete whole-line
  `=== CACHE_BREAKPOINT ===` (also immediately followed by `<project_context>`),
  the first `</INSTRUCTIONS>`, the first project-context marker, and the first
  accepted volatile line. `Date:` requires exactly `YYYY-MM-DD` and only space,
  tab, or `<|im_end|>` after it.
- Recognize only complete initial-user Codex and Claude envelopes, plus the
  complete leading-system `<project>/## Context/<instructions>` envelope.
  Ordinary conversation and quoted lookalikes are excluded.
- Map every raw boundary to an exact token frontier. Count non-boundaries as
  mapping skips; merge origin bits for coincident frontiers. The first token
  containing volatile bytes starts the cumulative SSD-ineligible suffix.

NInfer intentionally replaces upstream byte reconstruction with its tokenizer's
boundary-aware render/tokenize pipeline. Its equivalent fixtures are executable
in `tests/targets/qwen3_6/test_frontend.cpp` and consume the local corpus.
