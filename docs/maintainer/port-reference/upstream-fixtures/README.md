# Boundary discovery port fixture provenance

This directory is repository-local provenance for the bounded checkpoint recognizer.
The implementation and fixture semantics were ported from llama.cpp
`983f0aeb7c1b33dd234f16c086a44466e7da1b76`, `tools/server/server-common.cpp`
(SHA-256 `8c83f75fecae01d58154e6309ec5eb6ac5952aa37a88177abbce6142ee95f7b3`), under
the upstream MIT license.  The source artifact manifest and upstream LICENSE hash are recorded
in `../ninfer-boundary-port-reference-manifest.json`.

The cases exercised by NInfer's frontend regression are: fenced marker exclusion; marker plus
project tag; Codex AGENTS/INSTRUCTIONS/environment; Claude system-reminder/claudeMd/currentDate;
the complete leading-system project envelope; and volatility descendants.  They deliberately use
sanitized text and never require an external llama.cpp checkout.
