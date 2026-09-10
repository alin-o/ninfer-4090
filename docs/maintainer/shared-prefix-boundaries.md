# Shared prefix boundary discovery

The Qwen3.6 Frontend recognizes structural cache boundaries only while rendering
the initial folded System/Developer instruction span.  User, Tool and Assistant
content is deliberately outside that trusted region, except for the complete initial-user
Codex AGENTS/INSTRUCTIONS/environment and Claude system-reminder/claudeMd/currentDate
envelopes.  These bounded upstream exceptions, plus the complete leading-system
`<project>/## Context/<instructions>` envelope, are recognized as a whole; lookalikes do
not authorize a boundary.  The recognizer is ported
from llama.cpp `server_checkpoint_discover` at
`983f0aeb7c1b33dd234f16c086a44466e7da1b76`; the source hashes and MIT license
are retained in the repository-local port reference manifest described below.

Recognized whole lines are the final system terminator, `=== CACHE_BREAKPOINT ===`
(including its immediately-adjacent `<project_context>` form),
`</INSTRUCTIONS>`, `<project_context>`, and the first supported volatile
metadata field.  Horizontal whitespace is accepted and fenced blocks are
ignored. `Today:` requires an exact `YYYY-MM-DD`; `Date:` additionally permits
only an empty or final-terminator suffix. The boundary bytes are sent through the normal rendered-chat
tokenization call.  A candidate that does not land on an exact token frontier
is retained only as a mapping skip; it is never rounded into a cache prefix.

`PreparedContextCache::structural_checkpoints` publishes the merged origin bits,
role and SSD eligibility, which travel through CaptureGroup and publication to the shared
catalog's existing physical owner.  It does not introduce a second physical cache owner.
`first_volatile_token` is cumulative: a prefix
ending before that token remains eligible, while any checkpoint including it is
not.  Media prompts retain the same recognized structural checkpoints and may
produce the corresponding automatic shared-prefix opportunities; their
`ssd_eligible` metadata is false until a media-aware persistence policy exists.

The repository-local provenance artifact is
`docs/maintainer/port-reference/ninfer-boundary-port-reference-manifest.json`.
It records the upstream source revision and hashes; tests use local fixtures and
do not require an upstream checkout.

Complete durable export/import of an eligible shared owner is specified separately in
[Complete shared-prefix snapshot format](shared-prefix-snapshot.md). Boundary classification stays
in Frontend; the physical State/KV codec stays in Program.
