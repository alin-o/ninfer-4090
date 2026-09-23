# Tests

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — Python container, registered layout, quantization, and resource behavior;
- `ops/` — one identifiable qualification suite per semantic Op or closely related overload group,
  using independent numerical/state-transition oracles at real supported shapes;
- `ops/linear/` — weight/activation-profile-specific public Linear conformance tests plus their
  one shared input generator, FP64 GEMM oracle, tolerance registry, and output/effects mechanics;
- `ops/linear_add/`, `ops/linear_pair/`, `ops/linear_swiglu/` — fused-Op suites split by registered
  weight/activation profile, each evaluating its complete formula rather than composing production
  Ops;
- `targets/qwen3_6/` — shared tokenizer/template, multimodal preprocessing, MRoPE, prepared-prompt,
  stop/output decoding, hybrid topology, decoder/GDN and round-state layouts/views, shifted-MTP
  alignment, Vision control, and family runtime mechanisms;
- `targets/qwen3_6_27b/` — registered inventory, converter recipe, source verifier, artifact
  bindings, reference diagnostics, family Program/multimodal/MTP behavior, and the opt-in real-Engine
  prefix test and causal-scoring State/KV isolation test;
- `targets/qwen3_6_35b_a3b/` — registered inventory/converter contracts, artifact-native diagnostic
  reference, MoE oracle, typed binding, selected-expert row access, 256K INT8 memory calculation,
  and the opt-in real public-Engine route;
- `test_ninfer_artifact_reader.cpp` — C++ framing, directory, encoded-size, payload-span, and
  geometry behavior against a self-contained C++ fixture;
- `test_openai_schema.cpp`, `test_openai_responses.cpp`,
  `test_openai_responses_store.cpp`, `test_anthropic_schema.cpp`, and
  `test_tool_call_parser.cpp` — current protocol translation, Responses Item/state/SSE behavior,
  and incremental tool-call behavior;
- `test_request_log.cpp` — the consumed request JSONL schema and exact measurement fields, plus
  Serve-owned failure severity and exclusion of arbitrary client error text from operational
  records;
- `test_auto_save_writer.cpp` — bounded snapshot-consumer queue, delayed/failed completion,
  cancellation and shutdown settlement, plus queued/active/reserved accounting lifetime;
- `test_http_error_handler.cpp` — protocol-shaped payload-limit errors and application-error
  preservation;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v13 reports;
- `test_bench_matrix.py` — schema-v13 report consumption by the Python matrix summarizer;
- `test_serve_corpus.py` — current serving request-log identity at the measurement consumer;
- device/tensor/arena tests — reusable lower-component behavior; KV tests cover the core physical
  container, family runtime tests cover dimension-driven GDN storage/view mechanics, and Op tests
  cover mathematical state transitions at their own boundary.

Tests are grouped by observable risk, not by mirroring every source file or class.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/W8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

For cached side-request scheduling and SSD recovery, build and run only the affected checks:

```bash
cmake --build build-agent-verify --target ninfer_admission_policy_test \
  ninfer_resource_manager_test ninfer_qwen3_6_27b_prefix_real_test \
  ninfer_openai_cache_real_test -j
ctest --test-dir build-agent-verify \
  -R '^ninfer_(admission_policy|resource_manager)_test$' --output-on-failure
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
  NINFER_PREFIX_REAL_SCENARIO=prefill-interleave \
  build-agent-verify/tests/ninfer_qwen3_6_27b_prefix_real_test
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
  NINFER_OPENAI_CACHE_SCENARIO=ssd-side-request \
  build-agent-verify/tests/ninfer_openai_cache_real_test
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
  NINFER_OPENAI_CACHE_SCENARIO=ssd-memory-only-victim \
  build-agent-verify/tests/ninfer_openai_cache_real_test
```

The real-model cases require a free GPU. They check a cached short continuation completing during
long prefill, and an SSD-backed second-session harness loading beside an active private continuation
when an unused shared victim has no disk copy. They also compare generated tokens and MTP decisions
against isolated execution; the CPU checks cover fairness, protected owners, and cancelled recovery.
The memory-only victim case checks replacement of a non-durable user-content checkpoint, including
restoration after allocation failure and cancellation. SSD export eligibility remains enforced.

`ninfer_qwen3_8_shared_recovery_zero_output_test` verifies reservation cancellation and an
Aborted lifecycle fact for zero-output SSD recovery. `ninfer_qwen3_8_shared_import_errors_test`
checks typed checksum/stale/adoption failures, snapshot-query pin cleanup, and transactional
replacement without a rollback snapshot when physical capacity is available. Both use
`NINFER_QWEN3_8_27B_WEIGHTS`. Transfer fault hooks and their atomic counters are enabled only
with `BUILD_TESTING=ON`; application-only builds compile these call sites to no-ops.

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_w8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|w8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert \
  tests/test_bench_matrix.py tests/test_serve_corpus.py
```

The Python suites cover generic artifact framing and exact converter inventories, source recipes,
encoders, and payload verification. Model execution and real-artifact binding are tested through
the C++ target and Engine suites below; there is no Python inference implementation.

`ninfer_qwen3_6_digest_test` needs no GPU or model. It checks SHA-256 against published known-answer
vectors and fixed binary fixtures, including padding and chunk boundaries, incremental file
hashing, corruption at the start/middle/end of the payload, and cancellation during a large checksum.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

The supported Qwen3.8 groupwise route uses the same real-Engine test with its
own artifact variable.  Its `pressure-resume` and `host-restore` scenarios
exercise rk4v4-e8 KV with MTP:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=pressure-resume \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

`NINFER_PREFIX_REAL_SCENARIO=responses-host-continuation` exercises parsed `store:false`
Responses with full conversation history and `preserve_thinking`, matching the production prompt
semantics: the completed assistant response is retained in VRAM,
offloaded with State/Main/MTP under pressure, then restored for the next user turn. It checks
inferred head advancement, reuse through the assistant response (allowing the final pending token),
and exact generated-token and MTP-counter agreement with a cold continuation.

`ninfer_openai_cache_real_test` exercises the full OpenAI parsing/resolution/GenerationService
path with SSD disabled. A Responses request seeds a stable harness, and a Chat Completions request
with a different volatile instruction suffix and conversation restores it from RAM. It checks
explicit-only write suppression and existing-prefix reads, rejects a changed harness, and compares
generated tokens and MTP counters with a cold run. It then continues the completed assistant reply
through Responses, requiring reuse through that reply (allowing the final pending token) and another
exact cold comparison. The separate `responses-host-continuation` scenario forces the completed
private endpoint into RAM and requires at least 99% reuse of its longer prompt.
It uses `NINFER_QWEN3_8_27B_WEIGHTS`.

`ninfer_qwen3_8_concurrent_ingress_test` holds the execution lock while testing immutable candidate
discovery and request-log memory observations, then submits short requests during a long MTP
generation through SSD, warm, and cold routes. A control call must also get an execution boundary
before that generation finishes. It requires overlapping execution, multi-row decode,
exact cold-oracle output, and settled reservations. It also checks unchanged user-history reuse and
safe fallback after historical channel instructions are removed. The bounded profile uses three
lanes, six Device State slots, eight Host State slots and three shared owners.
For a separate 19K-token-prefix replay with the same assertions:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_OPENAI_CACHE_SCENARIO=concurrent-ingress-long \
  build-agent-verify/tests/ninfer_openai_cache_real_test
```

The long replay uses a 32,768-token per-request ceiling and 98,304-token shared KV capacity.
Its printed timings describe this synthetic current-build fixture, not production acceptance.

The calibration comparison and constrained four-request fallback are direct scenario invocations
so their stdout (including fixture hashes) remains available:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=cache-fixture-equivalence \
NINFER_REPORT_CACHE_FIXTURE_HASHES=1 \
  build/tests/ninfer_qwen3_6_27b_prefix_real_test

NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=four-request-root-fallback \
  build/tests/ninfer_qwen3_6_27b_prefix_real_test
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_score_real_test --output-on-failure
```

Run the peer 35B-A3B route independently:

```bash
NINFER_QWEN3_6_35B_A3B_WEIGHTS=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_35b_a3b_real_test --output-on-failure
```

Without the corresponding variable CTest marks each C++ integration test as skipped. Neither test
uses another numerical/execution path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares restored and cold greedy output, compares stripped and preserved closed-turn
prompt lengths, and verifies turn/response rewrite-checkpoint reuse paths plus Responses
inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- family Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## Generated reasoning continuation regression

`ninfer_qwen3_6_frontend_test` includes a self-contained generated-reasoning replay regression
that runs without an official tokenizer directory or GPU. It checks exact endpoint identity and
shortlist equality while preserving rejection of changed historical text. Runtime-mechanism
checks also cover position and media mismatches with request-local prefill boundaries.

The same frontend test includes deterministic alternate-BPE histories: generated `>` and `|`
remain separate when canonical encoding would merge them. It checks original token IDs and exact
endpoint digests, survival after rendered log text and the prepared prompt are consumed, token
counts/context limits, positions and capture/volatility boundaries, edited history, explicit
session isolation, special-token provenance, and descendants after bounded CPU-history eviction.
These cases are self-contained and run before the optional official-resource tests. Use:

```bash
ctest --test-dir build-agent-verify -R '^ninfer_qwen3_6_frontend_test$' --output-on-failure
```

The real-model integration below checks completion, endpoint reuse and save/restore with MTP.
Alternate tokenization itself uses deterministic token fixtures rather than requiring a sampled
model response to happen to contain a noncanonical BPE sequence. A cold continuation oracle must
use the same preserved history; a fresh canonical text-only run is a different model input.

The Qwen3.8 real-artifact case exercises MTP endpoint reuse and private snapshot round-trip
continuation, including old-version rejection:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_OPENAI_CACHE_SCENARIO=reasoning-continuation \
build-agent-verify/tests/ninfer_openai_cache_real_test
```

`ninfer_qwen3_6_27b_prefix_real_test` also covers endpoint continuation when a private long anchor
and shared checkpoint reference the same StateImage. It checks both a surviving shared owner and
full-capacity admission that evicts the shared owner, requires complete endpoint reuse, and compares
generated tokens and MTP decisions with cold execution. Run just these Qwen3.8 cases with:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=shared-anchor-endpoint \
build-agent-verify/tests/ninfer_qwen3_6_27b_prefix_real_test
```

The default prefix test also continues four generated responses in one named session with three
Device State slots. Completed SSD saves keep the older endpoints eligible for residency; they
must yield State capacity so each continuation reuses the
latest complete response. The fixture requires actual State reclamation and compares final generated
tokens and MTP decisions with cold execution. Run it alone with:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=latest-response-pressure \
build-agent-verify/tests/ninfer_qwen3_6_27b_prefix_real_test
```

The `latest-response-catalog` variant caps the private catalog at two continuations. It reproduces
the one-response lag caused by pricing eviction losses while ignoring consumed endpoint losses.
Both variants run in the default prefix test and require full latest-endpoint reuse plus cold MTP
parity.

The `superseded-retirement` variant continues six turns without capacity pressure. It requires
exactly the latest private owner plus one explicitly saved owner, exercises replacement of that
file binding, and checks that a restored SSD-backed source stays resident after reuse. Ordinary
superseded private owners must disappear without waiting for pressure. It also compares the last
warm response's tokens and MTP decisions with cold execution.

## Idle admission under cache pressure

The ResourceManager regression `bounded pressure fallback readiness` exhausts the optional
pressure search while a verified root fallback can satisfy both physical resources and private
publication capacity. Admission must keep that feasible fallback.

The default Qwen3.8 prefix test also fills all six Device and eight Host State slots across retained
conversations, requires State reclamation, then checks fresh reasoning admission and subsequent
Engine health with cold generated-token and MTP parity. Run the GPU case alone with:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=idle-cache-pressure \
build-agent-verify/tests/ninfer_qwen3_6_27b_prefix_real_test
```
