#include "serve/operational_log.h"
#include "serve/request_log.h"
#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    bool protected_artifact_rejected = false;
    try {
        JsonlRequestLog unsafe("same-path.ninfer", "same-path.ninfer");
    } catch (const std::invalid_argument&) { protected_artifact_rejected = true; }
    failures += check(protected_artifact_rejected,
                      "request log accepted the model artifact as its output path");

    ServeOptions options;
    options.artifact_path                  = "/models/qwen3_6_27b.ninfer";
    options.host                           = "127.0.0.1";
    options.port                           = 8123;
    options.api_key                        = "must-not-appear";
    options.model_id_override              = "deployment-alias";
    options.request_log_jsonl              = "requests.jsonl";
    options.max_context                    = 262144;
    options.kv_capacity                    = ninfer::KvCapacityPolicy::explicit_capacity(524288);
    options.prefill_chunk                  = 1024;
    options.log_stats_interval_ms          = 2500;
    options.kv_cache                       = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend            = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens       = 3;
    options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
    options.enable_vision                  = false;
    options.allow_prefix_reuse             = true;
    options.preserve_thinking              = true;
    options.default_thinking_budget        = 512;
    options.sampling_overrides.temperature = 0.6F;
    options.startup_argv = {"ninfer-serve", options.artifact_path, "--api-key", "<redacted>"};

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path                                   = options.artifact_path;
    engine_options.device                                          = options.device;
    engine_options.max_context                                     = options.max_context;
    engine_options.max_concurrency                                 = 2;
    engine_options.max_pending_requests                            = options.max_pending_requests;
    engine_options.pending_timeout_ms                              = options.pending_timeout_ms;
    engine_options.prefill_chunk                                   = options.prefill_chunk;
    engine_options.kv_cache                                        = options.kv_cache;
    engine_options.speculative                                     = options.speculative;
    engine_options.enable_vision                                   = options.enable_vision;
    engine_options.use_cuda_graph                                  = options.use_cuda_graph;
    engine_options.context_cache.device_state_slots                = 2;
    engine_options.context_cache.host_state_slots                  = 3;
    engine_options.context_cache.host_kv_capacity_bytes            = 64ULL << 20;
    engine_options.context_cache.max_private_continuations         = 4;
    engine_options.context_cache.max_shared_prefixes               = 2;
    engine_options.context_cache.max_long_anchors_per_continuation = 2;

    const ninfer::ModelSamplingDefaults sampling_defaults{
        .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
        .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.8F, .presence_penalty = 1.5F},
    };

    ninfer::LoadSummary load;
    load.target               = "qwen3_6_27b";
    load.model_id             = "qwen3.6-27b";
    load.weights_id           = "groupwise-int";
    load.load_seconds         = 1.234567890123;
    load.upload_seconds       = 0.345678901234;
    load.artifact_bytes_read  = 1000;
    load.host_to_device_bytes = 900;
    load.peak_staging_bytes   = 128;
    load.tensor_count         = 42;
    load.resource_count       = 6;
    load.context_cost         = {
                .transfer_source = ninfer::ContextCostPresetSource::External,
                .prefill_source  = ninfer::ContextCostPresetSource::CompiledDefault,
                .hardware_class  = "nvidia-geforce-rtx-5090-sm120",
                .model_id        = "qwen3.6-27b",
                .weights_id      = "groupwise-int",
                .preset_path     = "local-costs.json",
    };

    ninfer::MemorySummary memory;
    memory.max_context                 = 262144;
    memory.kv_capacity_mode            = ninfer::KvCapacityMode::Explicit;
    memory.kv_capacity                 = 524288;
    memory.kv_capacity_page_groups     = 8192;
    memory.kv_capacity_max_page_groups = 16384;
    memory.kv_cache                    = ninfer::KvCacheStorage::Fp8E4M3Row256;
    memory.weights.capacity_bytes      = 100;
    memory.sequence.capacity_bytes     = 200;
    memory.workspace.capacity_bytes    = 500;
    memory.vision_workspace            = ninfer::VisionWorkspaceMemorySummary{
                   .aggregate_prompt_tokens = 32768,
                   .max_item_tokens         = 16384,
                   .general_capacity_bytes  = 300,
                   .encode_peak_bytes       = 400,
                   .handoff_offset_bytes    = 300,
                   .handoff_capacity_bytes  = 200,
                   .handoff_active_bytes    = 0,
                   .handoff_peak_bytes      = 150,
    };
    memory.minimum_runtime_reservation_bytes = 1300;
    memory.kv_capacity_increment_bytes       = 100;
    memory.runtime_reservation_bytes         = 1600;
    memory.available_after_weights_bytes     = 1700;
    memory.available_after_startup_bytes     = 180;
    memory.planned_slack_bytes               = 100;
    memory.cuda_graph_allowance_bytes        = 600;
    memory.kv_payload_bytes                  = 400;
    memory.gdn_state_bytes                   = 8192;
    memory.checkpoint_state_image_bytes      = 8192;
    memory.device_main_kv_capacity_pages     = 100;
    memory.device_main_kv_occupied_pages     = 25;
    memory.device_main_kv_page_bytes         = 4096;
    memory.device_backend_kv_capacity_pages  = 50;
    memory.device_backend_kv_occupied_pages  = 60;
    memory.device_backend_kv_page_bytes      = 2048;
    memory.host_state_capacity_slots         = 3;
    memory.host_state_occupied_slots         = 1;
    memory.host_main_kv_page_bytes           = 5000;
    memory.host_backend_kv_page_bytes        = 3000;
    memory.host_kv_capacity_bytes            = 64ULL << 20;
    memory.host_kv_occupied_bytes            = 8ULL << 20;

    ServerLogEnvironment environment;
    environment.device                    = 0;
    environment.gpu_name                  = "NVIDIA GeForce RTX 5090";
    environment.gpu_uuid                  = "GPU-00000000-0000-0000-0000-000000000000";
    environment.total_device_memory_bytes = 32000000000ULL;
    environment.compute_capability_major  = 12;
    environment.compute_capability_minor  = 0;
    environment.cuda_compile_version      = "13.1";
    environment.cuda_runtime_version      = "13.1";
    environment.cuda_driver_version       = "13.1";

    const Json server = Json::parse(format_server_start_json(
        "serve-test", 1000, options, engine_options, sampling_defaults, "deployment-alias", load,
        memory, environment, std::uint64_t{123456}));
    failures += check(server.at("artifact_type") == kRequestLogArtifactType,
                      "server record artifact type mismatch");
    failures += check(server.at("schema_version") == kRequestLogSchemaVersion,
                      "server record schema mismatch");
    failures += check(server.at("event") == "server_start", "server event mismatch");
    failures += check(server.at("server").at("public_model_id") == "deployment-alias",
                      "resolved public model id missing");
    failures += check(server.at("artifact").at("target") == "qwen3_6_27b", "server target missing");
    failures += check(server.at("artifact").at("weights_id") == "groupwise-int",
                      "server weights id missing");
    failures += check(server.at("artifact").at("size_bytes") == 123456, "artifact size missing");
    failures += check(server.at("engine").at("max_context") == 262144, "max context missing");
    failures += check(server.at("engine").at("kv_capacity") == 524288, "KV capacity missing");
    failures += check(server.at("engine").at("kv_capacity_mode") == "explicit" &&
                          server.at("engine").at("kv_capacity_page_groups") == 8192 &&
                          server.at("engine").at("kv_capacity_max_page_groups") == 16384,
                      "KV capacity resolution metadata missing");
    failures +=
        check(server.at("engine").at("log_stats_interval_ms") == 2500, "stats interval missing");
    failures += check(server.at("server").at("request_log_jsonl") == "requests.jsonl",
                      "request log path missing");
    failures += check(server.at("server").at("default_thinking_budget") == 512,
                      "server thinking budget missing");
    failures += check(server.at("engine").at("kv_cache") == "fp8-e4m3-row256", "KV type missing");
    failures += check(server.at("engine").at("vision") == false, "Vision state missing");
    failures += check(server.at("engine").at("speculative_backend") == "mtp",
                      "speculative backend missing");
    failures +=
        check(server.at("engine").at("proposal_head") == "optimized", "proposal head missing");
    failures += check(
        server.at("engine").at("context_cost").at("transfer_source") == "external" &&
            server.at("engine").at("context_cost").at("prefill_source") == "compiled-default" &&
            server.at("engine").at("context_cost").at("hardware_class") ==
                "nvidia-geforce-rtx-5090-sm120" &&
            server.at("engine").at("context_cost").at("preset_path") == "local-costs.json",
        "resolved context-cost layers missing");
    failures += check(server.at("engine").at("prefix_reuse") == true, "prefix-reuse state missing");
    failures += check(
        server.at("engine").at("context_cache").at("device_state_slots") == 2 &&
            server.at("engine").at("context_cache").at("total_device_state_slots") == 4 &&
            server.at("engine").at("context_cache").at("host_state_slots") == 3 &&
            server.at("engine").at("context_cache").at("host_kv_capacity_bytes") == (64ULL << 20) &&
            server.at("engine").at("context_cache").at("max_private_continuations") == 4 &&
            server.at("engine").at("context_cache").at("max_shared_prefixes") == 2,
        "resolved context-cache configuration missing");
    failures += check(server.at("server").at("default_preserve_thinking") == true,
                      "server preserve-thinking default missing");
    failures +=
        check(server.at("sampling_defaults").at("thinking").at("temperature") == 1.0 &&
                  server.at("sampling_defaults").at("non_thinking").at("presence_penalty") == 1.5,
              "registered mode-specific sampling defaults missing");
    failures += check(
        server.at("sampling_defaults").at("server_overrides").at("temperature").get<float>() ==
                0.6F &&
            server.at("sampling_defaults").at("server_overrides").at("top_p").is_null(),
        "server sampling overrides lost omission state");
    failures += check(server.at("environment").at("gpu_name") == "NVIDIA GeForce RTX 5090",
                      "GPU name missing");
    failures +=
        check(server.at("memory").at("workspace").at("capacity_bytes") == 500 &&
                  server.at("memory").at("vision_workspace").at("general_capacity_bytes") == 300 &&
                  server.at("memory").at("vision_workspace").at("handoff_capacity_bytes") == 200 &&
                  server.at("memory").at("vision_workspace").at("handoff_peak_bytes") == 150,
              "Vision workspace layout missing");
    failures += check(server.at("memory").at("cuda_graph_allowance_bytes") == 600,
                      "CUDA Graph allowance missing");
    failures += check(server.at("memory").at("runtime_reservation_bytes") == 1600 &&
                          server.at("memory").at("available_after_weights_bytes") == 1700 &&
                          server.at("memory").at("available_after_startup_bytes") == 180 &&
                          server.at("memory").at("kv_capacity_headroom_bytes") == 0 &&
                          server.at("memory").at("planned_slack_bytes") == 100,
                      "adaptive KV memory ledger missing");
    failures += check(server.at("memory").at("host_state_capacity_slots") == 3 &&
                          server.at("memory").at("host_state_occupied_slots") == 1 &&
                          server.at("memory").at("host_kv_capacity_bytes") == (64ULL << 20) &&
                          server.at("memory").at("host_kv_occupied_bytes") == (8ULL << 20),
                      "Host context-cache memory ledger missing");
    failures += check(server.dump().find("must-not-appear") == std::string::npos,
                      "server JSON leaked the API key");
    failures += check(server.at("argv").at(3) == "<redacted>",
                      "server argv did not retain the redaction marker");

    GenerationRequest request;
    request.max_tokens = 4096;
    request.messages.resize(2);
    request.messages.front().content.push_back(ContentPart{.kind = ContentKind::Image});

    PreparedRequest prepared;
    prepared.enable_thinking                           = true;
    prepared.thinking_budget                           = 256;
    prepared.effective_reasoning_effort                = ninfer::ReasoningEffort::XHigh;
    prepared.preserve_thinking                         = true;
    prepared.sampling.temperature                      = 0.6F;
    prepared.sampling.top_p                            = 0.95F;
    prepared.sampling.top_k                            = 20;
    prepared.sampling.min_p                            = 0.0F;
    prepared.sampling.presence_penalty                 = 1.0F;
    prepared.sampling.frequency_penalty                = 0.0F;
    prepared.sampling.seed                             = 7632647173703958409ULL;
    prepared.acquisition_seconds                       = 0.004;
    prepared.preparation.seconds                       = 0.12;
    prepared.preparation.media_preprocess_seconds      = 0.08;
    prepared.preparation.media_preprocess_work_seconds = 0.31;
    prepared.preparation.tokenize_seconds              = 0.02;
    prepared.preparation.media_items                   = 1;
    prepared.preparation.media_cache_misses            = 1;
    prepared.preparation.built_patch_bytes             = 49152;
    prepared.preparation.stable_boundaries_recognized  = 3;
    prepared.preparation.stable_boundaries_capturable  = 2;
    prepared.preparation.stable_boundary_mapping_skips = 1;
    prepared.preparation.ssd_eligible_boundaries       = 1;

    const RequestLogMetadata metadata{
        .model                             = "qwen3.6-27b",
        .response_id                       = "chatcmpl-correlation-fixture",
        .stream                            = false,
        .output_tokens_explicit            = true,
        .preserve_thinking_semantic_change = true,
    };
    RequestLogContext context =
        make_request_log_context(7, "openai_chat_completions", request, metadata, prepared);
    context.kv_snapshot = make_kv_capacity_snapshot(memory);
    const Json started  = Json::parse(format_request_start_json("serve-test", 2000, context));
    failures +=
        check(started.at("request").at("request_id") == 7, "request id missing from start record");
    failures += check(started.at("request").at("response_id") == "chatcmpl-correlation-fixture",
                      "wire response id missing from start record");
    failures += check(started.at("request").at("requested_output_tokens") == 4096,
                      "request output budget missing");
    failures +=
        check(started.at("kv_capacity").at("device").at("main").at("capacity_pages") == 100 &&
                  started.at("kv_capacity").at("device").at("main").at("used_pages") == 25 &&
                  started.at("kv_capacity").at("device").at("main").at("free_pages") == 75 &&
                  started.at("kv_capacity").at("device").at("main").at("used_bytes") == 25 * 4096 &&
                  started.at("kv_capacity").at("device").at("backend").at("free_pages") == 0 &&
                  started.at("kv_capacity").at("device").at("backend").at("free_bytes") == 0 &&
                  started.at("kv_capacity").at("host").at("capacity_bytes") == (64ULL << 20) &&
                  started.at("kv_capacity").at("host").at("used_bytes") == (8ULL << 20) &&
                  started.at("kv_capacity").at("host").at("free_bytes") == (56ULL << 20),
              "KV snapshot arithmetic or saturated free capacity is wrong");
    failures += check(started.at("request").at("enable_thinking") == true,
                      "resolved thinking mode missing");
    failures += check(started.at("request").at("thinking_budget") == 256,
                      "resolved thinking budget missing");
    failures += check(started.at("request").at("requested_reasoning_effort").is_null() &&
                          started.at("request").at("resolved_reasoning_effort") == "xhigh",
                      "requested and resolved reasoning effort are not distinguished");
    failures += check(started.at("request").at("preserve_thinking") == true &&
                          started.at("request").at("preserve_thinking_semantic_change") == true,
                      "resolved preserve-thinking metadata missing");
    failures += check(started.at("request").at("sampling").at("seed") == 7632647173703958409ULL,
                      "resolved seed missing");
    failures +=
        check(started.at("request").at("media_item_count") == 1 &&
                  started.at("preparation_seconds").at("acquisition") == 0.004 &&
                  started.at("preparation_seconds").at("media_preprocess_work") == 0.31 &&
                  started.at("preparation_seconds").at("tokenize") == 0.02 &&
                  started.at("preparation_seconds").at("cache_misses") == 1 &&
                  started.at("preparation_seconds").at("stable_boundaries_recognized") == 3 &&
                  started.at("preparation_seconds").at("ssd_eligible_boundaries") == 1,
              "request-scoped media preparation diagnostics missing");

    ApiError preparation_error;
    preparation_error.status               = 400;
    preparation_error.type                 = "invalid_request_error";
    preparation_error.param                = "messages";
    preparation_error.code                 = "context_length_exceeded";
    preparation_error.message              = "sentinel-client-value\nsecond-record";
    preparation_error.checkpoint_lifecycle = {
        ninfer::CheckpointLifecycleFact{
            .frontier         = 128,
            .operation        = ninfer::CheckpointLifecycleOperation::Restored,
            .source_tier      = ninfer::CheckpointLifecycleTier::Ssd,
            .destination_tier = ninfer::CheckpointLifecycleTier::Host,
            .status           = ninfer::CheckpointLifecycleStatus::Committed,
        },
    };
    GenerationRequest rejected_request                = request;
    rejected_request.reasoning_effort                 = RequestedReasoningEffort::High;
    const RequestRejectionLogContext rejected_context = make_request_rejection_log_context(
        8, "anthropic_messages", rejected_request, metadata, preparation_error);
    const Json rejected =
        Json::parse(format_request_rejected_json("serve-test", 2500, rejected_context));
    failures +=
        check(rejected.at("event") == "request_rejected" && rejected.at("phase") == "prepare",
              "preparation rejection event or phase mismatch");
    failures +=
        check(rejected.at("request").at("request_id") == 8 &&
                  rejected.at("request").at("response_id") == "chatcmpl-correlation-fixture" &&
                  rejected.at("request").at("media_item_count") == 1 &&
                  rejected.at("request").at("message_count") == 2,
              "preparation rejection request shape missing");
    failures += check(rejected.at("request").at("requested_reasoning_effort") == "high" &&
                          rejected.at("request").at("resolved_reasoning_effort").is_null(),
                      "rejection log fabricated a resolved reasoning effort");
    failures += check(rejected.at("error").at("status") == 400 &&
                          rejected.at("error").at("code") == "context_length_exceeded" &&
                          rejected.at("error").at("param") == "messages" &&
                          rejected.at("error").at("message") == preparation_error.message,
                      "preparation rejection API error missing");
    failures += check(rejected.at("checkpoint_summary").at("reuse_loaded") == true &&
                          rejected.at("checkpoint_summary").at("restore_committed") == true,
                      "preparation rejection lost a committed durable restore summary");
    const OperationalRecord client_rejection = render_request_rejected(rejected_context);
    failures +=
        check(client_rejection.severity == OperationalSeverity::Info &&
                  client_rejection.message.find("status=rejected") != std::string::npos &&
                  client_rejection.message.find("error_code=\"context_length_exceeded\"") !=
                      std::string::npos &&
                  client_rejection.message.find("sentinel-client-value") == std::string::npos &&
                  client_rejection.message.find('\n') == std::string::npos,
              "operational rejection severity or client-data policy mismatch");
    RequestRejectionLogContext overload_context = rejected_context;
    overload_context.error.status               = 429;
    overload_context.error.code                 = "server_overloaded";
    failures +=
        check(render_request_rejected(overload_context).severity == OperationalSeverity::Warning,
              "operational overload rejection is not warning severity");

    GenerationOutcome outcome;
    outcome.generated_token_ids              = {17, 23, 42};
    outcome.prompt_tokens                    = 401;
    outcome.completion_tokens                = 3;
    outcome.finish_reason                    = ninfer::FinishReason::OutputLimit;
    outcome.metrics.prepare_seconds          = 0.1234567890123;
    outcome.metrics.ttft_seconds             = 0.3580246791357;
    outcome.metrics.vision_seconds           = 0.0;
    outcome.metrics.prefill_seconds          = 0.2345678901234;
    outcome.metrics.decode_seconds           = 5.3456789012345;
    outcome.metrics.total_seconds            = 5.7037035803702;
    outcome.metrics.prefix_cache_hit_tokens  = 101;
    outcome.metrics.prefix_reuse_path        = ninfer::PrefixReusePath::PrivateTurnClosure;
    outcome.metrics.durable_restore_frontier = 101;
    outcome.metrics.durable_loaded_from_ssd  = true;
    outcome.metrics.durable_fallback_reason  = "ssd-successful-replacement";
    outcome.metrics.engine_timing            = {
                   .queue_wait_seconds                   = 0.001,
                   .engine_boundary_exposed_seconds      = 0.001,
                   .program_submit_exposed_seconds       = 0.002,
                   .program_post_exposed_seconds         = 0.003,
                   .engine_commit_output_exposed_seconds = 0.004,
                   .engine_maintenance_exposed_seconds   = 0.005,
                   .device_wait_exposed_seconds          = 0.3,
                   .decode_host_exposed_seconds          = 0.01,
                   .decode_device_wait_exposed_seconds   = 0.2,
                   .prefill_units                        = 4,
                   .decode_rounds                        = 2,
                   .control_units                        = 1,
    };
    outcome.metrics.speculative_backend               = ninfer::SpeculativeBackend::Mtp;
    outcome.metrics.speculative_draft_window          = 3;
    outcome.metrics.speculative_rounds                = 300;
    outcome.metrics.speculative_draft_tokens          = 900;
    outcome.metrics.speculative_accepted_tokens       = 720;
    outcome.metrics.speculative_fallback_steps        = 2;
    outcome.metrics.speculative_accepted_per_position = {290, 240, 190};
    outcome.metrics.materialization                   = {
                          .predicted_now_ns           = 200000,
                          .predicted_future_loss_ns   = 50000,
                          .predicted_total_ns         = 250000,
                          .targets_evaluated          = 7,
                          .projection_work            = 31,
                          .planning_elapsed_ns        = 9000,
                          .search_elapsed_ns          = 6000,
                          .stop_reason                = ninfer::MaterializationStopReason::QueueExhausted,
                          .budget_exhausted           = false,
                          .selected_degradation_units = 2,
                          .selected_maximal_fallback  = false,
    };
    outcome.thinking           = ninfer::ThinkingBudgetStats{.configured_budget     = 256,
                                                             .model_thinking_tokens = 256,
                                                             .injected_tokens       = 19,
                                                             .applied               = true};
    outcome.checkpoints        = {.reuse_loaded      = true,
                                  .restore_committed = true,
                                  .offload_committed = true,
                                  .created_committed = 2,
                                  .capture_aborted   = 1};
    const auto checkpoint_fact = [](ninfer::CheckpointLifecycleOperation operation,
                                    ninfer::CheckpointLifecycleStatus status =
                                        ninfer::CheckpointLifecycleStatus::Committed) {
        return ninfer::CheckpointLifecycleFact{
            .frontier = 64, .operation = operation, .status = status};
    };
    outcome.checkpoint_lifecycle = {
        checkpoint_fact(ninfer::CheckpointLifecycleOperation::Loaded),
        checkpoint_fact(ninfer::CheckpointLifecycleOperation::Restored),
        checkpoint_fact(ninfer::CheckpointLifecycleOperation::Created),
        checkpoint_fact(ninfer::CheckpointLifecycleOperation::Created),
        checkpoint_fact(ninfer::CheckpointLifecycleOperation::Offloaded),
        checkpoint_fact(ninfer::CheckpointLifecycleOperation::Created,
                        ninfer::CheckpointLifecycleStatus::Aborted),
    };

    // Fork-local fields on the operational request line: upstream's restructure dropped
    // speculative decoding and host timings, and this fork restated them. The fixture above
    // pins every value: 720 accepted of 900 drafted over 300 rounds is 3.4 tokens per round
    // at 0.8 acceptance, the five host-exposed components sum to 15 ms, and 2 decode rounds
    // split 0.01 s of decode host work and 0.2 s of device wait.
    const std::string operational = render_request_done(context, outcome).message;
    failures +=
        check(operational.find(" speculative_backend=mtp") != std::string::npos &&
                  operational.find(" speculative_tokens_per_round=3.400") != std::string::npos &&
                  operational.find(" speculative_acceptance=0.800") != std::string::npos,
              "operational request line lost the speculative fields");
    // The fixture configures a 256-token budget with 256 model tokens, 19 injected control
    // tokens and applied=true, so every field on the thinking group is pinned.
    failures += check(operational.find(" thinking_budget_tokens=256") != std::string::npos &&
                          operational.find(" thinking_model_tokens=256") != std::string::npos &&
                          operational.find(" thinking_control_tokens=19") != std::string::npos &&
                          operational.find(" thinking_control=applied") != std::string::npos,
                      "operational request line lost the thinking accounting");
    failures +=
        check(operational.find(" host_exposed_ms=15.000") != std::string::npos &&
                  operational.find(" decode_host_us_per_round=5000.000") != std::string::npos &&
                  operational.find(" decode_wait_us_per_round=100000.000") != std::string::npos,
              "operational request line lost the host timings");

    const Json done = Json::parse(format_request_done_json("serve-test", 3000, context, outcome));
    failures += check(done.at("checkpoint_summary").at("reuse_loaded") == true &&
                          done.at("checkpoint_summary").at("restore_committed") == true &&
                          done.at("checkpoint_summary").at("created_committed") == 2 &&
                          done.at("checkpoint_summary").at("offload_committed") == true &&
                          done.at("checkpoint_summary").at("capture_aborted") == 1,
                      "request terminal lost immutable checkpoint facts");
    failures +=
        check(done.at("result").at("finish_reason") == "output_limit", "finish reason missing");
    failures += check(done.at("result").at("prompt_tokens") == 401, "prompt tokens missing");
    failures += check(done.at("result").at("generated_token_ids") == Json::array({17, 23, 42}),
                      "exact generated token ids missing");
    failures += check(done.at("result").at("computed_prefill_tokens") == 300,
                      "computed prefill tokens missing");
    failures += check(done.at("result").at("prefix_reuse_path") == "private_turn_closure",
                      "prefix reuse path missing");
    failures += check(done.at("result").at("durable_restore").at("frontier_tokens") == 101 &&
                          done.at("result").at("durable_restore").at("ssd_loaded") == true &&
                          done.at("result").at("durable_restore").at("fallback_reason") ==
                              "ssd-successful-replacement",
                      "durable restore classification missing");
    const auto durable_reason = [&](std::string reason) {
        GenerationOutcome classified               = outcome;
        classified.metrics.durable_loaded_from_ssd = false;
        classified.metrics.durable_fallback_reason = std::move(reason);
        return Json::parse(format_request_done_json("serve-test", 3002, context, classified))
            .at("result")
            .at("durable_restore")
            .at("fallback_reason");
    };
    failures += check(durable_reason("ssd-host-state-capacity") == "ssd-host-state-capacity" &&
                          durable_reason("ssd-io-failure") == "ssd-io-failure" &&
                          durable_reason("ssd-validation-failure") == "ssd-validation-failure",
                      "pre-I/O capacity, I/O, and validation classifications collapsed in JSONL");
    failures += check(done.at("result").at("thinking_budget") == 256 &&
                          done.at("result").at("model_thinking_tokens") == 256 &&
                          done.at("result").at("thinking_control_tokens") == 19 &&
                          done.at("result").at("thinking_control_applied") == true,
                      "thinking-control result accounting missing");
    outcome.metrics.prefix_reuse_path = ninfer::PrefixReusePath::PrivateResponseReplay;
    const Json response_restore =
        Json::parse(format_request_done_json("serve-test", 3001, context, outcome));
    failures +=
        check(response_restore.at("result").at("prefix_reuse_path") == "private_response_replay",
              "response checkpoint reuse path missing");
    failures += check(done.at("timings_seconds").at("decode").get<double>() ==
                          outcome.metrics.decode_seconds,
                      "decode time lost precision");
    failures +=
        check(done.at("timings_seconds").at("ttft").get<double>() == outcome.metrics.ttft_seconds,
              "TTFT missing or lost precision");
    failures += check(done.at("speculative").at("backend") == "mtp", "speculative backend missing");
    failures +=
        check(done.at("speculative").at("draft_window") == 3, "speculative draft window missing");
    failures += check(done.at("speculative").at("fallback_steps") == 2,
                      "speculative fallback count missing");
    failures +=
        check(done.at("speculative").at("accepted_per_position") == Json::array({290, 240, 190}),
              "speculative position counts missing");
    failures += check(done.at("materialization").at("predicted_total_ns") == 250000 &&
                          done.at("materialization").at("targets_evaluated") == 7 &&
                          done.at("materialization").at("stop_reason") == "queue_exhausted" &&
                          !done.at("materialization").contains("model_optimal") &&
                          !done.at("materialization").contains("absolute_bound_gap_ns"),
                      "request-owned materialization diagnostics missing");
    failures += check(
        done.at("engine_timing").at("queue_wait_seconds") == 0.001 &&
            std::abs(done.at("engine_timing").at("host_exposed_seconds").at("total").get<double>() -
                     0.015) < 1.0e-15 &&
            done.at("engine_timing").at("decode").at("rounds") == 2 &&
            done.at("engine_timing").at("units").at("prefill") == 4,
        "request Engine timing exposure is incomplete");

    const ninfer::CheckpointLifecycleFact lifecycle_fact{
        .key_digests      = {0x0123456789abcdefULL, 0xfedcba9876543210ULL},
        .frontier         = 192,
        .identity_tag     = 7,
        .ordinal          = 2,
        .role             = ninfer::CheckpointLifecycleRole::LongAnchor,
        .scope            = ninfer::CheckpointLifecycleScope::Private,
        .operation        = ninfer::CheckpointLifecycleOperation::Offloaded,
        .source_tier      = ninfer::CheckpointLifecycleTier::Device,
        .destination_tier = ninfer::CheckpointLifecycleTier::Host,
        .status           = ninfer::CheckpointLifecycleStatus::Committed,
        .state_images     = 1,
        .main_kv_pages    = 3,
        .backend_kv_pages = 4,
        .elapsed_ns       = 12345,
    };
    const Json lifecycle =
        Json::parse(format_checkpoint_lifecycle_json("serve-test", 3500, context, lifecycle_fact));
    failures += check(
        lifecycle.at("event") == "checkpoint_lifecycle" &&
            lifecycle.at("request").at("request_id") == 7 &&
            lifecycle.at("request").at("response_id") == "chatcmpl-correlation-fixture" &&
            lifecycle.at("checkpoint").at("key_digest") == "0123456789abcdeffedcba9876543210" &&
            lifecycle.at("checkpoint").at("frontier") == 192 &&
            lifecycle.at("checkpoint").at("role") == "long_anchor" &&
            lifecycle.at("checkpoint").at("scope") == "private" &&
            lifecycle.at("operation") == "offloaded" && lifecycle.at("status") == "committed" &&
            lifecycle.at("source_tier") == "device" && lifecycle.at("destination_tier") == "host" &&
            lifecycle.at("resources").at("state").at("bytes") == 8192 &&
            lifecycle.at("resources").at("main_kv").at("bytes") == 3 * 5000 &&
            lifecycle.at("resources").at("backend_kv").at("bytes") == 4 * 3000 &&
            lifecycle.at("elapsed_ns") == 12345 &&
            lifecycle.at("kv_capacity").at("device").at("main").at("used_pages") == 25,
        "checkpoint lifecycle identity, tier, quantity, correlation, or KV snapshot mismatch");
    ninfer::CheckpointLifecycleFact displaced = lifecycle_fact;
    displaced.key_digests                     = {0x1111, 0x2222};
    displaced.content_digest                  = std::string(64, 'd');
    displaced.role                            = ninfer::CheckpointLifecycleRole::SharedStablePrefix;
    displaced.scope                           = ninfer::CheckpointLifecycleScope::Shared;
    displaced.operation                       = ninfer::CheckpointLifecycleOperation::Evicted;
    displaced.source_tier                     = ninfer::CheckpointLifecycleTier::Host;
    displaced.destination_tier                = ninfer::CheckpointLifecycleTier::Ssd;
    ninfer::CheckpointLifecycleFact restored  = displaced;
    restored.key_digests                      = {0x3333, 0x4444};
    restored.content_digest                   = std::string(64, 'r');
    restored.operation                        = ninfer::CheckpointLifecycleOperation::Restored;
    restored.source_tier                      = ninfer::CheckpointLifecycleTier::Ssd;
    restored.destination_tier                 = ninfer::CheckpointLifecycleTier::Device;
    const Json displaced_json =
        Json::parse(format_checkpoint_lifecycle_json("serve-test", 3501, context, displaced));
    const Json restored_json =
        Json::parse(format_checkpoint_lifecycle_json("serve-test", 3502, context, restored));
    failures += check(
        displaced_json.at("operation") == "evicted" && displaced_json.at("status") == "committed" &&
            displaced_json.at("source_tier") == "host" &&
            displaced_json.at("destination_tier") == "ssd" &&
            displaced_json.at("checkpoint").at("content_digest") == std::string(64, 'd') &&
            restored_json.at("operation") == "restored" &&
            restored_json.at("status") == "committed" && restored_json.at("source_tier") == "ssd" &&
            restored_json.at("destination_tier") == "device" &&
            restored_json.at("checkpoint").at("content_digest") == std::string(64, 'r'),
        "replacement JSONL did not distinguish displaced and restored owners");
    RequestLogContext background_context;
    background_context.kv_snapshot = context.kv_snapshot;
    const Json background          = Json::parse(
        format_checkpoint_lifecycle_json("serve-test", 3501, background_context, lifecycle_fact));
    failures += check(background.at("request").at("request_id").is_null() &&
                          background.at("request").at("response_id").is_null(),
                      "background checkpoint activity fabricated request correlation");
    ApiError lifecycle_error;
    lifecycle_error.status               = 499;
    lifecycle_error.code                 = "client_disconnected";
    lifecycle_error.checkpoint_lifecycle = {lifecycle_fact};
    failures +=
        check(make_generation_request_failure(lifecycle_error).checkpoint_lifecycle.size() == 1,
              "request failure discarded aborted checkpoint lifecycle facts");
    const RequestFailure disconnected_after_reuse = attach_checkpoint_lifecycle(
        make_client_disconnected_failure(RequestFailurePhase::Transport),
        lifecycle_error.checkpoint_lifecycle);
    failures += check(
        disconnected_after_reuse.classification == RequestFailureClass::ClientDisconnected &&
            disconnected_after_reuse.phase == RequestFailurePhase::Transport &&
            disconnected_after_reuse.checkpoint_lifecycle.size() == 1 &&
            disconnected_after_reuse.checkpoint_lifecycle.front().key_digests ==
                lifecycle_fact.key_digests,
        "attaching settled lifecycle facts changed transport classification or lost identity");

    const std::string media_prompt = format_prompt_markdown(
        "model-visible <|vision_start|><|image_pad|><|vision_end|>",
        std::array<CapturedMediaMetadata, 1>{CapturedMediaMetadata{
            .kind       = ninfer::MediaKind::Image,
            .media_type = "image/png\nunsafe`field",
            .bytes      = 1234,
            .sha256     = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }});
    failures += check(media_prompt.starts_with("model-visible <|vision_start|>") &&
                          media_prompt.find("kind=`image`") != std::string::npos &&
                          media_prompt.find("bytes=`1234`") != std::string::npos &&
                          media_prompt.find("sha256=`aaaaaaaa") != std::string::npos &&
                          media_prompt.find("https://") == std::string::npos &&
                          media_prompt.find("data:") == std::string::npos,
                      "prompt Markdown lost safe media metadata or included a media source");

    const Json error = Json::parse(
        format_request_error_json("serve-test", 4000, context, "generation failed",
                                  summarize_checkpoint_lifecycle(outcome.checkpoint_lifecycle)));
    failures += check(error.at("event") == "request_error", "request error event mismatch");
    failures += check(error.at("error").at("message") == "generation failed",
                      "request error message missing");
    failures += check(error.at("checkpoint_summary").at("created_committed") == 2 &&
                          error.at("checkpoint_summary").at("restore_committed") == true &&
                          error.at("checkpoint_summary").at("capture_aborted") == 1,
                      "request error omitted its lifecycle-derived checkpoint summary");

    const OperationalRecord internal_failure = render_request_failure(
        context,
        make_internal_request_failure(RequestFailurePhase::Generation, "sentinel-internal-detail"));
    failures +=
        check(internal_failure.severity == OperationalSeverity::Error &&
                  internal_failure.message.find("sentinel-internal-detail") == std::string::npos,
              "operational internal failure severity or data policy mismatch");
    const OperationalRecord disconnected = render_request_failure(
        context, make_client_disconnected_failure(RequestFailurePhase::Transport));
    failures += check(disconnected.severity == OperationalSeverity::Info &&
                          disconnected.message.find("status=cancelled") != std::string::npos,
                      "client disconnect is not an informational cancellation");

    ThroughputReport throughput;
    throughput.interval_seconds                         = 2.0;
    throughput.computed_prefill_tokens                  = 100;
    throughput.committed_decode_tokens                  = 40;
    throughput.decode_rounds                            = 10;
    throughput.decode_row_rounds                        = 18;
    throughput.previous.root_selections                 = 2;
    throughput.previous.state_h2d_bytes                 = 100;
    throughput.current.running_requests                 = 2;
    throughput.current.prefilling_requests              = 1;
    throughput.current.decode_ready_requests            = 1;
    throughput.current.waiting_requests                 = 3;
    throughput.current.materializing_requests           = 1;
    throughput.current.capture_pending_requests         = 1;
    throughput.current.terminal_pending_requests        = 1;
    throughput.current.root_selections                  = 3;
    throughput.current.state_h2d_count                  = 1;
    throughput.current.state_h2d_bytes                  = 132;
    throughput.current.state_h2d_seconds                = 0.25;
    throughput.current.device_state_occupied_slots      = 3;
    throughput.current.host_state_occupied_slots        = 1;
    throughput.current.last_selected_frontier_tokens    = 64;
    throughput.current.pressure_spill_pages             = 4;
    throughput.current.pressure_private_owners_degraded = 1;
    throughput.current.pressure_checkpoints_dropped     = 1;
    throughput.current.pressure_searches                = 1;
    throughput.current.host_work                        = {
                               .engine_boundary_ns            = 1000000,
                               .program_submit_ns             = 2000000,
                               .program_post_ns               = 3000000,
                               .engine_commit_output_ns       = 4000000,
                               .engine_maintenance_ns         = 5000000,
                               .device_wait_ns                = 300000000,
                               .decode_host_ns                = 10000000,
                               .decode_device_wait_ns         = 200000000,
                               .prefill_host_ns               = 3000000,
                               .prefill_device_wait_ns        = 50000000,
                               .control_host_ns               = 2000000,
                               .control_device_wait_ns        = 50000000,
                               .prefill_units                 = 4,
                               .control_units                 = 1,
                               .admission_policy_ns           = 1000000,
                               .context_progress_ns           = 2000000,
                               .stats_publication_ns          = 250000,
                               .admission_policy_invocations  = 2,
                               .context_progress_invocations  = 4,
                               .stats_publication_invocations = 5,
    };
    const Json throughput_json =
        Json::parse(format_throughput_json("serve-test", 5000, throughput));
    failures += check(throughput_json.at("event") == "throughput", "throughput event mismatch");
    failures += check(throughput_json.at("tokens").at("computed_prefill") == 100 &&
                          throughput_json.at("tokens").at("committed_decode") == 40,
                      "throughput token deltas mismatch");
    failures += check(throughput_json.at("decode_batch").at("average_size") == 1.8,
                      "throughput batch average mismatch");
    failures += check(throughput_json.at("scheduler").at("materializing") == 1 &&
                          throughput_json.at("scheduler").at("capture_pending") == 1 &&
                          throughput_json.at("scheduler").at("terminal_pending") == 1,
                      "context scheduler gauges missing");
    failures += check(
        std::abs(throughput_json.at("host_work").at("elapsed_seconds").at("total").get<double>() -
                 0.015) < 1.0e-15 &&
            std::abs(throughput_json.at("host_work").at("device_wait_seconds").get<double>() -
                     0.3) < 1.0e-15 &&
            std::abs(throughput_json.at("host_work")
                         .at("decode_host_microseconds_per_round")
                         .get<double>() -
                     1000.0) < 1.0e-12 &&
            std::abs(throughput_json.at("host_work")
                         .at("decode_device_wait_microseconds_per_round")
                         .get<double>() -
                     20000.0) < 1.0e-12 &&
            std::abs(throughput_json.at("host_work")
                         .at("detail_microseconds_per_invocation")
                         .at("stats_publication")
                         .get<double>() -
                     50.0) < 1.0e-12,
        "throughput Host work deltas or normalization are incorrect");

    ThroughputReport zero_rounds;
    const Json zero_rounds_json =
        Json::parse(format_throughput_json("serve-test", 5001, zero_rounds));
    failures += check(
        zero_rounds_json.at("decode_batch").at("average_size").is_null() &&
            zero_rounds_json.at("host_work").at("decode_host_microseconds_per_round").is_null() &&
            zero_rounds_json.at("host_work")
                .at("detail_microseconds_per_invocation")
                .at("admission_policy")
                .is_null(),
        "zero Host-work denominators must serialize as null");
    failures += check(
        throughput_json.at("context_cache").at("selections").at("root") == 1 &&
            throughput_json.at("context_cache").at("state_transfers").at("h2d").at("bytes") == 32 &&
            throughput_json.at("context_cache").at("occupancy").at("device_state_slots") == 3 &&
            throughput_json.at("context_cache").at("pressure").at("spill_pages") == 4 &&
            throughput_json.at("context_cache").at("pressure").at("private_owners_degraded") == 1 &&
            !throughput_json.at("context_cache").contains("last_materialization"),
        "context-cache throughput statistics missing or not interval-scoped");

    const std::filesystem::path log_path = std::filesystem::temp_directory_path() /
                                           ("ninfer-request-log-test-" +
#ifdef _WIN32
                                            std::to_string(static_cast<long long>(::_getpid())) +
#else
                                            std::to_string(static_cast<long long>(::getpid())) +
#endif
                                            ".jsonl");
    std::filesystem::remove(log_path);
    {
        JsonlRequestLog writer(log_path.string());
        writer.write_request_start(context);
    }
    {
        JsonlRequestLog writer(log_path.string());
        writer.write_request_rejected(rejected_context);
        writer.write_request_error(context, "generation failed");
    }
    std::ifstream input(log_path);
    std::string first_line;
    std::string second_line;
    std::string third_line;
    std::string extra_line;
    std::getline(input, first_line);
    std::getline(input, second_line);
    std::getline(input, third_line);
    std::getline(input, extra_line);
    failures += check(!first_line.empty() && !second_line.empty() && !third_line.empty() &&
                          extra_line.empty(),
                      "JSONL writer did not append exactly one flushed line per event");
    if (!first_line.empty() && !second_line.empty() && !third_line.empty()) {
        failures += check(Json::parse(first_line).at("event") == "request_start",
                          "first appended event mismatch");
        failures += check(Json::parse(second_line).at("event") == "request_rejected",
                          "second appended event mismatch");
        failures += check(Json::parse(third_line).at("event") == "request_error",
                          "third appended event mismatch");
    }
    input.close();
    std::filesystem::remove(log_path);

    const std::filesystem::path capture_root =
        std::filesystem::temp_directory_path() /
        ("ninfer-request-content-test-" +
#ifdef _WIN32
         std::to_string(static_cast<long long>(::_getpid())));
#else
         std::to_string(static_cast<long long>(::getpid())));
#endif
    std::filesystem::remove_all(capture_root);
    std::filesystem::create_directories(capture_root);
    const std::filesystem::path capture_log = capture_root / "requests.jsonl";
    const std::filesystem::path content_dir = capture_root / "content";
    RequestLogContext captured_context      = context;
    const std::string rendered_prompt =
        "<|im_start|>user\nrendered prompt secret, not raw JSON<|im_end|>\n";
    captured_context.rendered_prompt   = std::make_shared<const std::string>(rendered_prompt);
    GenerationOutcome captured_outcome = outcome;
    captured_outcome.checkpoint_lifecycle.clear();
    captured_outcome.reasoning  = "private chain of thought";
    captured_outcome.text       = "final assistant content";
    captured_outcome.tool_calls = {{.name = "lookup", .arguments_json = R"({"key":"secret"})"}};
    std::string first_prompt_file;
    {
        JsonlRequestLog writer(capture_log.string(), {}, {}, content_dir);
        writer.write_request_start(captured_context);
        writer.write_request_done(captured_context, captured_outcome);
        failures += check(captured_context.prompt_file.status == "written" &&
                              captured_context.prompt_file.file.starts_with("prompt") &&
                              captured_context.prompt_file.file.ends_with(".md"),
                          "captured prompt did not publish a prefixed Markdown file");
        first_prompt_file = captured_context.prompt_file.file;
    }
    std::ifstream captured_input(capture_log);
    std::string captured_start_line;
    std::string captured_done_line;
    std::getline(captured_input, captured_start_line);
    std::getline(captured_input, captured_done_line);
    const Json captured_start       = Json::parse(captured_start_line);
    const Json captured_done        = Json::parse(captured_done_line);
    const Json& prompt_meta         = captured_start.at("content_files").at("prompt");
    const Json& response_meta       = captured_done.at("content_files").at("response");
    const std::string response_file = response_meta.at("file").get<std::string>();
    failures += check(prompt_meta.at("status") == "written" &&
                          prompt_meta.at("bytes") == rendered_prompt.size() &&
                          prompt_meta.at("sha256").get<std::string>().size() == 64,
                      "prompt content reference is incomplete");
    failures += check(response_meta.at("status") == "written" &&
                          response_file.starts_with("response") && response_file.ends_with(".md") &&
                          first_prompt_file.substr(6) == response_file.substr(8),
                      "prompt and response filenames do not share one request suffix");
    std::ifstream prompt_input(content_dir / first_prompt_file, std::ios::binary);
    const std::string stored_prompt((std::istreambuf_iterator<char>(prompt_input)), {});
    failures += check(stored_prompt == rendered_prompt,
                      "prompt Markdown is not the exact Frontend-rendered text");
    std::ifstream response_input(content_dir / response_file, std::ios::binary);
    const std::string stored_response((std::istreambuf_iterator<char>(response_input)), {});
    const auto sha256_hex = [](std::string_view value) {
        return ninfer::targets::qwen3_6::frontend_internal::sha256_hex(
            ninfer::targets::qwen3_6::frontend_internal::sha256(value));
    };
    failures += check(prompt_meta.at("sha256") == sha256_hex(stored_prompt) &&
                          response_meta.at("sha256") == sha256_hex(stored_response),
                      "JSONL digest does not match the atomically published Markdown file");
    failures += check(stored_response == format_response_markdown(captured_outcome) &&
                          stored_response.find("private chain of thought") != std::string::npos &&
                          stored_response.find("final assistant content") != std::string::npos &&
                          stored_response.find(R"({"key":"secret"})") != std::string::npos,
                      "response Markdown lost reasoning, content, or tool calls");
    failures +=
        check(captured_start_line.find("rendered prompt secret") == std::string::npos &&
                  captured_done_line.find("private chain of thought") == std::string::npos &&
                  captured_done_line.find("final assistant content") == std::string::npos &&
                  captured_done_line.find(R"({"key":"secret"})") == std::string::npos,
              "captured content leaked inline into JSONL");

    const std::filesystem::path protocol_log = capture_root / "protocols.jsonl";
    std::vector<std::string> protocol_responses;
    {
        JsonlRequestLog writer(protocol_log.string(), {}, {}, content_dir);
        const std::array<std::pair<const char*, bool>, 4> protocols{{
            {"openai_chat_completions", false},
            {"openai_responses", false},
            {"anthropic_messages", false},
            {"openai_chat_completions", true},
        }};
        for (std::size_t index = 0; index < protocols.size(); ++index) {
            RequestLogContext protocol_context = context;
            protocol_context.id                = 20 + index;
            protocol_context.protocol          = protocols[index].first;
            protocol_context.stream            = protocols[index].second;
            protocol_context.rendered_prompt = std::make_shared<const std::string>(rendered_prompt);
            writer.write_request_start(protocol_context);
            writer.write_request_done(protocol_context, captured_outcome);
            protocol_responses.push_back("response" + writer.server_instance_id() + "-request-" +
                                         std::to_string(protocol_context.id) + ".md");
        }
    }
    std::ifstream protocol_input(protocol_log);
    std::string protocol_jsonl((std::istreambuf_iterator<char>(protocol_input)), {});
    failures += check(protocol_jsonl.find(rendered_prompt) == std::string::npos &&
                          protocol_jsonl.find(captured_outcome.text) == std::string::npos &&
                          protocol_jsonl.find(captured_outcome.reasoning) == std::string::npos,
                      "one protocol path embedded captured content in JSONL");
    for (const std::string& file : protocol_responses) {
        std::ifstream response(content_dir / file, std::ios::binary);
        const std::string value((std::istreambuf_iterator<char>(response)), {});
        failures += check(value == format_response_markdown(captured_outcome),
                          "protocol or streaming mode changed final response Markdown");
    }

    RequestLogContext restarted_context = context;
    restarted_context.rendered_prompt   = std::make_shared<const std::string>("restart prompt");
    {
        JsonlRequestLog restarted((capture_root / "restart.jsonl").string(), {}, {}, content_dir);
        restarted.write_request_start(restarted_context);
    }
    failures += check(restarted_context.prompt_file.status == "written" &&
                          restarted_context.prompt_file.file != first_prompt_file,
                      "same request ID collided across server instances");

    const std::filesystem::path unusable = capture_root / "not-a-directory";
    { std::ofstream marker(unusable); }
    bool unusable_rejected = false;
    try {
        JsonlRequestLog rejected((capture_root / "unusable.jsonl").string(), {}, {}, unusable);
    } catch (const std::runtime_error&) { unusable_rejected = true; }
    failures += check(unusable_rejected, "unusable content directory did not fail startup");

    const std::filesystem::path failure_dir = capture_root / "failures";
    RequestLogContext failed_context        = context;
    failed_context.rendered_prompt = std::make_shared<const std::string>("still infer this");
    const std::filesystem::path failure_log = capture_root / "failures.jsonl";
    {
        bool prompt_fragment_failed = false;
        JsonlRequestLog writer(failure_log.string(), {}, {}, failure_dir,
                               [&](std::string_view stage, const std::filesystem::path&,
                                   const std::filesystem::path& final) {
                                   const std::string filename = final.filename().string();
                                   if (stage == "fragment_written" &&
                                       filename.starts_with("prompt") && !prompt_fragment_failed) {
                                       prompt_fragment_failed = true;
                                       throw std::runtime_error("injected fragment failure");
                                   }
                                   if (stage == "before_rename" &&
                                       filename.starts_with("response")) {
                                       std::filesystem::create_directory(final);
                                   }
                               });
        writer.write_request_start(failed_context);
        writer.write_request_done(failed_context, captured_outcome);
    }
    std::ifstream failure_input(failure_log);
    std::string failed_start_line;
    std::string failed_done_line;
    std::getline(failure_input, failed_start_line);
    std::getline(failure_input, failed_done_line);
    const Json failed_start = Json::parse(failed_start_line);
    const Json failed_done  = Json::parse(failed_done_line);
    failures +=
        check(failed_start.at("content_files").at("prompt").at("status") == "failed" &&
                  failed_start.at("content_files").at("prompt").at("file").is_null() &&
                  failed_start.at("content_files").at("prompt").at("error_class") ==
                      "format_or_io_failure" &&
                  failed_done.at("event") == "request_done" &&
                  failed_done.at("content_files").at("response").at("status") == "failed" &&
                  failed_done.at("content_files").at("response").at("file").is_null() &&
                  failed_done.at("content_files").at("response").at("error_class") ==
                      "atomic_rename_failed",
              "injected write/rename failure emitted a false reference or failed inference");
    bool partial_found = false;
    for (const auto& entry : std::filesystem::directory_iterator(failure_dir)) {
        if (entry.path().filename().string().find(".tmp") != std::string::npos) {
            partial_found = true;
        }
    }
    failures += check(!partial_found, "content write failure left a temporary partial file");
    std::filesystem::remove_all(capture_root);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
