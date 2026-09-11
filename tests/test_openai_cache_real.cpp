#include "serve/generation_service.h"
#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/request_events.h"
#include "serve/request_log.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::serve;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

RuntimeStats settled_stats(const GenerationService& service) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto stats = service.runtime_stats();
        if (stats.running_requests == 0 && stats.waiting_requests == 0 &&
            stats.materializing_requests == 0) {
            return stats;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("OpenAI cache request did not settle");
}

ServeOptions options(const char* artifact) {
    ServeOptions result;
    result.artifact_path                           = artifact;
    result.max_context                             = 4096;
    result.kv_capacity                             = KvCapacityPolicy::explicit_capacity(4096);
    result.max_concurrency                         = 1;
    result.pending_timeout_ms                      = 120000;
    result.prefill_chunk                           = 1024;
    result.kv_cache                                = KvCacheStorage::RK4V4E8;
    result.speculative.backend                     = SpeculativeBackend::Mtp;
    result.speculative.draft_tokens                = 3;
    result.speculative.proposal_head               = ProposalHead::Optimized;
    result.context_cache.device_state_slots        = 1;
    result.context_cache.host_state_slots          = 16;
    result.context_cache.host_kv_capacity_bytes    = 1ULL << 30;
    result.context_cache.max_private_continuations = 4;
    result.context_cache.max_shared_prefixes       = 8;
    result.context_cache.max_long_anchors_per_continuation = 0;
    result.enable_thinking                                 = false;
    result.preserve_thinking                               = true;
    result.greedy                                          = true;
    result.sampling_overrides.seed                         = 42;
    return result;
}

GenerationOutcome generate_history(GenerationService& service, const Json& history, bool responses,
                                   bool explicit_only = false) {
    Json body{{"model", "qwen3.8"}};
    if (explicit_only) { body["prompt_cache_options"] = Json{{"mode", "explicit"}}; }
    GenerationRequest generation;
    ContextCacheHints hints;
    if (responses) {
        body["store"]             = false;
        body["input"]             = history;
        body["max_output_tokens"] = 8;
        const auto request        = parse_openai_responses_create_request(body, RequestLimits{});
        OpenAIResponsesStore store(8, 1ULL << 20);
        auto resolved =
            resolve_openai_responses_prompt(request.prompt, store, "resp_cache", request.store);
        generation = std::move(resolved.generation);
        hints      = std::move(resolved.cache_hints);
    } else {
        body["messages"]   = history;
        body["max_tokens"] = 8;
        generation         = parse_chat_completion_request(body, RequestLimits{}).generation;
    }
    // Exercise the production merge of protocol markers/policy with Responses retention hints.
    auto prepared = service.prepare(generation, GenerationConsumerMode::Aggregate, {}, hints);
    auto result   = service.run(prepared, nullptr);
    (void)settled_stats(service);
    require(!result.generated_token_ids.empty(), "OpenAI cache fixture generated no tokens");
    return result;
}

GenerationOutcome generate(GenerationService& service, const std::string& instructions,
                           const std::string& user, bool responses, bool explicit_only = false) {
    return generate_history(service,
                            Json::array({Json{{"role", "system"}, {"content", instructions}},
                                         Json{{"role", "user"}, {"content", user}}}),
                            responses, explicit_only);
}

void exercise_disconnect_after_checkpoint_reuse(const char* artifact) {
    GenerationService service(options(artifact));
    const Json history =
        Json::array({Json{{"role", "user"}, {"content", "Retain this exact disconnect fixture."}}});
    const GenerationOutcome seeded = generate_history(service, history, false);
    require(seeded.checkpoints.created_committed != 0,
            "disconnect fixture did not create a reusable checkpoint");

    Json body{{"model", "qwen3.8"}, {"messages", history}, {"max_tokens", 8}, {"stream", true}};
    GenerationRequest request = parse_chat_completion_request(body, RequestLimits{}).generation;
    PreparedRequest prepared  = service.prepare(request, GenerationConsumerMode::Streaming);
    StreamSink sink;
    sink.on_content   = [](const std::string&) { throw ClientDisconnected(); };
    sink.on_reasoning = [](const std::string&) { throw ClientDisconnected(); };
    bool disconnected = false;
    try {
        (void)service.run(prepared, &sink);
    } catch (const ClientDisconnected&) { disconnected = true; }

    const auto loaded =
        std::find_if(prepared.failure_checkpoint_lifecycle.begin(),
                     prepared.failure_checkpoint_lifecycle.end(), [](const auto& fact) {
                         return (fact.operation == CheckpointLifecycleOperation::Loaded ||
                                 fact.operation == CheckpointLifecycleOperation::Restored) &&
                                fact.status == CheckpointLifecycleStatus::Committed &&
                                fact.frontier != 0 &&
                                fact.key_digests != std::array<std::uint64_t, 2>{};
                     });
    const RequestFailure failure = attach_checkpoint_lifecycle(
        make_client_disconnected_failure(RequestFailurePhase::Transport),
        prepared.failure_checkpoint_lifecycle);
    require(disconnected && loaded != prepared.failure_checkpoint_lifecycle.end() &&
                failure.classification == RequestFailureClass::ClientDisconnected &&
                summarize_checkpoint_lifecycle(failure.checkpoint_lifecycle).restore_committed,
            "disconnect after reuse lost settled facts or changed transport classification");
}

void exercise_harness(const char* artifact) {
    std::string harness;
    for (unsigned index = 0; index < 1500; ++index) { harness += "stable "; }
    harness += "\n=== CACHE_BREAKPOINT ===\n";
    const auto instructions = [&](const std::string& workspace) {
        return harness + "Current working directory: " + workspace;
    };
    GenerationOutcome restored;
    GenerationOutcome continued;
    Json continuation;
    {
        GenerationService service(options(artifact));
        // Explicit-only with no markers must not seed any automatic shared writes.
        (void)generate(service, instructions("explicit"), "Reply yes.", true, true);
        const auto seed = generate(service, instructions("first"), "Reply yes.", true);
        require(seed.metrics.prefix_cache_hit_tokens == 0,
                "explicit-only request unexpectedly published an automatic harness");
        const auto before = settled_stats(service);
        // Both the volatile instruction suffix and the conversation change. Private endpoints,
        // full instructions and last-content markers cannot match this request.
        restored         = generate(service, instructions("second"), "Reply no.", false);
        const auto after = settled_stats(service);
        std::cout << "openai_harness prompt_tokens=" << restored.prompt_tokens
                  << " reused_tokens=" << restored.metrics.prefix_cache_hit_tokens
                  << " path=" << static_cast<int>(restored.metrics.prefix_reuse_path)
                  << " state_d2h=" << before.state_d2h_count
                  << " state_h2d=" << after.state_h2d_count - before.state_h2d_count
                  << " main_h2d_pages=" << after.main_kv_h2d_pages - before.main_kv_h2d_pages
                  << " mtp_h2d_pages=" << after.backend_kv_h2d_pages - before.backend_kv_h2d_pages
                  << '\n';
        require(restored.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
                    restored.metrics.prefix_cache_hit_tokens >= 1500,
                "changed OpenAI conversation missed its shared harness");
        require(after.state_h2d_count > before.state_h2d_count,
                "shared harness regression did not exercise RAM State reload");
        // Explicit-only remains able to read an existing exact shared prefix.
        const auto explicit_read =
            generate(service, instructions("third"), "Reply yes.", true, true);
        require(explicit_read.metrics.prefix_reuse_path == PrefixReusePath::SharedStablePrefix &&
                    explicit_read.metrics.prefix_cache_hit_tokens ==
                        restored.metrics.prefix_cache_hit_tokens,
                "explicit-only request could not read the existing harness");
        const auto changed = generate(service,
                                      "A different harness.\n=== CACHE_BREAKPOINT ===\n"
                                      "Current working directory: second",
                                      "Reply no.", true);
        require(changed.metrics.prefix_cache_hit_tokens == 0,
                "changed harness incorrectly reused an unrelated prefix");
        require(restored.finish_reason == FinishReason::StopToken,
                "continuation fixture must include the completed assistant response");
        continuation                = Json::array({
            Json{{"role", "system"}, {"content", instructions("second")}},
            Json{{"role", "user"}, {"content", "Reply no."}},
            Json{{"role", "assistant"}, {"content", restored.text}},
            Json{{"role", "user"}, {"content", "Continue briefly."}},
        });
        const auto before_continue  = settled_stats(service);
        continued                   = generate_history(service, continuation, true);
        const auto after_continue   = settled_stats(service);
        const auto completed_tokens = restored.prompt_tokens + restored.completion_tokens;
        std::cout << "openai_continuation completed_tokens=" << completed_tokens
                  << " prompt_tokens=" << continued.prompt_tokens
                  << " reused_tokens=" << continued.metrics.prefix_cache_hit_tokens
                  << " path=" << static_cast<int>(continued.metrics.prefix_reuse_path)
                  << " state_h2d="
                  << after_continue.state_h2d_count - before_continue.state_h2d_count << '\n';
        require(continued.metrics.prefix_reuse_path == PrefixReusePath::PrivateEndpoint &&
                    continued.metrics.prefix_cache_hit_tokens >= completed_tokens - 1,
                "Responses continuation lost the completed assistant response");
    }
    auto cold_options               = options(artifact);
    cold_options.allow_prefix_reuse = false;
    GenerationService cold_service(cold_options);
    const auto cold = generate(cold_service, instructions("second"), "Reply no.", false);
    require(cold.metrics.prefix_cache_hit_tokens == 0 &&
                cold.generated_token_ids == restored.generated_token_ids &&
                cold.metrics.speculative_draft_tokens ==
                    restored.metrics.speculative_draft_tokens &&
                cold.metrics.speculative_accepted_tokens ==
                    restored.metrics.speculative_accepted_tokens,
            "RAM harness reuse diverged from cold tokens or MTP counters");
    std::cout << "openai_harness exact_cold_match=true\n";
    const auto cold_continuation = generate_history(cold_service, continuation, true);
    require(cold_continuation.metrics.prefix_cache_hit_tokens == 0 &&
                cold_continuation.generated_token_ids == continued.generated_token_ids &&
                cold_continuation.metrics.speculative_draft_tokens ==
                    continued.metrics.speculative_draft_tokens &&
                cold_continuation.metrics.speculative_accepted_tokens ==
                    continued.metrics.speculative_accepted_tokens,
            "Responses continuation diverged from cold tokens or MTP counters");
    std::cout << "openai_continuation exact_cold_match=true\n";
}

struct TemporaryDirectory {
    TemporaryDirectory() {
        std::string pattern = "/tmp/ninfer-request-capture-real-XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const char* created = ::mkdtemp(writable.data());
        if (created == nullptr) { throw std::runtime_error("capture mkdtemp failed"); }
        path = created;
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)), {});
}

std::string base64(std::span<const std::uint8_t> bytes) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size() + 2U) / 3U * 4U);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3U) {
        const std::uint32_t first  = bytes[offset];
        const std::uint32_t second = offset + 1U < bytes.size() ? bytes[offset + 1U] : 0U;
        const std::uint32_t third  = offset + 2U < bytes.size() ? bytes[offset + 2U] : 0U;
        const std::uint32_t value  = (first << 16U) | (second << 8U) | third;
        encoded.push_back(alphabet[(value >> 18U) & 63U]);
        encoded.push_back(alphabet[(value >> 12U) & 63U]);
        encoded.push_back(offset + 1U < bytes.size() ? alphabet[(value >> 6U) & 63U] : '=');
        encoded.push_back(offset + 2U < bytes.size() ? alphabet[value & 63U] : '=');
    }
    return encoded;
}

std::vector<std::uint8_t> capture_ppm() {
    constexpr int width      = 32;
    constexpr int height     = 32;
    const std::string header = "P6\n32 32\n255\n";
    std::vector<std::uint8_t> image(header.begin(), header.end());
    image.reserve(image.size() + width * height * 3U);
    for (int pixel = 0; pixel < width * height; ++pixel) {
        image.push_back(static_cast<std::uint8_t>(pixel));
        image.push_back(static_cast<std::uint8_t>(pixel * 3));
        image.push_back(static_cast<std::uint8_t>(pixel * 7));
    }
    return image;
}

struct CapturedGeneration {
    GenerationOutcome outcome;
    std::string prompt;
    std::string response;
};

CapturedGeneration capture_generation(GenerationService& service, JsonlRequestLog& writer,
                                      GenerationRequest request, std::string protocol,
                                      std::uint64_t request_id, bool stream) {
    PreparedRequest prepared  = service.prepare(request, stream ? GenerationConsumerMode::Streaming
                                                                : GenerationConsumerMode::Aggregate);
    RequestLogContext context = make_request_log_context(
        request_id, std::move(protocol), request,
        RequestLogMetadata{.model       = "qwen3.8",
                           .response_id = "capture-response-" + std::to_string(request_id),
                           .stream      = stream,
                           .output_tokens_explicit = true},
        prepared);
    writer.write_request_start(context);
    StreamSink sink;
    GenerationOutcome outcome = service.run(prepared, stream ? &sink : nullptr);
    writer.write_request_done(context, outcome);
    const std::string suffix =
        writer.server_instance_id() + "-request-" + std::to_string(request_id) + ".md";
    return {.outcome = std::move(outcome),
            .prompt  = read_file(service.options().request_log_content_dir / ("prompt" + suffix)),
            .response =
                read_file(service.options().request_log_content_dir / ("response" + suffix))};
}

void exercise_protocol_content_capture(const char* artifact) {
    TemporaryDirectory temporary;
    ServeOptions configured            = options(artifact);
    configured.request_log_jsonl       = temporary.path / "requests.jsonl";
    configured.request_log_content_dir = temporary.path / "content";
    configured.context_cache           = ContextCacheOptions{.enabled = false};
    configured.enable_vision           = true;
    GenerationService service(configured);
    JsonlRequestLog writer(configured.request_log_jsonl, artifact, {},
                           configured.request_log_content_dir);

    const std::vector<std::uint8_t> media_bytes = capture_ppm();
    const std::string media_payload             = base64(media_bytes);
    const std::string data_uri = "data:image/x-portable-pixmap;base64," + media_payload;
    const Json chat_body{
        {"model", "qwen3.8"},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content", Json::array({Json{{"type", "text"}, {"text", "capture-chat-sentinel"}},
                                      Json{{"type", "image_url"},
                                           {"image_url", Json{{"url", data_uri}}}}})}}})},
        {"max_tokens", 4}};
    GenerationRequest chat = parse_chat_completion_request(chat_body, RequestLimits{}).generation;
    const CapturedGeneration chat_aggregate =
        capture_generation(service, writer, chat, "openai_chat_completions", 1, false);
    GenerationRequest chat_stream =
        parse_chat_completion_request(chat_body, RequestLimits{}).generation;
    const CapturedGeneration streamed = capture_generation(service, writer, std::move(chat_stream),
                                                           "openai_chat_completions", 2, true);

    OpenAIResponsesStore store(4, 1ULL << 20);
    const auto responses_request = parse_openai_responses_create_request(
        Json{{"model", "qwen3.8"},
             {"input",
              Json::array({
                  Json{{"role", "user"},
                       {"content",
                        Json::array(
                            {Json{{"type", "input_text"}, {"text", "capture-responses-sentinel"}},
                             Json{{"type", "input_image"}, {"image_url", data_uri}}})}},
              })},
             {"store", false},
             {"max_output_tokens", 4}},
        RequestLimits{});
    auto responses = resolve_openai_responses_prompt(responses_request.prompt, store,
                                                     "capture-responses", false);
    const CapturedGeneration responses_capture = capture_generation(
        service, writer, std::move(responses.generation), "openai_responses", 3, false);

    AnthropicThinkingSigner::Key signing_key{};
    AnthropicThinkingSigner signer(signing_key);
    GenerationRequest anthropic =
        parse_anthropic_messages_request(
            Json{{"model", "qwen3.8"},
                 {"messages",
                  Json::array({Json{
                      {"role", "user"},
                      {"content",
                       Json::array({Json{{"type", "text"}, {"text", "capture-anthropic-sentinel"}},
                                    Json{{"type", "image"},
                                         {"source", Json{{"type", "base64"},
                                                         {"media_type", "image/x-portable-pixmap"},
                                                         {"data", media_payload}}}}})}}})},
                 {"max_tokens", 4}},
            RequestLimits{}, signer)
            .generation;
    const CapturedGeneration anthropic_capture =
        capture_generation(service, writer, std::move(anthropic), "anthropic_messages", 4, false);

    require(chat_aggregate.prompt.find("capture-chat-sentinel") != std::string::npos &&
                responses_capture.prompt.find("capture-responses-sentinel") != std::string::npos &&
                anthropic_capture.prompt.find("capture-anthropic-sentinel") != std::string::npos,
            "protocol capture did not persist Frontend-rendered prompt semantics");
    require(chat_aggregate.prompt.find("\"messages\"") == std::string::npos &&
                responses_capture.prompt.find("\"input\"") == std::string::npos,
            "protocol capture persisted raw HTTP JSON instead of rendered prompt text");
    require(chat_aggregate.prompt.find("## Non-text media") != std::string::npos &&
                responses_capture.prompt.find("## Non-text media") != std::string::npos &&
                anthropic_capture.prompt.find("## Non-text media") != std::string::npos &&
                chat_aggregate.prompt.find(media_payload) == std::string::npos &&
                responses_capture.prompt.find(data_uri) == std::string::npos &&
                anthropic_capture.prompt.find(media_payload) == std::string::npos,
            "protocol-acquired media was not reduced to safe type/size/digest metadata");
    require(streamed.outcome.text == chat_aggregate.outcome.text &&
                streamed.outcome.reasoning == chat_aggregate.outcome.reasoning &&
                streamed.response == chat_aggregate.response,
            "streaming and aggregate generation produced different final Markdown");
    require(chat_aggregate.response == format_response_markdown(chat_aggregate.outcome) &&
                responses_capture.response == format_response_markdown(responses_capture.outcome) &&
                anthropic_capture.response == format_response_markdown(anthropic_capture.outcome),
            "a protocol capture did not preserve the final logical response");
    const std::string jsonl = read_file(configured.request_log_jsonl);
    require(jsonl.find(media_payload) == std::string::npos &&
                jsonl.find("capture-chat-sentinel") == std::string::npos &&
                jsonl.find("capture-responses-sentinel") == std::string::npos &&
                jsonl.find("capture-anthropic-sentinel") == std::string::npos,
            "JSONL embedded acquired media or model-visible text");
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    int devices          = 0;
    if (artifact == nullptr || *artifact == '\0' || cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices == 0) {
        std::cout << "skip: OpenAI cache regression requires Qwen3.8 weights and CUDA\n";
        return 77;
    }
    try {
        exercise_harness(artifact);
        exercise_disconnect_after_checkpoint_reuse(artifact);
        exercise_protocol_content_capture(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
