#include "serve/serve_metrics.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::serve {

namespace {

void append_counter(std::string& out, const char* name, std::uint64_t value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %llu\n", name,
                  static_cast<unsigned long long>(value));
    out += line;
}

void append_counter(std::string& out, const char* name, double value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.6f\n", name, value);
    out += line;
}

} // namespace

void ServeMetrics::begin_request(std::uint64_t id, int prompt_tokens) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_[id] = prompt_tokens > 0 ? prompt_tokens : 0;
}

void ServeMetrics::end_request(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_.erase(id);
}

std::vector<std::pair<std::uint64_t, int>> ServeMetrics::active_snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {active_.begin(), active_.end()};
}

void ServeMetrics::record(const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    const std::uint64_t cached = m.prefix_cache_hit_tokens;
    const std::uint64_t prompt = outcome.prompt_tokens > 0
                                     ? static_cast<std::uint64_t>(outcome.prompt_tokens)
                                     : 0;

    const std::lock_guard<std::mutex> lock(mutex_);
    requests_total_ += 1;
    prefix_cache_hit_tokens_total_ += cached;
    speculative_draft_tokens_total_ += m.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += m.speculative_accepted_tokens;
    last_completed_.prompt_tokens = static_cast<int>(prompt);
    // Clamped like computed_prefill above: a cache figure reported larger
    // than the prompt must not advertise more resident tokens than exist.
    last_completed_.cached_tokens = static_cast<int>(std::min(cached, prompt));
}

ServeMetrics::LastCompleted ServeMetrics::last_completed() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return last_completed_;
}

std::string ServeMetrics::render(std::uint32_t max_concurrency,
                                 const ninfer::RuntimeStats& live) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t in_flight  = active_.size();
    const std::uint64_t processing = std::min<std::uint64_t>(in_flight, max_concurrency);
    std::string out;
    out.reserve(1024);
    append_counter(out, "llamacpp:prompt_tokens_total", live.computed_prefill_tokens);
    append_counter(out, "llamacpp:prompt_seconds_total", live.prefill_seconds_total);
    append_counter(out, "llamacpp:tokens_predicted_total", live.committed_decode_tokens);
    append_counter(out, "llamacpp:tokens_predicted_seconds_total", live.decode_seconds_total);
    append_counter(out, "llamacpp:requests_processing", processing);
    append_counter(out, "llamacpp:requests_deferred", in_flight - processing);
    append_counter(out, "ninfer:requests_total", requests_total_);
    append_counter(out, "ninfer:prefix_cache_hit_tokens_total", prefix_cache_hit_tokens_total_);
    append_counter(out, "ninfer:draft_tokens_total", speculative_draft_tokens_total_);
    append_counter(out, "ninfer:draft_accepted_tokens_total", speculative_accepted_tokens_total_);
    append_counter(out, "ninfer:auto_save_queued_jobs",
                   static_cast<std::uint64_t>(live.auto_save_queued_jobs));
    append_counter(out, "ninfer:auto_save_queued_bytes", live.auto_save_queued_bytes);
    append_counter(out, "ninfer:auto_save_in_flight_jobs",
                   static_cast<std::uint64_t>(live.auto_save_in_flight_jobs));
    append_counter(out, "ninfer:auto_save_in_flight_bytes", live.auto_save_in_flight_bytes);
    append_counter(out, "ninfer:auto_save_reserved_jobs",
                   static_cast<std::uint64_t>(live.auto_save_reserved_jobs));
    append_counter(out, "ninfer:auto_save_reserved_bytes", live.auto_save_reserved_bytes);
    append_counter(out, "ninfer:auto_save_rejected_jobs_total", live.auto_save_rejected_jobs);
    return out;
}

} // namespace ninfer::serve
