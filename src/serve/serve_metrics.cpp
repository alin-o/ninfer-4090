#include "serve/serve_metrics.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::serve {

namespace {

void append_counter(std::string& out, const char* name, std::uint64_t value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %llu\n", name, static_cast<unsigned long long>(value));
    out += line;
}

void append_counter(std::string& out, const char* name, double value) {
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.6f\n", name, value);
    out += line;
}

constexpr const char* cache_role_name(ContextCacheMetricRole role) {
    switch (role) {
    case ContextCacheMetricRole::Harness:
        return "harness";
    case ContextCacheMetricRole::Project:
        return "project";
    case ContextCacheMetricRole::ConversationHead:
        return "conversation_head";
    case ContextCacheMetricRole::Transient:
        return "transient";
    case ContextCacheMetricRole::Count:
        break;
    }
    return "unknown";
}

constexpr const char* cache_placement_name(ContextCacheMetricPlacement placement) {
    switch (placement) {
    case ContextCacheMetricPlacement::Device:
        return "device";
    case ContextCacheMetricPlacement::Host:
        return "host";
    case ContextCacheMetricPlacement::Both:
        return "both";
    case ContextCacheMetricPlacement::Count:
        break;
    }
    return "unknown";
}

constexpr const char* cache_pin_name(ContextCacheMetricPin pin) {
    switch (pin) {
    case ContextCacheMetricPin::Unpinned:
        return "unpinned";
    case ContextCacheMetricPin::Pinned:
        return "pinned";
    case ContextCacheMetricPin::Count:
        break;
    }
    return "unknown";
}

constexpr const char* cache_identity_name(ContextCacheMetricIdentity identity) {
    switch (identity) {
    case ContextCacheMetricIdentity::None:
        return "none";
    case ContextCacheMetricIdentity::Explicit:
        return "explicit";
    case ContextCacheMetricIdentity::InitialPrefix:
        return "initial_prefix";
    case ContextCacheMetricIdentity::Count:
        break;
    }
    return "unknown";
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
    const std::uint64_t prompt =
        outcome.prompt_tokens > 0 ? static_cast<std::uint64_t>(outcome.prompt_tokens) : 0;

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
    append_counter(out, "ninfer:shared_ssd_manifest_records",
                   static_cast<std::uint64_t>(live.shared_ssd_manifest_records));
    append_counter(out, "ninfer:shared_ssd_manifest_bytes", live.shared_ssd_manifest_bytes);
    append_counter(out, "ninfer:shared_ssd_transfer_jobs{state=\"queued\"}",
                   static_cast<std::uint64_t>(live.shared_ssd_queued_jobs));
    append_counter(out, "ninfer:shared_ssd_transfer_jobs{state=\"active\"}",
                   static_cast<std::uint64_t>(live.shared_ssd_active_jobs));
    append_counter(out, "ninfer:shared_ssd_staging_bytes", live.shared_ssd_staging_bytes);
    append_counter(out, "ninfer:shared_ssd_peak_staging_bytes", live.shared_ssd_peak_staging_bytes);
    append_counter(out, "ninfer:shared_ssd_writes_total{result=\"completed\"}",
                   live.shared_ssd_writes_completed);
    append_counter(out, "ninfer:shared_ssd_writes_total{result=\"failed\"}",
                   live.shared_ssd_writes_failed);
    append_counter(out, "ninfer:shared_ssd_writes_total{result=\"coalesced\"}",
                   live.shared_ssd_writes_coalesced);
    append_counter(out, "ninfer:shared_ssd_loads_total{result=\"completed\"}",
                   live.shared_ssd_loads_completed);
    append_counter(out, "ninfer:shared_ssd_loads_total{result=\"failed\"}",
                   live.shared_ssd_loads_failed);
    append_counter(out, "ninfer:shared_ssd_loads_total{result=\"coalesced\"}",
                   live.shared_ssd_loads_coalesced);
    append_counter(out, "ninfer:shared_ssd_hits_total{temperature=\"loaded\"}",
                   live.shared_ssd_loaded_hits);
    append_counter(out, "ninfer:shared_ssd_hits_total{temperature=\"warm\"}",
                   live.shared_ssd_warm_hits);
    append_counter(out, "ninfer:shared_ssd_quota_rejections_total",
                   live.shared_ssd_quota_rejections);
    append_counter(out, "ninfer:shared_ssd_corrupt_records_total", live.shared_ssd_corrupt_records);
    append_counter(out, "ninfer:shared_ssd_io_nanoseconds_total", live.shared_ssd_io_nanoseconds);
    append_counter(out, "ninfer:shared_ssd_validation_nanoseconds_total",
                   live.shared_ssd_validation_nanoseconds);
    append_counter(out, "ninfer:shared_ssd_adoption_nanoseconds_total",
                   live.shared_ssd_adoption_nanoseconds);
    append_counter(out, "ninfer:shared_ssd_export_claims{state=\"pending\"}",
                   static_cast<std::uint64_t>(live.shared_ssd_pending_export_claims));
    append_counter(out, "ninfer:shared_ssd_directory_records{state=\"unpublished\"}",
                   static_cast<std::uint64_t>(live.shared_ssd_unpublished_records));
    append_counter(out, "ninfer:shared_ssd_directory_bytes{state=\"unpublished\"}",
                   live.shared_ssd_unpublished_bytes);
    append_counter(out, "ninfer:session_publications_total{identity=\"explicit\"}",
                   live.session_publications_explicit_total);
    append_counter(out, "ninfer:session_publications_total{identity=\"initial_prefix\"}",
                   live.session_publications_initial_prefix_total);
    append_counter(out, "ninfer:session_supersessions_total", live.session_supersessions_total);
    append_counter(out, "ninfer:session_late_publications_rejected_total",
                   live.session_late_publications_rejected_total);
    append_counter(
        out,
        "ninfer:context_cache_reclaimed_capacity_total{tier=\"device\",resource=\"state_slots\"}",
        live.reclaimed_device_state_slots_total);
    append_counter(
        out,
        "ninfer:context_cache_reclaimed_capacity_total{tier=\"device\",resource=\"main_kv_pages\"}",
        live.reclaimed_device_main_kv_pages_total);
    append_counter(out,
                   "ninfer:context_cache_reclaimed_capacity_total{tier=\"device\",resource="
                   "\"backend_kv_pages\"}",
                   live.reclaimed_device_backend_kv_pages_total);
    append_counter(
        out,
        "ninfer:context_cache_reclaimed_capacity_total{tier=\"host\",resource=\"state_slots\"}",
        live.reclaimed_host_state_slots_total);
    append_counter(
        out, "ninfer:context_cache_reclaimed_capacity_total{tier=\"host\",resource=\"kv_bytes\"}",
        live.reclaimed_host_kv_bytes_total);
    for (std::uint8_t raw_role = 0;
         raw_role < static_cast<std::uint8_t>(ContextCacheMetricRole::Count); ++raw_role) {
        const auto role = static_cast<ContextCacheMetricRole>(raw_role);
        for (std::uint8_t raw_placement = 0;
             raw_placement < static_cast<std::uint8_t>(ContextCacheMetricPlacement::Count);
             ++raw_placement) {
            const auto placement = static_cast<ContextCacheMetricPlacement>(raw_placement);
            for (std::uint8_t raw_pin = 0;
                 raw_pin < static_cast<std::uint8_t>(ContextCacheMetricPin::Count); ++raw_pin) {
                const auto pin = static_cast<ContextCacheMetricPin>(raw_pin);
                for (std::uint8_t raw_identity = 0;
                     raw_identity < static_cast<std::uint8_t>(ContextCacheMetricIdentity::Count);
                     ++raw_identity) {
                    const auto identity = static_cast<ContextCacheMetricIdentity>(raw_identity);
                    char name[240];
                    std::snprintf(name, sizeof(name),
                                  "ninfer:context_cache_owners{role=\"%s\",placement=\"%s\",pin=\"%"
                                  "s\",identity=\"%s\"}",
                                  cache_role_name(role), cache_placement_name(placement),
                                  cache_pin_name(pin), cache_identity_name(identity));
                    append_counter(out, name,
                                   static_cast<std::uint64_t>(
                                       live.context_cache_owners[context_cache_owner_metric_index(
                                           role, placement, pin, identity)]));
                }
            }
        }
    }
    return out;
}

} // namespace ninfer::serve
