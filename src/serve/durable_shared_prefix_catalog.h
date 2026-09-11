#pragma once

#include "ninfer/engine.h"
#include "runtime/engine/durable_shared_snapshot_access.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::serve {

struct DurableSharedPrefixCatalogOptions {
    std::filesystem::path directory;
    std::uint32_t max_records   = 16;
    std::uint64_t max_bytes     = 64ULL << 30U;
    std::uint32_t workers       = 2;
    std::uint32_t max_jobs      = 4;
    std::uint64_t staging_bytes = 4ULL << 30U;
    // Deterministic crash-window seam. Production leaves it empty.
    std::function<void()> before_manifest_publish;
    // Deterministic single-flight/queue seam. Production leaves it empty.
    std::function<void()> before_payload_read;
    // Deterministic first-load registration race seam. Production leaves it empty.
    std::function<void()> before_load_registration;
};

struct DurableSharedPrefixCatalogStats {
    std::uint32_t manifest_records       = 0;
    std::uint64_t manifest_bytes         = 0;
    std::uint32_t queued_jobs            = 0;
    std::uint32_t active_jobs            = 0;
    std::uint64_t staging_bytes          = 0;
    std::uint64_t peak_staging_bytes     = 0;
    std::uint64_t writes_completed       = 0;
    std::uint64_t writes_failed          = 0;
    std::uint64_t writes_coalesced       = 0;
    std::uint64_t loads_completed        = 0;
    std::uint64_t loads_failed           = 0;
    std::uint64_t loads_coalesced        = 0;
    std::uint64_t loaded_hits            = 0;
    std::uint64_t warm_hits              = 0;
    std::uint64_t quota_rejections       = 0;
    std::uint64_t corrupt_records        = 0;
    std::uint64_t io_nanoseconds         = 0;
    std::uint64_t validation_nanoseconds = 0;
    std::uint64_t adoption_nanoseconds   = 0;
    std::uint32_t pending_export_claims  = 0;
    std::uint32_t unpublished_records    = 0;
    std::uint64_t unpublished_bytes      = 0;
};

struct DurableSharedPrefixRestore {
    std::uint32_t frontier = 0;
    bool loaded_from_ssd   = false;
    bool warm_available    = false;
    std::string fallback_reason;
};

// Gateway-owned filesystem transport. Engine supplies opaque, Program-produced records and owns
// adoption; this class owns bounded metadata, worker/I/O lifetimes and crash-consistent publish.
class DurableSharedPrefixCatalog {
public:
    using Candidate = runtime::DurableSharedSnapshotAccess::Candidate;
    using Snapshot  = runtime::DurableSharedSnapshotAccess::Snapshot;
    using Clock     = std::chrono::steady_clock;

    explicit DurableSharedPrefixCatalog(DurableSharedPrefixCatalogOptions options);
    ~DurableSharedPrefixCatalog() noexcept;

    DurableSharedPrefixCatalog(const DurableSharedPrefixCatalog&)            = delete;
    DurableSharedPrefixCatalog& operator=(const DurableSharedPrefixCatalog&) = delete;

    [[nodiscard]] DurableSharedPrefixRestore
    restore_matching(Engine& engine, const PreparedPrompt& prompt, Clock::time_point deadline,
                     const CancellationView& cancellation  = {},
                     const RequestOptions& request_options = {});
    void schedule_exports(Engine& engine);
    void observe_hit(const DurableSharedPrefixRestore& restore, std::uint32_t reused_tokens,
                     PrefixReusePath path) noexcept;
    void drain();
    [[nodiscard]] DurableSharedPrefixCatalogStats stats() const noexcept;

    // Narrow storage seams used by deterministic filesystem regressions.
    void enqueue(Snapshot snapshot, std::function<void(bool)> settlement = {});
    [[nodiscard]] std::shared_ptr<const std::vector<std::uint8_t>>
    load(const Candidate& candidate, Clock::time_point deadline,
         const CancellationView& cancellation = {});

private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace ninfer::serve
