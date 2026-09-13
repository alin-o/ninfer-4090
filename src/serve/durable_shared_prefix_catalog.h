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

namespace testing {
struct DurableSharedPrefixCatalogTestAccess;
}

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
    // Deterministic load-queue insertion failure seam. Production leaves it empty.
    std::function<void()> before_load_enqueue;
    // Deterministic pre-rename and failed-cleanup seams. Production leaves them empty.
    std::function<void()> before_record_rename;
    std::function<void()> before_temporary_remove;
    std::function<void()> before_manifest_rename;
    std::function<void()> before_manifest_temporary_remove;
    std::function<void(const ninfer::CheckpointLifecycleFact&)> lifecycle_observer;
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
    std::string content_digest;
    std::uint64_t serialized_bytes        = 0;
    std::uint64_t elapsed_ns              = 0;
    bool loaded_from_ssd                  = false;
    bool warm_available                   = false;
    std::uint64_t recovery_reservation_id = 0;
    std::shared_ptr<runtime::DeferredDurableRecovery> deferred_recovery;
    std::string fallback_reason;
    std::vector<ninfer::CheckpointLifecycleFact> lifecycle;
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
    [[nodiscard]] DurableSharedPrefixRestore
    stage_matching(Engine& engine, const PreparedPrompt& prompt, Clock::time_point deadline,
                   const CancellationView& cancellation = {});
    void load_staged(Engine& engine,
                     const std::shared_ptr<runtime::DeferredDurableRecovery>& recovery,
                     Clock::time_point deadline, const CancellationView& cancellation = {});
    [[nodiscard]] DurableSharedPrefixRestore
    settle_staged(const std::shared_ptr<runtime::DeferredDurableRecovery>& recovery) noexcept;
    void schedule_exports(Engine& engine);
    void observe_hit(const DurableSharedPrefixRestore& restore, std::uint32_t reused_tokens,
                     PrefixReusePath path) noexcept;
    void drain();
    void
    set_lifecycle_observer(std::function<void(const ninfer::CheckpointLifecycleFact&)> observer);
    [[nodiscard]] DurableSharedPrefixCatalogStats stats() const noexcept;

    // Narrow storage seams used by deterministic filesystem regressions.
    void enqueue(Snapshot snapshot, std::function<void(bool)> settlement = {});
    [[nodiscard]] std::shared_ptr<const std::vector<std::uint8_t>>
    load(const Candidate& candidate, Clock::time_point deadline,
         const CancellationView& cancellation = {});
    // Deterministic service-boundary deadline seam. Production never calls this.
    void set_before_payload_read_for_test(std::function<void()> callback);

private:
    struct State;

    struct LoadedRecord {
        std::shared_ptr<const std::vector<std::uint8_t>> bytes;
        std::string digest;
        std::string filename;
        std::uint64_t order = 0;
    };

    [[nodiscard]] LoadedRecord load_record(const Candidate& candidate, Clock::time_point deadline,
                                           const CancellationView& cancellation = {});
    void invalidate_loaded(const LoadedRecord& loaded) noexcept;

    friend struct testing::DurableSharedPrefixCatalogTestAccess;
    std::shared_ptr<State> state_;
};

} // namespace ninfer::serve
