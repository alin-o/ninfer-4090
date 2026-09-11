#pragma once

// Internal serving boundary for the Gateway-owned durable shared-prefix catalog. This deliberately
// is not a public anchor-reference protocol: callers can discover only exact candidates for an
// already prepared prompt and exchange opaque NINFSHR1 records.

#include "ninfer/engine.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::runtime {

struct DurableSharedSnapshotAccess {
    using Candidate = targets::qwen3_6::DurableSharedPrefixCandidate;
    using Snapshot  = targets::qwen3_6::RetainedSessionSnapshot;

    struct Export {
        Snapshot snapshot;
        std::uint32_t slot  = 0;
        std::uint64_t owner = 0;
    };

    struct ImportResult {
        std::uint32_t disposition            = 0;
        std::uint32_t slot                   = 0;
        std::uint32_t frontier               = 0;
        std::uint64_t validation_nanoseconds = 0;
        std::uint64_t adoption_nanoseconds   = 0;
    };

    enum class RecoverySource : std::uint8_t {
        None,
        Memory,
        Ssd,
    };

    struct RecoveryDecision {
        RecoverySource source = RecoverySource::None;
        Candidate candidate;
        std::uint32_t frontier                 = 0;
        std::uint64_t estimated_memory_cost_ns = 0;
        std::string reason;
    };

    class ValidationError final : public std::invalid_argument {
    public:
        using std::invalid_argument::invalid_argument;
    };

    [[nodiscard]] static std::vector<Candidate> candidates(Engine& engine,
                                                           const PreparedPrompt& prompt);
    [[nodiscard]] static RecoveryDecision
    decide_recovery(Engine& engine, const PreparedPrompt& prompt,
                    const RequestOptions& request_options,
                    std::span<const Candidate> available_ssd_candidates);
    [[nodiscard]] static ImportResult import(Engine& engine, const Candidate& candidate,
                                             std::shared_ptr<const std::vector<std::uint8_t>> bytes,
                                             const CancellationView& cancellation = {});
    [[nodiscard]] static bool resident(Engine& engine, const Candidate& candidate);
    [[nodiscard]] static bool settle_export(Engine& engine, std::uint32_t slot, std::uint64_t owner,
                                            bool committed);
    [[nodiscard]] static std::vector<Export>
    begin_exports(Engine& engine, const std::function<std::shared_ptr<void>(std::size_t)>& reserve,
                  const std::function<bool(std::uint32_t, std::uint64_t)>& claim,
                  const std::function<void(std::uint32_t, std::uint64_t)>& relinquish);
};

} // namespace ninfer::runtime
