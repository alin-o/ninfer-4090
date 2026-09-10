#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "runtime/engine/context_transfer_test_gate.h"
#include "targets/qwen3_6/impl/frontend/digest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Retained-session snapshot format (target-private).
//
// A snapshot is the complete host image of one catalogued continuation: the resident prefix
// (ledger + identity + shortlist digests), the paged Text/backend KV payload in logical page
// order, and the endpoint StateImage (GDN conv/recurrent state, continuation hidden, and any
// DFlash local state per the image layout). Byte order is the host's (x86 little-endian); a
// snapshot binds to the exact weights identity and KV configuration, so cross-endian
// portability is intentionally out of scope.
//
// Versions 1 and 2 described the pre-reconciliation lane-retained format (v2 appended the
// turn-checkpoint ring). Both are rejected by this build: the physical KV layout and the
// state model changed with the upstream reconciliation, so old files cannot be re-landed.
// Version 3 is the continuation-catalog format. Beside the endpoint it persists the rewrite
// checkpoint and long anchors (each an extra StateImage; the KV payload already covers every
// checkpoint frontier) - a multi-turn continuation diverges from the resident ledger just
// before the endpoint (the assistant header renders differently once the reply is input), so
// reuse of a restored session rides those turn-boundary checkpoints exactly as it does for a
// warm one. Restore degrades gracefully: checkpoints whose StateImage does not fit the state
// pools, or whose anchor ordinal exceeds the server's configured capacity, are dropped while
// the endpoint remains mandatory.

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr char kSessionSnapshotMagic[8]         = {'N', 'I', 'N', 'F', 'S', 'E', 'S', '1'};
constexpr std::uint32_t kSessionSnapshotVersion = 3;
constexpr char kSharedSnapshotMagic[8]          = {'N', 'I', 'N', 'F', 'S', 'H', 'R', '1'};
constexpr std::uint32_t kSharedSnapshotVersion  = 1;
constexpr std::size_t kSharedChecksumBytes      = 32;
constexpr std::size_t kSharedEnvelopeHeaderBytes =
    sizeof(kSharedSnapshotMagic) + sizeof(std::uint32_t) + 2U * sizeof(std::uint64_t) +
    kSharedChecksumBytes;
constexpr std::uint32_t kSharedIdentitySchema = 1;

constexpr std::uint32_t kKvFlagPackedV   = 1U << 0;
constexpr std::uint32_t kKvFlagRotateK   = 1U << 1;
constexpr std::uint32_t kKvFlagRotateV   = 1U << 2;
constexpr std::uint32_t kKvFlagPackedK   = 1U << 3;
constexpr std::uint32_t kKvFlagE8Lattice = 1U << 4;
constexpr std::uint32_t kKvFlagE8Root    = 1U << 5;

class SnapshotWriter {
public:
    explicit SnapshotWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out_.insert(out_.end(), begin, begin + count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    // Reserves a device-payload region and returns its offset; the caller fills it with
    // cudaMemcpyAsync once the full host image is sized (the vector no longer reallocates).
    std::size_t reserve_payload(std::size_t count) {
        const std::size_t offset = out_.size();
        out_.resize(out_.size() + count);
        return offset;
    }

private:
    std::vector<std::uint8_t>& out_;
};

class SnapshotReader {
public:
    explicit SnapshotReader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }

    template <class T>
    [[nodiscard]] T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }

    // Borrows a device-payload region without copying; valid for the snapshot's lifetime.
    [[nodiscard]] const std::uint8_t* payload(std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        const std::uint8_t* region = data_.data() + cursor_;
        cursor_ += count;
        return region;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

template <class T>
void write_vector(SnapshotWriter& writer, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writer.pod<std::uint64_t>(values.size());
    writer.bytes(values.data(), values.size() * sizeof(T));
}

template <class T>
void write_span(SnapshotWriter& writer, std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writer.pod<std::uint64_t>(values.size());
    writer.bytes(values.data(), values.size_bytes());
}

template <class T>
std::vector<T> read_vector(SnapshotReader& reader, std::size_t maximum_count, const char* label) {
    const std::uint64_t count = reader.pod<std::uint64_t>();
    if (count > maximum_count) {
        throw std::invalid_argument(std::string("session snapshot ") + label +
                                    " count is out of range");
    }
    std::vector<T> values(static_cast<std::size_t>(count));
    reader.bytes(values.data(), values.size() * sizeof(T));
    return values;
}

void write_vision_items(SnapshotWriter& writer, const std::vector<VisionItem>& items) {
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(items.size()));
    for (const VisionItem& item : items) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(item.modality));
        writer.pod<std::int32_t>(item.grid.temporal);
        writer.pod<std::int32_t>(item.grid.height);
        writer.pod<std::int32_t>(item.grid.width);
        writer.pod<std::uint64_t>(item.patch_begin);
        writer.pod<std::uint64_t>(item.patch_count);
        writer.bytes(item.content_digest.data(), item.content_digest.size());
        write_vector(writer, item.timestamps);
        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(item.token_spans.size()));
        for (const TokenSpan& span : item.token_spans) {
            writer.pod<std::uint64_t>(span.begin);
            writer.pod<std::uint64_t>(span.count);
        }
    }
}

std::vector<VisionItem> read_vision_items(SnapshotReader& reader, std::size_t tokens) {
    const std::uint32_t count = reader.pod<std::uint32_t>();
    if (count > tokens) {
        throw std::invalid_argument("session snapshot vision item count is out of range");
    }
    std::vector<VisionItem> items(count);
    for (VisionItem& item : items) {
        item.modality      = static_cast<PromptModality>(reader.pod<std::uint8_t>());
        item.grid.temporal = reader.pod<std::int32_t>();
        item.grid.height   = reader.pod<std::int32_t>();
        item.grid.width    = reader.pod<std::int32_t>();
        item.patch_begin   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        item.patch_count   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        reader.bytes(item.content_digest.data(), item.content_digest.size());
        item.timestamps           = read_vector<double>(reader, tokens, "vision timestamp");
        const std::uint32_t spans = reader.pod<std::uint32_t>();
        if (spans > tokens) {
            throw std::invalid_argument("session snapshot vision span count is out of range");
        }
        item.token_spans.resize(spans);
        for (TokenSpan& span : item.token_spans) {
            span.begin = static_cast<std::size_t>(reader.pod<std::uint64_t>());
            span.count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        }
    }
    return items;
}

struct SnapshotConfig {
    std::uint32_t kv_dtype            = 0;
    std::int32_t kv_quant_group       = 0;
    std::uint32_t kv_flags            = 0;
    std::uint32_t speculative_backend = 0;
    std::uint32_t draft_window        = 0;
    std::uint32_t page_size           = 0;
    std::uint64_t state_image_bytes   = 0;
    std::uint32_t text_plane_count    = 0;
    std::uint64_t text_page_stride    = 0;
    std::uint32_t backend_plane_count = 0;
    std::uint64_t backend_page_stride = 0;
};

struct SnapshotSession {
    std::uint32_t tokens                     = 0;
    std::uint32_t execution_frontier         = 0;
    std::uint32_t ledger_frontier            = 0;
    std::uint32_t text_kv_valid              = 0;
    std::uint32_t mtp_kv_valid               = 0;
    std::int32_t rope_delta                  = 0;
    std::uint8_t tail_hidden_valid           = 0;
    std::uint32_t text_committed_frontier    = 0;
    std::uint32_t backend_committed_frontier = 0;
    std::uint32_t text_pages                 = 0;
    std::uint32_t backend_pages              = 0;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;
};

struct SharedSnapshotConfig {
    SnapshotConfig physical;
    std::uint32_t max_context     = 0;
    std::uint32_t token_domain    = 0;
    std::uint32_t proposal_head   = 0;
    std::uint32_t identity_schema = 0;
    std::uint32_t identity_tag    = 0;
};

struct SharedSnapshotBoundary {
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    std::uint8_t tail_hidden_valid = 0;
    runtime::PrefillWork rebuild_work;
    std::uint32_t text_pages    = 0;
    std::uint32_t backend_pages = 0;
};

struct SharedImportBacking {
    std::vector<std::uint8_t> storage;
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    SharedSnapshotBoundary boundary;
    std::size_t state_offset   = 0;
    std::size_t text_offset    = 0;
    std::size_t backend_offset = 0;
};

void write_config(SnapshotWriter& writer, const SnapshotConfig& config);
SnapshotConfig read_config(SnapshotReader& reader);

void write_shared_config(SnapshotWriter& writer, const SharedSnapshotConfig& config) {
    write_config(writer, config.physical);
    writer.pod(config.max_context);
    writer.pod(config.token_domain);
    writer.pod(config.proposal_head);
    writer.pod(config.identity_schema);
    writer.pod(config.identity_tag);
}

SharedSnapshotConfig read_shared_config(SnapshotReader& reader) {
    SharedSnapshotConfig config;
    config.physical        = read_config(reader);
    config.max_context     = reader.pod<std::uint32_t>();
    config.token_domain    = reader.pod<std::uint32_t>();
    config.proposal_head   = reader.pod<std::uint32_t>();
    config.identity_schema = reader.pod<std::uint32_t>();
    config.identity_tag    = reader.pod<std::uint32_t>();
    return config;
}

void write_shared_boundary(SnapshotWriter& writer, const SharedSnapshotBoundary& boundary) {
    writer.pod(boundary.frontier);
    writer.pod(boundary.backend_frontier);
    writer.pod(boundary.rope_delta);
    writer.pod(boundary.tail_hidden_valid);
    writer.pod(boundary.rebuild_work);
    writer.pod(boundary.text_pages);
    writer.pod(boundary.backend_pages);
}

SharedSnapshotBoundary read_shared_boundary(SnapshotReader& reader) {
    SharedSnapshotBoundary boundary;
    boundary.frontier          = reader.pod<std::uint32_t>();
    boundary.backend_frontier  = reader.pod<std::uint32_t>();
    boundary.rope_delta        = reader.pod<std::int32_t>();
    boundary.tail_hidden_valid = reader.pod<std::uint8_t>();
    boundary.rebuild_work      = reader.pod<runtime::PrefillWork>();
    boundary.text_pages        = reader.pod<std::uint32_t>();
    boundary.backend_pages     = reader.pod<std::uint32_t>();
    return boundary;
}

void validate_durable_metadata(const qwen3_6::SharedPrefixPersistenceMetadata& metadata,
                               std::uint32_t frontier) {
    constexpr std::uint32_t classified_origins =
        qwen3_6::SharedPrefixSystemEnd | qwen3_6::SharedPrefixCacheMarker |
        qwen3_6::SharedPrefixInstructionsEnd | qwen3_6::SharedPrefixProjectContext;
    if (!metadata.ssd_eligible || metadata.structural_role == 0 ||
        metadata.structural_role > static_cast<std::uint8_t>(qwen3_6::SharedPrefixRole::Project) ||
        (metadata.structural_origins & classified_origins) == 0 ||
        !has_shared_candidate_evidence(metadata.evidence,
                                       SharedCandidateEvidence::EngineStructural) ||
        (metadata.first_volatile_token && frontier >= *metadata.first_volatile_token)) {
        throw std::invalid_argument(
            "shared snapshot candidate is volatile or lacks durable structural provenance "
            "(eligible=" +
            std::to_string(metadata.ssd_eligible) +
            ", role=" + std::to_string(metadata.structural_role) +
            ", origins=" + std::to_string(metadata.structural_origins) +
            ", evidence=" + std::to_string(static_cast<std::uint8_t>(metadata.evidence)) +
            ", frontier=" + std::to_string(frontier) + ", cutoff=" +
            (metadata.first_volatile_token ? std::to_string(*metadata.first_volatile_token)
                                           : std::string("none")) +
            ")");
    }
}

std::size_t checked_snapshot_sum(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("shared snapshot size overflows host accounting");
    }
    return left + right;
}

std::size_t checked_snapshot_product(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("shared snapshot size overflows host accounting");
    }
    return left * right;
}

std::size_t shared_snapshot_max_bytes(std::uint32_t capacity, std::size_t state_bytes,
                                      std::size_t text_page_stride,
                                      std::size_t backend_page_stride) {
    constexpr std::size_t binding_bytes = sizeof(std::uint32_t) + 4096U;
    constexpr std::size_t config_bytes  = 6U * sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                                         sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                                         sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                                         5U * sizeof(std::uint32_t);
    constexpr std::size_t boundary_bytes = 3U * sizeof(std::uint32_t) + sizeof(std::uint8_t) +
                                           sizeof(runtime::PrefillWork) +
                                           2U * sizeof(std::uint32_t);
    constexpr std::size_t provenance_bytes = sizeof(std::uint8_t) + sizeof(std::uint32_t) +
                                             3U * sizeof(std::uint8_t) + sizeof(std::uint32_t);
    constexpr std::size_t identity_fixed_bytes = 6U * sizeof(std::uint64_t) + sizeof(std::uint32_t);
    constexpr std::size_t identity_bytes_per_token =
        sizeof(TokenId) + sizeof(std::uint8_t) + 3U * sizeof(std::int32_t) + sizeof(std::uint32_t);
    constexpr std::size_t identity_directory_bytes =
        kSharedChecksumBytes + 3U * sizeof(std::uint64_t);

    std::size_t maximum = kSharedEnvelopeHeaderBytes;
    maximum             = checked_snapshot_sum(maximum, binding_bytes);
    maximum             = checked_snapshot_sum(maximum, config_bytes);
    maximum             = checked_snapshot_sum(maximum, boundary_bytes);
    maximum             = checked_snapshot_sum(maximum, provenance_bytes);
    maximum             = checked_snapshot_sum(maximum, identity_directory_bytes);
    maximum             = checked_snapshot_sum(maximum, identity_fixed_bytes);
    maximum =
        checked_snapshot_sum(maximum, checked_snapshot_product(capacity, identity_bytes_per_token));
    maximum                 = checked_snapshot_sum(maximum, state_bytes);
    const std::size_t pages = kv_pages_for_frontier(capacity);
    maximum = checked_snapshot_sum(maximum, checked_snapshot_product(pages, text_page_stride));
    maximum = checked_snapshot_sum(maximum, checked_snapshot_product(pages, backend_page_stride));
    return maximum;
}

void write_config(SnapshotWriter& writer, const SnapshotConfig& config) {
    writer.pod(config.kv_dtype);
    writer.pod(config.kv_quant_group);
    writer.pod(config.kv_flags);
    writer.pod(config.speculative_backend);
    writer.pod(config.draft_window);
    writer.pod(config.page_size);
    writer.pod(config.state_image_bytes);
    writer.pod(config.text_plane_count);
    writer.pod(config.text_page_stride);
    writer.pod(config.backend_plane_count);
    writer.pod(config.backend_page_stride);
}

SnapshotConfig read_config(SnapshotReader& reader) {
    SnapshotConfig config;
    config.kv_dtype            = reader.pod<std::uint32_t>();
    config.kv_quant_group      = reader.pod<std::int32_t>();
    config.kv_flags            = reader.pod<std::uint32_t>();
    config.speculative_backend = reader.pod<std::uint32_t>();
    config.draft_window        = reader.pod<std::uint32_t>();
    config.page_size           = reader.pod<std::uint32_t>();
    config.state_image_bytes   = reader.pod<std::uint64_t>();
    config.text_plane_count    = reader.pod<std::uint32_t>();
    config.text_page_stride    = reader.pod<std::uint64_t>();
    config.backend_plane_count = reader.pod<std::uint32_t>();
    config.backend_page_stride = reader.pod<std::uint64_t>();
    return config;
}

void write_session(SnapshotWriter& writer, const SnapshotSession& session) {
    writer.pod(session.tokens);
    writer.pod(session.execution_frontier);
    writer.pod(session.ledger_frontier);
    writer.pod(session.text_kv_valid);
    writer.pod(session.mtp_kv_valid);
    writer.pod(session.rope_delta);
    writer.pod(session.tail_hidden_valid);
    writer.pod(session.text_committed_frontier);
    writer.pod(session.backend_committed_frontier);
    writer.pod(session.text_pages);
    writer.pod(session.backend_pages);
    writer.pod(session.rebuild_work);
    writer.pod(session.rebuild_tail_begin);
}

SnapshotSession read_session(SnapshotReader& reader) {
    SnapshotSession session;
    session.tokens                     = reader.pod<std::uint32_t>();
    session.execution_frontier         = reader.pod<std::uint32_t>();
    session.ledger_frontier            = reader.pod<std::uint32_t>();
    session.text_kv_valid              = reader.pod<std::uint32_t>();
    session.mtp_kv_valid               = reader.pod<std::uint32_t>();
    session.rope_delta                 = reader.pod<std::int32_t>();
    session.tail_hidden_valid          = reader.pod<std::uint8_t>();
    session.text_committed_frontier    = reader.pod<std::uint32_t>();
    session.backend_committed_frontier = reader.pod<std::uint32_t>();
    session.text_pages                 = reader.pod<std::uint32_t>();
    session.backend_pages              = reader.pod<std::uint32_t>();
    session.rebuild_work               = reader.pod<runtime::PrefillWork>();
    session.rebuild_tail_begin         = reader.pod<std::uint32_t>();
    return session;
}

// Session identity: FNV-1a 64 over the resident ledger's token bytes, rendered as 16 hex
// chars. Deterministic across processes on one endianness, which snapshot compatibility
// already requires. The shared prefix form lives in program_impl.h so checkpoint digests
// hash identically.
std::string ledger_digest(const std::vector<TokenId>& ledger) {
    return ledger_prefix_digest(std::span<const TokenId>(ledger.data(), ledger.size()));
}

} // namespace

std::uint32_t
ProgramImplCore::continuation_depth(const ContinuationHandle& continuation) const noexcept {
    if (!valid_continuation(continuation)) { return 0; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return static_cast<std::uint32_t>(sequence.ledger.size());
}

std::string ProgramImplCore::continuation_digest(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return ledger_digest(sequence.ledger);
}

std::vector<SlotCheckpoint>
ProgramImplCore::continuation_checkpoints(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    const std::uint32_t depth     = static_cast<std::uint32_t>(sequence.ledger.size());

    std::vector<std::uint32_t> frontiers;
    frontiers.reserve(sequence.long_anchors.size() + 2U);
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        frontiers.push_back(anchor.frontier);
    }
    if (sequence.rewrite_checkpoint.valid) {
        frontiers.push_back(sequence.rewrite_checkpoint.frontier);
    }
    if (sequence.endpoint_valid) { frontiers.push_back(sequence.ledger_frontier); }
    std::sort(frontiers.begin(), frontiers.end());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());

    std::vector<SlotCheckpoint> out;
    out.reserve(frontiers.size());
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier > depth) { continue; }
        out.push_back(SlotCheckpoint{frontier, ledger_prefix_digest(std::span<const TokenId>(
                                                   sequence.ledger.data(), frontier))});
    }
    return out;
}

qwen3_6::ContinuationSummary
ProgramImplCore::continuation_summary(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    return continuation_summary(continuation_states[ContractAccess::index(continuation)]);
}

qwen3_6::RetainedSessionSnapshot
ProgramImplCore::save_continuation(const ContinuationHandle& continuation,
                                   std::string_view model_binding) {
    qwen3_6::RetainedSessionSnapshot snapshot =
        begin_save_continuation(continuation, model_binding);
    if (snapshot.await_transfer) { snapshot.await_transfer(snapshot.bytes); }
    retire_ready_snapshot_sources();
    snapshot.await_transfer = {};
    return snapshot;
}

qwen3_6::RetainedSessionSnapshot ProgramImplCore::begin_save_continuation(
    const ContinuationHandle& continuation, std::string_view model_binding,
    const std::function<std::shared_ptr<void>(std::size_t)>& reserve) {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    if (model_binding.size() > 4096) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    if (pending_transaction_ || (has_context_transaction() && !snapshot_save_window_)) {
        throw std::logic_error("cannot snapshot a session during a pending transaction");
    }
    // ResourceManager may start an eviction spill immediately after Program has reserved a
    // materialization topology and before the first physical transaction step.  That window is
    // safe: no source mutation has been submitted, while the reservation prevents a competing
    // topology from invalidating the selected immutable ranges.  Once a generated round is
    // pending or physical progress begins, retain the ordinary no-snapshot rule above.
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("session persistence does not support the DFlash backend");
    }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];

    const std::size_t tokens = sequence.ledger.size();
    if (tokens == 0 || tokens > capacity || sequence.prefix_identity.size() != tokens ||
        sequence.prefix_digests.size() != tokens || sequence.ledger_frontier != tokens ||
        sequence.execution_frontier > tokens || tokens - sequence.execution_frontier > 1 ||
        !sequence.endpoint_valid || !sequence.kv) {
        throw std::logic_error("retained session ledger and identity are inconsistent");
    }
    if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
        !state_store->valid(sequence.state.read)) {
        throw std::logic_error("retained session state binding is not a settled endpoint");
    }
    const StateImageHandle state             = sequence.state.read;
    const StateImageHostLayout& state_layout = state_images->host_layout();

    // Checkpoint directory: every checkpoint names one entry in a deduplicated StateImage
    // table (a rewrite checkpoint or anchor may alias the endpoint image).
    std::vector<StateImageHandle> unique_states;
    unique_states.reserve(2U + sequence.long_anchors.size());
    const auto image_index = [&](StateImageHandle handle) -> std::int32_t {
        if (!state_store->valid(handle) ||
            state_store->residency(handle) == StateReplicaResidency::None) {
            throw std::logic_error("retained session StateImage has no published replica");
        }
        for (std::size_t index = 0; index < unique_states.size(); ++index) {
            if (unique_states[index] == handle) { return static_cast<std::int32_t>(index); }
        }
        unique_states.push_back(handle);
        return static_cast<std::int32_t>(unique_states.size() - 1U);
    };
    const std::int32_t endpoint_image = image_index(state);
    std::int32_t rewrite_image        = -1;
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("retained rewrite checkpoint has no StateImage");
        }
        rewrite_image = image_index(*sequence.rewrite_state);
    }
    std::vector<std::int32_t> anchor_images;
    anchor_images.reserve(sequence.long_anchors.size());
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        anchor_images.push_back(image_index(anchor.state));
    }

    const DeviceKVPagePool& text_pool    = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout   = plan_host_kv_page_layout(text_pool.geometry());
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }

    const KVAddressSpaceHandle text_address = sequence.kv->text;
    const std::uint32_t text_committed      = text_kv_addresses->committed_frontier(text_address);
    const std::uint32_t text_pages          = kv_pages_for_frontier(text_committed);
    if (text_pages == 0 || text_pages > text_kv_addresses->mapped_pages(text_address) ||
        sequence.execution_frontier > text_committed) {
        throw std::logic_error("retained session Text KV coverage is inconsistent");
    }
    std::uint32_t backend_committed = 0;
    std::uint32_t backend_pages     = 0;
    if (backend != nullptr) {
        if (!sequence.kv->backend) {
            throw std::logic_error("retained session has no backend KV address");
        }
        backend_committed = backend_kv_addresses->committed_frontier(*sequence.kv->backend);
        backend_pages     = kv_pages_for_frontier(backend_committed);
        if (backend_pages > backend_kv_addresses->mapped_pages(*sequence.kv->backend)) {
            throw std::logic_error("retained session backend KV coverage is inconsistent");
        }
    } else if (sequence.kv->backend) {
        throw std::logic_error("retained session backend KV address has no backing cache");
    }

    SnapshotConfig config;
    config.kv_dtype       = static_cast<std::uint32_t>(kv_dtype);
    config.kv_quant_group = kv_quant_group;
    config.kv_flags = (kv_packed_v ? kKvFlagPackedV : 0U) | (kv_rotate_k ? kKvFlagRotateK : 0U) |
                      (kv_rotate_v ? kKvFlagRotateV : 0U) | (kv_packed_k ? kKvFlagPackedK : 0U) |
                      (kv_e8_lattice ? kKvFlagE8Lattice : 0U) | (kv_e8_root ? kKvFlagE8Root : 0U);
    config.speculative_backend = static_cast<std::uint32_t>(speculative_backend);
    config.draft_window        = draft_window;
    config.page_size           = static_cast<std::uint32_t>(kPagedKVPageSize);
    config.state_image_bytes   = state_layout.image_bytes;
    config.text_plane_count    = static_cast<std::uint32_t>(text_pool.plane_count());
    config.text_page_stride    = text_layout.page_stride;
    if (backend != nullptr) {
        config.backend_plane_count = static_cast<std::uint32_t>(backend->page_pool().plane_count());
        config.backend_page_stride = backend_layout->page_stride;
    }

    SnapshotSession session;
    session.tokens                     = static_cast<std::uint32_t>(tokens);
    session.execution_frontier         = sequence.execution_frontier;
    session.ledger_frontier            = sequence.ledger_frontier;
    session.text_kv_valid              = sequence.text_kv_valid;
    session.mtp_kv_valid               = sequence.mtp_kv_valid;
    session.rope_delta                 = sequence.rope_delta;
    session.tail_hidden_valid          = sequence.tail_hidden_valid ? 1 : 0;
    session.text_committed_frontier    = text_committed;
    session.backend_committed_frontier = backend_committed;
    session.text_pages                 = text_pages;
    session.backend_pages              = backend_pages;
    session.rebuild_work               = sequence.rebuild_work;
    session.rebuild_tail_begin         = sequence.rebuild_tail_begin;

    qwen3_6::RetainedSessionSnapshot snapshot;
    snapshot.tokens         = session.tokens;
    snapshot.session_digest = ledger_digest(sequence.ledger);
    SnapshotWriter writer(snapshot.bytes);
    writer.bytes(kSessionSnapshotMagic, sizeof(kSessionSnapshotMagic));
    writer.pod(kSessionSnapshotVersion);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(model_binding.size()));
    writer.bytes(model_binding.data(), model_binding.size());
    write_config(writer, config);
    write_session(writer, session);
    write_vector(writer, sequence.ledger);
    write_vector(writer, sequence.prefix_identity.token_types());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        write_vector(writer, sequence.prefix_identity.position_axis(axis));
    }
    write_vision_items(writer, sequence.prefix_identity.vision_items());
    write_vector(writer, sequence.prefix_identity.rewrite_execution_frontiers());
    write_vector(writer, sequence.prefix_digests.image());

    // Checkpoint directory: rewrite checkpoint, long anchors, and the StateImage table map.
    writer.pod<std::uint8_t>(sequence.rewrite_checkpoint.valid ? 1 : 0);
    writer.pod<std::uint8_t>(static_cast<std::uint8_t>(sequence.rewrite_checkpoint.kind));
    writer.pod<std::uint32_t>(sequence.rewrite_checkpoint.frontier);
    writer.pod(sequence.rewrite_checkpoint.rebuild_work);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(sequence.long_anchors.size()));
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        writer.pod<std::uint32_t>(anchor.frontier);
        writer.pod<std::uint32_t>(anchor.ordinal);
        writer.pod(anchor.rebuild_work);
    }
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(unique_states.size()));
    writer.pod<std::int32_t>(endpoint_image);
    writer.pod<std::int32_t>(rewrite_image);
    for (const std::int32_t index : anchor_images) { writer.pod<std::int32_t>(index); }

    // Account before growing the pageable assembly image.  During submission it coexists with
    // the pinned D2H backing; during consumption a freshly assembled pageable image coexists
    // with that backing.  Two complete images is therefore the actual bounded footprint.
    const std::size_t header_bytes  = snapshot.bytes.size();
    const std::size_t state_bytes   = state_layout.image_bytes * unique_states.size();
    const std::size_t text_bytes    = config.text_page_stride * session.text_pages;
    const std::size_t backend_bytes = config.backend_page_stride * session.backend_pages;
    if (state_bytes > std::numeric_limits<std::size_t>::max() - header_bytes ||
        text_bytes > std::numeric_limits<std::size_t>::max() - header_bytes - state_bytes ||
        backend_bytes >
            std::numeric_limits<std::size_t>::max() - header_bytes - state_bytes - text_bytes) {
        throw std::overflow_error("session snapshot size overflows host accounting");
    }
    const std::size_t transfer_bytes = header_bytes + state_bytes + text_bytes + backend_bytes;
    if (transfer_bytes > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("session snapshot double residency overflows host accounting");
    }
    if (runtime::testing::snapshot_transfer_gate() != nullptr) {
        runtime::testing::note_snapshot_host_accounting(runtime::testing::SnapshotHostAccounting{
            .metadata_bytes   = header_bytes,
            .state_bytes      = state_bytes,
            .main_kv_bytes    = text_bytes,
            .backend_kv_bytes = backend_bytes,
            .transfer_bytes   = transfer_bytes,
            .staging_bytes    = transfer_bytes,
            .resident_bytes   = transfer_bytes * 2U,
        });
    }

    // Reserve the complete bounded writer footprint before allocating pinned backing or
    // submitting D2H.  A full queue therefore drops an involuntary spill without placing work
    // on the execution thread's transfer dependency chain.
    if (reserve) {
        snapshot.queue_reservation = reserve(transfer_bytes * 2U);
        if (!snapshot.queue_reservation) { return {}; }
    }

    // Allocate the complete assembly capacity once after the reservation succeeds.  Growing the
    // three payload regions independently can retain geometric vector capacity above the final
    // image size while the equally large pinned backing is live, violating the advertised Host
    // bound.  A single exact reserve prevents intermediate payload reallocations.
    snapshot.bytes.reserve(transfer_bytes);
    if (snapshot.bytes.capacity() != transfer_bytes) {
        // The supported libstdc++ implementation reserves exactly.  Refuse a conforming but
        // over-allocating implementation rather than silently exceeding queue accounting.
        throw std::runtime_error("session snapshot assembly capacity exceeds its Host reservation");
    }

    // Size the payload after the reservation succeeds. CUDA must never receive the ordinary
    // vector storage as an asynchronous D2H destination.
    const std::size_t state_offset      = writer.reserve_payload(state_bytes);
    const std::size_t text_kv_offset    = writer.reserve_payload(text_bytes);
    const std::size_t backend_kv_offset = writer.reserve_payload(backend_bytes);

    auto transfer_backing = std::make_shared<PinnedHostBuffer>(snapshot.bytes.size());
    std::memcpy(transfer_backing->data(), snapshot.bytes.data(), snapshot.bytes.size());
    std::uint8_t* base = static_cast<std::uint8_t*>(transfer_backing->data());

    // Own submission cleanup before the first possible CUDA copy.  If later validation or a
    // copier throws, destruction records/drains transfer-stream work before backing is freed.
    struct PendingTransferSettlement {
        DeviceContext* device = nullptr;
        std::shared_ptr<PinnedHostBuffer> backing;
        std::shared_ptr<CudaCompletionEvent> completion;
        std::shared_ptr<CudaCompletionEvent> producer;
        std::shared_ptr<void> queue_reservation;
        StateImageStore* state_store = nullptr;
        std::vector<StateImageHandle> state_sources;
        std::vector<std::pair<LogicalKVPageStore*, LogicalKVPageHandle>> kv_sources;
        bool submitted = false;
        bool recorded  = false;
        std::exception_ptr failure;
        std::once_flag event_settlement;
        bool sources_retired            = false;
        bool test_ownership_observed    = false;
        std::size_t test_backing_bytes  = 0;
        std::size_t test_pinned_sources = 0;

        void settle_event() noexcept {
            std::call_once(event_settlement, [&] {
                if (!submitted || !device || !completion) { return; }
                try {
                    device->bind_to_current_thread();
                    if (!recorded) {
                        completion->record(device->transfer_stream);
                        recorded = true;
                    }
                    completion->synchronize();
                } catch (...) { failure = std::current_exception(); }
            });
        }

        void retire_sources() noexcept {
            if (sources_retired) { return; }
            settle_event();
            // cudaEventSynchronize reports an asynchronous copy failure only after the stream
            // has reached its terminal event.  Either outcome is settled, so pins must not
            // leak and no partial image is published (the consumer receives the exception).
            try {
                for (const StateImageHandle source : state_sources) {
                    state_store->unpin_snapshot_source(source);
                }
                state_sources.clear();
                for (const auto& [pages, source] : kv_sources) { pages->unpin_source(source); }
                kv_sources.clear();
            } catch (...) {}
            if (test_ownership_observed && test_pinned_sources != 0) {
                runtime::testing::note_snapshot_transfer_pins_released(test_pinned_sources);
                test_pinned_sources = 0;
            }
            sources_retired = true;
        }

        ~PendingTransferSettlement() {
            retire_sources();
            if (test_ownership_observed) {
                runtime::testing::note_snapshot_transfer_backing_released(test_backing_bytes);
            }
        }
    };

    auto pending               = std::make_shared<PendingTransferSettlement>();
    pending->device            = &device;
    pending->backing           = transfer_backing;
    pending->completion        = std::make_shared<CudaCompletionEvent>(device);
    pending->producer          = std::make_shared<CudaCompletionEvent>(device);
    pending->queue_reservation = snapshot.queue_reservation;
    pending->state_store       = state_store.get();
    // Establish the execution producer dependency before transfer-stream reads.  The pins are
    // Program ownership capabilities, not merely CUDA ordering, and survive continuation
    // release until completion settlement.
    pending->producer->record(device.stream);
    pending->producer->wait(device.transfer_stream);
    if (const auto* gate = runtime::testing::snapshot_transfer_gate(); gate != nullptr) {
        if (gate->enqueue == nullptr) {
            throw std::logic_error("snapshot transfer test gate has no enqueue hook");
        }
        cuda_check(gate->enqueue(gate->context, device.transfer_stream),
                   "enqueue snapshot transfer test delay", __FILE__, __LINE__);
        runtime::testing::note_snapshot_transfer_gate_wait();
    }
    for (std::size_t index = 0; index < unique_states.size(); ++index) {
        const StateImageHandle image  = unique_states[index];
        std::uint8_t* const image_out = base + state_offset + index * state_layout.image_bytes;
        if (state_store->residency(image) == StateReplicaResidency::HostOnly) {
            const qwen3_6::HostStateImageConstView view = state_store->host_view(image);
            std::memcpy(image_out, view.data, state_layout.image_bytes);
        } else {
            state_store->pin_snapshot_source(image);
            pending->state_sources.push_back(image);
            pending->submitted = true;
            state_images->copy_to_host(
                state_store->physical_slot(image),
                qwen3_6::HostStateImageView{reinterpret_cast<std::byte*>(image_out), &state_layout},
                device.transfer_stream);
            ++snapshot_traffic_.state_d2h_count;
            snapshot_traffic_.state_d2h_bytes += state_layout.image_bytes;
        }
    }

    // KV pages: device-resident runs go through the pool's page copier; demoted pages are read
    // from their published Host replicas without touching the device.
    const auto copy_address_pages = [&](const KVAddressSpaceStore& addresses,
                                        LogicalKVPageStore& pages, const DeviceKVPagePool& pool,
                                        KVAddressSpaceHandle address, std::uint32_t page_count,
                                        const HostKVPageLayout& layout, std::size_t payload_offset,
                                        std::uint64_t& d2h_pages, std::uint64_t& d2h_bytes) {
        std::vector<DeviceKVPageHandle> run;
        run.reserve(page_count);
        std::uint32_t run_begin = 0;
        const auto flush_run    = [&] {
            if (run.empty()) { return; }
            pending->submitted = true;
            pool.copy_to_host(std::span<const DeviceKVPageHandle>(run.data(), run.size()),
                                 reinterpret_cast<std::byte*>(base + payload_offset +
                                                              static_cast<std::size_t>(run_begin) *
                                                                  layout.page_stride),
                                 layout, device.transfer_stream);
            d2h_pages += run.size();
            d2h_bytes += run.size() * layout.page_stride;
            run.clear();
        };
        for (std::uint32_t page = 0; page < page_count; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.device_resident(logical)) {
                if (!pages.can_pin_source(logical)) {
                    throw std::logic_error("retained session KV snapshot source is not stable");
                }
                pages.pin_source(logical);
                pending->kv_sources.emplace_back(&pages, logical);
                if (run.empty()) { run_begin = page; }
                run.push_back(pages.physical(logical));
                continue;
            }
            flush_run();
            if (!pages.host_resident(logical) || !pages.host_replica_current(logical)) {
                throw std::logic_error("retained session KV page has no current replica");
            }
            if (!host_kv_extents) {
                throw std::logic_error("retained session Host replica has no extent store");
            }
            const HostKVPageReplica& replica     = pages.host_replica(logical);
            const HostKVAllocationConstView view = host_kv_extents->view(replica.extent);
            if (view.layout().page_stride != layout.page_stride ||
                replica.page_offset >= view.page_count()) {
                throw std::logic_error("retained session Host replica layout is inconsistent");
            }
            std::memcpy(base + payload_offset + static_cast<std::size_t>(page) * layout.page_stride,
                        view.data() +
                            static_cast<std::size_t>(replica.page_offset) * layout.page_stride,
                        layout.page_stride);
        }
        flush_run();
    };
    copy_address_pages(*text_kv_addresses, *text_kv_pages, text_pool, text_address,
                       session.text_pages, text_layout, text_kv_offset,
                       snapshot_traffic_.main_kv_d2h_pages, snapshot_traffic_.main_kv_d2h_bytes);
    if (backend != nullptr && session.backend_pages != 0) {
        copy_address_pages(*backend_kv_addresses, *backend_kv_pages, backend->page_pool(),
                           *sequence.kv->backend, session.backend_pages, *backend_layout,
                           backend_kv_offset, snapshot_traffic_.backend_kv_d2h_pages,
                           snapshot_traffic_.backend_kv_d2h_bytes);
    }
    if (runtime::testing::snapshot_transfer_gate() != nullptr && pending->submitted) {
        pending->test_ownership_observed = true;
        pending->test_backing_bytes      = transfer_backing->size();
        pending->test_pinned_sources = pending->state_sources.size() + pending->kv_sources.size();
        runtime::testing::note_snapshot_transfer_ownership_acquired(pending->test_backing_bytes,
                                                                    pending->test_pinned_sources);
    }
    // Keep the completion event with the immutable host payload. The Engine's bounded writer
    // waits for it off the execution worker. Source pins and the Program-owned retirement list
    // prevent conflicting mutation/reuse until the event settles; do not feed this event back
    // into device.stream, because that stream also carries already admitted independent work.
    // Later transfer-stream users are already ordered by stream FIFO.
    pending->completion->record(device.transfer_stream);
    pending->recorded = true;
    if (snapshot.bytes.size() != transfer_bytes) {
        throw std::logic_error("session snapshot payload sizing changed after reservation");
    }
    snapshot.transfer_bytes = transfer_bytes;
    snapshot.bytes.clear();
    snapshot.bytes.shrink_to_fit();
    snapshot.await_transfer = [pending, transfer_bytes](std::vector<std::uint8_t>& bytes) {
        pending->settle_event();
        if (pending->failure) { std::rethrow_exception(pending->failure); }
        // This bounded Host-consumer step runs after the producer event. It is the only point
        // that creates pageable file bytes, so the Engine worker remains free while D2H runs.
        bytes.resize(transfer_bytes);
        std::memcpy(bytes.data(), pending->backing->data(), transfer_bytes);
    };
    snapshot.settle_transfer = [pending] { pending->settle_event(); };
    snapshot_source_retirements_.push_back(SnapshotSourceRetirement{
        .ready  = [pending] { return !pending->submitted || pending->completion->ready(); },
        .retire = [pending] { pending->retire_sources(); },
    });
    return snapshot;
}

ContinuationHandle ProgramImplCore::restore_continuation(std::span<const std::uint8_t> snapshot,
                                                         std::string_view model_binding) {
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("cannot restore a session during a resource transaction");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("session persistence does not support the DFlash backend");
    }

    SnapshotReader reader(snapshot);
    char magic[sizeof(kSessionSnapshotMagic)] = {};
    reader.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kSessionSnapshotMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("file is not a session snapshot");
    }
    const std::uint32_t version = reader.pod<std::uint32_t>();
    if (version < kSessionSnapshotVersion) {
        throw std::invalid_argument(
            "session snapshot predates the context-cache reconciliation and cannot be restored");
    }
    if (version != kSessionSnapshotVersion) {
        throw std::invalid_argument("session snapshot version is unsupported");
    }
    const std::uint32_t binding_bytes = reader.pod<std::uint32_t>();
    if (binding_bytes > 4096) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    std::string binding(binding_bytes, '\0');
    reader.bytes(binding.data(), binding_bytes);
    if (binding != model_binding) {
        throw std::invalid_argument("session snapshot was saved for a different model");
    }

    const StateImageHostLayout& state_layout = state_images->host_layout();
    const DeviceKVPagePool& text_pool        = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout       = plan_host_kv_page_layout(text_pool.geometry());
    qwen3_6::PagedKVCache* backend           = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }

    const SnapshotConfig config = read_config(reader);
    const std::uint32_t expected_flags =
        (kv_packed_v ? kKvFlagPackedV : 0U) | (kv_rotate_k ? kKvFlagRotateK : 0U) |
        (kv_rotate_v ? kKvFlagRotateV : 0U) | (kv_packed_k ? kKvFlagPackedK : 0U) |
        (kv_e8_lattice ? kKvFlagE8Lattice : 0U) | (kv_e8_root ? kKvFlagE8Root : 0U);
    if (config.kv_dtype != static_cast<std::uint32_t>(kv_dtype) ||
        config.kv_quant_group != kv_quant_group || config.kv_flags != expected_flags ||
        config.page_size != static_cast<std::uint32_t>(kPagedKVPageSize) ||
        config.text_plane_count != static_cast<std::uint32_t>(text_pool.plane_count()) ||
        config.text_page_stride != text_layout.page_stride) {
        throw std::invalid_argument("session snapshot KV configuration does not match the server");
    }
    if (config.speculative_backend != static_cast<std::uint32_t>(speculative_backend) ||
        config.draft_window != draft_window) {
        throw std::invalid_argument(
            "session snapshot speculative configuration does not match the server");
    }
    const std::uint32_t backend_plane_count =
        backend != nullptr ? static_cast<std::uint32_t>(backend->page_pool().plane_count()) : 0U;
    const std::uint64_t backend_page_stride = backend != nullptr ? backend_layout->page_stride : 0U;
    if (config.backend_plane_count != backend_plane_count ||
        config.backend_page_stride != backend_page_stride) {
        throw std::invalid_argument(
            "session snapshot backend KV configuration does not match the server");
    }
    if (config.state_image_bytes != state_layout.image_bytes) {
        throw std::invalid_argument("session snapshot state geometry does not match the server");
    }

    const SnapshotSession session = read_session(reader);
    if (session.tokens == 0 || session.tokens > capacity) {
        throw std::invalid_argument("session snapshot depth exceeds the server context");
    }
    if (session.ledger_frontier != session.tokens || session.execution_frontier > session.tokens ||
        session.tokens - session.execution_frontier > 1 ||
        session.text_kv_valid > session.text_committed_frontier ||
        session.execution_frontier > session.text_committed_frontier ||
        session.text_committed_frontier > capacity ||
        session.mtp_kv_valid > session.backend_committed_frontier ||
        session.backend_committed_frontier > capacity ||
        session.rebuild_work.tokens != session.execution_frontier ||
        session.rebuild_tail_begin > session.execution_frontier) {
        throw std::invalid_argument("session snapshot frontiers are inconsistent");
    }
    if (session.text_pages != kv_pages_for_frontier(session.text_committed_frontier) ||
        session.text_pages == 0 ||
        session.backend_pages != kv_pages_for_frontier(session.backend_committed_frontier) ||
        (backend == nullptr && session.backend_pages != 0)) {
        throw std::invalid_argument("session snapshot page counts are out of range");
    }

    std::vector<TokenId> ledger = read_vector<TokenId>(reader, session.tokens, "ledger");
    if (ledger.size() != session.tokens) {
        throw std::invalid_argument("session snapshot ledger does not match its depth");
    }
    for (const TokenId id : ledger) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("session snapshot ledger token is out of domain");
        }
    }
    std::vector<std::uint8_t> token_types =
        read_vector<std::uint8_t>(reader, session.tokens, "token type");
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis = read_vector<std::int32_t>(reader, session.tokens, "position");
    }
    std::vector<VisionItem> vision_items = read_vision_items(reader, session.tokens);
    std::vector<std::uint32_t> rewrite_frontiers =
        read_vector<std::uint32_t>(reader, session.tokens, "rewrite frontier");
    std::vector<std::array<std::uint64_t, 2>> digest_image =
        read_vector<std::array<std::uint64_t, 2>>(
            reader, static_cast<std::size_t>(session.tokens) + 1U, "shortlist digest");
    if (token_types.size() != session.tokens || positions[0].size() != session.tokens ||
        positions[1].size() != session.tokens || positions[2].size() != session.tokens ||
        digest_image.size() != static_cast<std::size_t>(session.tokens) + 1U) {
        throw std::invalid_argument("session snapshot identity does not match its depth");
    }
    if (!vision_items.empty() && !vision_enabled) {
        throw std::invalid_argument("session snapshot holds media but Vision is disabled");
    }

    // Checkpoint directory.
    const std::uint8_t rewrite_valid_flag   = reader.pod<std::uint8_t>();
    const std::uint8_t rewrite_kind_value   = reader.pod<std::uint8_t>();
    const std::uint32_t rewrite_frontier    = reader.pod<std::uint32_t>();
    const runtime::PrefillWork rewrite_work = reader.pod<runtime::PrefillWork>();
    if (rewrite_valid_flag != 0 &&
        (rewrite_frontier == 0 || rewrite_frontier > session.tokens ||
         rewrite_work.tokens != rewrite_frontier ||
         rewrite_kind_value > static_cast<std::uint8_t>(RewriteCheckpointKind::ResponseReplay))) {
        throw std::invalid_argument("session snapshot rewrite checkpoint is inconsistent");
    }

    struct SnapshotAnchor {
        std::uint32_t frontier = 0;
        std::uint32_t ordinal  = 0;
        runtime::PrefillWork rebuild_work;
        std::int32_t image = -1;
    };

    const std::uint32_t anchor_count = reader.pod<std::uint32_t>();
    if (anchor_count > 64U) {
        throw std::invalid_argument("session snapshot anchor count is out of range");
    }
    std::vector<SnapshotAnchor> anchors(anchor_count);
    for (SnapshotAnchor& anchor : anchors) {
        anchor.frontier     = reader.pod<std::uint32_t>();
        anchor.ordinal      = reader.pod<std::uint32_t>();
        anchor.rebuild_work = reader.pod<runtime::PrefillWork>();
        if (anchor.frontier == 0 || anchor.frontier > session.tokens || anchor.ordinal == 0 ||
            anchor.rebuild_work.tokens != anchor.frontier) {
            throw std::invalid_argument("session snapshot long anchor is inconsistent");
        }
    }
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (anchors[previous].ordinal == anchors[index].ordinal) {
                throw std::invalid_argument("session snapshot anchor ordinals are not unique");
            }
        }
    }
    const std::uint32_t image_count = reader.pod<std::uint32_t>();
    if (image_count == 0 || image_count > 2U + anchor_count) {
        throw std::invalid_argument("session snapshot StateImage table is out of range");
    }
    const auto read_image_index = [&](bool required) -> std::int32_t {
        const std::int32_t index = reader.pod<std::int32_t>();
        if ((required && index < 0) || index >= static_cast<std::int32_t>(image_count) ||
            (!required && index < -1)) {
            throw std::invalid_argument("session snapshot StateImage index is out of range");
        }
        return index;
    };
    const std::int32_t endpoint_image = read_image_index(true);
    const std::int32_t rewrite_image  = read_image_index(false);
    if ((rewrite_valid_flag != 0) != (rewrite_image >= 0)) {
        throw std::invalid_argument("session snapshot rewrite StateImage index is inconsistent");
    }
    for (SnapshotAnchor& anchor : anchors) { anchor.image = read_image_index(true); }

    std::vector<const std::uint8_t*> image_payloads(image_count);
    for (std::uint32_t index = 0; index < image_count; ++index) {
        image_payloads[index] = reader.payload(state_layout.image_bytes);
    }
    const std::uint8_t* state_payload = image_payloads[static_cast<std::size_t>(endpoint_image)];
    const std::uint8_t* text_payload = reader.payload(config.text_page_stride * session.text_pages);
    const std::uint8_t* backend_payload =
        session.backend_pages != 0
            ? reader.payload(config.backend_page_stride * session.backend_pages)
            : nullptr;
    if (reader.remaining() != 0) {
        throw std::invalid_argument("session snapshot has trailing bytes");
    }

    if (text_pool.available_pages() < session.text_pages ||
        (backend != nullptr && backend->page_pool().available_pages() < session.backend_pages)) {
        throw std::invalid_argument(
            "session snapshot does not fit the free KV capacity; evict other sessions first");
    }

    // Page uploads run through a temporarily activated address space, which needs one execution
    // row; rows are lane-indexed and held only by active requests, so any idle lane's row works.
    std::optional<std::int32_t> free_row;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (requests[lane].lifecycle == Lifecycle::Empty &&
            active_continuations[lane] == continuation_capacity) {
            free_row = static_cast<std::int32_t>(lane);
            break;
        }
    }
    if (!free_row) { throw std::logic_error("session restore requires an idle execution lane"); }

    std::optional<std::uint32_t> slot_index = allocate_continuation_slot();
    if (!slot_index) {
        throw std::invalid_argument(
            "session snapshot does not fit the continuation catalog; evict other sessions first");
    }

    std::optional<StateImageHandle> state;
    std::vector<std::optional<StateImageHandle>> extra_images(image_count);
    std::optional<KVAddressSpaceHandle> text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    try {
        // Every restored image prefers a Device slot and falls back to a HostOnly replica (the
        // shape a demoted checkpoint has) when the Device pool is occupied by other sessions.
        const auto stage_image =
            [&](const std::uint8_t* payload) -> std::optional<StateImageHandle> {
            const qwen3_6::HostStateImageConstView view{reinterpret_cast<const std::byte*>(payload),
                                                        &state_layout};
            std::optional<StateImageHandle> handle = state_store->reserve_reset(device.stream);
            if (handle) {
                state_images->copy_from_host(view, state_store->physical_slot(*handle),
                                             device.stream);
                ++snapshot_traffic_.state_h2d_count;
                snapshot_traffic_.state_h2d_bytes += state_layout.image_bytes;
                return handle;
            }
            return state_store->adopt_host_image(view);
        };
        state = stage_image(state_payload);
        if (!state) {
            throw std::invalid_argument(
                "session snapshot does not fit the free state capacity; evict other sessions "
                "first");
        }

        // Optional checkpoint images: an image that fits neither pool just drops the
        // checkpoints naming it; the endpoint above stays mandatory.
        const std::uint32_t anchor_capacity =
            context_cache.max_long_anchors_per_continuation.value_or(0);
        const auto upload_image = [&](std::int32_t index) {
            const auto slot_index = static_cast<std::size_t>(index);
            if (index == endpoint_image || extra_images[slot_index]) { return; }
            extra_images[slot_index] = stage_image(image_payloads[slot_index]);
        };
        if (rewrite_valid_flag != 0) { upload_image(rewrite_image); }
        for (const SnapshotAnchor& anchor : anchors) {
            if (anchor.ordinal <= anchor_capacity) { upload_image(anchor.image); }
        }

        const auto build_address = [&](KVAddressSpaceStore& addresses, DeviceKVPagePool& pool,
                                       std::uint32_t page_count, std::uint32_t committed,
                                       const HostKVPageLayout& layout, const std::uint8_t* payload,
                                       std::uint64_t& h2d_pages,
                                       std::uint64_t& h2d_bytes) -> KVAddressSpaceHandle {
            std::optional<KVAddressSpaceHandle> address = addresses.create_inactive();
            if (!address) {
                throw std::invalid_argument(
                    "session snapshot does not fit the KV address capacity; evict other "
                    "sessions first");
            }
            if (page_count == 0) { return *address; }
            try {
                addresses.activate(*address, page_count, *free_row);
                addresses.materialize_to_tokens(*address, committed, device.stream);
                addresses.commit_frontier(*address, committed);
                std::vector<DeviceKVPageHandle> destinations;
                destinations.reserve(page_count);
                for (std::uint32_t page = 0; page < page_count; ++page) {
                    destinations.push_back(addresses.physical_page(*address, page));
                }
                pool.copy_from_host(
                    reinterpret_cast<const std::byte*>(payload), layout,
                    std::span<const DeviceKVPageHandle>(destinations.data(), destinations.size()),
                    device.stream);
                h2d_pages += page_count;
                h2d_bytes += static_cast<std::uint64_t>(page_count) * layout.page_stride;
            } catch (...) {
                if (addresses.active(*address)) { addresses.deactivate(*address); }
                (void)addresses.release(*address);
                throw;
            }
            return *address;
        };
        text_address =
            build_address(*text_kv_addresses, text_kv_pages->physical_pool(), session.text_pages,
                          session.text_committed_frontier, text_layout, text_payload,
                          snapshot_traffic_.main_kv_h2d_pages, snapshot_traffic_.main_kv_h2d_bytes);
        if (backend != nullptr) {
            backend_address = build_address(
                *backend_kv_addresses, backend->page_pool(), session.backend_pages,
                session.backend_committed_frontier, *backend_layout, backend_payload,
                snapshot_traffic_.backend_kv_h2d_pages, snapshot_traffic_.backend_kv_h2d_bytes);
        }
        device.synchronize();

        // Point of adoption: the sequence owns the physical handles from here, so the local
        // optionals are disarmed as they are handed over and the failure path collapses to
        // release_continuation_slot_best_effort.
        SequenceState& sequence = continuation_states[*slot_index];
        if (state_store->role(*state) == StateImageRole::ActiveMutable) {
            state_store->freeze(*state);
        }
        sequence.state = ActiveStateBinding{.read = *state, .write = *state};
        state.reset();
        sequence.kv.emplace(SequenceKVBundle{.text = *text_address, .backend = backend_address});
        text_address.reset();
        backend_address.reset();
        unbind_sequence_kv(sequence);
        sequence.lane = static_cast<std::uint32_t>(*free_row);

        sequence.ledger.assign(ledger.begin(), ledger.end());
        sequence.prefix_identity.restore(std::move(token_types), std::move(positions),
                                         std::move(vision_items), std::move(rewrite_frontiers));
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.restore(std::move(digest_image));
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.execution_frontier      = session.execution_frontier;
        sequence.ledger_frontier         = session.ledger_frontier;
        sequence.text_kv_valid           = session.text_kv_valid;
        sequence.mtp_kv_valid            = session.mtp_kv_valid;
        sequence.dflash_context_frontier = 0;
        sequence.rope_delta              = session.rope_delta;
        sequence.mtp_draft_count         = 0;
        sequence.tail_hidden_valid       = session.tail_hidden_valid != 0;
        sequence.state_source_retained   = false;
        sequence.endpoint_valid          = true;
        sequence.rewrite_checkpoint      = {};
        sequence.rewrite_state.reset();
        sequence.reserved_state.reset();
        sequence.rebuild_work       = session.rebuild_work;
        sequence.rebuild_tail_begin = session.rebuild_tail_begin;

        // Adopt the surviving checkpoints: freeze the uploaded images and hand them to the
        // sequence with one checkpoint reference per naming checkpoint (the endpoint keeps
        // zero references, matching finish()).
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image && state_store->role(*image) == StateImageRole::ActiveMutable) {
                state_store->freeze(*image);
            }
        }
        const auto resolve_image = [&](std::int32_t index) -> std::optional<StateImageHandle> {
            if (index == endpoint_image) { return sequence.state.read; }
            return extra_images[static_cast<std::size_t>(index)];
        };
        if (rewrite_valid_flag != 0) {
            if (const std::optional<StateImageHandle> handle = resolve_image(rewrite_image)) {
                sequence.rewrite_state      = *handle;
                sequence.rewrite_checkpoint = RewriteCheckpoint{
                    .valid        = true,
                    .kind         = static_cast<RewriteCheckpointKind>(rewrite_kind_value),
                    .frontier     = rewrite_frontier,
                    .rebuild_work = rewrite_work,
                };
                state_store->retain_checkpoint_reference(*handle);
            }
        }
        for (const SnapshotAnchor& anchor : anchors) {
            if (anchor.ordinal > anchor_capacity) { continue; }
            const std::optional<StateImageHandle> handle = resolve_image(anchor.image);
            if (!handle) { continue; }
            sequence.long_anchors.push_back(LongAnchorCheckpoint{
                .state        = *handle,
                .frontier     = anchor.frontier,
                .ordinal      = anchor.ordinal,
                .rebuild_work = anchor.rebuild_work,
            });
            state_store->retain_checkpoint_reference(*handle);
        }
        refresh_state_views(sequence);

        text_kv_addresses->set_checkpoint_requirement(sequence.kv->text,
                                                      sequence.execution_frontier);
        if (sequence.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*sequence.kv->backend,
                                                             backend_kv_valid(sequence));
        }
        // Mint the endpoint summary once so a malformed rebuild surfaces here instead of at the
        // Engine catalog's adoption.
        (void)continuation_summary(sequence);

        continuation_slots[*slot_index].role = ContinuationSlotRole::Catalogued;
        advance_resource_revision();
        return ContractAccess::make_continuation(this, *slot_index,
                                                 continuation_slots[*slot_index].generation);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        if (backend_address) {
            if (backend_kv_addresses->active(*backend_address)) {
                backend_kv_addresses->deactivate(*backend_address);
            }
            (void)backend_kv_addresses->release(*backend_address);
        }
        if (text_address) {
            if (text_kv_addresses->active(*text_address)) {
                text_kv_addresses->deactivate(*text_address);
            }
            (void)text_kv_addresses->release(*text_address);
        }
        if (state) { (void)state_store->release(*state); }
        // Images already adopted by the sequence carry checkpoint references, so this release
        // refuses them and release_continuation_slot_best_effort below owns their teardown
        // instead.
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image) { (void)state_store->release(*image); }
        }
        if (slot_index) { release_continuation_slot_best_effort(*slot_index); }
        throw;
    }
}

qwen3_6::RetainedSessionSnapshot
ProgramImplCore::export_shared_prefix(const SharedPrefixHandle& handle,
                                      std::string_view model_binding,
                                      const qwen3_6::SharedPrefixPersistenceMetadata& metadata) {
    qwen3_6::RetainedSessionSnapshot snapshot =
        begin_export_shared_prefix(handle, model_binding, metadata);
    if (snapshot.await_transfer) { snapshot.await_transfer(snapshot.bytes); }
    retire_ready_snapshot_sources();
    snapshot.await_transfer = {};
    return snapshot;
}

qwen3_6::RetainedSessionSnapshot ProgramImplCore::begin_export_shared_prefix(
    const SharedPrefixHandle& handle, std::string_view model_binding,
    const qwen3_6::SharedPrefixPersistenceMetadata& metadata,
    const std::function<std::shared_ptr<void>(std::size_t)>& reserve) {
    if (!valid_shared_prefix(handle)) {
        throw std::invalid_argument("shared snapshot source is not catalogued");
    }
    if (model_binding.size() > 4096) {
        throw std::invalid_argument("shared snapshot model binding is too long");
    }
    if (pending_transaction_ || has_context_transaction()) {
        throw std::logic_error("cannot export a shared prefix during a resource transaction");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("shared snapshot does not support the DFlash backend");
    }

    const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(handle)];
    if (!shared.kv || !shared.identity || !shared.identity->backing || shared.frontier == 0 ||
        shared.identity->shortlist_key.frontier != shared.frontier ||
        shared.identity->ledger().size() != shared.frontier || !state_store->valid(shared.state) ||
        state_store->residency(shared.state) == StateReplicaResidency::None) {
        throw std::logic_error("shared snapshot source is incomplete");
    }
    validate_durable_metadata(metadata, shared.frontier);
    const auto* identity = shared.identity->prefix_identity();
    if (identity == nullptr || identity->size() < shared.frontier) {
        throw std::logic_error("shared snapshot exact identity is incomplete");
    }
    if (!identity->vision_items().empty()) {
        throw std::invalid_argument("shared snapshot media persistence is not supported");
    }

    const DeviceKVPagePool& text_pool    = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout   = plan_host_kv_page_layout(text_pool.geometry());
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }
    if ((backend != nullptr) != shared.kv->backend.has_value()) {
        throw std::logic_error("shared snapshot backend KV ownership is inconsistent");
    }
    const std::uint32_t text_committed = text_kv_addresses->committed_frontier(shared.kv->text);
    const std::uint32_t backend_committed =
        shared.kv->backend ? backend_kv_addresses->committed_frontier(*shared.kv->backend) : 0U;
    const bool backend_coverage =
        shared.kv->backend
            ? backend_committed >= shared.backend_frontier && backend_committed <= shared.frontier
            : backend_committed == 0;
    if (text_committed != shared.frontier || !backend_coverage) {
        throw std::logic_error("shared snapshot KV frontiers do not match the boundary (Main=" +
                               std::to_string(text_committed) + "/" +
                               std::to_string(shared.frontier) +
                               ", backend=" + std::to_string(backend_committed) + "/" +
                               std::to_string(shared.backend_frontier) + ")");
    }

    SharedSnapshotConfig config;
    config.physical.kv_dtype       = static_cast<std::uint32_t>(kv_dtype);
    config.physical.kv_quant_group = kv_quant_group;
    config.physical.kv_flags =
        (kv_packed_v ? kKvFlagPackedV : 0U) | (kv_rotate_k ? kKvFlagRotateK : 0U) |
        (kv_rotate_v ? kKvFlagRotateV : 0U) | (kv_packed_k ? kKvFlagPackedK : 0U) |
        (kv_e8_lattice ? kKvFlagE8Lattice : 0U) | (kv_e8_root ? kKvFlagE8Root : 0U);
    config.physical.speculative_backend = static_cast<std::uint32_t>(speculative_backend);
    config.physical.draft_window        = draft_window;
    config.physical.page_size           = static_cast<std::uint32_t>(kPagedKVPageSize);
    config.physical.state_image_bytes   = state_images->host_layout().image_bytes;
    config.physical.text_plane_count    = static_cast<std::uint32_t>(text_pool.plane_count());
    config.physical.text_page_stride    = text_layout.page_stride;
    if (backend != nullptr) {
        config.physical.backend_plane_count =
            static_cast<std::uint32_t>(backend->page_pool().plane_count());
        config.physical.backend_page_stride = backend_layout->page_stride;
    }
    config.max_context     = capacity;
    config.token_domain    = TextConfig::token_domain;
    config.proposal_head   = static_cast<std::uint32_t>(proposal_head);
    config.identity_schema = kSharedIdentitySchema;
    config.identity_tag    = shared.identity->shortlist_key.identity_tag;

    SharedSnapshotBoundary boundary;
    boundary.frontier          = shared.frontier;
    boundary.backend_frontier  = shared.backend_frontier;
    boundary.rope_delta        = shared.rope_delta;
    boundary.tail_hidden_valid = shared.tail_hidden_valid ? 1U : 0U;
    boundary.rebuild_work      = validated_rebuild_work(shared.rebuild_work, shared.frontier);
    boundary.text_pages        = kv_pages_for_frontier(shared.frontier);
    boundary.backend_pages     = kv_pages_for_frontier(shared.backend_frontier);
    if (boundary.text_pages == 0 ||
        boundary.text_pages > text_kv_addresses->mapped_pages(shared.kv->text) ||
        (shared.kv->backend &&
         boundary.backend_pages > backend_kv_addresses->mapped_pages(*shared.kv->backend))) {
        throw std::logic_error("shared snapshot KV coverage is incomplete");
    }

    std::vector<std::uint8_t> identity_bytes;
    SnapshotWriter identity_writer(identity_bytes);
    write_span(identity_writer, shared.identity->ledger());
    write_span(identity_writer,
               std::span<const std::uint8_t>(identity->token_types()).first(shared.frontier));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        write_span(
            identity_writer,
            std::span<const std::int32_t>(identity->position_axis(axis)).first(shared.frontier));
    }
    write_vision_items(identity_writer, {});
    const auto& rewrite_frontiers = identity->rewrite_execution_frontiers();
    const auto rewrite_end =
        std::upper_bound(rewrite_frontiers.begin(), rewrite_frontiers.end(), shared.frontier);
    write_span(identity_writer,
               std::span<const std::uint32_t>(rewrite_frontiers.begin(), rewrite_end));
    const auto identity_digest = frontend_internal::sha256(identity_bytes);

    qwen3_6::RetainedSessionSnapshot snapshot;
    snapshot.tokens         = shared.frontier;
    snapshot.session_digest = ledger_prefix_digest(shared.identity->ledger());
    snapshot.content_digest = frontend_internal::sha256_hex(identity_digest);
    SnapshotWriter writer(snapshot.bytes);
    writer.bytes(kSharedSnapshotMagic, sizeof(kSharedSnapshotMagic));
    writer.pod(kSharedSnapshotVersion);
    const std::size_t total_size_offset = snapshot.bytes.size();
    writer.pod<std::uint64_t>(0);
    const std::size_t payload_size_offset = snapshot.bytes.size();
    writer.pod<std::uint64_t>(0);
    const std::size_t checksum_offset = snapshot.bytes.size();
    std::array<std::uint8_t, kSharedChecksumBytes> empty_checksum{};
    writer.bytes(empty_checksum.data(), empty_checksum.size());
    if (snapshot.bytes.size() != kSharedEnvelopeHeaderBytes) {
        throw std::logic_error("shared snapshot envelope geometry changed");
    }

    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(model_binding.size()));
    writer.bytes(model_binding.data(), model_binding.size());
    write_shared_config(writer, config);
    write_shared_boundary(writer, boundary);
    writer.pod<std::uint8_t>(static_cast<std::uint8_t>(metadata.evidence));
    writer.pod(metadata.structural_origins);
    writer.pod(metadata.structural_role);
    writer.pod<std::uint8_t>(metadata.ssd_eligible ? 1U : 0U);
    writer.pod<std::uint8_t>(metadata.first_volatile_token ? 1U : 0U);
    writer.pod<std::uint32_t>(metadata.first_volatile_token.value_or(0));
    writer.bytes(identity_digest.data(), identity_digest.size());
    writer.pod(shared.identity->shortlist_key.digests[0]);
    writer.pod(shared.identity->shortlist_key.digests[1]);
    writer.pod<std::uint64_t>(identity_bytes.size());
    writer.bytes(identity_bytes.data(), identity_bytes.size());

    const std::size_t state_bytes = state_images->host_layout().image_bytes;
    const std::size_t text_bytes =
        static_cast<std::size_t>(boundary.text_pages) * text_layout.page_stride;
    const std::size_t backend_bytes =
        static_cast<std::size_t>(boundary.backend_pages) * config.physical.backend_page_stride;
    std::size_t transfer_bytes = checked_snapshot_sum(snapshot.bytes.size(), state_bytes);
    transfer_bytes             = checked_snapshot_sum(transfer_bytes, text_bytes);
    transfer_bytes             = checked_snapshot_sum(transfer_bytes, backend_bytes);
    if (transfer_bytes > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("shared snapshot double residency overflows host accounting");
    }
    const std::uint64_t total_u64   = transfer_bytes;
    const std::uint64_t payload_u64 = transfer_bytes - kSharedEnvelopeHeaderBytes;
    std::memcpy(snapshot.bytes.data() + total_size_offset, &total_u64, sizeof(total_u64));
    std::memcpy(snapshot.bytes.data() + payload_size_offset, &payload_u64, sizeof(payload_u64));

    if (reserve) {
        snapshot.queue_reservation = reserve(transfer_bytes * 2U);
        if (!snapshot.queue_reservation) { return {}; }
    }
    snapshot.bytes.reserve(transfer_bytes);
    if (snapshot.bytes.capacity() != transfer_bytes) {
        throw std::runtime_error("shared snapshot assembly capacity exceeds its Host reservation");
    }
    const std::size_t state_offset   = writer.reserve_payload(state_bytes);
    const std::size_t text_offset    = writer.reserve_payload(text_bytes);
    const std::size_t backend_offset = writer.reserve_payload(backend_bytes);
    auto transfer_backing            = std::make_shared<PinnedHostBuffer>(transfer_bytes);
    std::memcpy(transfer_backing->data(), snapshot.bytes.data(), transfer_bytes);

    struct SharedTransferSettlement {
        DeviceContext* device = nullptr;
        std::shared_ptr<PinnedHostBuffer> backing;
        std::shared_ptr<CudaCompletionEvent> completion;
        std::shared_ptr<CudaCompletionEvent> producer;
        std::shared_ptr<void> queue_reservation;
        StateImageStore* states = nullptr;
        std::vector<StateImageHandle> state_sources;
        std::vector<std::pair<LogicalKVPageStore*, LogicalKVPageHandle>> kv_sources;
        bool submitted = false;
        bool recorded  = false;
        std::exception_ptr failure;
        std::once_flag settlement;
        bool retired = false;

        void settle() noexcept {
            std::call_once(settlement, [&] {
                if (!submitted || !device || !completion) { return; }
                try {
                    device->bind_to_current_thread();
                    if (!recorded) {
                        completion->record(device->transfer_stream);
                        recorded = true;
                    }
                    completion->synchronize();
                } catch (...) { failure = std::current_exception(); }
            });
        }

        void retire() noexcept {
            if (retired) { return; }
            settle();
            try {
                for (const StateImageHandle source : state_sources) {
                    states->unpin_snapshot_source(source);
                }
                for (const auto& [store, source] : kv_sources) { store->unpin_source(source); }
            } catch (...) {}
            state_sources.clear();
            kv_sources.clear();
            retired = true;
        }

        ~SharedTransferSettlement() { retire(); }
    };

    auto pending               = std::make_shared<SharedTransferSettlement>();
    pending->device            = &device;
    pending->backing           = transfer_backing;
    pending->completion        = std::make_shared<CudaCompletionEvent>(device);
    pending->producer          = std::make_shared<CudaCompletionEvent>(device);
    pending->queue_reservation = snapshot.queue_reservation;
    pending->states            = state_store.get();
    pending->producer->record(device.stream);
    pending->producer->wait(device.transfer_stream);
    auto* base = static_cast<std::uint8_t*>(transfer_backing->data());

    if (state_store->residency(shared.state) == StateReplicaResidency::HostOnly) {
        const auto view = state_store->host_view(shared.state);
        std::memcpy(base + state_offset, view.data, state_bytes);
    } else {
        state_store->pin_snapshot_source(shared.state);
        pending->state_sources.push_back(shared.state);
        pending->submitted = true;
        state_images->copy_to_host(
            state_store->physical_slot(shared.state),
            qwen3_6::HostStateImageView{reinterpret_cast<std::byte*>(base + state_offset),
                                        &state_images->host_layout()},
            device.transfer_stream);
        ++snapshot_traffic_.state_d2h_count;
        snapshot_traffic_.state_d2h_bytes += state_bytes;
    }

    const auto copy_pages = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                const DeviceKVPagePool& pool, KVAddressSpaceHandle address,
                                std::uint32_t count, const HostKVPageLayout& layout,
                                std::size_t offset, std::uint64_t& d2h_pages,
                                std::uint64_t& d2h_bytes) {
        std::vector<DeviceKVPageHandle> run;
        run.reserve(count);
        std::uint32_t run_begin = 0;
        const auto flush        = [&] {
            if (run.empty()) { return; }
            pending->submitted = true;
            pool.copy_to_host(
                run,
                reinterpret_cast<std::byte*>(
                    base + offset + static_cast<std::size_t>(run_begin) * layout.page_stride),
                layout, device.transfer_stream);
            d2h_pages += run.size();
            d2h_bytes += run.size() * layout.page_stride;
            run.clear();
        };
        for (std::uint32_t page = 0; page < count; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.device_resident(logical)) {
                if (!pages.can_pin_source(logical)) {
                    throw std::logic_error("shared snapshot KV source is not immutable");
                }
                pages.pin_source(logical);
                pending->kv_sources.emplace_back(&pages, logical);
                if (run.empty()) { run_begin = page; }
                run.push_back(pages.physical(logical));
                continue;
            }
            flush();
            if (!pages.host_replica_current(logical) || !host_kv_extents) {
                throw std::logic_error("shared snapshot KV page has no current replica");
            }
            const HostKVPageReplica& replica     = pages.host_replica(logical);
            const HostKVAllocationConstView view = host_kv_extents->view(replica.extent);
            if (view.layout().page_stride != layout.page_stride ||
                replica.page_offset >= view.page_count()) {
                throw std::logic_error("shared snapshot Host KV geometry is inconsistent");
            }
            std::memcpy(base + offset + static_cast<std::size_t>(page) * layout.page_stride,
                        view.data() +
                            static_cast<std::size_t>(replica.page_offset) * layout.page_stride,
                        layout.page_stride);
        }
        flush();
    };
    copy_pages(*text_kv_addresses, *text_kv_pages, text_pool, shared.kv->text, boundary.text_pages,
               text_layout, text_offset, snapshot_traffic_.main_kv_d2h_pages,
               snapshot_traffic_.main_kv_d2h_bytes);
    if (shared.kv->backend) {
        copy_pages(*backend_kv_addresses, *backend_kv_pages, backend->page_pool(),
                   *shared.kv->backend, boundary.backend_pages, *backend_layout, backend_offset,
                   snapshot_traffic_.backend_kv_d2h_pages, snapshot_traffic_.backend_kv_d2h_bytes);
    }
    if (pending->submitted) {
        pending->completion->record(device.transfer_stream);
        pending->recorded = true;
    }
    snapshot.transfer_bytes = transfer_bytes;
    snapshot.bytes.clear();
    snapshot.bytes.shrink_to_fit();
    snapshot.await_transfer = [pending, transfer_bytes, checksum_offset](auto& bytes) {
        pending->settle();
        if (pending->failure) { std::rethrow_exception(pending->failure); }
        bytes.resize(transfer_bytes);
        std::memcpy(bytes.data(), pending->backing->data(), transfer_bytes);
        const auto checksum = frontend_internal::sha256(
            std::span<const std::uint8_t>(bytes).subspan(kSharedEnvelopeHeaderBytes));
        std::memcpy(bytes.data() + checksum_offset, checksum.data(), checksum.size());
    };
    snapshot.settle_transfer = [pending] { pending->settle(); };
    snapshot_source_retirements_.push_back(SnapshotSourceRetirement{
        .ready  = [pending] { return !pending->submitted || pending->completion->ready(); },
        .retire = [pending] { pending->retire(); },
    });
    return snapshot;
}

qwen3_6::ValidatedSharedPrefixImport<Variant>
ProgramImplCore::parse_shared_prefix(std::span<const std::uint8_t> snapshot,
                                     std::string_view model_binding,
                                     const std::function<void()>& cancellation_checkpoint) const {
    if (snapshot.size() < kSharedEnvelopeHeaderBytes) {
        throw std::invalid_argument("shared snapshot is truncated");
    }
    SnapshotReader envelope(snapshot);
    char magic[sizeof(kSharedSnapshotMagic)]{};
    envelope.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kSharedSnapshotMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("file is not a shared snapshot");
    }
    if (envelope.pod<std::uint32_t>() != kSharedSnapshotVersion) {
        throw std::invalid_argument("shared snapshot version is unsupported");
    }
    const std::uint64_t total_size   = envelope.pod<std::uint64_t>();
    const std::uint64_t payload_size = envelope.pod<std::uint64_t>();
    std::array<std::uint8_t, kSharedChecksumBytes> expected_checksum{};
    envelope.bytes(expected_checksum.data(), expected_checksum.size());
    if (total_size != snapshot.size() ||
        payload_size != snapshot.size() - kSharedEnvelopeHeaderBytes) {
        throw std::invalid_argument("shared snapshot envelope lengths are inconsistent");
    }
    const DeviceKVPagePool& text_pool    = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout   = plan_host_kv_page_layout(text_pool.geometry());
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }
    const std::size_t maximum_size = shared_snapshot_max_bytes(
        capacity, state_images->host_layout().image_bytes, text_layout.page_stride,
        backend_layout ? backend_layout->page_stride : 0U);
    if (snapshot.size() > maximum_size) {
        throw std::invalid_argument("shared snapshot exceeds the configured geometry bound");
    }
    if (cancellation_checkpoint) { cancellation_checkpoint(); }
    const auto actual_checksum = frontend_internal::sha256(
        snapshot.subspan(kSharedEnvelopeHeaderBytes), cancellation_checkpoint);
    if (actual_checksum != expected_checksum) {
        throw std::invalid_argument("shared snapshot payload checksum does not match");
    }

    SnapshotReader reader(snapshot.subspan(kSharedEnvelopeHeaderBytes));
    const std::uint32_t binding_size = reader.pod<std::uint32_t>();
    if (binding_size > 4096U) {
        throw std::invalid_argument("shared snapshot model binding is too long");
    }
    std::string binding(binding_size, '\0');
    reader.bytes(binding.data(), binding.size());
    if (binding != model_binding) {
        throw std::invalid_argument("shared snapshot was saved for a different model");
    }

    const SharedSnapshotConfig config = read_shared_config(reader);
    const std::uint32_t expected_flags =
        (kv_packed_v ? kKvFlagPackedV : 0U) | (kv_rotate_k ? kKvFlagRotateK : 0U) |
        (kv_rotate_v ? kKvFlagRotateV : 0U) | (kv_packed_k ? kKvFlagPackedK : 0U) |
        (kv_e8_lattice ? kKvFlagE8Lattice : 0U) | (kv_e8_root ? kKvFlagE8Root : 0U);
    const std::uint32_t backend_planes =
        backend ? static_cast<std::uint32_t>(backend->page_pool().plane_count()) : 0U;
    const std::uint64_t backend_stride = backend ? backend_layout->page_stride : 0U;
    if (config.physical.kv_dtype != static_cast<std::uint32_t>(kv_dtype) ||
        config.physical.kv_quant_group != kv_quant_group ||
        config.physical.kv_flags != expected_flags ||
        config.physical.speculative_backend != static_cast<std::uint32_t>(speculative_backend) ||
        config.physical.draft_window != draft_window ||
        config.physical.page_size != static_cast<std::uint32_t>(kPagedKVPageSize) ||
        config.physical.state_image_bytes != state_images->host_layout().image_bytes ||
        config.physical.text_plane_count != static_cast<std::uint32_t>(text_pool.plane_count()) ||
        config.physical.text_page_stride != text_layout.page_stride ||
        config.physical.backend_plane_count != backend_planes ||
        config.physical.backend_page_stride != backend_stride || config.max_context != capacity ||
        config.token_domain != TextConfig::token_domain ||
        config.proposal_head != static_cast<std::uint32_t>(proposal_head) ||
        config.identity_schema != kSharedIdentitySchema ||
        config.identity_tag != capture_identity_tag(speculative_backend, proposal_head, kv_dtype)) {
        throw std::invalid_argument("shared snapshot execution configuration does not match");
    }

    const SharedSnapshotBoundary boundary = read_shared_boundary(reader);
    if (boundary.frontier == 0 || boundary.frontier > capacity ||
        boundary.rebuild_work.tokens != boundary.frontier ||
        boundary.text_pages != kv_pages_for_frontier(boundary.frontier) ||
        boundary.backend_pages != kv_pages_for_frontier(boundary.backend_frontier) ||
        (speculative_backend == SpeculativeBackend::Mtp
             ? boundary.backend_frontier + 1U != boundary.frontier
             : boundary.backend_frontier != 0) ||
        (backend == nullptr && boundary.backend_pages != 0)) {
        throw std::invalid_argument("shared snapshot boundary frontiers are inconsistent");
    }
    qwen3_6::SharedPrefixPersistenceMetadata metadata;
    metadata.evidence           = static_cast<SharedCandidateEvidence>(reader.pod<std::uint8_t>());
    metadata.structural_origins = reader.pod<std::uint32_t>();
    metadata.structural_role    = reader.pod<std::uint8_t>();
    metadata.ssd_eligible       = reader.pod<std::uint8_t>() != 0;
    const bool has_cutoff       = reader.pod<std::uint8_t>() != 0;
    const std::uint32_t cutoff  = reader.pod<std::uint32_t>();
    if (has_cutoff) { metadata.first_volatile_token = cutoff; }
    validate_durable_metadata(metadata, boundary.frontier);

    std::array<std::uint8_t, kSharedChecksumBytes> expected_identity_digest{};
    reader.bytes(expected_identity_digest.data(), expected_identity_digest.size());
    const std::array<std::uint64_t, 2> expected_shortlist{reader.pod<std::uint64_t>(),
                                                          reader.pod<std::uint64_t>()};
    const std::uint64_t identity_size = reader.pod<std::uint64_t>();
    const std::size_t fixed_payload   = checked_snapshot_sum(
        state_images->host_layout().image_bytes,
        checked_snapshot_sum(static_cast<std::size_t>(boundary.text_pages) *
                                   text_layout.page_stride,
                               static_cast<std::size_t>(boundary.backend_pages) * backend_stride));
    if (identity_size > reader.remaining() || fixed_payload > reader.remaining() - identity_size ||
        reader.remaining() - identity_size != fixed_payload) {
        throw std::invalid_argument("shared snapshot payload lengths are inconsistent");
    }
    const std::uint8_t* identity_payload = reader.payload(static_cast<std::size_t>(identity_size));
    const auto actual_identity_digest    = frontend_internal::sha256(
        std::span<const std::uint8_t>(identity_payload, static_cast<std::size_t>(identity_size)),
        cancellation_checkpoint);
    if (actual_identity_digest != expected_identity_digest) {
        throw std::invalid_argument("shared snapshot identity digest does not match");
    }

    SnapshotReader identity_reader(
        std::span<const std::uint8_t>(identity_payload, static_cast<std::size_t>(identity_size)));
    std::vector<TokenId> ledger =
        read_vector<TokenId>(identity_reader, boundary.frontier, "shared ledger");
    std::vector<std::uint8_t> token_types =
        read_vector<std::uint8_t>(identity_reader, boundary.frontier, "shared token type");
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis = read_vector<std::int32_t>(identity_reader, boundary.frontier, "shared position");
    }
    std::vector<VisionItem> vision_items = read_vision_items(identity_reader, boundary.frontier);
    std::vector<std::uint32_t> rewrite_frontiers =
        read_vector<std::uint32_t>(identity_reader, boundary.frontier, "shared rewrite frontier");
    if (identity_reader.remaining() != 0 || ledger.size() != boundary.frontier ||
        token_types.size() != boundary.frontier || positions[0].size() != boundary.frontier ||
        positions[1].size() != boundary.frontier || positions[2].size() != boundary.frontier ||
        !vision_items.empty()) {
        throw std::invalid_argument("shared snapshot exact identity is inconsistent");
    }
    for (const TokenId token : ledger) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::invalid_argument("shared snapshot token is out of domain");
        }
    }
    auto capture_backing    = std::make_shared<PreparedCaptureBacking>();
    capture_backing->ledger = ledger;
    capture_backing->prefix_identity.restore(std::move(token_types), std::move(positions), {},
                                             std::move(rewrite_frontiers));
    PreparedPromptData prompt;
    prompt.token_ids   = ledger;
    prompt.token_types = capture_backing->prefix_identity.token_types();
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto& values = capture_backing->prefix_identity.position_axis(axis);
        prompt.positions.insert(prompt.positions.end(), values.begin(), values.end());
    }
    prompt.identity.rewrite_execution_frontiers =
        capture_backing->prefix_identity.rewrite_execution_frontiers();
    PrefixShortlistDigests digests;
    digests.assign(prompt);
    if (digests.at(boundary.frontier) != expected_shortlist) {
        throw std::invalid_argument("shared snapshot shortlist digest does not match its identity");
    }
    const PrefixShortlistKey shortlist{
        .digests      = expected_shortlist,
        .frontier     = boundary.frontier,
        .identity_tag = config.identity_tag,
    };
    auto capture_identity = std::make_shared<PreparedCaptureIdentity>(PreparedCaptureIdentity{
        .backing       = std::move(capture_backing),
        .shortlist_key = shortlist,
        .rebuild_work  = boundary.rebuild_work,
    });

    auto backing = std::make_shared<SharedImportBacking>();
    backing->storage.assign(snapshot.begin(), snapshot.end());
    backing->identity                         = std::move(capture_identity);
    backing->boundary                         = boundary;
    const std::size_t consumed_before_payload = snapshot.size() - reader.remaining();
    backing->state_offset                     = consumed_before_payload;
    backing->text_offset =
        checked_snapshot_sum(backing->state_offset, state_images->host_layout().image_bytes);
    backing->backend_offset =
        checked_snapshot_sum(backing->text_offset, static_cast<std::size_t>(boundary.text_pages) *
                                                       text_layout.page_stride);
    (void)reader.payload(state_images->host_layout().image_bytes);
    (void)reader.payload(static_cast<std::size_t>(boundary.text_pages) * text_layout.page_stride);
    (void)reader.payload(static_cast<std::size_t>(boundary.backend_pages) * backend_stride);
    if (reader.remaining() != 0) {
        throw std::invalid_argument("shared snapshot has trailing bytes");
    }

    qwen3_6::SharedPrefixSummary summary{
        .checkpoint =
            {
                .ref             = {.kind     = runtime::CheckpointKind::SharedStablePrefix,
                                    .frontier = boundary.frontier},
                .scope           = runtime::CheckpointScope::Shared,
                .shortlist_key   = shortlist,
                .state_residency = runtime::ReplicaResidency::HostOnly,
                .required_kv     = {.main_frontier    = boundary.frontier,
                                    .backend_frontier = boundary.backend_frontier,
                                    .main_pages       = boundary.text_pages,
                                    .backend_pages    = boundary.backend_pages},
                .rebuild_work    = boundary.rebuild_work,
            },
        .active_references = 0,
    };
    return ContractAccess::make_shared_import(
        std::static_pointer_cast<const void>(backing), summary, metadata,
        frontend_internal::sha256_hex(actual_identity_digest));
}

bool ProgramImplCore::shared_prefix_matches(
    const qwen3_6::ValidatedSharedPrefixImport<Variant>& imported,
    const SharedPrefixHandle& resident) const {
    if (!imported || !valid_shared_prefix(resident)) { return false; }
    const auto backing = std::static_pointer_cast<const SharedImportBacking>(
        ContractAccess::implementation(imported));
    const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(resident)];
    return backing && backing->identity && shared.identity &&
           backing->identity->shortlist_key == shared.identity->shortlist_key &&
           backing->identity->prefix_equals(*shared.identity);
}

qwen3_6::SharedPrefixPublication<Variant> ProgramImplCore::adopt_shared_prefix(
    const qwen3_6::ValidatedSharedPrefixImport<Variant>& imported) {
    if (!imported) { throw std::invalid_argument("shared import plan is empty"); }
    if (pending_transaction_ || has_context_transaction()) {
        throw std::logic_error("cannot adopt a shared prefix during a resource transaction");
    }
    validate_durable_metadata(imported.metadata(), imported.summary().checkpoint.ref.frontier);
    const auto backing = std::static_pointer_cast<const SharedImportBacking>(
        ContractAccess::implementation(imported));
    if (!backing || !backing->identity || backing->boundary.frontier == 0 ||
        backing->identity->shortlist_key != imported.summary().checkpoint.shortlist_key) {
        throw std::logic_error("validated shared import changed before adoption");
    }

    std::optional<std::uint32_t> shared_index;
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if (shared_prefix_slots[index].role == SharedPrefixSlotRole::Free) {
            shared_prefix_slots[index].role = SharedPrefixSlotRole::ReservedCapture;
            shared_index                    = index;
            break;
        }
    }
    if (!shared_index) {
        throw std::invalid_argument("shared snapshot does not fit the shared-prefix catalog");
    }

    const StateImageHostLayout& state_layout = state_images->host_layout();
    const HostKVPageLayout text_layout =
        plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry());
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }
    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    bool shared_populated = false;
    try {
        state = state_store->adopt_host_image(qwen3_6::HostStateImageConstView{
            reinterpret_cast<const std::byte*>(backing->storage.data() + backing->state_offset),
            &state_layout});
        if (!state) {
            throw std::invalid_argument("shared snapshot does not fit the Host State capacity");
        }
        runtime::testing::shared_snapshot_import_checkpoint(
            runtime::testing::SharedSnapshotImportStage::StateAllocated);

        std::optional<std::int32_t> free_row;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (requests[lane].lifecycle == Lifecycle::Empty &&
                active_continuations[lane] == continuation_capacity) {
                free_row = static_cast<std::int32_t>(lane);
                break;
            }
        }
        if (!free_row) {
            throw std::invalid_argument("shared snapshot adoption requires an idle execution lane");
        }
        if (!host_kv_extents) {
            throw std::invalid_argument("shared snapshot adoption requires Host KV capacity");
        }
        const auto build_address = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                       const HostKVPageLayout& layout, std::uint32_t frontier,
                                       const std::uint8_t* payload) {
            std::optional<KVAddressSpaceHandle> address = addresses.create_inactive();
            if (!address) {
                throw std::invalid_argument("shared snapshot does not fit the KV address capacity");
            }
            try {
                const std::uint32_t count = kv_pages_for_frontier(frontier);
                addresses.activate(*address, count, *free_row);
                addresses.materialize_to_tokens(*address, frontier, device.stream);
                addresses.commit_frontier(*address, frontier);
                std::vector<DeviceKVPageHandle> destinations;
                std::vector<LogicalKVPageHandle> logical;
                destinations.reserve(count);
                logical.reserve(count);
                for (std::uint32_t page = 0; page < count; ++page) {
                    destinations.push_back(addresses.physical_page(*address, page));
                    logical.push_back(addresses.logical_page(*address, page));
                }
                pages.physical_pool().copy_from_host(reinterpret_cast<const std::byte*>(payload),
                                                     layout, destinations, device.stream);
                device.synchronize();
                addresses.deactivate(*address);
                std::optional<HostKVExtentReservation> host =
                    host_kv_extents->prepare(pages, logical);
                if (!host) {
                    throw std::invalid_argument(
                        "shared snapshot does not fit the Host KV capacity");
                }
                HostKVAllocationView destination = host_kv_extents->writable_view(*host);
                std::memcpy(destination.data(), payload,
                            static_cast<std::size_t>(count) * layout.page_stride);
                (void)host_kv_extents->publish(std::move(*host));
                addresses.set_checkpoint_requirement(*address, frontier);
                return *address;
            } catch (...) {
                if (addresses.active(*address)) { addresses.deactivate(*address); }
                (void)addresses.release(*address);
                (void)host_kv_extents->release_unreferenced();
                throw;
            }
        };
        text_address = build_address(*text_kv_addresses, *text_kv_pages, text_layout,
                                     backing->boundary.frontier,
                                     backing->storage.data() + backing->text_offset);
        runtime::testing::shared_snapshot_import_checkpoint(
            runtime::testing::SharedSnapshotImportStage::MainKvAllocated);
        if (backing->boundary.backend_frontier != 0) {
            backend_address = build_address(*backend_kv_addresses, *backend_kv_pages,
                                            *backend_layout, backing->boundary.backend_frontier,
                                            backing->storage.data() + backing->backend_offset);
        }

        SharedPrefixState& shared = shared_prefix_states[*shared_index];
        state_store->retain_checkpoint_reference(*state);
        shared.state    = *state;
        shared.kv       = SequenceKVBundle{.text = *text_address, .backend = backend_address};
        shared.identity = backing->identity;
        shared.frontier = backing->boundary.frontier;
        shared.backend_frontier                    = backing->boundary.backend_frontier;
        shared.rope_delta                          = backing->boundary.rope_delta;
        shared.tail_hidden_valid                   = backing->boundary.tail_hidden_valid != 0;
        shared.rebuild_work                        = backing->boundary.rebuild_work;
        shared.active_references                   = 0;
        shared_populated                           = true;
        const qwen3_6::SharedPrefixSummary summary = shared_prefix_summary(shared);
        if (summary != imported.summary()) {
            throw std::logic_error("shared snapshot summary changed during sealed adoption");
        }
        shared_prefix_slots[*shared_index].role = SharedPrefixSlotRole::Catalogued;
        state.reset();
        text_address.reset();
        backend_address.reset();
        advance_resource_revision();
        return {.handle = ContractAccess::make_shared_prefix(
                    this, *shared_index, shared_prefix_slots[*shared_index].generation),
                .summary = summary};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        if (shared_populated) {
            shared_prefix_states[*shared_index] = {};
            if (state) { state_store->release_checkpoint_reference(*state); }
        }
        if (backend_address) { (void)backend_kv_addresses->release(*backend_address); }
        if (text_address) { (void)text_kv_addresses->release(*text_address); }
        if (state) { (void)state_store->release(*state); }
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        if (*shared_index < shared_prefix_capacity &&
            shared_prefix_slots[*shared_index].role == SharedPrefixSlotRole::ReservedCapture) {
            shared_prefix_slots[*shared_index].role = SharedPrefixSlotRole::Free;
        }
        throw;
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
