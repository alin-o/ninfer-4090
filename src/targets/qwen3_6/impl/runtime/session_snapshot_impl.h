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
// Version 4 removes rendering/scheduling frontiers from model-input identity and its digests.
// Earlier versions are rejected instead of mixing the old and new shortlist semantics.

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr char kSessionSnapshotMagic[8]         = {'N', 'I', 'N', 'F', 'S', 'E', 'S', '1'};
constexpr std::uint32_t kSessionSnapshotVersion = 4;
constexpr char kSharedSnapshotMagic[8]          = {'N', 'I', 'N', 'F', 'S', 'H', 'R', '1'};
constexpr std::uint32_t kSharedSnapshotVersion  = 2;
constexpr std::size_t kSharedChecksumBytes      = 32;
constexpr std::size_t kSharedEnvelopeHeaderBytes =
    sizeof(kSharedSnapshotMagic) + sizeof(std::uint32_t) + 2U * sizeof(std::uint64_t) +
    kSharedChecksumBytes;
constexpr std::uint32_t kSharedIdentitySchema = 2;

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
    std::vector<std::uint8_t> owned_storage;
    std::shared_ptr<const std::vector<std::uint8_t>> retained_storage;
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    SharedSnapshotBoundary boundary;
    std::size_t state_offset   = 0;
    std::size_t text_offset    = 0;
    std::size_t backend_offset = 0;

    [[nodiscard]] const std::uint8_t* data() const noexcept {
        return retained_storage ? retained_storage->data() : owned_storage.data();
    }
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
    constexpr std::size_t identity_fixed_bytes = 5U * sizeof(std::uint64_t) + sizeof(std::uint32_t);
    constexpr std::size_t identity_bytes_per_token =
        sizeof(TokenId) + sizeof(std::uint8_t) + 3U * sizeof(std::int32_t);
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
            "session snapshot uses an obsolete prefix identity format and cannot be restored");
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
                                         std::move(vision_items));
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

std::vector<qwen3_6::DurableSharedPrefixCandidate>
ProgramImplCore::durable_shared_prefix_candidates(const PreparedPromptData& prompt) {
    std::vector<qwen3_6::DurableSharedPrefixCandidate> candidates;
    if (!prompt.vision_items.empty() || prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3U * prompt.token_ids.size()) {
        return candidates;
    }

    for (const qwen3_6::PreparedCacheOpportunity& opportunity :
         prompt.context_cache.opportunities) {
        if (opportunity.kind != PromptCacheMarkerKind::SharedStablePrefix ||
            opportunity.frontier == 0 || opportunity.frontier > prompt.token_ids.size()) {
            continue;
        }
        const qwen3_6::SharedPrefixPersistenceMetadata metadata{
            .evidence             = opportunity.evidence,
            .structural_origins   = opportunity.structural_origins,
            .structural_role      = static_cast<std::uint8_t>(opportunity.structural_role),
            .ssd_eligible         = opportunity.ssd_eligible,
            .first_volatile_token = prompt.context_cache.first_volatile_token,
        };
        try {
            validate_durable_metadata(metadata, opportunity.frontier);
        } catch (const std::invalid_argument&) { continue; }

        std::vector<std::uint8_t> identity_bytes;
        SnapshotWriter writer(identity_bytes);
        write_span(writer, std::span<const TokenId>(prompt.token_ids).first(opportunity.frontier));
        write_span(writer,
                   std::span<const std::uint8_t>(prompt.token_types).first(opportunity.frontier));
        for (std::size_t axis = 0; axis < 3; ++axis) {
            write_span(writer, std::span<const std::int32_t>(prompt.positions)
                                   .subspan(axis * prompt.token_ids.size(), opportunity.frontier));
        }
        write_vision_items(writer, {});
        candidates.push_back(qwen3_6::DurableSharedPrefixCandidate{
            .content_digest =
                frontend_internal::sha256_hex(frontend_internal::sha256(identity_bytes)),
            .frontier = opportunity.frontier,
        });
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& left, const auto& right) { return left.frontier > right.frontier; });
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

struct DurableSharedImportPhysicalPlan {
    struct HandleIdentity {
        std::uint32_t index      = 0;
        std::uint64_t generation = 0;
    };

    const ProgramImplCore* owner = nullptr;
    runtime::ProgramResourceRevision resource_revision;
    std::optional<HandleIdentity> replacement;
    std::optional<HandleIdentity> host_private;
    std::optional<HandleIdentity> host_shared;
    std::optional<StateImageHandle> reclaimed_private_state;
    std::vector<runtime::CheckpointRef> reclaimed_private_checkpoints;
    std::vector<StateImageHandle> duplicate_host_states;
    std::vector<HostKVPageReplicaRelease> duplicate_host_releases;
    runtime::UniquePhysicalReclamation reclamation;
};

runtime::DurableImportAssessment ProgramImplCore::inspect_durable_shared_prefix_import(
    std::uint32_t frontier, const SharedPrefixHandle* replacement,
    const ContinuationHandle* host_private, const SharedPrefixHandle* host_shared) const {
    runtime::DurableImportAssessment out{.resource_revision = resource_revision_};
    if (frontier == 0 || frontier > capacity || speculative_backend == SpeculativeBackend::DFlash ||
        !state_store || !text_kv_addresses || !text_kv_pages) {
        out.feasibility = runtime::DurableImportFeasibility::Unsupported;
        return out;
    }
    if (!host_state_images) {
        out.feasibility = runtime::DurableImportFeasibility::HostStateCapacity;
        return out;
    }
    if (!host_kv_arena || !host_kv_extents) {
        out.feasibility = runtime::DurableImportFeasibility::HostKvCapacity;
        return out;
    }
    if (pending_transaction_ || has_context_transaction()) {
        out.feasibility = runtime::DurableImportFeasibility::TransactionConflict;
        return out;
    }
    if ((host_private != nullptr && host_shared != nullptr) ||
        (host_private != nullptr && !valid_continuation(*host_private)) ||
        (host_shared != nullptr && !valid_shared_prefix(*host_shared)) ||
        (replacement != nullptr && host_shared == replacement)) {
        out.feasibility = runtime::DurableImportFeasibility::TransactionConflict;
        return out;
    }

    const SharedPrefixState* victim = nullptr;
    if (replacement != nullptr) {
        if (!valid_shared_prefix(*replacement)) {
            out.feasibility = runtime::DurableImportFeasibility::TransactionConflict;
            return out;
        }
        const std::uint32_t index = ContractAccess::index(*replacement);
        if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
            out.feasibility = runtime::DurableImportFeasibility::TransactionConflict;
            return out;
        }
        victim                                  = &shared_prefix_states[index];
        const detail::PhysicalResources removed = resident_resources(*victim);
        out.reclamation                         = {
                                    .device_state_slots      = removed.device.state_slots,
                                    .device_main_kv_pages    = removed.device.main_kv_pages,
                                    .device_backend_kv_pages = removed.device.backend_kv_pages,
                                    .host_state_slots        = removed.host.state_slots,
                                    .host_kv_bytes           = removed.host.kv_bytes,
        };
        if (state_store->checkpoint_references(victim->state) != 1) {
            out.reclamation.device_state_slots = 0;
            out.reclamation.host_state_slots   = 0;
        }
    } else if (std::none_of(
                   shared_prefix_slots.begin(), shared_prefix_slots.end(),
                   [](const auto& slot) { return slot.role == SharedPrefixSlotRole::Free; })) {
        out.feasibility = runtime::DurableImportFeasibility::Unsupported;
        return out;
    }

    std::vector<StateImageHandle> duplicate_host_states;
    std::vector<HostKVPageReplicaRelease> duplicate_host_releases;
    const auto append_state = [&](StateImageHandle state) {
        if (victim != nullptr && state == victim->state) { return; }
        if (std::find(duplicate_host_states.begin(), duplicate_host_states.end(), state) !=
            duplicate_host_states.end()) {
            return;
        }
        if (state_store->residency(state) == StateReplicaResidency::Both &&
            state_store->source_pins(state) == 0) {
            duplicate_host_states.push_back(state);
        }
    };
    const auto append_host_pages = [&](const SequenceKVBundle& kv) {
        const auto append = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address) {
            for (std::uint32_t index = 0; index < addresses.mapped_pages(address); ++index) {
                const LogicalKVPageHandle page = addresses.logical_page(address, index);
                const bool duplicate = pages.device_resident(page) && pages.host_resident(page) &&
                                       pages.source_pins(page) == 0 &&
                                       host_kv_extents->can_release_page_replica(pages, page);
                const bool already =
                    std::any_of(duplicate_host_releases.begin(), duplicate_host_releases.end(),
                                [&](const HostKVPageReplicaRelease& item) {
                                    return item.pages == &pages && item.page == page;
                                });
                if (duplicate && !already) {
                    duplicate_host_releases.push_back({.pages = &pages, .page = page});
                }
            }
        };
        append(*text_kv_addresses, *text_kv_pages, kv.text);
        if (kv.backend) { append(*backend_kv_addresses, *backend_kv_pages, *kv.backend); }
    };
    if (host_private != nullptr) {
        const SequenceState& sequence = continuation_states[ContractAccess::index(*host_private)];
        if (sequence.endpoint_valid) { append_state(sequence.state.read); }
        if (sequence.rewrite_state) { append_state(*sequence.rewrite_state); }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            append_state(anchor.state);
        }
        if (sequence.kv) { append_host_pages(*sequence.kv); }
    } else if (host_shared != nullptr) {
        const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(*host_shared)];
        if (shared.active_references != 0 || !shared.kv) {
            out.feasibility = runtime::DurableImportFeasibility::TransactionConflict;
            return out;
        }
        append_state(shared.state);
        append_host_pages(*shared.kv);
    }
    std::optional<runtime::CheckpointRef> replacement_private_coverage;
    if (victim != nullptr && host_private != nullptr) {
        const SequenceState& sequence = continuation_states[ContractAccess::index(*host_private)];
        const qwen3_6::ContinuationSummary summary      = continuation_summary(sequence);
        const qwen3_6::CheckpointSummary victim_summary = shared_prefix_summary(*victim).checkpoint;
        const auto kv_prefix_aliases =
            [&](const KVAddressSpaceStore& addresses, KVAddressSpaceHandle private_address,
                KVAddressSpaceHandle shared_address, std::uint32_t pages) {
                if (addresses.mapped_pages(private_address) < pages ||
                    addresses.mapped_pages(shared_address) < pages) {
                    return false;
                }
                for (std::uint32_t page = 0; page < pages; ++page) {
                    if (addresses.logical_page(private_address, page) !=
                        addresses.logical_page(shared_address, page)) {
                        return false;
                    }
                }
                return true;
            };
        const bool kv_aliases =
            sequence.kv &&
            kv_prefix_aliases(*text_kv_addresses, sequence.kv->text, victim->kv->text,
                              victim_summary.required_kv.main_pages) &&
            (victim_summary.required_kv.backend_pages == 0 ||
             (sequence.kv->backend && victim->kv->backend && backend_kv_addresses &&
              kv_prefix_aliases(*backend_kv_addresses, *sequence.kv->backend, *victim->kv->backend,
                                victim_summary.required_kv.backend_pages)));
        const auto covers = [&](const std::optional<qwen3_6::CheckpointSummary>& checkpoint,
                                std::optional<StateImageHandle> state) {
            if (replacement_private_coverage || !checkpoint || !state || !kv_aliases ||
                *state != victim->state ||
                checkpoint->ref.frontier != victim_summary.ref.frontier ||
                checkpoint->shortlist_key != victim_summary.shortlist_key ||
                checkpoint->required_kv != victim_summary.required_kv) {
                return;
            }
            replacement_private_coverage = checkpoint->ref;
        };
        covers(summary.endpoint,
               sequence.endpoint_valid ? std::optional(sequence.state.read) : std::nullopt);
        covers(summary.rewrite, sequence.rewrite_state);
        for (std::size_t index = 0;
             index < summary.long_anchors.size() && index < sequence.long_anchors.size(); ++index) {
            covers(std::optional(summary.long_anchors[index]),
                   std::optional(sequence.long_anchors[index].state));
        }
        out.replacement_alternate_coverage = replacement_private_coverage.has_value();
    }
    std::optional<StateImageHandle> reclaimed_private_state;
    std::vector<runtime::CheckpointRef> reclaimed_private_checkpoints;
    std::uint32_t victim_state_descriptors =
        victim != nullptr && state_store->checkpoint_references(victim->state) == 1 ? 1U : 0U;
    if (state_store->occupied() - victim_state_descriptors >= state_store->capacity() &&
        host_private != nullptr) {
        const SequenceState& sequence = continuation_states[ContractAccess::index(*host_private)];
        const qwen3_6::ContinuationSummary summary = continuation_summary(sequence);

        struct Candidate {
            StateImageHandle state;
            std::vector<runtime::CheckpointRef> checkpoints;
            std::uint32_t deepest_frontier = 0;
        };

        std::vector<Candidate> candidates;
        const auto append = [&](StateImageHandle state, runtime::CheckpointRef checkpoint) {
            if (state == sequence.state.read || state_store->source_pins(state) != 0 ||
                (replacement_private_coverage && checkpoint == *replacement_private_coverage)) {
                return;
            }
            const StateReplicaResidency residency = state_store->residency(state);
            if (residency != StateReplicaResidency::HostOnly) { return; }
            auto found =
                std::find_if(candidates.begin(), candidates.end(),
                             [&](const Candidate& candidate) { return candidate.state == state; });
            if (found == candidates.end()) {
                candidates.push_back(Candidate{.state = state});
                found = std::prev(candidates.end());
            }
            found->checkpoints.push_back(checkpoint);
            found->deepest_frontier = std::max(found->deepest_frontier, checkpoint.frontier);
        };
        if (sequence.rewrite_state && summary.rewrite) {
            append(*sequence.rewrite_state, summary.rewrite->ref);
        }
        for (std::size_t index = 0;
             index < sequence.long_anchors.size() && index < summary.long_anchors.size(); ++index) {
            append(sequence.long_anchors[index].state, summary.long_anchors[index].ref);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& left, const Candidate& right) {
                      return std::tuple{left.deepest_frontier, left.checkpoints.size()} <
                             std::tuple{right.deepest_frontier, right.checkpoints.size()};
                  });
        for (Candidate& candidate : candidates) {
            const std::uint32_t victim_reference =
                victim != nullptr && victim->state == candidate.state ? 1U : 0U;
            if (state_store->checkpoint_references(candidate.state) !=
                    candidate.checkpoints.size() + victim_reference ||
                !inspect_checkpoint_drop_option(sequence, candidate.checkpoints)) {
                continue;
            }
            reclaimed_private_state               = candidate.state;
            reclaimed_private_checkpoints         = std::move(candidate.checkpoints);
            victim_state_descriptors              = 1;
            const StateReplicaResidency residency = state_store->residency(candidate.state);
            out.reclamation.device_state_slots += residency == StateReplicaResidency::DeviceOnly ||
                                                          residency == StateReplicaResidency::Both
                                                      ? 1U
                                                      : 0U;
            out.reclamation.host_state_slots += residency == StateReplicaResidency::HostOnly ||
                                                        residency == StateReplicaResidency::Both
                                                    ? 1U
                                                    : 0U;
            break;
        }
    }
    if (state_store->occupied() - victim_state_descriptors >= state_store->capacity()) {
        out.feasibility = runtime::DurableImportFeasibility::LogicalStateCapacity;
        return out;
    }
    if (reclaimed_private_state) { std::erase(duplicate_host_states, *reclaimed_private_state); }
    if (out.reclamation.host_state_slots > host_state_images->occupied()) {
        out.feasibility = runtime::DurableImportFeasibility::HostStateCapacity;
        return out;
    }
    const std::uint32_t host_state_after_victim =
        host_state_images->occupied() - out.reclamation.host_state_slots;
    const std::uint32_t required_duplicate_host_states =
        host_state_after_victim < host_state_images->capacity()
            ? 0U
            : host_state_after_victim - host_state_images->capacity() + 1U;
    if (required_duplicate_host_states > duplicate_host_states.size()) {
        out.feasibility = runtime::DurableImportFeasibility::HostStateCapacity;
        return out;
    }
    duplicate_host_states.resize(required_duplicate_host_states);
    out.reclamation.host_state_slots += required_duplicate_host_states;

    const std::uint32_t text_pages = kv_pages_for_frontier(frontier);
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp ? frontier - 1U : 0U;
    const std::uint32_t backend_pages = kv_pages_for_frontier(backend_frontier);
    std::vector<HostKVPageReplicaRelease> host_releases;
    const auto collect = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                             KVAddressSpaceHandle address) {
        for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) == 1 && pages.host_resident(logical)) {
                host_releases.push_back({.pages = &pages, .page = logical});
            }
        }
    };
    if (victim != nullptr) {
        collect(*text_kv_addresses, *text_kv_pages, victim->kv->text);
        if (victim->kv->backend) {
            collect(*backend_kv_addresses, *backend_kv_pages, *victim->kv->backend);
        }
    }
    std::erase_if(duplicate_host_releases, [&](const HostKVPageReplicaRelease& duplicate) {
        return std::any_of(host_releases.begin(), host_releases.end(),
                           [&](const HostKVPageReplicaRelease& released) {
                               return duplicate.pages == released.pages &&
                                      duplicate.page == released.page;
                           });
    });
    const auto address_fits = [&](const KVAddressSpaceStore& addresses,
                                  const LogicalKVPageStore& pages, std::uint32_t required,
                                  bool releases_address) {
        const std::uint32_t freed_address = releases_address ? 1U : 0U;
        std::uint32_t freed_pages         = 0;
        if (releases_address) {
            const KVAddressSpaceHandle address =
                &addresses == text_kv_addresses.get() ? victim->kv->text : *victim->kv->backend;
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.address_references(logical) == 1) { ++freed_pages; }
            }
        }
        return addresses.occupied() - freed_address < addresses.capacity() &&
               required <= pages.capacity() - pages.occupied() + freed_pages;
    };
    if (!address_fits(*text_kv_addresses, *text_kv_pages, text_pages, victim != nullptr) ||
        (backend_pages != 0 &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !address_fits(*backend_kv_addresses, *backend_kv_pages, backend_pages,
                        victim != nullptr && victim->kv->backend.has_value())))) {
        out.feasibility = runtime::DurableImportFeasibility::DeviceCapacity;
        return out;
    }
    const HostKVPageLayout text_layout =
        plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry());
    std::optional<HostKVPageLayout> backend_layout;
    std::array<HostKVAllocationRequest, 2> allocations{};
    allocations[0]               = {.layout = &text_layout, .pages = text_pages};
    std::size_t allocation_count = 1;
    if (backend_pages != 0) {
        backend_layout = plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry());
        allocations[allocation_count++] = {.layout = &*backend_layout, .pages = backend_pages};
    }
    try {
        std::vector<HostKVPageReplicaRelease> selected_host_duplicates;
        selected_host_duplicates.reserve(duplicate_host_releases.size());
        const auto allocations_span =
            std::span<const HostKVAllocationRequest>(allocations.data(), allocation_count);
        while (!host_kv_extents->can_prepare_after_page_releases(selected_host_duplicates,
                                                                 host_releases, allocations_span)) {
            if (selected_host_duplicates.size() == duplicate_host_releases.size()) {
                out.feasibility = runtime::DurableImportFeasibility::HostKvCapacity;
                return out;
            }
            selected_host_duplicates.push_back(
                duplicate_host_releases[selected_host_duplicates.size()]);
        }
        duplicate_host_releases = std::move(selected_host_duplicates);
        for (const HostKVPageReplicaRelease& release : duplicate_host_releases) {
            const std::size_t stride = release.pages == text_kv_pages.get()
                                           ? text_host_kv_page_stride
                                           : backend_host_kv_page_stride;
            if (stride > std::numeric_limits<std::size_t>::max() - out.reclamation.host_kv_bytes) {
                out.feasibility = runtime::DurableImportFeasibility::HostKvCapacity;
                return out;
            }
            out.reclamation.host_kv_bytes += stride;
        }
        if (!host_kv_extents->can_prepare_after_page_releases(duplicate_host_releases,
                                                              host_releases, allocations_span)) {
            out.feasibility = runtime::DurableImportFeasibility::HostKvCapacity;
            return out;
        }
    } catch (...) {
        out.feasibility = runtime::DurableImportFeasibility::HostKvCapacity;
        return out;
    }
    out.feasibility         = runtime::DurableImportFeasibility::Feasible;
    auto plan               = std::make_shared<DurableSharedImportPhysicalPlan>();
    plan->owner             = this;
    plan->resource_revision = resource_revision_;
    const auto identity     = [](const auto& handle) {
        return DurableSharedImportPhysicalPlan::HandleIdentity{
                .index = ContractAccess::index(handle), .generation = ContractAccess::epoch(handle)};
    };
    if (replacement != nullptr) { plan->replacement = identity(*replacement); }
    if (host_private != nullptr) { plan->host_private = identity(*host_private); }
    if (host_shared != nullptr) { plan->host_shared = identity(*host_shared); }
    plan->duplicate_host_states         = std::move(duplicate_host_states);
    plan->duplicate_host_releases       = std::move(duplicate_host_releases);
    plan->reclaimed_private_state       = reclaimed_private_state;
    plan->reclaimed_private_checkpoints = std::move(reclaimed_private_checkpoints);
    plan->reclamation                   = out.reclamation;
    out.physical_plan                   = std::move(plan);
    return out;
}

qwen3_6::RetainedSessionSnapshot ProgramImplCore::begin_export_shared_prefix(
    const SharedPrefixHandle& handle, std::string_view model_binding,
    const qwen3_6::SharedPrefixPersistenceMetadata& metadata,
    const std::function<std::shared_ptr<void>(std::size_t)>& reserve) {
    return begin_shared_prefix_snapshot(handle, model_binding, &metadata, reserve);
}

// A null metadata pointer selects a transaction-local payload, with no durable envelope.
// The caller retains the exact in-memory identity; SSD policy applies only to publication.
qwen3_6::RetainedSessionSnapshot ProgramImplCore::begin_shared_prefix_snapshot(
    const SharedPrefixHandle& handle, std::string_view model_binding,
    const qwen3_6::SharedPrefixPersistenceMetadata* metadata,
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
    if (metadata && speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("shared snapshot does not support the DFlash backend");
    }

    const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(handle)];
    if (!shared.kv || !shared.identity || !shared.identity->backing || shared.frontier == 0 ||
        shared.identity->shortlist_key.frontier != shared.frontier ||
        shared.identity->ledger().size() != shared.frontier || !state_store->valid(shared.state) ||
        state_store->residency(shared.state) == StateReplicaResidency::None) {
        throw std::logic_error("shared snapshot source is incomplete");
    }
    if (metadata) { validate_durable_metadata(*metadata, shared.frontier); }
    const auto* identity = shared.identity->prefix_identity();
    if (identity == nullptr || identity->size() < shared.frontier) {
        throw std::logic_error("shared snapshot exact identity is incomplete");
    }
    if (metadata && !identity->vision_items().empty()) {
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

    qwen3_6::RetainedSessionSnapshot snapshot;
    snapshot.tokens = shared.frontier;
    SnapshotWriter writer(snapshot.bytes);
    std::size_t total_size_offset   = 0;
    std::size_t payload_size_offset = 0;
    std::optional<std::size_t> checksum_offset;
    if (metadata) {
        std::vector<std::uint8_t> identity_bytes;
        SnapshotWriter identity_writer(identity_bytes);
        write_span(identity_writer, shared.identity->ledger());
        write_span(identity_writer,
                   std::span<const std::uint8_t>(identity->token_types()).first(shared.frontier));
        for (std::size_t axis = 0; axis < 3; ++axis) {
            write_span(identity_writer, std::span<const std::int32_t>(identity->position_axis(axis))
                                            .first(shared.frontier));
        }
        write_vision_items(identity_writer, {});
        const auto identity_digest = frontend_internal::sha256(identity_bytes);

        snapshot.session_digest = ledger_prefix_digest(shared.identity->ledger());
        snapshot.content_digest = frontend_internal::sha256_hex(identity_digest);
        writer.bytes(kSharedSnapshotMagic, sizeof(kSharedSnapshotMagic));
        writer.pod(kSharedSnapshotVersion);
        total_size_offset = snapshot.bytes.size();
        writer.pod<std::uint64_t>(0);
        payload_size_offset = snapshot.bytes.size();
        writer.pod<std::uint64_t>(0);
        checksum_offset = snapshot.bytes.size();
        std::array<std::uint8_t, kSharedChecksumBytes> empty_checksum{};
        writer.bytes(empty_checksum.data(), empty_checksum.size());
        if (snapshot.bytes.size() != kSharedEnvelopeHeaderBytes) {
            throw std::logic_error("shared snapshot envelope geometry changed");
        }

        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(model_binding.size()));
        writer.bytes(model_binding.data(), model_binding.size());
        write_shared_config(writer, config);
        write_shared_boundary(writer, boundary);
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(metadata->evidence));
        writer.pod(metadata->structural_origins);
        writer.pod(metadata->structural_role);
        writer.pod<std::uint8_t>(metadata->ssd_eligible ? 1U : 0U);
        writer.pod<std::uint8_t>(metadata->first_volatile_token ? 1U : 0U);
        writer.pod<std::uint32_t>(metadata->first_volatile_token.value_or(0));
        writer.bytes(identity_digest.data(), identity_digest.size());
        writer.pod(shared.identity->shortlist_key.digests[0]);
        writer.pod(shared.identity->shortlist_key.digests[1]);
        writer.pod<std::uint64_t>(identity_bytes.size());
        writer.bytes(identity_bytes.data(), identity_bytes.size());
    }

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
    if (metadata) {
        const std::uint64_t total_u64   = transfer_bytes;
        const std::uint64_t payload_u64 = transfer_bytes - kSharedEnvelopeHeaderBytes;
        std::memcpy(snapshot.bytes.data() + total_size_offset, &total_u64, sizeof(total_u64));
        std::memcpy(snapshot.bytes.data() + payload_size_offset, &payload_u64, sizeof(payload_u64));
    }

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
                    runtime::testing::note_shared_snapshot_export_pin_released();
                }
                for (const auto& [store, source] : kv_sources) {
                    store->unpin_source(source);
                    runtime::testing::note_shared_snapshot_export_pin_released();
                }
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
        // Tracking insertion may allocate. Until it succeeds this scope owns the pin; afterward
        // SharedTransferSettlement owns it through transfer completion and retirement.
        state_store->pin_snapshot_source(shared.state);
        runtime::testing::note_shared_snapshot_export_pin_acquired();
        try {
            runtime::testing::shared_snapshot_export_checkpoint(
                runtime::testing::SharedSnapshotExportStage::StatePinnedBeforeRegistration);
            pending->state_sources.push_back(shared.state);
        } catch (...) {
            state_store->unpin_snapshot_source(shared.state);
            runtime::testing::note_shared_snapshot_export_pin_released();
            throw;
        }
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
                runtime::testing::note_shared_snapshot_export_pin_acquired();
                try {
                    runtime::testing::shared_snapshot_export_checkpoint(
                        runtime::testing::SharedSnapshotExportStage::KvPinnedBeforeRegistration);
                    pending->kv_sources.emplace_back(&pages, logical);
                } catch (...) {
                    pages.unpin_source(logical);
                    runtime::testing::note_shared_snapshot_export_pin_released();
                    throw;
                }
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
        if (checksum_offset) {
            const auto checksum = frontend_internal::sha256(
                std::span<const std::uint8_t>(bytes).subspan(kSharedEnvelopeHeaderBytes));
            std::memcpy(bytes.data() + *checksum_offset, checksum.data(), checksum.size());
        }
    };
    snapshot.settle_transfer = [pending] { pending->settle(); };
    snapshot_source_retirements_.push_back(SnapshotSourceRetirement{
        .ready =
            [pending] {
                runtime::testing::shared_snapshot_export_checkpoint(
                    runtime::testing::SharedSnapshotExportStage::BeforeReadinessQuery);
                return !pending->submitted || pending->completion->ready();
            },
        .retire = [pending] { pending->retire(); },
    });
    return snapshot;
}

qwen3_6::ValidatedSharedPrefixImport<Variant> ProgramImplCore::parse_shared_prefix(
    std::span<const std::uint8_t> snapshot, std::string_view model_binding,
    const std::function<void()>& cancellation_checkpoint,
    std::shared_ptr<const std::vector<std::uint8_t>> retained_storage) const {
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
        throw runtime::DurableImportError(runtime::DurableImportErrorKind::ChecksumMismatch,
                                          "shared snapshot payload checksum does not match");
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
    capture_backing->prefix_identity.restore(std::move(token_types), std::move(positions), {});
    PreparedPromptData prompt;
    prompt.token_ids   = ledger;
    prompt.token_types = capture_backing->prefix_identity.token_types();
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto& values = capture_backing->prefix_identity.position_axis(axis);
        prompt.positions.insert(prompt.positions.end(), values.begin(), values.end());
    }
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
    if (retained_storage && retained_storage->data() == snapshot.data() &&
        retained_storage->size() == snapshot.size()) {
        backing->retained_storage = std::move(retained_storage);
    } else {
        backing->owned_storage.assign(snapshot.begin(), snapshot.end());
    }
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
    return adopt_shared_prefix_impl(imported);
}

void ProgramImplCore::duplicate_shared_prefix_to_device_for_test(const SharedPrefixHandle& handle) {
    if (!valid_shared_prefix(handle) || pending_transaction_ || has_context_transaction()) {
        throw std::logic_error("test shared-prefix materialization is not available");
    }
    SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(handle)];
    if (!shared.kv || !host_kv_extents) {
        throw std::logic_error("test shared-prefix materialization has no Host KV source");
    }
    std::optional<StateImageTransfer> state_transfer;
    if (state_store->residency(shared.state) == StateReplicaResidency::HostOnly) {
        auto transfer = state_store->begin_host_to_device(shared.state, device.transfer_stream);
        if (!transfer) {
            throw std::logic_error("test shared-prefix Device State capacity is exhausted");
        }
        state_transfer.emplace(std::move(*transfer));
    }

    struct PendingDeviceKVDuplicate {
        LogicalKVPageStore* pages = nullptr;
        std::vector<LogicalKVPageHandle> missing;
        std::optional<DeviceKVPageReservation> reservation;
    };

    const auto prepare_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address) {
        PendingDeviceKVDuplicate pending{.pages = &pages};
        pending.missing.reserve(addresses.mapped_pages(address));
        for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (!pages.device_resident(logical)) {
                if (!pages.host_resident(logical)) {
                    throw std::logic_error("test shared-prefix page has no complete replica");
                }
                pending.missing.push_back(logical);
            }
        }
        if (pending.missing.empty()) { return pending; }
        pending.reservation =
            pages.physical_pool().reserve(static_cast<std::uint32_t>(pending.missing.size()));
        if (!pending.reservation) {
            throw std::logic_error(
                "test shared-prefix Device KV capacity is exhausted (requested " +
                std::to_string(pending.missing.size()) + ", available " +
                std::to_string(pages.physical_pool().available_pages()) + ")");
        }
        return pending;
    };
    const auto materialize_kv = [&](PendingDeviceKVDuplicate& pending) {
        std::vector<DeviceKVPageHandle> destinations;
        destinations.reserve(pending.missing.size());
        for (const LogicalKVPageHandle logical : pending.missing) {
            destinations.push_back(
                pending.pages->reserve_device_replica(logical, *pending.reservation));
        }
        for (std::size_t index = 0; index < pending.missing.size(); ++index) {
            const HostKVPageReplica replica = pending.pages->host_replica(pending.missing[index]);
            const HostKVAllocationConstView source =
                host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
            const std::array destination{destinations[index]};
            pending.pages->physical_pool().copy_from_host(source, destination,
                                                          device.transfer_stream);
        }
    };
    const auto abort_kv = [](PendingDeviceKVDuplicate& pending) noexcept {
        if (!pending.reservation) { return; }
        for (const LogicalKVPageHandle page : pending.missing) {
            pending.pages->abort_device_replica(page, *pending.reservation);
        }
    };
    auto text = prepare_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text);
    std::optional<PendingDeviceKVDuplicate> backend;
    if (shared.kv->backend) {
        backend.emplace(prepare_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend));
    }
    try {
        materialize_kv(text);
        if (backend) { materialize_kv(*backend); }
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    } catch (...) {
        if (backend) { abort_kv(*backend); }
        abort_kv(text);
        throw;
    }
    for (const LogicalKVPageHandle page : text.missing) {
        text_kv_pages->publish_device_replica(page);
    }
    if (backend) {
        for (const LogicalKVPageHandle page : backend->missing) {
            backend_kv_pages->publish_device_replica(page);
        }
    }
    if (state_transfer) { state_store->publish_transfer(std::move(*state_transfer), true); }
    advance_resource_revision();
}

void ProgramImplCore::fragment_shared_prefix_host_kv_for_test(
    const SharedPrefixHandle& victim_handle, const SharedPrefixHandle& separator_handle) {
    if (!valid_shared_prefix(victim_handle) || !valid_shared_prefix(separator_handle) ||
        (ContractAccess::index(victim_handle) == ContractAccess::index(separator_handle) &&
         ContractAccess::epoch(victim_handle) == ContractAccess::epoch(separator_handle)) ||
        pending_transaction_ || has_context_transaction() || !host_kv_arena || !host_kv_extents) {
        throw std::logic_error("test shared-prefix fragmentation is not available");
    }
    SharedPrefixState& victim    = shared_prefix_states[ContractAccess::index(victim_handle)];
    SharedPrefixState& separator = shared_prefix_states[ContractAccess::index(separator_handle)];
    if (!victim.kv || !separator.kv || victim.kv->backend || separator.kv->backend) {
        throw std::logic_error("test shared-prefix fragmentation requires Main-only checkpoints");
    }

    const auto membership = [&](const SharedPrefixState& shared) {
        std::vector<LogicalKVPageHandle> pages;
        const std::uint32_t count = text_kv_addresses->mapped_pages(shared.kv->text);
        pages.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            pages.push_back(text_kv_addresses->logical_page(shared.kv->text, index));
        }
        return pages;
    };
    const std::vector<LogicalKVPageHandle> victim_pages    = membership(victim);
    const std::vector<LogicalKVPageHandle> separator_pages = membership(separator);
    if (victim_pages.size() != 2 || separator_pages.size() != 2) {
        throw std::logic_error("test shared-prefix fragmentation requires two pages per owner");
    }
    const HostKVPageLayout& layout = host_kv_extents->page_layout(*text_kv_pages);
    if (host_kv_arena->occupied_bytes() != host_kv_arena->capacity_bytes() ||
        host_kv_arena->capacity_bytes() != 4U * layout.page_stride) {
        throw std::logic_error(
            "test shared-prefix fragmentation requires an exact four-page arena");
    }

    struct SavedPage {
        LogicalKVPageHandle page;
        std::vector<std::byte> bytes;
    };

    const auto save = [&](LogicalKVPageHandle page) {
        if (!text_kv_pages->device_resident(page) || !text_kv_pages->host_resident(page) ||
            text_kv_pages->source_pins(page) != 0) {
            throw std::logic_error("test shared-prefix fragmentation page is not settled Both");
        }
        const HostKVPageReplica replica = text_kv_pages->host_replica(page);
        const HostKVAllocationConstView source =
            host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
        SavedPage saved{.page = page, .bytes = std::vector<std::byte>(layout.page_stride)};
        std::memcpy(saved.bytes.data(), source.data(), layout.page_stride);
        return saved;
    };
    std::array<SavedPage, 4> ordered{save(victim_pages[0]), save(separator_pages[0]),
                                     save(victim_pages[1]), save(separator_pages[1])};
    std::array<HostKVPageReplicaRelease, 4> releases{};
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        releases[index] = {.pages = text_kv_pages.get(), .page = ordered[index].page};
    }
    if (!host_kv_extents->release_page_replicas(releases) || host_kv_arena->occupied_bytes() != 0) {
        throw std::logic_error("test shared-prefix fragmentation could not release its arena");
    }
    for (const SavedPage& saved : ordered) {
        const std::array page{saved.page};
        auto reservation = host_kv_extents->prepare_unpinned(*text_kv_pages, page);
        if (!reservation) {
            throw std::logic_error("test shared-prefix fragmentation could not rebuild one page");
        }
        std::memcpy(host_kv_extents->writable_view(*reservation).data(), saved.bytes.data(),
                    layout.page_stride);
        (void)host_kv_extents->publish(std::move(*reservation));
    }
    if (host_kv_arena->occupied_bytes() != host_kv_arena->capacity_bytes()) {
        throw std::logic_error("test shared-prefix fragmentation did not refill its arena");
    }
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        if (host_kv_extents->page_byte_offset(*text_kv_pages, ordered[index].page) !=
            index * layout.page_stride) {
            throw std::logic_error("test shared-prefix fragmentation placement changed");
        }
    }
    advance_resource_revision();
}

void ProgramImplCore::prepare_private_host_reclamation_for_test(
    const ContinuationHandle& continuation) {
    if (!valid_continuation(continuation) || pending_transaction_ || has_context_transaction() ||
        !host_kv_extents || !host_state_images) {
        throw std::logic_error("test private Host reclamation is not available");
    }
    SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    if (!sequence.endpoint_valid || !sequence.kv || sequence.state.read != sequence.state.write ||
        !state_store->valid(sequence.state.read)) {
        throw std::logic_error("test private Host reclamation requires a settled endpoint");
    }
    bool distinct_non_head =
        sequence.rewrite_state && *sequence.rewrite_state != sequence.state.read;
    distinct_non_head =
        distinct_non_head || std::any_of(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& anchor) {
                                             return anchor.state != sequence.state.read;
                                         });
    if (!distinct_non_head) {
        throw std::logic_error("test private Host reclamation requires a distinct non-head state");
    }
    if (state_store->residency(sequence.state.read) == StateReplicaResidency::DeviceOnly) {
        auto transfer =
            state_store->begin_device_to_host(sequence.state.read, device.transfer_stream);
        if (!transfer) {
            throw std::logic_error("test private Host reclamation has no Host State capacity");
        }
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        state_store->publish_transfer(std::move(*transfer), true);
    }
    if (state_store->residency(sequence.state.read) != StateReplicaResidency::Both) {
        throw std::logic_error("test private endpoint did not become mixed-resident");
    }

    const auto duplicate_pages = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                     KVAddressSpaceHandle address) {
        std::optional<LogicalKVPageHandle> aliased;
        std::optional<LogicalKVPageHandle> unaliased;
        for (std::uint32_t index = 0; index < addresses.mapped_pages(address); ++index) {
            const LogicalKVPageHandle page = addresses.logical_page(address, index);
            if (!pages.device_resident(page) || pages.host_resident(page) ||
                pages.source_pins(page) != 0) {
                continue;
            }
            if (pages.address_references(page) > 1 && !aliased) {
                aliased = page;
            } else if (pages.address_references(page) == 1) {
                unaliased = page;
            }
        }
        if (!aliased || !unaliased) {
            throw std::logic_error(
                "test private Host reclamation requires aliased and private KV pages");
        }
        const std::array selected{*aliased, *unaliased};
        auto reservation = host_kv_extents->prepare(pages, selected);
        if (!reservation) {
            throw std::logic_error("test private Host reclamation has no Host KV capacity");
        }
        const auto sources = host_kv_extents->device_sources(*reservation);
        pages.physical_pool().copy_to_host(sources, host_kv_extents->writable_view(*reservation),
                                           device.transfer_stream);
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        (void)host_kv_extents->publish(std::move(*reservation));
    };
    duplicate_pages(*text_kv_addresses, *text_kv_pages, sequence.kv->text);
    if (sequence.kv->backend) {
        duplicate_pages(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend);
    }
    advance_resource_revision();
}

qwen3_6::PrivateHostReclamationTestObservation
ProgramImplCore::private_host_reclamation_observation_for_test(
    const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) {
        throw std::logic_error("test private Host reclamation owner is invalid");
    }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    qwen3_6::PrivateHostReclamationTestObservation out;
    const qwen3_6::ContinuationSummary summary = continuation_summary(sequence);
    out.endpoint_frontier    = summary.endpoint ? summary.endpoint->ref.frontier : 0;
    const auto observe_state = [&](StateImageHandle state, std::uint32_t frontier) {
        if (state != sequence.state.read && out.unchanged_checkpoint_frontier == 0) {
            out.unchanged_checkpoint_frontier = frontier;
        }
        switch (state_store->residency(state)) {
        case StateReplicaResidency::DeviceOnly:
            ++out.device_only_state_checkpoints;
            break;
        case StateReplicaResidency::HostOnly:
            ++out.host_only_state_checkpoints;
            break;
        case StateReplicaResidency::Both:
            ++out.both_state_checkpoints;
            break;
        case StateReplicaResidency::None:
            break;
        }
    };
    observe_state(sequence.state.read, out.endpoint_frontier);
    if (sequence.rewrite_state && summary.rewrite) {
        observe_state(*sequence.rewrite_state, summary.rewrite->ref.frontier);
    }
    for (std::size_t index = 0;
         index < sequence.long_anchors.size() && index < summary.long_anchors.size(); ++index) {
        observe_state(sequence.long_anchors[index].state, summary.long_anchors[index].ref.frontier);
    }
    const auto observe_pages = [&](const KVAddressSpaceStore& addresses,
                                   const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                   std::uint32_t& host, std::uint32_t& aliased,
                                   std::uint32_t& pinned) {
        for (std::uint32_t index = 0; index < addresses.mapped_pages(address); ++index) {
            const LogicalKVPageHandle page = addresses.logical_page(address, index);
            if (!pages.host_resident(page)) { continue; }
            ++host;
            if (pages.address_references(page) > 1) { ++aliased; }
            if (pages.source_pins(page) != 0) { ++pinned; }
        }
    };
    observe_pages(*text_kv_addresses, *text_kv_pages, sequence.kv->text, out.main_host_pages,
                  out.main_aliased_host_pages, out.main_pinned_host_pages);
    if (sequence.kv->backend) {
        observe_pages(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                      out.backend_host_pages, out.backend_aliased_host_pages,
                      out.backend_pinned_host_pages);
    }
    return out;
}

qwen3_6::SharedPrefixPublication<Variant> ProgramImplCore::adopt_shared_prefix(
    const qwen3_6::ValidatedSharedPrefixImport<Variant>& imported, SharedPrefixHandle* replacement,
    runtime::CancellationFlagView cancellation, const std::function<void()>& commit_checkpoint,
    const ContinuationHandle* host_private, const SharedPrefixHandle* host_shared,
    std::shared_ptr<const void> physical_plan) {
    if (replacement == nullptr && host_private == nullptr && host_shared == nullptr) {
        return adopt_shared_prefix_impl(imported, commit_checkpoint);
    }
    const auto plan =
        std::static_pointer_cast<const DurableSharedImportPhysicalPlan>(physical_plan);
    const auto same_optional_handle = [](const auto& planned, const auto* supplied) {
        return planned.has_value() == (supplied != nullptr) &&
               (!planned || (planned->index == ContractAccess::index(*supplied) &&
                             planned->generation == ContractAccess::epoch(*supplied)));
    };
    if (!plan || plan->owner != this || plan->resource_revision != resource_revision_ ||
        !same_optional_handle(plan->replacement, replacement) ||
        !same_optional_handle(plan->host_private, host_private) ||
        !same_optional_handle(plan->host_shared, host_shared)) {
        throw runtime::DurableImportError(runtime::DurableImportErrorKind::StalePlan,
                                          "durable shared import physical plan is stale");
    }

    // A logical replacement need not reclaim physical storage before adoption. Keep the old
    // owner intact while staging the import whenever the current pools have enough room.
    // Only the publication cell is borrowed; failure restores it without a snapshot or D2H wait.
    if (replacement != nullptr && host_private == nullptr && host_shared == nullptr) {
        const std::uint32_t victim_index       = ContractAccess::index(*replacement);
        const SharedPrefixSlot original_slot   = shared_prefix_slots[victim_index];
        SharedPrefixState original             = std::move(shared_prefix_states[victim_index]);
        shared_prefix_slots[victim_index].role = SharedPrefixSlotRole::Free;
        if (++shared_prefix_slots[victim_index].generation == 0) {
            ++shared_prefix_slots[victim_index].generation;
        }
        try {
            const auto without_reclamation = inspect_durable_shared_prefix_import(
                imported.summary().checkpoint.ref.frontier, nullptr, nullptr, nullptr);
            if (without_reclamation.feasibility == runtime::DurableImportFeasibility::Feasible) {
                if (cancellation.requested()) {
                    throw RequestError(RequestErrorKind::Cancelled,
                                       "shared snapshot replacement was cancelled before commit");
                }
                auto publication = adopt_shared_prefix_impl(imported, commit_checkpoint);
                const std::uint32_t published_index = ContractAccess::index(publication.handle);
                if (published_index == victim_index) {
                    SharedPrefixState published = std::move(shared_prefix_states[victim_index]);
                    shared_prefix_states[victim_index] = std::move(original);
                    shared_prefix_slots[victim_index]  = original_slot;
                    (void)release_shared_prefix_state_strict(victim_index,
                                                             SharedPrefixSlotRole::Catalogued);
                    shared_prefix_states[victim_index]     = std::move(published);
                    shared_prefix_slots[victim_index].role = SharedPrefixSlotRole::Catalogued;
                } else {
                    shared_prefix_states[victim_index] = std::move(original);
                    shared_prefix_slots[victim_index]  = original_slot;
                    (void)release_shared_prefix_state_strict(victim_index,
                                                             SharedPrefixSlotRole::Catalogued);
                }
                ContractAccess::consume(*replacement);
                return publication;
            }
        } catch (...) {
            shared_prefix_states[victim_index] = std::move(original);
            shared_prefix_slots[victim_index]  = original_slot;
            throw;
        }
        shared_prefix_states[victim_index] = std::move(original);
        shared_prefix_slots[victim_index]  = original_slot;
    }

    const std::vector<StateImageHandle>& duplicate_host_states = plan->duplicate_host_states;
    std::vector<LogicalKVPageHandle> text_host_pages;
    std::vector<LogicalKVPageHandle> backend_host_pages;
    for (const HostKVPageReplicaRelease& release : plan->duplicate_host_releases) {
        if (release.pages == text_kv_pages.get()) {
            text_host_pages.push_back(release.page);
        } else if (release.pages == backend_kv_pages.get()) {
            backend_host_pages.push_back(release.page);
        } else {
            throw std::logic_error("durable shared import Host release plan changed stores");
        }
    }
    std::vector<CheckpointLifecycleFact> reclaimed_checkpoints;
    const auto lifecycle_role = [](runtime::CheckpointKind kind) {
        switch (kind) {
        case runtime::CheckpointKind::SessionEndpoint:
            return CheckpointLifecycleRole::SessionEndpoint;
        case runtime::CheckpointKind::TurnClosure:
            return CheckpointLifecycleRole::TurnClosure;
        case runtime::CheckpointKind::ResponseReplay:
            return CheckpointLifecycleRole::ResponseReplay;
        case runtime::CheckpointKind::SharedStablePrefix:
            return CheckpointLifecycleRole::SharedStablePrefix;
        case runtime::CheckpointKind::LongAnchor:
            return CheckpointLifecycleRole::LongAnchor;
        }
        return CheckpointLifecycleRole::SessionEndpoint;
    };
    const auto append_reclamation = [&](const qwen3_6::CheckpointSummary& checkpoint,
                                        std::uint32_t state_images, std::uint32_t main_kv_pages,
                                        std::uint32_t backend_kv_pages, bool logical_drop = false) {
        if (!logical_drop && state_images == 0 && main_kv_pages == 0 && backend_kv_pages == 0) {
            return;
        }
        const auto same_identity = [&](const CheckpointLifecycleFact& fact) {
            return fact.frontier == checkpoint.ref.frontier &&
                   fact.ordinal == checkpoint.ref.ordinal &&
                   fact.role == lifecycle_role(checkpoint.ref.kind) &&
                   fact.key_digests == checkpoint.shortlist_key.digests &&
                   fact.identity_tag == checkpoint.shortlist_key.identity_tag;
        };
        const auto found =
            std::find_if(reclaimed_checkpoints.begin(), reclaimed_checkpoints.end(), same_identity);
        if (found != reclaimed_checkpoints.end()) {
            found->state_images += state_images;
            found->main_kv_pages += main_kv_pages;
            found->backend_kv_pages += backend_kv_pages;
            return;
        }
        CheckpointLifecycleFact fact;
        fact.key_digests  = checkpoint.shortlist_key.digests;
        fact.frontier     = checkpoint.ref.frontier;
        fact.identity_tag = checkpoint.shortlist_key.identity_tag;
        fact.ordinal      = checkpoint.ref.ordinal;
        fact.role         = lifecycle_role(checkpoint.ref.kind);
        fact.scope        = checkpoint.scope == runtime::CheckpointScope::Shared
                                ? CheckpointLifecycleScope::Shared
                                : CheckpointLifecycleScope::Private;
        fact.operation    = CheckpointLifecycleOperation::Evicted;
        fact.source_tier  = CheckpointLifecycleTier::Host;
        fact.destination_tier =
            logical_drop ? CheckpointLifecycleTier::None : CheckpointLifecycleTier::Device;
        fact.status           = CheckpointLifecycleStatus::Committed;
        fact.state_images     = state_images;
        fact.main_kv_pages    = main_kv_pages;
        fact.backend_kv_pages = backend_kv_pages;
        reclaimed_checkpoints.push_back(std::move(fact));
    };
    if (host_private != nullptr) {
        const SequenceState& sequence = continuation_states[ContractAccess::index(*host_private)];
        const qwen3_6::ContinuationSummary summary = continuation_summary(*host_private);
        std::vector<StateImageHandle> attributed_states;
        const auto append_state = [&](const std::optional<qwen3_6::CheckpointSummary>& checkpoint,
                                      std::optional<StateImageHandle> state) {
            if (!checkpoint || !state ||
                std::find(duplicate_host_states.begin(), duplicate_host_states.end(), *state) ==
                    duplicate_host_states.end() ||
                std::find(attributed_states.begin(), attributed_states.end(), *state) !=
                    attributed_states.end()) {
                return;
            }
            attributed_states.push_back(*state);
            append_reclamation(*checkpoint, 1, 0, 0);
        };
        append_state(summary.endpoint,
                     sequence.endpoint_valid ? std::optional(sequence.state.read) : std::nullopt);
        append_state(summary.rewrite, sequence.rewrite_state);
        for (std::size_t index = 0;
             index < summary.long_anchors.size() && index < sequence.long_anchors.size(); ++index) {
            append_state(std::optional(summary.long_anchors[index]),
                         std::optional(sequence.long_anchors[index].state));
        }
        const qwen3_6::CheckpointSummary* kv_checkpoint = nullptr;
        const auto consider_kv = [&](const qwen3_6::CheckpointSummary* checkpoint) {
            if (checkpoint != nullptr && (kv_checkpoint == nullptr ||
                                          checkpoint->ref.frontier > kv_checkpoint->ref.frontier)) {
                kv_checkpoint = checkpoint;
            }
        };
        consider_kv(summary.endpoint ? &*summary.endpoint : nullptr);
        consider_kv(summary.rewrite ? &*summary.rewrite : nullptr);
        for (const auto& anchor : summary.long_anchors) { consider_kv(&anchor); }
        if (kv_checkpoint != nullptr) {
            append_reclamation(*kv_checkpoint, 0,
                               static_cast<std::uint32_t>(text_host_pages.size()),
                               static_cast<std::uint32_t>(backend_host_pages.size()));
        }
        bool private_state_attributed = false;
        for (const runtime::CheckpointRef dropped : plan->reclaimed_private_checkpoints) {
            const auto matches = [&](const qwen3_6::CheckpointSummary& checkpoint) {
                return checkpoint.ref == dropped;
            };
            if (summary.rewrite && matches(*summary.rewrite)) {
                const bool owns_physical_state =
                    !replacement ||
                    sequence.rewrite_state !=
                        std::optional(
                            shared_prefix_states[ContractAccess::index(*replacement)].state);
                const std::uint32_t state_images =
                    owns_physical_state && !private_state_attributed ? 1U : 0U;
                private_state_attributed = private_state_attributed || owns_physical_state;
                append_reclamation(*summary.rewrite, state_images, 0, 0, true);
                continue;
            }
            const auto anchor =
                std::find_if(summary.long_anchors.begin(), summary.long_anchors.end(), matches);
            if (anchor != summary.long_anchors.end()) {
                const std::size_t index =
                    static_cast<std::size_t>(anchor - summary.long_anchors.begin());
                const bool owns_physical_state =
                    replacement == nullptr ||
                    sequence.long_anchors[index].state !=
                        shared_prefix_states[ContractAccess::index(*replacement)].state;
                const std::uint32_t state_images =
                    owns_physical_state && !private_state_attributed ? 1U : 0U;
                private_state_attributed = private_state_attributed || owns_physical_state;
                append_reclamation(*anchor, state_images, 0, 0, true);
            }
        }
    } else if (host_shared != nullptr) {
        const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(*host_shared)];
        const qwen3_6::SharedPrefixSummary summary = shared_prefix_summary(shared);
        append_reclamation(summary.checkpoint,
                           std::find(duplicate_host_states.begin(), duplicate_host_states.end(),
                                     shared.state) != duplicate_host_states.end()
                               ? 1U
                               : 0U,
                           static_cast<std::uint32_t>(text_host_pages.size()),
                           static_cast<std::uint32_t>(backend_host_pages.size()));
    }
    enum class RollbackHostPageSource : std::uint8_t {
        DeviceReplica,
        VictimMainSnapshot,
        VictimBackendSnapshot,
    };

    struct RollbackHostPage {
        LogicalKVPageStore* pages     = nullptr;
        std::size_t byte_offset       = 0;
        RollbackHostPageSource source = RollbackHostPageSource::DeviceReplica;
        LogicalKVPageHandle stable_page;
        std::uint32_t snapshot_page = 0;
    };

    struct RollbackHostRun {
        LogicalKVPageStore* pages = nullptr;
        std::vector<RollbackHostPage> members;
    };

    std::vector<RollbackHostPage> rollback_host_pages;
    rollback_host_pages.reserve(plan->duplicate_host_releases.size());
    for (const HostKVPageReplicaRelease& release : plan->duplicate_host_releases) {
        rollback_host_pages.push_back(RollbackHostPage{
            .pages         = release.pages,
            .byte_offset   = host_kv_extents->page_byte_offset(*release.pages, release.page),
            .source        = RollbackHostPageSource::DeviceReplica,
            .stable_page   = release.page,
            .snapshot_page = 0,
        });
    }
    const auto release_host_duplicates = [&] {
        for (const StateImageHandle state : duplicate_host_states) {
            if (!state_store->drop_host_replica(state)) { std::terminate(); }
        }
        if (!plan->duplicate_host_releases.empty() &&
            !host_kv_extents->release_page_replicas(plan->duplicate_host_releases)) {
            std::terminate();
        }
    };
    if (cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "shared snapshot replacement was cancelled before commit");
    }
    for (const StateImageHandle state : duplicate_host_states) {
        if (state_store->residency(state) != StateReplicaResidency::Both ||
            state_store->source_pins(state) != 0) {
            throw runtime::DurableImportError(
                runtime::DurableImportErrorKind::StalePlan,
                "durable shared import Host State release plan is stale");
        }
    }
    if (!host_kv_extents->can_release_page_replicas(plan->duplicate_host_releases)) {
        throw runtime::DurableImportError(runtime::DurableImportErrorKind::StalePlan,
                                          "durable shared import Host KV release plan is stale");
    }
    if (commit_checkpoint) { commit_checkpoint(); }

    // Seal an exact in-memory rollback image while every victim replica and source pin is still
    // intact. The imported allocation/copy path remains genuinely fallible after teardown; any
    // exception reconstructs the victim and rewrites the caller-owned capability before it can
    // escape to ResourceManager. This is deliberately not an evict-first failure seam.
    std::shared_ptr<SharedImportBacking> rollback_backing;
    StateReplicaResidency rollback_state_residency = StateReplicaResidency::None;
    std::vector<std::pair<bool, bool>> rollback_text_residency;
    std::vector<std::pair<bool, bool>> rollback_backend_residency;
    std::vector<LogicalKVPageHandle> rollback_text_aliases;
    std::vector<LogicalKVPageHandle> rollback_backend_aliases;
    std::optional<std::uint32_t> rollback_shared_index;
    std::optional<StateImageHandle> rollback_state_alias;

    struct PrivateCheckpointRollback {
        SequenceState* sequence = nullptr;
        StateImageHandle original_state;
        StateReplicaResidency residency = StateReplicaResidency::None;
        std::optional<StateImageHandle> rewrite_state;
        RewriteCheckpoint rewrite_checkpoint;
        Tensor rewrite_checkpoint_hidden;
        std::vector<LongAnchorCheckpoint> long_anchors;
        std::vector<std::uint8_t> state_bytes;
        bool shared_state_alias = false;
        bool detached           = false;
    } private_rollback;

    if (replacement != nullptr) {
        const SharedPrefixState& original =
            shared_prefix_states[ContractAccess::index(*replacement)];
        rollback_shared_index    = ContractAccess::index(*replacement);
        rollback_state_residency = state_store->residency(original.state);
        if (state_store->checkpoint_references(original.state) > 1 &&
            (!plan->reclaimed_private_state || original.state != *plan->reclaimed_private_state)) {
            rollback_state_alias = original.state;
        }
        const auto capture_residency = [&](const KVAddressSpaceStore& addresses,
                                           LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                           std::vector<std::pair<bool, bool>>& output,
                                           std::vector<LogicalKVPageHandle>& aliases,
                                           RollbackHostPageSource host_source) {
            output.reserve(addresses.mapped_pages(address));
            aliases.reserve(addresses.mapped_pages(address));
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                output.emplace_back(pages.device_resident(logical), pages.host_resident(logical));
                const bool aliased = pages.address_references(logical) > 1;
                aliases.push_back(aliased ? logical : LogicalKVPageHandle{});
                if (pages.host_resident(logical) && !aliased) {
                    rollback_host_pages.push_back(RollbackHostPage{
                        .pages         = &pages,
                        .byte_offset   = host_kv_extents->page_byte_offset(pages, logical),
                        .source        = host_source,
                        .stable_page   = {},
                        .snapshot_page = page,
                    });
                }
            }
        };
        capture_residency(*text_kv_addresses, *text_kv_pages, original.kv->text,
                          rollback_text_residency, rollback_text_aliases,
                          RollbackHostPageSource::VictimMainSnapshot);
        if (original.kv->backend) {
            capture_residency(*backend_kv_addresses, *backend_kv_pages, *original.kv->backend,
                              rollback_backend_residency, rollback_backend_aliases,
                              RollbackHostPageSource::VictimBackendSnapshot);
        }
        auto rollback_snapshot = begin_shared_prefix_snapshot(*replacement, {}, nullptr, {});
        if (rollback_snapshot.await_transfer) {
            rollback_snapshot.await_transfer(rollback_snapshot.bytes);
        }
        retire_ready_snapshot_sources();
        rollback_backing                = std::make_shared<SharedImportBacking>();
        rollback_backing->owned_storage = std::move(rollback_snapshot.bytes);
        rollback_backing->identity      = original.identity;
        rollback_backing->boundary      = SharedSnapshotBoundary{
                 .frontier          = original.frontier,
                 .backend_frontier  = original.backend_frontier,
                 .rope_delta        = original.rope_delta,
                 .tail_hidden_valid = static_cast<std::uint8_t>(original.tail_hidden_valid),
                 .rebuild_work      = original.rebuild_work,
                 .text_pages        = kv_pages_for_frontier(original.frontier),
                 .backend_pages     = kv_pages_for_frontier(original.backend_frontier),
        };
        rollback_backing->text_offset    = state_images->host_layout().image_bytes;
        rollback_backing->backend_offset = checked_snapshot_sum(
            rollback_backing->text_offset,
            static_cast<std::size_t>(rollback_backing->boundary.text_pages) *
                plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry()).page_stride);
    }
    if (!plan->reclaimed_private_checkpoints.empty()) {
        if (host_private == nullptr || !plan->reclaimed_private_state) {
            throw std::invalid_argument("durable logical State reclamation lost its private owner");
        }
        SequenceState& sequence         = continuation_states[ContractAccess::index(*host_private)];
        private_rollback.sequence       = &sequence;
        private_rollback.original_state = *plan->reclaimed_private_state;
        private_rollback.shared_state_alias =
            replacement != nullptr &&
            shared_prefix_states[ContractAccess::index(*replacement)].state ==
                private_rollback.original_state;
        private_rollback.residency     = state_store->residency(private_rollback.original_state);
        private_rollback.rewrite_state = sequence.rewrite_state;
        private_rollback.rewrite_checkpoint        = sequence.rewrite_checkpoint;
        private_rollback.rewrite_checkpoint_hidden = sequence.rewrite_checkpoint_hidden;
        private_rollback.long_anchors              = sequence.long_anchors;
        const std::uint32_t victim_reference =
            replacement != nullptr &&
                    shared_prefix_states[ContractAccess::index(*replacement)].state ==
                        private_rollback.original_state
                ? 1U
                : 0U;
        if (private_rollback.residency != StateReplicaResidency::HostOnly ||
            state_store->source_pins(private_rollback.original_state) != 0 ||
            state_store->checkpoint_references(private_rollback.original_state) !=
                plan->reclaimed_private_checkpoints.size() + victim_reference ||
            !inspect_checkpoint_drop_option(sequence, plan->reclaimed_private_checkpoints)) {
            throw std::invalid_argument(
                "durable logical State reclamation plan changed before commit");
        }
        const qwen3_6::HostStateImageConstView source =
            state_store->host_view(private_rollback.original_state);
        private_rollback.state_bytes.resize(source.layout->image_bytes);
        std::memcpy(private_rollback.state_bytes.data(), source.data,
                    private_rollback.state_bytes.size());
    }
    // Remember each released suballocation's exact arena placement, rather than only aggregate
    // page counts. A victim can occupy several runs separated by retained aliases or unrelated
    // allocations. Once failed-import cleanup returns its allocations, reserving these original
    // byte ranges cannot be defeated by first-fit fragmentation or adjacent-run coalescing.
    std::vector<RollbackHostRun> rollback_host_runs;
    std::sort(rollback_host_pages.begin(), rollback_host_pages.end(),
              [](const RollbackHostPage& left, const RollbackHostPage& right) {
                  return left.byte_offset < right.byte_offset;
              });
    for (RollbackHostPage& page : rollback_host_pages) {
        const std::size_t stride = host_kv_extents->page_layout(*page.pages).page_stride;
        if (rollback_host_runs.empty() || rollback_host_runs.back().pages != page.pages ||
            rollback_host_runs.back().members.back().byte_offset + stride != page.byte_offset) {
            rollback_host_runs.push_back(
                RollbackHostRun{.pages = page.pages, .members = {std::move(page)}});
        } else {
            rollback_host_runs.back().members.push_back(std::move(page));
        }
    }
    if (cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "shared snapshot replacement was cancelled before commit");
    }
    if (commit_checkpoint) { commit_checkpoint(); }
    const auto detach_private_checkpoints = [&]() noexcept {
        if (private_rollback.sequence == nullptr) { return; }
        try {
            SequenceState& sequence = *private_rollback.sequence;
            for (const runtime::CheckpointRef checkpoint : plan->reclaimed_private_checkpoints) {
                if ((checkpoint.kind == runtime::CheckpointKind::TurnClosure ||
                     checkpoint.kind == runtime::CheckpointKind::ResponseReplay) &&
                    sequence.rewrite_state &&
                    *sequence.rewrite_state == private_rollback.original_state &&
                    sequence.rewrite_checkpoint.valid &&
                    checkpoint_kind(sequence.rewrite_checkpoint.kind) == checkpoint.kind &&
                    sequence.rewrite_checkpoint.frontier == checkpoint.frontier) {
                    state_store->release_checkpoint_reference(*sequence.rewrite_state);
                    sequence.rewrite_state.reset();
                    sequence.rewrite_checkpoint        = {};
                    sequence.rewrite_checkpoint_hidden = {};
                    continue;
                }
                if (checkpoint.kind == runtime::CheckpointKind::LongAnchor) {
                    const auto anchor = std::find_if(
                        sequence.long_anchors.begin(), sequence.long_anchors.end(),
                        [&](const LongAnchorCheckpoint& candidate) {
                            return candidate.state == private_rollback.original_state &&
                                   candidate.frontier == checkpoint.frontier &&
                                   candidate.ordinal == checkpoint.ordinal;
                        });
                    if (anchor != sequence.long_anchors.end()) {
                        state_store->release_checkpoint_reference(anchor->state);
                        sequence.long_anchors.erase(anchor);
                        continue;
                    }
                }
                std::terminate();
            }
            private_rollback.detached = true;
            if ((replacement == nullptr ||
                 shared_prefix_states[ContractAccess::index(*replacement)].state !=
                     private_rollback.original_state) &&
                !state_store->release(private_rollback.original_state)) {
                std::terminate();
            }
        } catch (...) { std::terminate(); }
    };
    detach_private_checkpoints();
    release_host_duplicates();
    if (replacement != nullptr) {
        const std::uint32_t victim_index = ContractAccess::index(*replacement);
        (void)release_shared_prefix_state_strict(victim_index, SharedPrefixSlotRole::Catalogued);
        ContractAccess::consume(*replacement);
    }
    try {
        auto publication = adopt_shared_prefix_impl(imported, commit_checkpoint);
        if (host_private != nullptr) {
            publication.reclaimed_private_summary = continuation_summary(*host_private);
        }
        if (host_shared != nullptr) {
            publication.reclaimed_shared_summary =
                shared_prefix_summary(shared_prefix_states[ContractAccess::index(*host_shared)]);
        }
        publication.reclaimed_checkpoints = std::move(reclaimed_checkpoints);
        return publication;
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        try {
            std::optional<StateImageHandle> restored_state;
            std::optional<StateImageTransfer> restored_state_transfer;
            std::optional<KVAddressSpaceHandle> restored_text;
            std::optional<KVAddressSpaceHandle> restored_backend;
            if (rollback_backing) {
                if (!rollback_shared_index || !replacement) {
                    throw std::logic_error("durable rollback lost its publication slot");
                }
                if (shared_prefix_slots[*rollback_shared_index].role !=
                    SharedPrefixSlotRole::Free) {
                    throw std::logic_error("durable rollback slot is unavailable");
                }
                const qwen3_6::HostStateImageConstView state_view{
                    reinterpret_cast<const std::byte*>(rollback_backing->data() +
                                                       rollback_backing->state_offset),
                    &state_images->host_layout()};
                if (rollback_state_alias && state_store->valid(*rollback_state_alias)) {
                    restored_state = *rollback_state_alias;
                } else if (rollback_state_residency == StateReplicaResidency::DeviceOnly) {
                    restored_state =
                        state_store->adopt_device_image(state_view, device.transfer_stream);
                } else {
                    restored_state = state_store->adopt_host_image(state_view);
                    if (restored_state && rollback_state_residency == StateReplicaResidency::Both) {
                        auto transfer = state_store->begin_host_to_device(*restored_state,
                                                                          device.transfer_stream);
                        if (transfer) { restored_state_transfer.emplace(std::move(*transfer)); }
                    }
                }
                if (!restored_state || (rollback_state_residency == StateReplicaResidency::Both &&
                                        !rollback_state_alias && !restored_state_transfer)) {
                    throw std::logic_error("durable rollback State capacity was not reserved");
                }
                const auto restore_kv = [&](KVAddressSpaceStore& addresses,
                                            LogicalKVPageStore& pages,
                                            const std::vector<LogicalKVPageHandle>& aliases,
                                            const std::vector<std::pair<bool, bool>>& residency,
                                            std::uint32_t frontier, const std::uint8_t* payload,
                                            const HostKVPageLayout& layout) {
                    std::vector<std::uint8_t> device_residency;
                    device_residency.reserve(residency.size());
                    for (const auto [device_resident, host_resident] : residency) {
                        (void)host_resident;
                        device_residency.push_back(device_resident ? 1U : 0U);
                    }
                    KVAddressSpaceHandle address =
                        addresses.restore_inactive_checkpoint(aliases, frontier, device_residency);
                    try {
                        for (std::uint32_t page = 0; page < residency.size(); ++page) {
                            const LogicalKVPageHandle logical =
                                addresses.logical_page(address, page);
                            if (!aliases[page].valid()) {
                                if (residency[page].first) {
                                    std::array<DeviceKVPageHandle, 1> destination{
                                        pages.physical(logical)};
                                    pages.physical_pool().copy_from_host(
                                        reinterpret_cast<const std::byte*>(
                                            payload +
                                            static_cast<std::size_t>(page) * layout.page_stride),
                                        layout, destination, device.transfer_stream);
                                }
                            }
                        }
                        return address;
                    } catch (...) {
                        (void)addresses.release(address);
                        (void)host_kv_extents->release_unreferenced();
                        throw;
                    }
                };
                const HostKVPageLayout text_layout =
                    plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry());
                restored_text = restore_kv(
                    *text_kv_addresses, *text_kv_pages, rollback_text_aliases,
                    rollback_text_residency, rollback_backing->boundary.frontier,
                    rollback_backing->data() + rollback_backing->text_offset, text_layout);
                if (!rollback_backend_residency.empty()) {
                    const HostKVPageLayout backend_layout =
                        plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry());
                    restored_backend = restore_kv(
                        *backend_kv_addresses, *backend_kv_pages, rollback_backend_aliases,
                        rollback_backend_residency, rollback_backing->boundary.backend_frontier,
                        rollback_backing->data() + rollback_backing->backend_offset,
                        backend_layout);
                }
            }

            std::vector<StateImageTransfer> restored_duplicate_states;
            restored_duplicate_states.reserve(duplicate_host_states.size());
            for (const StateImageHandle state : duplicate_host_states) {
                auto transfer = state_store->begin_device_to_host(state, device.transfer_stream);
                if (!transfer) {
                    throw std::logic_error("durable rollback Host State capacity was not reserved");
                }
                restored_duplicate_states.push_back(std::move(*transfer));
            }

            std::vector<HostKVExtentReservation> restored_host_runs;
            restored_host_runs.reserve(rollback_host_runs.size());
            for (const RollbackHostRun& run : rollback_host_runs) {
                std::vector<LogicalKVPageHandle> pages;
                pages.reserve(run.members.size());
                for (const RollbackHostPage& member : run.members) {
                    switch (member.source) {
                    case RollbackHostPageSource::DeviceReplica:
                        pages.push_back(member.stable_page);
                        break;
                    case RollbackHostPageSource::VictimMainSnapshot:
                        if (!restored_text) {
                            throw std::logic_error("durable rollback lost its Main KV address");
                        }
                        pages.push_back(
                            text_kv_addresses->logical_page(*restored_text, member.snapshot_page));
                        break;
                    case RollbackHostPageSource::VictimBackendSnapshot:
                        if (!restored_backend) {
                            throw std::logic_error("durable rollback lost its backend KV address");
                        }
                        pages.push_back(backend_kv_addresses->logical_page(*restored_backend,
                                                                           member.snapshot_page));
                        break;
                    }
                }
                std::optional<HostKVExtentReservation> reserved =
                    host_kv_extents->prepare_unpinned_at(*run.pages, pages,
                                                         run.members.front().byte_offset);
                if (!reserved) {
                    throw std::logic_error("durable rollback Host KV run was not preserved");
                }
                restored_host_runs.push_back(std::move(*reserved));
                HostKVExtentReservation& host  = restored_host_runs.back();
                HostKVAllocationView view      = host_kv_extents->writable_view(host);
                const HostKVPageLayout& layout = host_kv_extents->page_layout(*run.pages);
                for (std::size_t index = 0; index < run.members.size(); ++index) {
                    const RollbackHostPage& member = run.members[index];
                    if (member.source == RollbackHostPageSource::DeviceReplica) {
                        std::array<DeviceKVPageHandle, 1> source{run.pages->physical(pages[index])};
                        run.pages->physical_pool().copy_to_host(
                            source, view.subview(static_cast<std::uint32_t>(index), 1),
                            device.transfer_stream);
                        continue;
                    }
                    if (!rollback_backing) {
                        throw std::logic_error("durable rollback lost its sealed KV bytes");
                    }
                    const std::size_t source_offset =
                        member.source == RollbackHostPageSource::VictimMainSnapshot
                            ? rollback_backing->text_offset
                            : rollback_backing->backend_offset;
                    std::memcpy(view.data() + index * layout.page_stride,
                                rollback_backing->data() + source_offset +
                                    static_cast<std::size_t>(member.snapshot_page) *
                                        layout.page_stride,
                                layout.page_stride);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            if (restored_state_transfer) {
                state_store->publish_transfer(std::move(*restored_state_transfer), true);
            }
            for (StateImageTransfer& transfer : restored_duplicate_states) {
                state_store->publish_transfer(std::move(transfer), true);
            }
            for (HostKVExtentReservation& host : restored_host_runs) {
                (void)host_kv_extents->publish(std::move(host));
            }

            if (rollback_backing) {
                SharedPrefixState& restored = shared_prefix_states[*rollback_shared_index];
                shared_prefix_slots[*rollback_shared_index].role =
                    SharedPrefixSlotRole::ReservedCapture;
                state_store->retain_checkpoint_reference(*restored_state);
                restored.state = *restored_state;
                restored.kv = SequenceKVBundle{.text = *restored_text, .backend = restored_backend};
                restored.identity          = rollback_backing->identity;
                restored.frontier          = rollback_backing->boundary.frontier;
                restored.backend_frontier  = rollback_backing->boundary.backend_frontier;
                restored.rope_delta        = rollback_backing->boundary.rope_delta;
                restored.tail_hidden_valid = rollback_backing->boundary.tail_hidden_valid != 0;
                restored.rebuild_work      = rollback_backing->boundary.rebuild_work;
                restored.active_references = 0;
                shared_prefix_slots[*rollback_shared_index].role = SharedPrefixSlotRole::Catalogued;
                SharedPrefixHandle restored_handle = ContractAccess::make_shared_prefix(
                    this, *rollback_shared_index,
                    shared_prefix_slots[*rollback_shared_index].generation);
                std::destroy_at(replacement);
                std::construct_at(replacement, std::move(restored_handle));
                advance_resource_revision();
            }

            if (private_rollback.detached) {
                std::optional<StateImageHandle> private_state;
                if (private_rollback.shared_state_alias && rollback_backing && restored_state) {
                    private_state = *restored_state;
                } else {
                    private_state = state_store->adopt_host_image(qwen3_6::HostStateImageConstView{
                        reinterpret_cast<const std::byte*>(private_rollback.state_bytes.data()),
                        &state_images->host_layout()});
                }
                if (!private_state) {
                    throw std::logic_error(
                        "durable rollback logical State capacity was not preserved");
                }
                SequenceState& sequence = *private_rollback.sequence;
                sequence.rewrite_state  = private_rollback.rewrite_state;
                if (sequence.rewrite_state &&
                    *sequence.rewrite_state == private_rollback.original_state) {
                    sequence.rewrite_state = *private_state;
                }
                sequence.rewrite_checkpoint        = private_rollback.rewrite_checkpoint;
                sequence.rewrite_checkpoint_hidden = private_rollback.rewrite_checkpoint_hidden;
                sequence.long_anchors              = private_rollback.long_anchors;
                for (LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                    if (anchor.state == private_rollback.original_state) {
                        anchor.state = *private_state;
                    }
                }
                for (std::size_t index = 0; index < plan->reclaimed_private_checkpoints.size();
                     ++index) {
                    state_store->retain_checkpoint_reference(*private_state);
                }
            }
        } catch (...) {
            // The exact victim just released these resources and imported cleanup returned every
            // partial allocation. Failure to restore that sealed image is an internal invariant
            // breach, not a recoverable admission rejection.
            std::terminate();
        }
        std::rethrow_exception(failure);
    }
}

qwen3_6::SharedPrefixPublication<Variant> ProgramImplCore::adopt_shared_prefix_impl(
    const qwen3_6::ValidatedSharedPrefixImport<Variant>& imported,
    const std::function<void()>& commit_checkpoint) {
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
            reinterpret_cast<const std::byte*>(backing->data() + backing->state_offset),
            &state_layout});
        if (!state) {
            throw std::invalid_argument("shared snapshot does not fit the Host State capacity");
        }
        runtime::testing::shared_snapshot_import_checkpoint(
            runtime::testing::SharedSnapshotImportStage::StateAllocated);

        if (!host_kv_extents) {
            throw std::invalid_argument("shared snapshot adoption requires Host KV capacity");
        }
        const auto build_address = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                       const HostKVPageLayout& layout, std::uint32_t frontier,
                                       const std::uint8_t* payload) {
            const std::uint32_t count = kv_pages_for_frontier(frontier);
            std::vector<LogicalKVPageHandle> retained(count);
            std::vector<std::uint8_t> device_residency(count, 0U);
            std::optional<KVAddressSpaceHandle> address;
            try {
                address =
                    addresses.restore_inactive_checkpoint(retained, frontier, device_residency);
                std::vector<LogicalKVPageHandle> logical;
                logical.reserve(count);
                for (std::uint32_t page = 0; page < count; ++page) {
                    logical.push_back(addresses.logical_page(*address, page));
                }
                std::optional<HostKVExtentReservation> host =
                    host_kv_extents->prepare_unpinned(pages, logical);
                if (!host) {
                    throw std::invalid_argument(
                        "shared snapshot does not fit the Host KV capacity");
                }
                HostKVAllocationView destination = host_kv_extents->writable_view(*host);
                std::memcpy(destination.data(), payload,
                            static_cast<std::size_t>(count) * layout.page_stride);
                (void)host_kv_extents->publish(std::move(*host));
                return *address;
            } catch (...) {
                if (address) { (void)addresses.release(*address); }
                (void)host_kv_extents->release_unreferenced();
                throw;
            }
        };
        text_address =
            build_address(*text_kv_addresses, *text_kv_pages, text_layout,
                          backing->boundary.frontier, backing->data() + backing->text_offset);
        runtime::testing::shared_snapshot_import_checkpoint(
            runtime::testing::SharedSnapshotImportStage::MainKvAllocated);
        if (backing->boundary.backend_frontier != 0) {
            backend_address = build_address(*backend_kv_addresses, *backend_kv_pages,
                                            *backend_layout, backing->boundary.backend_frontier,
                                            backing->data() + backing->backend_offset);
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
        if (commit_checkpoint) { commit_checkpoint(); }
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
            if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
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
