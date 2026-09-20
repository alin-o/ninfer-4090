#include "targets/qwen3_6/impl/frontend/generated_history.h"

#include <algorithm>
#include <limits>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

// Use raw bytes: presentation decoding may repair UTF-8 or hide control tokens.
std::vector<std::size_t> token_offsets(const Tokenizer& tokenizer, std::span<const TokenId> tokens,
                                       std::string_view text) {
    std::vector<std::size_t> offsets;
    offsets.reserve(tokens.size() + 1U);
    offsets.push_back(0);
    for (const TokenId id : tokens) {
        const auto bytes = tokenizer.decode_token_bytes(id);
        if (bytes.empty() || !text.substr(offsets.back()).starts_with(bytes)) { return {}; }
        offsets.push_back(offsets.back() + bytes.size());
    }
    if (offsets.back() != text.size()) { return {}; }
    return offsets;
}

bool inside_assistant(const GeneratedHistorySource& entry, const RenderedChat& rendered,
                      std::span<const ChatRole> roles) {
    if (rendered.message_boundaries.size() != roles.size() + 1U) { return false; }
    for (std::size_t index = 0; index != roles.size(); ++index) {
        const auto begin = rendered.message_boundaries[index];
        const auto end   = rendered.message_boundaries[index + 1U];
        if (roles[index] == ChatRole::Assistant && begin && end && *begin <= entry.source_bytes &&
            entry.source_bytes < *end && entry.text.size() <= *end) {
            return true;
        }
    }
    return false;
}

bool same_special_tokens(const Tokenizer& tokenizer, std::span<const TokenId> before,
                         std::span<const std::size_t> before_offsets,
                         std::span<const TokenId> after,
                         std::span<const std::size_t> after_offsets) {
    std::size_t next = 0;
    for (std::size_t index = 0; index != before.size(); ++index) {
        if (!tokenizer.is_special_token(before[index])) { continue; }
        while (next != after.size() && !tokenizer.is_special_token(after[next])) { ++next; }
        if (next == after.size() || before[index] != after[next] ||
            before_offsets[index] != after_offsets[next]) {
            return false;
        }
        ++next;
    }
    return std::none_of(after.begin() + next, after.end(),
                        [&](TokenId id) { return tokenizer.is_special_token(id); });
}

void remap_frontiers(EncodedChat& encoded, std::span<const std::size_t> canonical_offsets,
                     std::span<const std::size_t> preserved_offsets, std::size_t canonical_end) {
    const auto exact = [&](std::uint32_t frontier) -> std::optional<std::uint32_t> {
        if (frontier >= canonical_end) {
            return static_cast<std::uint32_t>(frontier - canonical_end + preserved_offsets.size() -
                                              1U);
        }
        const auto byte = canonical_offsets[frontier];
        const auto found =
            std::lower_bound(preserved_offsets.begin(), preserved_offsets.end(), byte);
        if (found == preserved_offsets.end() || *found != byte) { return std::nullopt; }
        return static_cast<std::uint32_t>(found - preserved_offsets.begin());
    };
    for (auto& frontier : encoded.message_boundaries) {
        if (frontier) { frontier = exact(*frontier); }
    }
    for (auto& frontier : encoded.cache_boundaries) {
        if (frontier) { frontier = exact(*frontier); }
    }
    for (auto& boundary : encoded.structural_boundaries) {
        if (boundary.frontier) { boundary.frontier = exact(*boundary.frontier); }
    }
    if (encoded.rewrite_checkpoint) {
        if (const auto frontier = exact(encoded.rewrite_checkpoint->frontier)) {
            encoded.rewrite_checkpoint->frontier = *frontier;
        } else {
            encoded.rewrite_checkpoint.reset();
        }
    }
    std::vector<std::uint32_t> execution;
    for (const auto frontier : encoded.rewrite_execution_frontiers) {
        if (const auto mapped = exact(frontier)) { execution.push_back(*mapped); }
    }
    encoded.rewrite_execution_frontiers = std::move(execution);
    if (encoded.first_volatile_token) {
        const auto original = *encoded.first_volatile_token;
        if (const auto mapped = exact(original)) {
            encoded.first_volatile_token = *mapped;
        } else {
            // A token containing any volatile byte is already volatile, even when a preserved
            // token crosses the former boundary. Never promote that token into a stable prefix.
            const auto found = std::upper_bound(preserved_offsets.begin(), preserved_offsets.end(),
                                                canonical_offsets[original]);
            encoded.first_volatile_token =
                static_cast<std::uint32_t>(found - preserved_offsets.begin() - 1);
        }
    }
}

} // namespace

std::size_t GeneratedHistorySource::storage_bytes() const noexcept {
    return sizeof(*this) + (canonical_source.capacity() + token_ids.capacity()) * sizeof(TokenId) +
           text.capacity() + (explicit_session ? explicit_session->capacity() : 0U);
}

std::size_t GeneratedHistory::tokenization_limit(const RenderedChat& rendered,
                                                 const std::optional<std::string>& explicit_session,
                                                 std::uint32_t max_context) {
    std::size_t allowance = 0;
    std::lock_guard lock(mutex_);
    for (const auto& entry : entries_) {
        if (entry->explicit_session == explicit_session && rendered.text.starts_with(entry->text)) {
            // Every valid token decodes to at least one byte. This bounds the greatest possible
            // reduction without re-tokenizing the entry on the Engine's completion thread.
            allowance = std::max(allowance, entry->text.size() - entry->token_ids.size());
        }
    }
    return static_cast<std::size_t>(max_context) + allowance + 1U;
}

std::shared_ptr<const GeneratedHistorySource>
GeneratedHistory::prepare(const Tokenizer& tokenizer, const RenderedChat& rendered,
                          EncodedChat& encoded, std::span<const ChatRole> roles,
                          const std::optional<std::string>& explicit_session) {
    if (capacity_bytes_ == 0 || maximum_entries_ == 0 || !encoded.media_token_runs.empty()) {
        return {};
    }
    const auto canonical_offsets = token_offsets(tokenizer, encoded.input_ids, rendered.text);
    if (canonical_offsets.empty()) { return {}; }
    auto source              = std::make_shared<GeneratedHistorySource>();
    source->canonical_source = encoded.input_ids;
    source->source_bytes     = rendered.text.size();
    source->explicit_session = explicit_session;
    source->text             = rendered.text;

    // A preparation uses one immutable publication snapshot. Completion never holds this mutex
    // while preparing tokens, and a concurrently evicted entry remains valid for this reader.
    std::vector<std::shared_ptr<const GeneratedHistorySource>> candidates;
    {
        std::lock_guard lock(mutex_);
        for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
            const auto& entry = **it;
            if (entry.explicit_session == explicit_session &&
                rendered.text.starts_with(entry.text) && inside_assistant(entry, rendered, roles) &&
                entry.canonical_source.size() <= encoded.input_ids.size() &&
                std::equal(entry.canonical_source.begin(), entry.canonical_source.end(),
                           encoded.input_ids.begin())) {
                candidates.push_back(*it);
            }
        }
    }
    // Deepest verified history wins; equal text chooses the most recently completed trajectory.
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a->text.size() > b->text.size(); });
    for (const auto& entry : candidates) {
        const auto end = std::lower_bound(canonical_offsets.begin(), canonical_offsets.end(),
                                          entry->text.size());
        if (end == canonical_offsets.end() || *end != entry->text.size()) { continue; }
        const auto canonical_end = static_cast<std::size_t>(end - canonical_offsets.begin());
        const auto offsets       = token_offsets(tokenizer, entry->token_ids, entry->text);
        if (offsets.empty() ||
            !same_special_tokens(tokenizer, std::span(encoded.input_ids).first(canonical_end),
                                 canonical_offsets, entry->token_ids, offsets)) {
            continue;
        }
        const auto count = entry->token_ids.size() + encoded.input_ids.size() - canonical_end;
        if (count > std::numeric_limits<std::uint32_t>::max()) { continue; }
        std::vector<TokenId> resolved;
        resolved.reserve(count);
        resolved.insert(resolved.end(), entry->token_ids.begin(), entry->token_ids.end());
        resolved.insert(resolved.end(), encoded.input_ids.begin() + canonical_end,
                        encoded.input_ids.end());
        remap_frontiers(encoded, canonical_offsets, offsets, canonical_end);
        encoded.input_ids = std::move(resolved);
        break;
    }
    source->token_ids = encoded.input_ids;
    if (source->storage_bytes() > capacity_bytes_) { return {}; }
    return source;
}

void GeneratedHistory::remember(const Tokenizer& tokenizer,
                                std::shared_ptr<const GeneratedHistorySource> source,
                                std::span<const TokenId> generated) noexcept {
    if (!source || generated.empty() || maximum_entries_ == 0) { return; }
    try {
        auto entry = std::make_shared<GeneratedHistorySource>(*source);
        entry->token_ids.insert(entry->token_ids.end(), generated.begin(), generated.end());
        for (const TokenId id : generated) {
            const auto bytes = tokenizer.decode_token_bytes(id);
            if (bytes.empty()) { return; }
            entry->text += bytes;
        }
        const auto bytes = entry->storage_bytes();
        if (bytes > capacity_bytes_) { return; }
        std::lock_guard lock(mutex_);
        // Allocate the container slot before eviction: a failed publication leaves history intact.
        entries_.push_back(std::move(entry));
        retained_bytes_ += bytes;
        while (retained_bytes_ > capacity_bytes_ || entries_.size() > maximum_entries_) {
            retained_bytes_ -= entries_.front()->storage_bytes();
            entries_.pop_front();
        }
    } catch (...) {
        // Retaining CPU history is optional. Exact canonical preparation remains available.
    }
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
