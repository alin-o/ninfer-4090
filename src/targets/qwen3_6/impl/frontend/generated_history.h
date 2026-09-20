#pragma once

#include "targets/qwen3_6/impl/frontend/processor.h"

#include <deque>
#include <mutex>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Frontend semantics, independent of retained KV/State and of the movable log text.
struct GeneratedHistorySource {
    std::vector<TokenId> canonical_source;
    std::vector<TokenId> token_ids;
    std::string text;
    std::size_t source_bytes = 0;
    std::optional<std::string> explicit_session;

    [[nodiscard]] std::size_t storage_bytes() const noexcept;
};

class GeneratedHistory {
public:
    explicit GeneratedHistory(std::size_t capacity_bytes  = 64ULL * 1024ULL * 1024ULL,
                              std::size_t maximum_entries = 128)
        : capacity_bytes_(capacity_bytes), maximum_entries_(maximum_entries) {}

    // A preserved prefix can be shorter than its canonical encoding. Keep early tokenization
    // bounded while allowing enough canonical tokens to resolve any matching retained prefix.
    [[nodiscard]] std::size_t tokenization_limit(const RenderedChat& rendered,
                                                 const std::optional<std::string>& explicit_session,
                                                 std::uint32_t max_context);

    // Text-only. Resolve before computing positions, context opportunities or prefix identity.
    [[nodiscard]] std::shared_ptr<const GeneratedHistorySource>
    prepare(const Tokenizer& tokenizer, const RenderedChat& rendered, EncodedChat& encoded,
            std::span<const ChatRole> roles, const std::optional<std::string>& explicit_session);

    // Optional retention must never turn successful generation into an Engine failure.
    void remember(const Tokenizer& tokenizer, std::shared_ptr<const GeneratedHistorySource> source,
                  std::span<const TokenId> generated) noexcept;

private:
    const std::size_t capacity_bytes_;
    const std::size_t maximum_entries_;
    std::mutex mutex_;
    std::deque<std::shared_ptr<const GeneratedHistorySource>> entries_;
    std::size_t retained_bytes_ = 0;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
