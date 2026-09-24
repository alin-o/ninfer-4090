#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {

// Output budgets count the last sampled token; KV does not necessarily consume it.
// Arithmetic stays wide until the logical context ceiling has been applied.
struct ReservationHorizon {
    std::uint32_t prompt_tokens;
    std::uint32_t output_tokens;
    std::uint32_t context_tokens;
    std::uint32_t quantum;
    std::uint32_t page_tokens;

    void validate() const {
        if (prompt_tokens == 0 || prompt_tokens > context_tokens || quantum == 0 ||
            quantum > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            page_tokens == 0) {
            throw std::invalid_argument("invalid generation reservation horizon");
        }
    }

    [[nodiscard]] std::uint32_t completion_tokens() const noexcept {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(
            context_tokens, static_cast<std::uint64_t>(prompt_tokens) +
                                (output_tokens == 0 ? 0ULL : output_tokens - 1ULL)));
    }

    [[nodiscard]] std::uint32_t pages(std::uint64_t tokens) const noexcept {
        tokens = std::min<std::uint64_t>(tokens, context_tokens);
        return tokens == 0 ? 0U : static_cast<std::uint32_t>(1 + (tokens - 1) / page_tokens);
    }

    [[nodiscard]] std::uint32_t initial_pages() const noexcept {
        return pages(static_cast<std::uint64_t>(prompt_tokens) +
                     (output_tokens == 0 ? 0ULL : std::min(output_tokens, quantum) - 1ULL));
    }
};

} // namespace ninfer::runtime
