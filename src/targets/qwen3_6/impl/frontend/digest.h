#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {

using Sha256Digest = std::array<std::uint8_t, 32>;

// Incremental form used by bounded file publishers. It retains one SHA-256 block regardless of
// payload size and produces the same digest as the one-shot helpers below.
class Sha256Hasher {
public:
    Sha256Hasher() noexcept;

    void update(std::span<const std::uint8_t> input);
    void update(std::string_view input);
    [[nodiscard]] Sha256Digest finalize();

private:
    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> tail_{};
    std::uint64_t total_bytes_ = 0;
    std::size_t tail_size_     = 0;
    bool finalized_            = false;
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);

} // namespace ninfer::targets::qwen3_6::frontend_internal
