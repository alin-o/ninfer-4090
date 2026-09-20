#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

struct evp_md_ctx_st;

namespace ninfer::targets::qwen3_6::frontend_internal {

using Sha256Digest = std::array<std::uint8_t, 32>;

// Incremental form used by bounded file publishers. OpenSSL selects the CPU-accelerated SHA-256
// implementation; state stays bounded regardless of payload size. All helpers hash every byte.
class Sha256Hasher {
public:
    Sha256Hasher();

    void update(std::span<const std::uint8_t> input);
    void update(std::string_view input);
    [[nodiscard]] Sha256Digest finalize();

private:
    struct ContextDeleter {
        void operator()(evp_md_ctx_st* context) const noexcept;
    };

    std::unique_ptr<evp_md_ctx_st, ContextDeleter> context_;
    std::uint64_t total_bytes_ = 0;
    bool finalized_            = false;
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);

} // namespace ninfer::targets::qwen3_6::frontend_internal
