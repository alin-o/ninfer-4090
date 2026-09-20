#include "targets/qwen3_6/impl/frontend/digest.h"

#include <openssl/evp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::frontend_internal {

void Sha256Hasher::ContextDeleter::operator()(evp_md_ctx_st* context) const noexcept {
    EVP_MD_CTX_free(context);
}

Sha256Hasher::Sha256Hasher() : context_(EVP_MD_CTX_new()) {
    if (!context_) { throw std::bad_alloc(); }
    if (EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("SHA-256 initialization failed");
    }
}

void Sha256Hasher::update(std::span<const std::uint8_t> input) {
    if (finalized_) { throw std::logic_error("SHA-256 hasher is already finalized"); }
    constexpr std::uint64_t kMaximumBytes = std::numeric_limits<std::uint64_t>::max() / 8ULL;
    if (total_bytes_ > kMaximumBytes || input.size() > kMaximumBytes - total_bytes_) {
        throw std::invalid_argument("payload is too large to fingerprint");
    }
    if (!input.empty() && EVP_DigestUpdate(context_.get(), input.data(), input.size()) != 1) {
        throw std::runtime_error("SHA-256 update failed");
    }
    total_bytes_ += input.size();
}

void Sha256Hasher::update(std::string_view input) {
    update(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(input.data()),
                                         input.size()));
}

Sha256Digest Sha256Hasher::finalize() {
    if (finalized_) { throw std::logic_error("SHA-256 hasher is already finalized"); }
    finalized_ = true;
    Sha256Digest digest{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context_.get(), digest.data(), &size) != 1 || size != digest.size()) {
        throw std::runtime_error("SHA-256 finalization failed");
    }
    return digest;
}

Sha256Digest sha256(std::span<const std::uint8_t> input) { return sha256(input, {}); }

Sha256Digest sha256(std::span<const std::uint8_t> input, const std::function<void()>& checkpoint) {
    if (input.size() > std::numeric_limits<std::uint64_t>::max() / 8ULL) {
        throw std::invalid_argument("payload is too large to fingerprint");
    }
    Sha256Hasher hasher;
    if (checkpoint) {
        // Bound import/media cancellation latency while letting the crypto implementation
        // process large contiguous blocks. The same full payload is hashed on every route.
        constexpr std::size_t kChunk = 1U << 20;
        while (!input.empty()) {
            checkpoint();
            const std::size_t count = std::min(input.size(), kChunk);
            hasher.update(input.first(count));
            input = input.subspan(count);
        }
        checkpoint();
    } else {
        hasher.update(input);
    }
    return hasher.finalize();
}

Sha256Digest sha256(std::string_view input) {
    return sha256(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(input.data()),
                                                input.size()));
}

std::string sha256_hex(const Sha256Digest& digest) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '\0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        result[2 * i]     = hex[digest[i] >> 4U];
        result[2 * i + 1] = hex[digest[i] & 0x0fU];
    }
    return result;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
