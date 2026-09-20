#include "targets/qwen3_6/impl/frontend/digest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

void check_digest(std::span<const std::uint8_t> bytes, std::string_view expected) {
    require(fi::sha256_hex(fi::sha256(bytes)) == expected, "full SHA-256 differs from oracle");
    require(fi::sha256_hex(fi::sha256(bytes, [] {})) == expected,
            "cancellable SHA-256 differs from oracle");
    // Exercise the same incremental API as request-content file publication, including empty
    // writes and updates that split SHA blocks and the final padding boundary.
    for (const std::size_t chunk : {1U, 63U, 65U, 4093U}) {
        if (chunk == 1 && bytes.size() > 1024) { continue; }
        fi::Sha256Hasher hasher;
        hasher.update(std::string_view{});
        for (std::size_t offset = 0; offset < bytes.size();) {
            const auto part = bytes.subspan(offset, std::min(chunk, bytes.size() - offset));
            hasher.update(part);
            offset += part.size();
        }
        hasher.update(std::span<const std::uint8_t>{});
        require(fi::sha256_hex(hasher.finalize()) == expected,
                "incremental SHA-256 differs from oracle");
    }
}

void check_text(std::string_view text, std::string_view expected) {
    require(fi::sha256_hex(fi::sha256(text)) == expected, "text SHA-256 differs from oracle");
    check_digest(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()),
                                               text.size()),
                 expected);
}

void test_digests() {
    // Published SHA-256 known-answer vectors, including the million-'a' long-message vector.
    check_text("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check_text("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    check_text("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    check_text(std::string(1000000, 'a'),
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    struct Vector {
        std::size_t bytes;
        std::string_view digest;
    };

    // Independent fixtures generated with hashlib.sha256(bytes(i % 251 for i in range(n))).
    // Cover SHA padding/block boundaries and the cancellation chunk boundaries.
    constexpr std::array vectors{
        Vector{55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
        Vector{56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        Vector{63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        Vector{64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
        Vector{65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        Vector{127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"},
        Vector{128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"},
        Vector{129, "5099c6a56203f9687f7d33f4bfdf576d31dc91f6b695ecea38b2770c87631135"},
        Vector{1048575, "8b55c6a2d3444092a316985030e00c1e6f8608eaf912c896dee7c374d440ed11"},
        Vector{1048576, "631b84027d6b9e52b539c4e8373622d23032dfadc64d60af87339c9037e4f769"},
        Vector{1048577, "5769f52bc3eef28afa39c6fc68cadb7d0bd69812ae3a3d71452f519ec3c7aa56"},
        Vector{2097169, "b057f26faa652e9916aa08eed7e50906151f83de0e878173e9d06e359c0dd75e"},
    };
    std::vector<std::uint8_t> bytes(vectors.back().bytes);
    for (std::size_t i = 0; i < bytes.size(); ++i) { bytes[i] = i % 251; }
    for (const auto& vector : vectors) {
        check_digest(std::span<const std::uint8_t>(bytes).first(vector.bytes), vector.digest);
    }
    for (const auto offset : {std::size_t{0}, bytes.size() / 2, bytes.size() - 1}) {
        bytes[offset] ^= 1;
        require(fi::sha256_hex(fi::sha256(bytes)) != vectors.back().digest,
                "SHA-256 missed corruption outside a sampled region");
        bytes[offset] ^= 1;
    }

    struct Cancelled {};

    std::size_t checkpoints = 0;
    bool cancelled          = false;
    try {
        (void)fi::sha256(bytes, [&] {
            if (++checkpoints == 2) { throw Cancelled{}; }
        });
    } catch (const Cancelled&) { cancelled = true; }
    require(cancelled, "large checksum did not propagate mid-payload cancellation");
    // An interrupted checksum cannot leave global digest state dirty for another request.
    check_text("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

} // namespace

int main() {
    try {
        test_digests();
        std::cout << "SHA-256 oracle, full-payload integrity and cancellation checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
