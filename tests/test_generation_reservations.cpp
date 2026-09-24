#include "runtime/generation/reservation_horizon.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using ninfer::runtime::ReservationHorizon;
    for (std::uint32_t prompt = 1; prompt <= 257; ++prompt) {
        for (std::uint32_t output = 0; output <= 300; ++output) {
            for (const auto quantum : {1U, 2U, 64U, 16384U}) {
                const ReservationHorizon horizon{prompt, output, 512, quantum, 128};
                horizon.validate();
                const auto output_kv  = output == 0 ? 0 : output - 1;
                const auto completion = std::min(512U, prompt + output_kv);
                const auto initial =
                    std::min(512U, prompt + (output == 0 ? 0 : std::min(output, quantum) - 1));
                if (horizon.completion_tokens() != completion ||
                    horizon.initial_tokens() != initial ||
                    horizon.initial_pages() != (initial + 127) / 128 ||
                    horizon.initial_pages() < (prompt + 127) / 128 ||
                    horizon.initial_pages() > horizon.pages(completion)) {
                    std::cerr << "reservation lost prompt/final-token/context boundary\n";
                    return 1;
                }
                for (const auto drafts : {1U, 3U, 8U}) {
                    // Independent token-by-token footprint: alignment covers the target
                    // prefix; the next proposal writes at most drafts - 1 extra columns.
                    auto backend_end = initial;
                    for (unsigned d = 1; d < drafts && backend_end < 512; ++d) { ++backend_end; }
                    if (horizon.mtp_backend_pages(initial, drafts) != (backend_end + 127) / 128) {
                        std::cerr << "MTP provisional tail escaped its initial reservation\n";
                        return 1;
                    }
                }
            }
        }
    }
    const ReservationHorizon large{1, std::numeric_limits<std::uint32_t>::max(), 131072, 16384,
                                   128};
    if (large.completion_tokens() != 131072 || large.initial_pages() != 128) { return 1; }
    for (const auto quantum : {0U, 2147483648U, 4294967295U}) {
        bool rejected = false;
        try {
            ReservationHorizon{1, 300, 512, quantum, 128}.validate();
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) {
            std::cerr << "invalid reservation quantum accepted\n";
            return 1;
        }
    }
    std::cout << "exhaustive horizon arithmetic passed\n";
}
