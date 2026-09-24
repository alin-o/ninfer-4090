#pragma once

#include <ninfer/types.h>
#include <nlohmann/json.hpp>
#include <string_view>

namespace ninfer::serve {
[[nodiscard]] std::string_view context_monitor_page();
[[nodiscard]] nlohmann::json context_memory_json(const ContextMemory& memory);
[[nodiscard]] nlohmann::json context_monitor_json(const ContextSnapshot& snapshot, bool ready,
                                                  std::uint32_t concurrency);
} // namespace ninfer::serve
