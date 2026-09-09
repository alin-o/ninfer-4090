#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <optional>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6 {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   FrontendOptions options);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedContextCache structural_diagnostics(
        const std::vector<std::pair<std::optional<std::uint32_t>, std::uint32_t>>& boundaries);
};

} // namespace ninfer::targets::qwen3_6
