#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::media_acquire {

enum class SourceKind {
    Path,
    Url,
    Data,
    Bytes,
};

struct Source {
    SourceKind kind = SourceKind::Path;
    std::string value;
    std::string media_type;
    std::vector<std::uint8_t> bytes;
};

[[nodiscard]] inline std::string data_uri_media_type(std::string_view value) {
    if (!value.starts_with("data:")) { return {}; }
    const std::size_t parameter = value.find(';', 5);
    const std::size_t payload   = value.find(',', 5);
    const std::size_t end       = parameter == std::string_view::npos ? payload : parameter;
    if (end == std::string_view::npos || end == 5) { return {}; }
    return std::string(value.substr(5, end - 5));
}

} // namespace ninfer::product::media_acquire
