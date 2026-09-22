#pragma once

#include <ninfer/types.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace ninfer::serve {

// Serving policy shared by generation and token counting. Retry only preparation,
// before inference or checkpoint publication. Never change the client's history.
// Keep the newest media-bearing message intact, including multi-image comparisons
// and tool-result screenshots; earlier messages lose media oldest-first.
template <typename Prepare>
auto prepare_with_media_budget_recovery(PromptInput input, Prepare&& prepare,
                                       std::size_t& omitted_media) {
    omitted_media = 0;
    std::vector<std::size_t> historical_media;
    for (std::size_t index = 0; index < input.messages.size(); ++index) {
        const auto& parts = input.messages[index].parts;
        if (std::any_of(parts.begin(), parts.end(), [](const MessagePart& part) {
                return part.kind == MessagePartKind::Media;
            })) {
            historical_media.push_back(index);
        }
    }
    if (!historical_media.empty()) { historical_media.pop_back(); }

    for (std::size_t next = 0;; ++next) {
        // No rollback copy is needed for text-only requests or the final attempt.
        if (next == historical_media.size()) { return prepare(std::move(input)); }
        try {
            return prepare(input);
        } catch (const RequestError& error) {
            if (error.kind() != RequestErrorKind::MediaBudgetExceeded) { throw; }
        }
        for (MessagePart& part : input.messages[historical_media[next]].parts) {
            if (part.kind != MessagePartKind::Media) { continue; }
            const bool video = part.media.kind == MediaKind::Video;
            part = MessagePart{
                .kind = MessagePartKind::Text,
                .text = video ? "[Earlier video omitted to fit the media budget.]"
                              : "[Earlier image omitted to fit the media budget.]",
            };
            ++omitted_media;
        }
        // Replacing parts in place preserves cache-marker indices, roles, tool
        // call/result links, and all text. The Engine recomputes prefix identity
        // from the resulting prompt; no cached state from the old prompt is forced.
    }
}

} // namespace ninfer::serve
