#include "serve/media_budget_recovery.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::serve::prepare_with_media_budget_recovery;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

MessagePart image(std::uint8_t id) {
    return MessagePart{.kind = MessagePartKind::Media,
                       .media = OwnedMedia{.bytes = {id}, .media_type = "image/png"}};
}

std::vector<std::uint8_t> images(const PromptInput& input) {
    std::vector<std::uint8_t> result;
    for (const auto& message : input.messages) {
        for (const auto& part : message.parts) {
            if (part.kind == MessagePartKind::Media) { result.push_back(part.media.bytes.at(0)); }
        }
    }
    return result;
}

PromptInput conversation() {
    PromptInput input;
    input.messages = {
        {.role = ChatRole::System, .parts = {{.text = "Keep the game working."}}},
        {.role = ChatRole::User, .parts = {{.text = "Original screenshot"}, image(1)}},
        {.role = ChatRole::Assistant,
         .parts = {{.text = "Inspect the board"}},
         .reasoning_content = "Check cleared rows",
         .tool_calls = {{.id = "read-old", .name = "read", .arguments_json = "{}"}}},
        {.role = ChatRole::Tool,
         .parts = {{.text = "Read image file"}, image(2)},
         .tool_call_id = "read-old"},
        {.role = ChatRole::Assistant,
         .tool_calls = {{.id = "read-new", .name = "read", .arguments_json = "{}"}}},
        {.role = ChatRole::Tool,
         .parts = {{.text = "Latest comparison"}, image(3), image(4)},
         .tool_call_id = "read-new"},
        {.role = ChatRole::User, .parts = {{.text = "Continue in this session."}}},
    };
    return input;
}

int test_recovery_and_replay() {
    int failures = 0;
    const auto original = conversation();
    // Both a first request and a later stateless replay must recover without
    // editing the saved transcript or relying on provider-specific session state.
    for (int replay = 0; replay < 2; ++replay) {
        std::vector<std::vector<std::uint8_t>> attempts;
        std::size_t omitted = 0;
        const auto recovered = prepare_with_media_budget_recovery(
            original,
            [&](PromptInput input) {
                attempts.push_back(images(input));
                if (attempts.back().size() > 2) {
                    throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                                       "vision raw patches exceed processor budget");
                }
                return input;
            },
            omitted);
        failures += check(attempts == std::vector<std::vector<std::uint8_t>>{
                                          {1, 2, 3, 4}, {2, 3, 4}, {3, 4}},
                          "oldest media messages removed only as needed");
        failures += check(omitted == 2 && images(original).size() == 4,
                          "omission count and untouched original history");
        failures += check(recovered.messages.size() == original.messages.size(),
                          "all messages retained");
        for (std::size_t i = 0; i < original.messages.size(); ++i) {
            const auto& before = original.messages[i];
            const auto& after = recovered.messages[i];
            failures += check(before.role == after.role && before.parts.size() == after.parts.size() &&
                                  before.tool_call_id == after.tool_call_id &&
                                  before.reasoning_content == after.reasoning_content &&
                                  before.tool_calls.size() == after.tool_calls.size(),
                              "message ordering, content positions and tool links preserved");
            for (std::size_t p = 0; p < before.parts.size(); ++p) {
                if (before.parts[p].kind == MessagePartKind::Text) {
                    failures += check(before.parts[p].text == after.parts[p].text,
                                      "existing text preserved");
                }
            }
        }
        failures += check(recovered.messages[1].parts[1].kind == MessagePartKind::Text &&
                              recovered.messages[1].parts[1].text.find("omitted") != std::string::npos,
                          "model sees an omission marker");
    }
    return failures;
}

int test_retry_boundaries() {
    int failures = 0;
    for (const auto kind : {RequestErrorKind::MediaBudgetExceeded, RequestErrorKind::InvalidMedia,
                            RequestErrorKind::ContextLengthExceeded, RequestErrorKind::Cancelled}) {
        int attempts = 0;
        std::size_t omitted = 0;
        try {
            prepare_with_media_budget_recovery(
                conversation(),
                [&](PromptInput input) -> int {
                    ++attempts;
                    if (kind == RequestErrorKind::MediaBudgetExceeded && attempts == 3) {
                        failures += check(images(input) == std::vector<std::uint8_t>{3, 4},
                                          "newest image group is never stripped to force success");
                    }
                    throw RequestError(kind, "rejected");
                },
                omitted);
            failures += check(false, "irreducible errors propagated");
        } catch (const RequestError& error) {
            failures += check(error.kind() == kind, "original error kind preserved");
        }
        failures += check(attempts == (kind == RequestErrorKind::MediaBudgetExceeded ? 3 : 1),
                          "recovery is bounded and only retries media budget errors");
    }
    std::size_t omitted = 99;
    int attempts = 0;
    auto unchanged = prepare_with_media_budget_recovery(
        conversation(), [&](PromptInput input) { ++attempts; return input; }, omitted);
    failures += check(attempts == 1 && omitted == 0 && images(unchanged).size() == 4,
                      "requests within budget retain every image");
    return failures;
}

} // namespace

int main() {
    const int failures = test_recovery_and_replay() + test_retry_boundaries();
    if (failures == 0) { std::cout << "media budget recovery tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
