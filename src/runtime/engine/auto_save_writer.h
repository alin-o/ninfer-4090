#pragma once

#include "ninfer/types.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::runtime {

// Engine-owned bounded Host consumer for immutable Program snapshots. The producer callback may
// wait for CUDA, but the worker never mutates Program stores or the logical resource catalog.
class AutoSaveWriter {
public:
    using Snapshot = targets::qwen3_6::RetainedSessionSnapshot;
    using Publish  = std::function<void(const std::string&, const std::vector<std::uint8_t>&)>;

    AutoSaveWriter(std::uint32_t max_jobs, std::size_t max_bytes,
                   std::function<void(const SlotAutoSaveEvent&)> listener, Publish publish);
    ~AutoSaveWriter() noexcept;

    AutoSaveWriter(const AutoSaveWriter&)            = delete;
    AutoSaveWriter& operator=(const AutoSaveWriter&) = delete;

    [[nodiscard]] std::shared_ptr<void> reserve(std::size_t bytes);
    void enqueue(std::string path, Snapshot&& snapshot);
    void note_authoritative(const std::string& path, std::uint32_t tokens);
    void drain();
    void stop() noexcept;
    [[nodiscard]] RuntimeStats populate_stats(RuntimeStats stats) const noexcept;

private:
    struct State;
    static void worker_loop(const std::shared_ptr<State>& state);
    std::shared_ptr<State> state_;
};

} // namespace ninfer::runtime
