#include "runtime/engine/auto_save_writer.h"

#include "runtime/engine/slot_spill_guard.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer::runtime {

struct AutoSaveWriter::State {
    struct PendingWrite {
        std::string path;
        Snapshot snapshot;
    };

    State(std::uint32_t jobs, std::size_t bytes,
          std::function<void(const SlotAutoSaveEvent&)> event_listener, Publish file_publish)
        : max_jobs(jobs), max_bytes(bytes), listener(std::move(event_listener)),
          publish(std::move(file_publish)) {}

    const std::uint32_t max_jobs;
    const std::size_t max_bytes;
    std::function<void(const SlotAutoSaveEvent&)> listener;
    Publish publish;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<PendingWrite> pending;
    std::size_t queued_payload_bytes = 0;
    std::size_t reserved_bytes       = 0;
    std::uint32_t reserved_jobs      = 0;
    std::uint64_t rejected_jobs      = 0;
    std::size_t active_payload_bytes = 0;
    bool active                      = false;
    bool stop_requested              = false;
    std::thread worker;
    SlotSpillGuard spill_guard;
};

namespace {

std::uint64_t resident_bytes(std::uint64_t payload) noexcept {
    return payload > std::numeric_limits<std::uint64_t>::max() / 2U
               ? std::numeric_limits<std::uint64_t>::max()
               : payload * 2U;
}

} // namespace

void AutoSaveWriter::worker_loop(const std::shared_ptr<State>& state) {
    std::unique_lock lock(state->mutex);
    while (true) {
        state->cv.wait(lock, [&] { return state->stop_requested || !state->pending.empty(); });
        if (state->pending.empty()) { break; }
        AutoSaveWriter::State::PendingWrite item = std::move(state->pending.front());
        state->pending.pop_front();
        state->queued_payload_bytes -= item.snapshot.transfer_bytes;
        state->active_payload_bytes = item.snapshot.transfer_bytes;
        state->active               = true;
        lock.unlock();

        SlotAutoSaveEvent event;
        event.path         = item.path;
        event.tokens       = item.snapshot.tokens;
        event.bytes        = item.snapshot.transfer_bytes;
        const auto started = std::chrono::steady_clock::now();
        try {
            if (item.snapshot.await_transfer) { item.snapshot.await_transfer(item.snapshot.bytes); }
            if (const std::optional<std::uint32_t> deeper =
                    state->spill_guard.blocks(item.path, item.snapshot.tokens)) {
                event.skipped_behind_tokens = deeper;
            } else {
                state->publish(item.path, item.snapshot.bytes);
                state->spill_guard.note_spilled(item.path, item.snapshot.tokens);
            }
        } catch (const std::exception& error) { event.error = error.what(); } catch (...) {
            event.error = "unknown auto-save failure";
        }
        event.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (state->listener) {
            try {
                state->listener(event);
            } catch (...) {}
        }

        // Release the worker's pageable and pinned-buffer owners before publishing idle. A
        // Program retirement may intentionally retain the shared reservation/source pins.
        item.snapshot.release_storage();
        lock.lock();
        state->active_payload_bytes = 0;
        state->active               = false;
        state->cv.notify_all();
    }
    state->cv.notify_all();
}

AutoSaveWriter::AutoSaveWriter(std::uint32_t max_jobs, std::size_t max_bytes,
                               std::function<void(const SlotAutoSaveEvent&)> listener,
                               Publish publish)
    : state_(
          std::make_shared<State>(max_jobs, max_bytes, std::move(listener), std::move(publish))) {
    if (max_jobs == 0 || max_bytes == 0 || !state_->publish) {
        throw std::invalid_argument("auto-save writer bounds and publisher must be valid");
    }
    state_->worker = std::thread([state = state_] { worker_loop(state); });
}

AutoSaveWriter::~AutoSaveWriter() noexcept { stop(); }

std::shared_ptr<void> AutoSaveWriter::reserve(std::size_t bytes) {
    const std::shared_ptr<State> state = state_;
    std::unique_lock lock(state->mutex);
    if (state->stop_requested || state->reserved_jobs >= state->max_jobs ||
        bytes > state->max_bytes - state->reserved_bytes) {
        ++state->rejected_jobs;
        return {};
    }
    ++state->reserved_jobs;
    state->reserved_bytes += bytes;
    return std::shared_ptr<void>(state.get(), [state, bytes](void*) noexcept {
        std::scoped_lock release_lock(state->mutex);
        --state->reserved_jobs;
        state->reserved_bytes -= bytes;
        state->cv.notify_all();
    });
}

void AutoSaveWriter::enqueue(std::string path, Snapshot&& snapshot) {
    const std::shared_ptr<State> state = state_;
    std::unique_lock lock(state->mutex);
    if (!snapshot.queue_reservation) { return; }
    // Returning leaves ownership with the caller, whose snapshot destructor settles any CUDA
    // work before releasing storage. Never clear a rejected snapshot's settlement callback.
    if (state->stop_requested) { return; }
    state->pending.push_back(State::PendingWrite{std::move(path), std::move(snapshot)});
    state->queued_payload_bytes += state->pending.back().snapshot.transfer_bytes;
    lock.unlock();
    state->cv.notify_one();
}

void AutoSaveWriter::note_authoritative(const std::string& path, std::uint32_t tokens) {
    state_->spill_guard.note_authoritative(path, tokens);
}

void AutoSaveWriter::drain() {
    const std::shared_ptr<State> state = state_;
    std::unique_lock lock(state->mutex);
    state->cv.wait(lock, [&] { return state->pending.empty() && !state->active; });
}

void AutoSaveWriter::stop() noexcept {
    const std::shared_ptr<State> state = state_;
    if (!state) { return; }
    {
        std::scoped_lock lock(state->mutex);
        state->stop_requested = true;
    }
    state->cv.notify_all();
    if (state->worker.joinable()) {
        try {
            state->worker.join();
        } catch (...) {}
    }
}

RuntimeStats AutoSaveWriter::populate_stats(RuntimeStats stats) const noexcept {
    const std::shared_ptr<State> state = state_;
    std::lock_guard lock(state->mutex);
    stats.auto_save_queued_jobs     = static_cast<std::uint32_t>(state->pending.size());
    stats.auto_save_queued_bytes    = resident_bytes(state->queued_payload_bytes);
    stats.auto_save_in_flight_jobs  = state->active ? 1U : 0U;
    stats.auto_save_in_flight_bytes = resident_bytes(state->active_payload_bytes);
    stats.auto_save_reserved_jobs   = state->reserved_jobs;
    stats.auto_save_reserved_bytes  = state->reserved_bytes;
    stats.auto_save_rejected_jobs   = state->rejected_jobs;
    return stats;
}

} // namespace ninfer::runtime
