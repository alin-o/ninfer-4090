#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace ninfer::runtime {

// Serialize Engine execution boundaries in arrival order. A decode worker must not reacquire
// the boundary ahead of already-waiting snapshot/export or control operations: those operations
// may release pins needed to admit another request. No mutex is held while waiting for a ticket.
class ExecutionMutex {
public:
    void lock() {
        std::unique_lock lock(mutex_);
        const std::uint64_t ticket = next_++;
        ready_.wait(lock, [&] { return serving_ == ticket; });
    }

    void unlock() noexcept {
        {
            std::lock_guard lock(mutex_);
            ++serving_;
        }
        ready_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::uint64_t next_    = 0;
    std::uint64_t serving_ = 0;
};

} // namespace ninfer::runtime
