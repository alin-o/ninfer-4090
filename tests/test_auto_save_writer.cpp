#include "runtime/engine/auto_save_writer.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace std::chrono_literals;
using ninfer::runtime::AutoSaveWriter;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_full_queue_failure_and_retirement_accounting() {
    std::mutex mutex;
    std::condition_variable cv;
    bool transfer_started = false;
    bool transfer_ready   = false;
    std::vector<ninfer::SlotAutoSaveEvent> events;
    std::uint32_t publishes = 0;

    AutoSaveWriter writer(
        1, 128,
        [&](const ninfer::SlotAutoSaveEvent& event) {
            std::scoped_lock lock(mutex);
            events.push_back(event);
            cv.notify_all();
        },
        [&](const std::string&, const std::vector<std::uint8_t>& bytes) {
            expect(bytes == std::vector<std::uint8_t>({1, 2, 3}),
                   "writer published incomplete transfer bytes");
            ++publishes;
        });

    std::shared_ptr<void> reservation = writer.reserve(128);
    expect(static_cast<bool>(reservation), "first bounded writer reservation was rejected");
    expect(!writer.reserve(1), "full writer queue accepted another reservation");
    std::shared_ptr<void> program_retirement = reservation;

    AutoSaveWriter::Snapshot snapshot;
    snapshot.queue_reservation = std::move(reservation);
    snapshot.transfer_bytes    = 64;
    snapshot.tokens            = 7;
    snapshot.await_transfer    = [&](std::vector<std::uint8_t>& bytes) {
        std::unique_lock lock(mutex);
        transfer_started = true;
        cv.notify_all();
        cv.wait(lock, [&] { return transfer_ready; });
        bytes = {1, 2, 3};
    };
    writer.enqueue("session", std::move(snapshot));

    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return transfer_started; });
    }
    const ninfer::RuntimeStats active = writer.populate_stats({});
    expect(active.auto_save_queued_jobs == 0 && active.auto_save_queued_bytes == 0 &&
               active.auto_save_in_flight_jobs == 1 && active.auto_save_in_flight_bytes == 128 &&
               active.auto_save_reserved_jobs == 1 && active.auto_save_reserved_bytes == 128 &&
               active.auto_save_rejected_jobs == 1,
           "active/full writer accounting was incomplete");

    {
        std::scoped_lock lock(mutex);
        transfer_ready = true;
    }
    cv.notify_all();
    writer.drain();
    const ninfer::RuntimeStats retired_pending = writer.populate_stats({});
    expect(publishes == 1 && events.size() == 1 && events.front().error.empty(),
           "completed transfer was not published exactly once");
    expect(retired_pending.auto_save_queued_jobs == 0 &&
               retired_pending.auto_save_in_flight_jobs == 0 &&
               retired_pending.auto_save_in_flight_bytes == 0 &&
               retired_pending.auto_save_reserved_jobs == 1 &&
               retired_pending.auto_save_reserved_bytes == 128,
           "Program-retained reservation disappeared with writer completion");
    program_retirement.reset();
    const ninfer::RuntimeStats clean = writer.populate_stats({});
    expect(clean.auto_save_reserved_jobs == 0 && clean.auto_save_reserved_bytes == 0,
           "terminal Program retirement leaked writer accounting");

    std::uint32_t settlements = 0;
    AutoSaveWriter::Snapshot failed;
    failed.queue_reservation = writer.reserve(128);
    failed.transfer_bytes    = 64;
    failed.await_transfer    = [&](std::vector<std::uint8_t>&) {
        ++settlements;
        throw std::runtime_error("controlled transfer failure");
    };
    writer.enqueue("failed", std::move(failed));
    writer.drain();
    expect(settlements == 1 && events.size() == 2 &&
               events.back().error == "controlled transfer failure" && publishes == 1,
           "failed transfer published bytes or lost its terminal error");
    const ninfer::RuntimeStats failed_clean = writer.populate_stats({});
    expect(failed_clean.auto_save_reserved_jobs == 0 &&
               failed_clean.auto_save_reserved_bytes == 0 &&
               failed_clean.auto_save_in_flight_bytes == 0,
           "failed transfer leaked terminal accounting");
}

void test_cancellation_and_shutdown_settle_before_release() {
    AutoSaveWriter writer(2, 256, {}, [](const std::string&, const std::vector<std::uint8_t>&) {});
    std::mutex mutex;
    std::condition_variable cv;
    bool settlement_started = false;
    bool settlement_ready   = false;

    auto cancellation = std::async(std::launch::async, [&] {
        AutoSaveWriter::Snapshot snapshot;
        snapshot.queue_reservation = writer.reserve(128);
        snapshot.settle_transfer   = [&] {
            std::unique_lock lock(mutex);
            settlement_started = true;
            cv.notify_all();
            cv.wait(lock, [&] { return settlement_ready; });
        };
    });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return settlement_started; });
    }
    expect(writer.populate_stats({}).auto_save_reserved_jobs == 1 &&
               cancellation.wait_for(0ms) != std::future_status::ready,
           "cancelled snapshot released its live transfer reservation");
    {
        std::scoped_lock lock(mutex);
        settlement_ready = true;
    }
    cv.notify_all();
    cancellation.get();
    expect(writer.populate_stats({}).auto_save_reserved_jobs == 0,
           "cancelled snapshot did not clean reservation after settlement");

    bool transfer_started = false;
    bool transfer_ready   = false;
    AutoSaveWriter::Snapshot pending;
    pending.queue_reservation = writer.reserve(128);
    pending.transfer_bytes    = 64;
    pending.await_transfer    = [&](std::vector<std::uint8_t>& bytes) {
        std::unique_lock lock(mutex);
        transfer_started = true;
        cv.notify_all();
        cv.wait(lock, [&] { return transfer_ready; });
        bytes = {9};
    };
    writer.enqueue("shutdown", std::move(pending));
    std::shared_ptr<void> late_reservation = writer.reserve(128);
    expect(static_cast<bool>(late_reservation),
           "shutdown fixture could not reserve a second bounded job");
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return transfer_started; });
    }
    auto shutdown = std::async(std::launch::async, [&] { writer.stop(); });
    expect(shutdown.wait_for(0ms) != std::future_status::ready,
           "writer shutdown released a pending transfer without settlement");
    {
        std::scoped_lock lock(mutex);
        transfer_ready = true;
    }
    cv.notify_all();
    shutdown.get();
    std::uint32_t late_settlements = 0;
    {
        AutoSaveWriter::Snapshot late;
        late.queue_reservation = std::move(late_reservation);
        late.settle_transfer   = [&] { ++late_settlements; };
        writer.enqueue("after-shutdown", std::move(late));
    }
    const ninfer::RuntimeStats clean = writer.populate_stats({});
    expect(late_settlements == 1 && clean.auto_save_queued_jobs == 0 &&
               clean.auto_save_in_flight_jobs == 0 && clean.auto_save_in_flight_bytes == 0 &&
               clean.auto_save_reserved_jobs == 0 && clean.auto_save_reserved_bytes == 0,
           "writer shutdown dropped settlement or left nonterminal resource accounting");
}

} // namespace

int main() {
    test_full_queue_failure_and_retirement_accounting();
    test_cancellation_and_shutdown_settle_before_release();
    if (failures != 0) {
        std::cerr << failures << " auto-save writer checks failed\n";
        return 1;
    }
    std::cout << "auto-save writer checks passed\n";
    return 0;
}
