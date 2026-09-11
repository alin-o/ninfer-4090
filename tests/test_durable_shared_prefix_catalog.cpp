#include "serve/durable_shared_prefix_catalog.h"

#include <atomic>
#include <barrier>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using ninfer::serve::DurableSharedPrefixCatalog;
using ninfer::serve::DurableSharedPrefixCatalogOptions;

struct TemporaryDirectory {
    TemporaryDirectory() {
        std::string pattern = "/tmp/ninfer-durable-shared-XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* created = ::mkdtemp(writable.data());
        if (created == nullptr) { throw std::runtime_error("mkdtemp failed"); }
        path = created;
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

DurableSharedPrefixCatalog::Snapshot snapshot(char digest_character, std::uint32_t frontier,
                                              std::vector<std::uint8_t> bytes) {
    DurableSharedPrefixCatalog::Snapshot result;
    result.content_digest = std::string(64, digest_character);
    result.session_digest = "test";
    result.tokens         = frontier;
    result.transfer_bytes = bytes.size();
    result.bytes          = std::move(bytes);
    return result;
}

DurableSharedPrefixCatalogOptions options(const std::filesystem::path& path) {
    return {.directory     = path,
            .max_records   = 4,
            .max_bytes     = 4096,
            .workers       = 1,
            .max_jobs      = 2,
            .staging_bytes = 4096};
}

std::filesystem::path record_path(const std::filesystem::path& directory, std::string_view digest) {
    std::error_code error;
    for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end;
         it.increment(error)) {
        const std::string name = it->path().filename().string();
        if (name.starts_with(digest) && name.ends_with(".nsh")) { return it->path(); }
    }
    return {};
}

void test_restart_lazy_load_and_duplicate_write() {
    TemporaryDirectory temporary;
    const std::vector<std::uint8_t> payload{1, 2, 3, 4, 5};
    {
        DurableSharedPrefixCatalog catalog(options(temporary.path));
        std::atomic<std::uint32_t> committed{0};
        catalog.enqueue(snapshot('a', 17, payload),
                        [&](bool value) { committed.fetch_add(value ? 1U : 100U); });
        catalog.enqueue(snapshot('a', 17, payload),
                        [&](bool value) { committed.fetch_add(value ? 1U : 100U); });
        catalog.drain();
        const auto stats = catalog.stats();
        expect(stats.manifest_records == 1 && stats.manifest_bytes == payload.size(),
               "duplicate publication did not coalesce into one bounded record");
        expect(stats.writes_completed == 1 && stats.writes_coalesced >= 1,
               "duplicate publication counters are incorrect");
        expect(committed.load() == 2,
               "coalesced publications did not settle every owner as durable");
    }
    {
        std::promise<void> read_entered;
        std::promise<void> release_read;
        std::shared_future<void> release = release_read.get_future().share();
        auto configured                  = options(temporary.path);
        configured.max_jobs              = 1;
        configured.before_payload_read   = [&] {
            read_entered.set_value();
            release.wait();
        };
        DurableSharedPrefixCatalog catalog(std::move(configured));
        expect(catalog.stats().manifest_records == 1,
               "restart did not load bounded manifest metadata");
        const DurableSharedPrefixCatalog::Candidate candidate{
            .content_digest = std::string(64, 'a'), .frontier = 17};
        std::atomic<std::uint32_t> matching_loads{0};
        const auto load_once = [&] {
            const auto loaded = catalog.load(candidate, DurableSharedPrefixCatalog::Clock::now() +
                                                            std::chrono::seconds(1));
            if (loaded && *loaded == payload) { ++matching_loads; }
        };
        std::thread first(load_once);
        read_entered.get_future().wait();
        std::thread second(load_once);
        while (catalog.stats().loads_coalesced == 0) { std::this_thread::yield(); }
        std::atomic<bool> queue_rejected{false};
        catalog.enqueue(snapshot('9', 18, payload),
                        [&](bool committed) { queue_rejected.store(!committed); });
        release_read.set_value();
        first.join();
        second.join();
        catalog.drain();
        const auto stats = catalog.stats();
        expect(matching_loads.load() == 2 && stats.loads_completed == 1 &&
                   stats.loads_coalesced >= 1,
               "identical pending lazy loads did not coalesce");
        expect(queue_rejected.load() && stats.quota_rejections >= 1,
               "full transfer queue accepted another staged job");
    }
}

void test_truncation_quota_and_cancelled_load_cleanup() {
    TemporaryDirectory temporary;
    auto configured        = options(temporary.path);
    configured.max_records = 1;
    {
        DurableSharedPrefixCatalog catalog(configured);
        catalog.enqueue(snapshot('b', 9, {8, 7, 6, 5}));
        catalog.drain();
        catalog.enqueue(snapshot('c', 10, {1, 1, 1, 1}));
        catalog.drain();
        expect(catalog.stats().manifest_records == 1 && catalog.stats().quota_rejections >= 1,
               "full record quota published an additional record");
    }
    std::filesystem::resize_file(record_path(temporary.path, std::string(64, 'b')), 2);
    {
        DurableSharedPrefixCatalog catalog(configured);
        // Size mismatch is rejected while loading startup metadata, so the truncated payload is
        // unavailable without ever entering adoption or receiving hit credit.
        expect(catalog.stats().manifest_records == 0,
               "truncated record remained available after restart validation");
    }

    TemporaryDirectory cancelled_directory;
    {
        DurableSharedPrefixCatalog catalog(options(cancelled_directory.path));
        catalog.enqueue(snapshot('d', 12, std::vector<std::uint8_t>(1024, 3)));
        catalog.drain();
        std::atomic<bool> cancelled{true};
        const auto loaded =
            catalog.load({.content_digest = std::string(64, 'd'), .frontier = 12},
                         DurableSharedPrefixCatalog::Clock::now() + std::chrono::seconds(1),
                         ninfer::CancellationView([&] { return cancelled.load(); }));
        expect(!loaded, "cancelled lazy load returned a payload");
        catalog.drain();
        expect(catalog.stats().staging_bytes == 0,
               "cancelled lazy load retained its staging reservation");
    }
}

void test_interrupted_manifest_preserves_previous_commit() {
    TemporaryDirectory temporary;
    const auto configured = options(temporary.path);
    {
        DurableSharedPrefixCatalog catalog(configured);
        catalog.enqueue(snapshot('e', 20, {2, 0}));
        catalog.drain();
    }
    {
        auto interrupted                    = configured;
        interrupted.before_manifest_publish = [] { throw std::runtime_error("injected crash"); };
        DurableSharedPrefixCatalog catalog(std::move(interrupted));
        std::atomic<bool> counted_as_durable{false};
        catalog.enqueue(snapshot('f', 21, {2, 1}),
                        [&](bool committed) { counted_as_durable.store(committed); });
        catalog.drain();
        expect(catalog.stats().writes_failed == 1 && catalog.stats().manifest_records == 1,
               "interrupted manifest publication altered the committed index");
        expect(!counted_as_durable.load(),
               "interrupted manifest publication was counted as durable recovery");
    }
    {
        DurableSharedPrefixCatalog catalog(configured);
        auto prior =
            catalog.load({.content_digest = std::string(64, 'e'), .frontier = 20},
                         DurableSharedPrefixCatalog::Clock::now() + std::chrono::seconds(1));
        auto interrupted =
            catalog.load({.content_digest = std::string(64, 'f'), .frontier = 21},
                         DurableSharedPrefixCatalog::Clock::now() + std::chrono::seconds(1));
        expect(prior && !interrupted,
               "restart did not preserve the previous commit and discard the orphan replacement");
        prior.reset();
        interrupted.reset();
        catalog.drain();
    }
}

void test_first_load_registration_race_releases_loser_outside_mutex() {
    TemporaryDirectory temporary;
    const auto candidate = DurableSharedPrefixCatalog::Candidate{
        .content_digest = std::string(64, '7'), .frontier = 23};
    const std::vector<std::uint8_t> payload(128, 7);
    {
        DurableSharedPrefixCatalog catalog(options(temporary.path));
        catalog.enqueue(snapshot('7', candidate.frontier, payload));
        catalog.drain();
    }

    std::barrier registration_race(2);
    auto configured                     = options(temporary.path);
    configured.max_jobs                 = 2;
    configured.before_load_registration = [&] { registration_race.arrive_and_wait(); };
    DurableSharedPrefixCatalog catalog(std::move(configured));
    std::atomic<std::uint32_t> loaded{0};
    const auto load_once = [&] {
        const auto result = catalog.load(candidate, DurableSharedPrefixCatalog::Clock::now() +
                                                        std::chrono::seconds(2));
        if (result && *result == payload) { ++loaded; }
    };
    std::thread first(load_once);
    std::thread second(load_once);
    first.join();
    second.join();
    catalog.drain();
    const auto stats = catalog.stats();
    expect(loaded.load() == 2 && stats.loads_completed == 1 && stats.loads_coalesced == 1,
           "simultaneous first loads did not coalesce without deadlocking");
    expect(stats.staging_bytes == 0,
           "redundant first-load reservation was not released after catalog unlock");
}

void test_replacement_and_repeated_manifest_failures_stay_bounded() {
    TemporaryDirectory temporary;
    auto configured        = options(temporary.path);
    configured.max_records = 3;
    configured.max_bytes   = 64;
    const std::string digest(64, '8');
    {
        DurableSharedPrefixCatalog catalog(configured);
        catalog.enqueue(snapshot('8', 24, {1, 2, 3, 4}));
        catalog.drain();
    }
    {
        auto interrupted                    = configured;
        interrupted.before_manifest_publish = [] { throw std::runtime_error("injected crash"); };
        DurableSharedPrefixCatalog catalog(std::move(interrupted));
        catalog.enqueue(snapshot('8', 24, {4, 3, 2, 1}));
        for (char value = '1'; value <= '6'; ++value) {
            catalog.enqueue(snapshot(value, 24, std::vector<std::uint8_t>(8, value)));
            catalog.drain();
        }
        catalog.drain();
        const auto stats = catalog.stats();
        expect(stats.manifest_records == 1 && stats.manifest_bytes == 4 &&
                   stats.unpublished_records == 0 && stats.unpublished_bytes == 0,
               "failed manifest publications escaped directory quota accounting");
        std::uint32_t record_files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(temporary.path)) {
            if (entry.path().filename().string().ends_with(".nsh")) { ++record_files; }
        }
        expect(record_files == 1,
               "failed replacement/publications left unbounded committed-looking payloads");
    }
    {
        DurableSharedPrefixCatalog catalog(configured);
        auto prior =
            catalog.load({.content_digest = digest, .frontier = 24},
                         DurableSharedPrefixCatalog::Clock::now() + std::chrono::seconds(1));
        expect(prior && *prior == std::vector<std::uint8_t>({1, 2, 3, 4}),
               "interrupted replacement did not preserve the prior committed record");
        prior.reset();
        catalog.drain();
        catalog.enqueue(snapshot('8', 24, {4, 3, 2, 1}));
        catalog.drain();
    }
    {
        DurableSharedPrefixCatalog catalog(configured);
        auto replaced =
            catalog.load({.content_digest = digest, .frontier = 24},
                         DurableSharedPrefixCatalog::Clock::now() + std::chrono::seconds(1));
        expect(replaced && *replaced == std::vector<std::uint8_t>({4, 3, 2, 1}),
               "validated replacement did not survive restart");
        replaced.reset();
        catalog.drain();
    }
}

void test_orphan_cleanup_preserves_unrelated_files() {
    TemporaryDirectory temporary;
    const std::filesystem::path unrelated_temporary = temporary.path / "keep.tmp.user";
    const std::filesystem::path unrelated_record    = temporary.path / "keep.nsh";
    std::ofstream(unrelated_temporary) << "keep";
    std::ofstream(unrelated_record) << "keep";

    DurableSharedPrefixCatalog catalog(options(temporary.path));
    expect(std::filesystem::exists(unrelated_temporary) &&
               std::filesystem::exists(unrelated_record),
           "orphan cleanup removed files outside the durable catalog namespace");
}

} // namespace

int main() {
    test_restart_lazy_load_and_duplicate_write();
    test_truncation_quota_and_cancelled_load_cleanup();
    test_interrupted_manifest_preserves_previous_commit();
    test_first_load_registration_race_releases_loser_outside_mutex();
    test_replacement_and_repeated_manifest_failures_stay_bounded();
    test_orphan_cleanup_preserves_unrelated_files();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
