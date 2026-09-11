#include "serve/durable_shared_prefix_catalog.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::string_view kManifestName      = "catalog.manifest";
constexpr std::string_view kManifestHeader    = "NINFER_SHARED_CATALOG\t1\n";
constexpr std::size_t kMaximumManifestBytes   = 1U << 20U;
constexpr std::uint32_t kMaximumOrphanCleanup = 32;

bool valid_digest(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
}

bool decimal_digits(std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(),
                                         [](unsigned char ch) { return ch >= '0' && ch <= '9'; });
}

bool catalog_record_name(std::string_view name);

bool catalog_temporary_name(std::string_view name) {
    constexpr std::string_view manifest_prefix = "catalog.manifest.tmp.";
    if (name.starts_with(manifest_prefix)) {
        return decimal_digits(name.substr(manifest_prefix.size()));
    }
    constexpr std::string_view temporary_separator = ".tmp.";
    const std::size_t temporary                    = name.rfind(temporary_separator);
    return temporary != std::string_view::npos && catalog_record_name(name.substr(0, temporary)) &&
           decimal_digits(name.substr(temporary + temporary_separator.size()));
}

bool catalog_record_name(std::string_view name) {
    constexpr std::string_view suffix = ".nsh";
    if (!name.ends_with(suffix)) { return false; }
    name.remove_suffix(suffix.size());
    if (name.size() == 64U) { return valid_digest(name); }
    if (name.size() <= 65U || name[64] != '.' || !valid_digest(name.substr(0, 64))) {
        return false;
    }
    return decimal_digits(name.substr(65));
}

bool record_filename_for_digest(std::string_view filename, std::string_view digest) {
    return valid_digest(digest) && filename.starts_with(digest) && catalog_record_name(filename) &&
           (filename.size() == digest.size() + 4U || filename[digest.size()] == '.');
}

std::string record_filename(std::string_view digest, std::uint64_t order) {
    return std::string(digest) + '.' + std::to_string(order) + ".nsh";
}

std::string error_text(std::string_view operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

void write_all(int fd, std::span<const std::uint8_t> bytes) {
    while (!bytes.empty()) {
        const ssize_t written = ::write(fd, bytes.data(), bytes.size());
        if (written < 0) {
            if (errno == EINTR) { continue; }
            throw std::runtime_error(error_text("durable shared-prefix write failed"));
        }
        if (written == 0) {
            throw std::runtime_error("durable shared-prefix write made no progress");
        }
        bytes = bytes.subspan(static_cast<std::size_t>(written));
    }
}

void sync_directory(const std::filesystem::path& directory) {
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(error_text("durable shared-prefix directory open failed"));
    }
    const int result      = ::fsync(fd);
    const int saved_errno = errno;
    ::close(fd);
    if (result != 0) {
        errno = saved_errno;
        throw std::runtime_error(error_text("durable shared-prefix directory sync failed"));
    }
}

void atomic_write(const std::filesystem::path& directory, const std::filesystem::path& final_path,
                  std::span<const std::uint8_t> bytes, std::uint64_t nonce,
                  bool* renamed = nullptr) {
    if (renamed != nullptr) { *renamed = false; }
    const std::filesystem::path temporary =
        directory / (final_path.filename().string() + ".tmp." + std::to_string(nonce));
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        throw std::runtime_error(error_text("durable shared-prefix temporary open failed"));
    }
    try {
        write_all(fd, bytes);
        if (::fsync(fd) != 0) {
            throw std::runtime_error(error_text("durable shared-prefix file sync failed"));
        }
        if (::close(fd) != 0) {
            throw std::runtime_error(error_text("durable shared-prefix file close failed"));
        }
    } catch (...) {
        const int saved_errno = errno;
        ::close(fd);
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        errno = saved_errno;
        throw;
    }
    if (::rename(temporary.c_str(), final_path.c_str()) != 0) {
        const int saved_errno = errno;
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        errno = saved_errno;
        throw std::runtime_error(error_text("durable shared-prefix rename failed"));
    }
    if (renamed != nullptr) { *renamed = true; }
    sync_directory(directory);
}

std::vector<std::string_view> split_tabs(std::string_view line) {
    std::vector<std::string_view> fields;
    while (true) {
        const std::size_t tab = line.find('\t');
        fields.push_back(line.substr(0, tab));
        if (tab == std::string_view::npos) { break; }
        line.remove_prefix(tab + 1U);
    }
    return fields;
}

std::uint64_t parse_decimal(std::string_view value) {
    if (value.empty()) { throw std::invalid_argument("empty decimal"); }
    std::uint64_t result = 0;
    for (const char ch : value) {
        if (ch < '0' || ch > '9' || result > (UINT64_MAX - static_cast<unsigned>(ch - '0')) / 10U) {
            throw std::invalid_argument("invalid decimal");
        }
        result = result * 10U + static_cast<unsigned>(ch - '0');
    }
    return result;
}

} // namespace

struct DurableSharedPrefixCatalog::State : std::enable_shared_from_this<State> {
    struct Record {
        std::string digest;
        std::string filename;
        std::uint32_t frontier = 0;
        std::uint64_t bytes    = 0;
        std::uint64_t order    = 0;
    };

    struct LoadJob {
        struct Payload {
            std::vector<std::uint8_t> bytes;
            std::shared_ptr<void> reservation;
        };

        std::mutex mutex;
        std::condition_variable cv;
        std::shared_ptr<Payload> payload;
        std::string error;
        bool done = false;
    };

    explicit State(DurableSharedPrefixCatalogOptions configured) : options(std::move(configured)) {
        if (options.directory.empty() || options.max_records == 0 || options.max_bytes == 0 ||
            options.workers == 0 || options.max_jobs == 0 || options.staging_bytes == 0) {
            throw std::invalid_argument("durable shared-prefix catalog bounds must be nonzero");
        }
        std::filesystem::create_directories(options.directory);
        try {
            load_manifest();
        } catch (const std::invalid_argument&) {
            records.clear();
            catalog_bytes = 0;
            next_order    = 1;
            ++values.corrupt_records;
        }
        cleanup_orphans();
    }

    ~State() = default;

    void start() {
        const auto self = shared_from_this();
        threads.reserve(options.workers);
        for (std::uint32_t index = 0; index < options.workers; ++index) {
            threads.emplace_back([self] { self->worker_loop(); });
        }
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        cv.notify_all();
        for (std::thread& thread : threads) {
            if (thread.joinable()) { thread.join(); }
        }
    }

    std::shared_ptr<void> reserve(std::size_t bytes) {
        std::lock_guard lock(mutex);
        if (stopping || reserved_jobs >= options.max_jobs || bytes > options.staging_bytes ||
            reserved_bytes > options.staging_bytes - bytes) {
            ++values.quota_rejections;
            return {};
        }
        ++reserved_jobs;
        reserved_bytes += bytes;
        values.peak_staging_bytes = std::max(values.peak_staging_bytes, reserved_bytes);
        const auto self           = shared_from_this();
        return std::shared_ptr<void>(self.get(), [self, bytes](void*) {
            std::lock_guard release_lock(self->mutex);
            --self->reserved_jobs;
            self->reserved_bytes -= bytes;
            self->drained.notify_all();
        });
    }

    void push(std::function<void()> task) {
        {
            std::lock_guard lock(mutex);
            if (stopping) { throw std::runtime_error("durable shared-prefix catalog is stopping"); }
            jobs.push_back(std::move(task));
        }
        cv.notify_one();
    }

    void worker_loop() noexcept {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (stopping && jobs.empty()) { return; }
                task = std::move(jobs.front());
                jobs.pop_front();
                ++active_jobs;
            }
            try {
                task();
            } catch (...) {}
            {
                std::lock_guard lock(mutex);
                --active_jobs;
                drained.notify_all();
            }
        }
    }

    void load_manifest() {
        const std::filesystem::path path = options.directory / kManifestName;
        std::error_code error;
        const std::uint64_t size = std::filesystem::file_size(path, error);
        if (error) { return; }
        if (size > kMaximumManifestBytes) {
            throw std::invalid_argument("durable catalog manifest is too large");
        }
        std::ifstream input(path, std::ios::binary);
        std::string contents(static_cast<std::size_t>(size), '\0');
        input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!input || !contents.starts_with(kManifestHeader)) {
            throw std::invalid_argument("durable catalog manifest is invalid");
        }
        std::size_t cursor = kManifestHeader.size();
        while (cursor < contents.size()) {
            const std::size_t end = contents.find('\n', cursor);
            if (end == std::string::npos) {
                throw std::invalid_argument("durable catalog manifest is truncated");
            }
            const auto fields = split_tabs(std::string_view(contents).substr(cursor, end - cursor));
            cursor            = end + 1U;
            if (fields.size() != 5 || !valid_digest(fields[0])) {
                throw std::invalid_argument("durable catalog manifest record is invalid");
            }
            const std::uint64_t frontier = parse_decimal(fields[1]);
            const std::uint64_t bytes    = parse_decimal(fields[2]);
            const std::uint64_t order    = parse_decimal(fields[4]);
            const std::string filename(fields[3]);
            if (!record_filename_for_digest(filename, fields[0]) || frontier == 0 ||
                frontier > UINT32_MAX || bytes == 0 || order == UINT64_MAX ||
                records.contains(std::string(fields[0])) || bytes > options.max_bytes ||
                records.size() >= options.max_records ||
                catalog_bytes > options.max_bytes - bytes) {
                throw std::invalid_argument("durable catalog manifest exceeds configured bounds");
            }
            const std::filesystem::path record_path = options.directory / filename;
            const std::uint64_t actual = std::filesystem::file_size(record_path, error);
            if (error || actual != bytes) { continue; }
            Record record{.digest   = std::string(fields[0]),
                          .filename = filename,
                          .frontier = static_cast<std::uint32_t>(frontier),
                          .bytes    = bytes,
                          .order    = order};
            catalog_bytes += bytes;
            next_order = std::max(next_order, order + 1U);
            records.emplace(record.digest, std::move(record));
        }
    }

    void cleanup_orphans() noexcept {
        std::uint32_t removed = 0;
        std::error_code error;
        for (std::filesystem::directory_iterator it(options.directory, error), end;
             !error && it != end; it.increment(error)) {
            if (!it->is_regular_file(error)) { continue; }
            const std::string name    = it->path().filename().string();
            const bool temporary      = catalog_temporary_name(name);
            const bool record         = catalog_record_name(name);
            const auto indexed_record = record ? records.find(name.substr(0, 64)) : records.end();
            const bool indexed =
                indexed_record != records.end() && indexed_record->second.filename == name;
            if (temporary || (record && !indexed)) {
                bool removed_now = false;
                if (removed < kMaximumOrphanCleanup) {
                    removed_now = std::filesystem::remove(it->path(), error);
                    if (!error && removed_now) { ++removed; }
                }
                if (!removed_now) {
                    error.clear();
                    const std::uint64_t bytes = it->file_size(error);
                    unpublished_records =
                        unpublished_records == UINT32_MAX ? UINT32_MAX : unpublished_records + 1U;
                    unpublished_bytes = error || unpublished_bytes > UINT64_MAX - bytes
                                            ? UINT64_MAX
                                            : unpublished_bytes + bytes;
                }
                error.clear();
            }
        }
    }

    std::string manifest_contents(const std::optional<Record>& added,
                                  std::string_view removed_digest = {}) {
        std::vector<Record> ordered;
        ordered.reserve(records.size() + (added ? 1U : 0U));
        for (const auto& [digest, record] : records) {
            if (digest != removed_digest && (!added || digest != added->digest)) {
                ordered.push_back(record);
            }
        }
        if (added) { ordered.push_back(*added); }
        std::sort(ordered.begin(), ordered.end(),
                  [](const Record& left, const Record& right) { return left.order < right.order; });
        std::string result(kManifestHeader);
        for (const Record& record : ordered) {
            result += record.digest + '\t' + std::to_string(record.frontier) + '\t' +
                      std::to_string(record.bytes) + '\t' + record.filename + '\t' +
                      std::to_string(record.order) + '\n';
        }
        if (result.size() > kMaximumManifestBytes) {
            throw std::runtime_error("durable catalog manifest metadata bound exceeded");
        }
        return result;
    }

    [[nodiscard]] bool directory_quota_available(std::uint64_t bytes,
                                                 std::uint32_t record_count) const noexcept {
        const std::uint64_t occupied_bytes = catalog_bytes > UINT64_MAX - unpublished_bytes
                                                 ? UINT64_MAX
                                                 : catalog_bytes + unpublished_bytes;
        const std::uint64_t occupied_records =
            static_cast<std::uint64_t>(records.size()) + unpublished_records;
        return bytes <= options.max_bytes && occupied_bytes <= options.max_bytes - bytes &&
               occupied_records <= options.max_records &&
               record_count <= options.max_records - occupied_records;
    }

    void charge_unpublished(std::uint64_t bytes) noexcept {
        unpublished_records =
            unpublished_records == UINT32_MAX ? UINT32_MAX : unpublished_records + 1U;
        unpublished_bytes =
            unpublished_bytes > UINT64_MAX - bytes ? UINT64_MAX : unpublished_bytes + bytes;
    }

    void release_unpublished(std::uint64_t bytes) noexcept {
        if (unpublished_records != UINT32_MAX) { --unpublished_records; }
        if (unpublished_bytes != UINT64_MAX) { unpublished_bytes -= bytes; }
    }

    void remove_unpublished(const std::filesystem::path& path, std::uint64_t bytes) noexcept {
        std::error_code error;
        const bool removed = std::filesystem::remove(path, error);
        if (!error && !removed) {
            std::lock_guard lock(mutex);
            release_unpublished(bytes);
            return;
        }
        if (!error) {
            try {
                sync_directory(options.directory);
                std::lock_guard lock(mutex);
                release_unpublished(bytes);
                return;
            } catch (...) {}
        }
        // The file may still exist, or its deletion may not be crash-durable. Keep the charge
        // conservatively; a restart's bounded orphan cleanup will settle it.
    }

    [[nodiscard]] bool record_matches(const Record& record,
                                      std::span<const std::uint8_t> expected) const {
        if (record.bytes != expected.size()) { return false; }
        std::ifstream input(options.directory / record.filename, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != static_cast<std::streamsize>(expected.size())) {
            return false;
        }
        input.seekg(0);
        std::array<std::uint8_t, 64U << 10U> buffer{};
        std::size_t offset = 0;
        while (offset < expected.size()) {
            const std::size_t count = std::min(buffer.size(), expected.size() - offset);
            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count));
            if (!input || !std::equal(buffer.begin(), buffer.begin() + count,
                                      expected.begin() + static_cast<std::ptrdiff_t>(offset))) {
                return false;
            }
            offset += count;
        }
        return true;
    }

    void settle_write(std::string_view digest, bool committed) noexcept {
        std::vector<std::function<void(bool)>> settlements;
        {
            std::lock_guard lock(mutex);
            const auto found = pending_writes.find(std::string(digest));
            if (found != pending_writes.end()) {
                settlements = std::move(found->second);
                pending_writes.erase(found);
            }
        }
        for (auto& settlement : settlements) {
            if (!settlement) { continue; }
            try {
                settlement(committed);
            } catch (...) {}
        }
    }

    void publish(Snapshot snapshot) {
        const std::string digest     = snapshot.content_digest;
        const std::uint32_t frontier = snapshot.tokens;
        try {
            if (snapshot.await_transfer) { snapshot.await_transfer(snapshot.bytes); }
            snapshot.await_transfer = {};
            if (!valid_digest(digest) || frontier == 0 || snapshot.bytes.empty()) {
                throw std::invalid_argument("Program produced invalid durable snapshot metadata");
            }

            const auto io_started = Clock::now();
            bool committed        = false;
            {
                std::scoped_lock publish_lock(publish_mutex);
                bool already_committed = false;
                bool quota_rejected    = false;
                std::optional<Record> previous;
                {
                    std::lock_guard lock(mutex);
                    const auto found = records.find(digest);
                    if (found != records.end()) { previous = found->second; }
                }
                if (previous && previous->frontier == frontier &&
                    record_matches(*previous, snapshot.bytes)) {
                    std::lock_guard lock(mutex);
                    ++values.writes_coalesced;
                    already_committed = true;
                } else {
                    std::lock_guard lock(mutex);
                    if (!directory_quota_available(snapshot.bytes.size(), 1U)) {
                        ++values.quota_rejections;
                        quota_rejected = true;
                    } else {
                        charge_unpublished(snapshot.bytes.size());
                    }
                }
                committed = already_committed;
                if (!already_committed && !quota_rejected) {
                    Record record{.digest   = digest,
                                  .filename = record_filename(digest, next_order),
                                  .frontier = frontier,
                                  .bytes    = snapshot.bytes.size(),
                                  .order    = next_order++};
                    const std::filesystem::path record_path = options.directory / record.filename;
                    bool record_published                   = false;
                    bool retain_unpublished                 = false;
                    try {
                        atomic_write(options.directory, record_path, snapshot.bytes,
                                     record.order * 2U, &record_published);
                        const std::string manifest = manifest_contents(record, digest);
                        if (options.before_manifest_publish) { options.before_manifest_publish(); }
                        bool manifest_renamed = false;
                        try {
                            atomic_write(options.directory, options.directory / kManifestName,
                                         std::span<const std::uint8_t>(
                                             reinterpret_cast<const std::uint8_t*>(manifest.data()),
                                             manifest.size()),
                                         record.order * 2U + 1U, &manifest_renamed);
                        } catch (...) {
                            // After manifest rename, the commit marker may survive a crash even if
                            // its parent fsync failed. Keep the referenced payload and its quota
                            // charge; restart will resolve which manifest version persisted.
                            retain_unpublished = manifest_renamed;
                            throw;
                        }
                        {
                            std::lock_guard lock(mutex);
                            if (previous) {
                                catalog_bytes -= previous->bytes;
                                // The superseded file remains charged until unlink + parent sync.
                                charge_unpublished(previous->bytes);
                            }
                            records.insert_or_assign(digest, record);
                            catalog_bytes += record.bytes;
                            release_unpublished(record.bytes);
                            ++values.writes_completed;
                            values.io_nanoseconds += static_cast<std::uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                                     io_started)
                                    .count());
                        }
                        if (previous && previous->filename != record.filename) {
                            remove_unpublished(options.directory / previous->filename,
                                               previous->bytes);
                        }
                    } catch (...) {
                        if (record_published && !retain_unpublished) {
                            remove_unpublished(record_path, record.bytes);
                        } else if (!record_published) {
                            std::lock_guard lock(mutex);
                            release_unpublished(record.bytes);
                        }
                        throw;
                    }
                    committed = true;
                }
            }
            settle_write(digest, committed);
        } catch (...) {
            {
                std::lock_guard lock(mutex);
                ++values.writes_failed;
            }
            settle_write(digest, false);
        }
        snapshot.release_storage();
    }

    void invalidate(std::string_view digest) noexcept {
        try {
            std::scoped_lock publish_lock(publish_mutex);
            std::optional<Record> invalid;
            {
                std::lock_guard lock(mutex);
                const auto found = records.find(std::string(digest));
                if (found == records.end()) { return; }
                invalid = found->second;
            }
            const std::string manifest = manifest_contents(std::nullopt, digest);
            const std::uint64_t nonce  = next_order++ * 2U + 1U;
            atomic_write(
                options.directory, options.directory / kManifestName,
                std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(manifest.data()), manifest.size()),
                nonce);
            {
                std::lock_guard lock(mutex);
                const auto found = records.find(std::string(digest));
                if (found == records.end() || found->second.filename != invalid->filename) {
                    return;
                }
                catalog_bytes -= invalid->bytes;
                charge_unpublished(invalid->bytes);
                records.erase(found);
            }
            remove_unpublished(options.directory / invalid->filename, invalid->bytes);
        } catch (...) {
            // If invalidation cannot commit, the old manifest remains authoritative. A later
            // restore validates again, and a later export performs byte-exact replacement.
        }
    }

    DurableSharedPrefixCatalogOptions options;
    mutable std::mutex mutex;
    std::mutex publish_mutex;
    std::condition_variable cv;
    std::condition_variable drained;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread> threads;
    std::unordered_map<std::string, Record> records;
    std::unordered_map<std::string, std::weak_ptr<LoadJob>> pending_loads;
    std::unordered_map<std::string, std::vector<std::function<void(bool)>>> pending_writes;
    std::unordered_set<std::string> exported_owners;
    DurableSharedPrefixCatalogStats values;
    std::uint64_t catalog_bytes       = 0;
    std::uint64_t next_order          = 1;
    std::size_t reserved_bytes        = 0;
    std::uint32_t reserved_jobs       = 0;
    std::uint32_t active_jobs         = 0;
    std::uint32_t unpublished_records = 0;
    std::uint64_t unpublished_bytes   = 0;
    bool stopping                     = false;
};

DurableSharedPrefixCatalog::DurableSharedPrefixCatalog(DurableSharedPrefixCatalogOptions options)
    : state_(std::make_shared<State>(std::move(options))) {
    state_->start();
}

DurableSharedPrefixCatalog::~DurableSharedPrefixCatalog() noexcept {
    if (state_) { state_->stop(); }
}

void DurableSharedPrefixCatalog::enqueue(Snapshot snapshot, std::function<void(bool)> settlement) {
    const auto state              = state_;
    const auto settle_immediately = [&](bool committed) noexcept {
        if (!settlement) { return; }
        try {
            settlement(committed);
        } catch (...) {}
    };
    const auto settle_snapshot_source = [&]() noexcept {
        if (!snapshot.settle_transfer) { return; }
        try {
            snapshot.settle_transfer();
        } catch (...) {}
    };
    if (!snapshot.queue_reservation) {
        const std::size_t bytes =
            snapshot.transfer_bytes != 0 ? snapshot.transfer_bytes : snapshot.bytes.size();
        if (bytes > SIZE_MAX / 2U) {
            settle_snapshot_source();
            settle_immediately(false);
            return;
        }
        snapshot.queue_reservation = state->reserve(bytes * 2U);
        if (!snapshot.queue_reservation) {
            settle_snapshot_source();
            settle_immediately(false);
            return;
        }
    }
    if (!valid_digest(snapshot.content_digest)) {
        settle_snapshot_source();
        settle_immediately(false);
        return;
    }
    {
        std::lock_guard lock(state->mutex);
        const auto pending = state->pending_writes.find(snapshot.content_digest);
        if (pending != state->pending_writes.end()) {
            pending->second.push_back(std::move(settlement));
            ++state->values.writes_coalesced;
            return;
        }
        state->pending_writes.emplace(
            snapshot.content_digest, std::vector<std::function<void(bool)>>{std::move(settlement)});
    }
    auto holder = std::make_shared<Snapshot>(std::move(snapshot));
    try {
        state->push([state, holder] { state->publish(std::move(*holder)); });
    } catch (...) {
        {
            std::lock_guard lock(state->mutex);
            ++state->values.writes_failed;
        }
        if (holder->settle_transfer) {
            try {
                holder->settle_transfer();
            } catch (...) {}
        }
        state->settle_write(holder->content_digest, false);
    }
}

std::shared_ptr<const std::vector<std::uint8_t>>
DurableSharedPrefixCatalog::load(const Candidate& candidate, Clock::time_point deadline,
                                 const CancellationView& cancellation) {
    const auto state = state_;
    std::shared_ptr<State::LoadJob> job;
    State::Record record;
    {
        std::lock_guard lock(state->mutex);
        const auto found = state->records.find(candidate.content_digest);
        if (found == state->records.end() || found->second.frontier != candidate.frontier) {
            return {};
        }
        record = found->second;
        if (const auto pending = state->pending_loads[candidate.content_digest].lock()) {
            job = pending;
            ++state->values.loads_coalesced;
        }
    }
    if (!job) {
        if (record.bytes > SIZE_MAX / 2U) { return {}; }
        auto reservation = state->reserve(static_cast<std::size_t>(record.bytes) * 2U);
        if (!reservation) { return {}; }
        job                       = std::make_shared<State::LoadJob>();
        job->payload              = std::make_shared<State::LoadJob::Payload>();
        job->payload->reservation = std::move(reservation);
        if (state->options.before_load_registration) { state->options.before_load_registration(); }
        bool enqueue_load = false;
        std::shared_ptr<State::LoadJob> redundant;
        {
            std::lock_guard lock(state->mutex);
            if (const auto pending = state->pending_loads[candidate.content_digest].lock()) {
                redundant = std::move(job);
                job       = pending;
                ++state->values.loads_coalesced;
            } else {
                state->pending_loads[candidate.content_digest] = job;
                enqueue_load                                   = true;
            }
        }
        // Payload destruction releases the staging reservation by locking State::mutex. Never
        // destroy the losing first-load job while that mutex is held.
        redundant.reset();
        if (enqueue_load) {
            const auto queued_job = job;
            try {
                state->push([state, queued_job, record] {
                    const auto started = Clock::now();
                    try {
                        if (state->options.before_payload_read) {
                            state->options.before_payload_read();
                        }
                        std::ifstream input(state->options.directory / record.filename,
                                            std::ios::binary | std::ios::ate);
                        const std::streamsize size = input.tellg();
                        if (!input || size < 0 ||
                            static_cast<std::uint64_t>(size) != record.bytes) {
                            throw std::invalid_argument(
                                "durable shared-prefix record size changed");
                        }
                        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
                        input.seekg(0);
                        input.read(reinterpret_cast<char*>(bytes.data()), size);
                        if (!input) {
                            throw std::invalid_argument(
                                "durable shared-prefix record is truncated");
                        }
                        std::lock_guard job_lock(queued_job->mutex);
                        queued_job->payload->bytes = std::move(bytes);
                    } catch (const std::exception& error) {
                        std::lock_guard job_lock(queued_job->mutex);
                        queued_job->error = error.what();
                    }
                    {
                        std::lock_guard job_lock(queued_job->mutex);
                        queued_job->done = true;
                    }
                    queued_job->cv.notify_all();
                    bool succeeded = false;
                    {
                        std::lock_guard job_lock(queued_job->mutex);
                        succeeded = queued_job->error.empty();
                    }
                    std::lock_guard lock(state->mutex);
                    state->pending_loads.erase(record.digest);
                    if (succeeded) {
                        ++state->values.loads_completed;
                    } else {
                        ++state->values.loads_failed;
                    }
                    state->values.io_nanoseconds += static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
                            .count());
                });
            } catch (...) {
                std::lock_guard lock(state->mutex);
                state->pending_loads.erase(record.digest);
                throw;
            }
        }
    }

    std::unique_lock lock(job->mutex);
    while (!job->done) {
        if (cancellation.requested() || Clock::now() >= deadline) { return {}; }
        job->cv.wait_for(lock, std::chrono::milliseconds(10));
    }
    if (!job->error.empty()) { return {}; }
    return std::shared_ptr<const std::vector<std::uint8_t>>(job->payload, &job->payload->bytes);
}

DurableSharedPrefixRestore DurableSharedPrefixCatalog::restore_matching(
    Engine& engine, const PreparedPrompt& prompt, Clock::time_point deadline,
    const CancellationView& cancellation, const RequestOptions& request_options) {
    DurableSharedPrefixRestore observation;
    const auto candidates = runtime::DurableSharedSnapshotAccess::candidates(engine, prompt);
    std::vector<Candidate> available;
    {
        std::lock_guard lock(state_->mutex);
        available.reserve(candidates.size());
        for (const Candidate& candidate : candidates) {
            const auto found = state_->records.find(candidate.content_digest);
            if (found != state_->records.end() && found->second.frontier == candidate.frontier) {
                available.push_back(candidate);
            }
        }
    }
    for (;;) {
        const auto decision = runtime::DurableSharedSnapshotAccess::decide_recovery(
            engine, prompt, request_options, available);
        if (decision.source == runtime::DurableSharedSnapshotAccess::RecoverySource::Memory) {
            observation.fallback_reason = decision.reason;
            observation.frontier        = decision.frontier;
            observation.warm_available  = true;
            return observation;
        }
        if (decision.source != runtime::DurableSharedSnapshotAccess::RecoverySource::Ssd) {
            if (observation.fallback_reason.empty()) {
                observation.fallback_reason = decision.reason;
            }
            return observation;
        }
        observation.fallback_reason = decision.reason;
        const Candidate candidate   = decision.candidate;
        auto bytes                  = load(candidate, deadline, cancellation);
        if (!bytes) {
            observation.fallback_reason =
                Clock::now() >= deadline ? "ssd-deadline" : "ssd-unavailable";
            available.erase(std::remove(available.begin(), available.end(), candidate),
                            available.end());
            if (cancellation.requested() || Clock::now() >= deadline) { return observation; }
            continue;
        }
        try {
            const auto imported = runtime::DurableSharedSnapshotAccess::import(engine, candidate,
                                                                               bytes, cancellation);
            {
                std::lock_guard lock(state_->mutex);
                state_->values.validation_nanoseconds += imported.validation_nanoseconds;
                state_->values.adoption_nanoseconds += imported.adoption_nanoseconds;
            }
            observation.frontier        = imported.frontier;
            observation.loaded_from_ssd = true;
            return observation;
        } catch (const RequestError& error) {
            if (error.kind() == RequestErrorKind::Cancelled) { throw; }
            observation.fallback_reason = "ssd-adoption-unavailable";
        } catch (const runtime::DurableSharedSnapshotAccess::ValidationError&) {
            {
                std::lock_guard lock(state_->mutex);
                ++state_->values.corrupt_records;
            }
            state_->invalidate(candidate.content_digest);
            observation.fallback_reason = "ssd-validation";
        } catch (const std::invalid_argument&) {
            observation.fallback_reason = "ssd-adoption-unavailable";
        }
        available.erase(std::remove(available.begin(), available.end(), candidate),
                        available.end());
    }
}

void DurableSharedPrefixCatalog::schedule_exports(Engine& engine) {
    const auto state = state_;
    auto exports     = runtime::DurableSharedSnapshotAccess::begin_exports(
        engine, [state](std::size_t bytes) { return state->reserve(bytes); },
        [state](std::uint32_t slot, std::uint64_t owner) {
            const std::string key = std::to_string(slot) + ':' + std::to_string(owner);
            std::lock_guard lock(state->mutex);
            return state->exported_owners.insert(key).second;
        },
        [state](std::uint32_t slot, std::uint64_t owner) {
            const std::string key = std::to_string(slot) + ':' + std::to_string(owner);
            std::lock_guard lock(state->mutex);
            state->exported_owners.erase(key);
        });
    for (auto& pending : exports) {
        Engine* const owner_engine = &engine;
        enqueue(std::move(pending.snapshot),
                [state, owner_engine, slot = pending.slot, owner = pending.owner](bool committed) {
                    const std::string key = std::to_string(slot) + ':' + std::to_string(owner);
                    (void)runtime::DurableSharedSnapshotAccess::settle_export(*owner_engine, slot,
                                                                              owner, committed);
                    std::lock_guard lock(state->mutex);
                    state->exported_owners.erase(key);
                });
    }
}

void DurableSharedPrefixCatalog::observe_hit(const DurableSharedPrefixRestore& restore,
                                             std::uint32_t reused_tokens,
                                             PrefixReusePath path) noexcept {
    if (restore.frontier == 0 || path != PrefixReusePath::SharedStablePrefix ||
        reused_tokens < restore.frontier) {
        return;
    }
    std::lock_guard lock(state_->mutex);
    if (restore.loaded_from_ssd) { ++state_->values.loaded_hits; }
    if (restore.warm_available) { ++state_->values.warm_hits; }
}

void DurableSharedPrefixCatalog::drain() {
    std::unique_lock lock(state_->mutex);
    state_->drained.wait(lock, [&] {
        return state_->jobs.empty() && state_->active_jobs == 0 && state_->reserved_jobs == 0;
    });
}

DurableSharedPrefixCatalogStats DurableSharedPrefixCatalog::stats() const noexcept {
    std::lock_guard lock(state_->mutex);
    auto result                  = state_->values;
    result.manifest_records      = static_cast<std::uint32_t>(state_->records.size());
    result.manifest_bytes        = state_->catalog_bytes;
    result.queued_jobs           = static_cast<std::uint32_t>(state_->jobs.size());
    result.active_jobs           = state_->active_jobs;
    result.staging_bytes         = state_->reserved_bytes;
    result.pending_export_claims = static_cast<std::uint32_t>(state_->exported_owners.size());
    result.unpublished_records   = state_->unpublished_records;
    result.unpublished_bytes     = state_->unpublished_bytes;
    return result;
}

} // namespace ninfer::serve
