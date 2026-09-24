#include "serve/context_monitor.h"
#include "serve/http_server.h"
#include <chrono>
#include <thread>

#include <iostream>
#include <stdexcept>

namespace ninfer::serve::testing {
struct HttpServerTestAccess {
    static httplib::Server& transport(HttpServer& server) { return server.server_; }
};
} // namespace ninfer::serve::testing

namespace {
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
} // namespace

void check_http_routes() {
    ninfer::serve::ServeOptions options;
    options.api_key = "monitor-test-key";
    ninfer::serve::HttpServer server(options, nullptr);
    auto& transport = ninfer::serve::testing::HttpServerTestAccess::transport(server);
    const int port  = transport.bind_to_any_port("127.0.0.1");
    require(port > 0, "bind test HTTP transport");
    std::thread listener([&] { transport.listen_after_bind(); });

    struct Cleanup {
        httplib::Server& server;
        std::thread& thread;

        ~Cleanup() {
            server.stop();
            thread.join();
        }
    } cleanup{transport, listener};

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!transport.is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(transport.is_running(), "start test HTTP transport");
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(2, 0);
    const auto page = client.Get("/monitor");
    require(page && page->status == 200 &&
                page->get_header_value("Content-Type").starts_with("text/html") &&
                page->body.starts_with("<!doctype html>"),
            "public dashboard HTML");
    const auto denied = client.Get("/monitor/data");
    require(denied && denied->status == 401, "monitor data requires authentication");
    const auto data =
        client.Get("/monitor/data", httplib::Headers{{"Authorization", "Bearer monitor-test-key"}});
    require(data && data->status == 200 && data->get_header_value("Cache-Control") == "no-store",
            "authenticated monitor data");
    const auto json = nlohmann::json::parse(data->body);
    require(!json.at("ready") && json.at("slots").empty(), "HTTP loading snapshot");
    const auto slots =
        client.Get("/slots", httplib::Headers{{"Authorization", "Bearer monitor-test-key"}});
    require(slots && slots->status == 200, "existing slots endpoint");
    const auto slot_json = nlohmann::json::parse(slots->body);
    require(slot_json.is_array() && slot_json.at(0).contains("is_processing") &&
                !slot_json.at(0).contains("memory"),
            "existing slots shape preserved");
}

int main() {
    using namespace ninfer;
    try {
        check_http_routes();
        const auto empty = serve::context_monitor_json({}, false, 8);
        require(!empty.at("ready") && empty.at("slots").empty(), "loading state");
        require(empty.at("max_concurrency") == 8, "configured concurrency");
        require(empty.at("generation").at("decode_seconds") == 0, "initial decode time");

        ContextSnapshot snapshot;
        snapshot.memory.device_main_kv_capacity_pages = 100;
        snapshot.memory.device_main_kv_occupied_pages = 60;
        snapshot.memory.device_main_kv_reserved_pages = 20;
        snapshot.memory.device_main_kv_page_bytes     = 4096;
        snapshot.memory.kv_capacity                   = 12800;
        snapshot.memory.host_kv_capacity_bytes        = 1048576;
        snapshot.memory.host_kv_occupied_bytes        = 262144;
        snapshot.runtime.committed_decode_tokens      = 120;
        snapshot.runtime.decode_seconds_total         = 2;
        snapshot.runtime.running_requests             = 2;
        snapshot.runtime.waiting_requests             = 1;
        snapshot.slots.resize(3);
        auto& running                       = snapshot.slots[0];
        running.processing                  = true;
        running.request_id                  = 7;
        running.lane                        = 0;
        running.mapped_tokens               = 256;
        running.reserved_tokens             = 512;
        running.memory                      = {.device_state_bytes = 100, .device_kv_bytes = 8192};
        auto& suspended                     = snapshot.slots[1];
        suspended.processing                = true;
        suspended.request_id                = 8;
        suspended.lane                      = 1;
        suspended.suspended                 = true;
        suspended.memory.device_state_bytes = 100;
        suspended.host_backing_bytes        = 16384;
        suspended.host_backing_used_bytes   = 8192;
        auto& retained                      = snapshot.slots[2];
        retained.retained                   = true;
        retained.session_digest             = "0123456789abcdef";
        // Two checkpoints reference the same KV. Their rows must not be summed into pool usage.
        retained.checkpoints = {
            {128, "a", {.device_state_bytes = 100, .device_kv_bytes = 4096}},
            {256, "b", {.host_state_bytes = 256, .device_kv_bytes = 8192, .host_kv_bytes = 4096}}};
        const auto result = serve::context_monitor_json(snapshot, true, 2);
        require(result.at("pools").at("main_occupied_pages") == 60, "unique pool usage");
        require(result.at("pools").at("main_reserved_pages") == 20, "reservation separated");
        const auto& slots = result.at("slots");
        require(slots.at(0).at("request_id") == 7 && slots.at(1).at("request_id") == 8,
                "parallel request identities");
        require(slots.at(1).at("suspended") &&
                    slots.at(1).at("memory").at("device_state_bytes") == 100 &&
                    slots.at(1).at("memory").at("device_kv_bytes") == 0,
                "suspension retains state but evacuates KV");
        require(slots.at(1).at("host_backing_bytes") == 16384 &&
                    slots.at(1).at("host_backing_used_bytes") == 8192,
                "RAM allocated versus copied");
        const auto& checkpoint = slots.at(2).at("checkpoints").at(1);
        require(checkpoint.at("frontier") == 256 &&
                    checkpoint.at("memory").at("host_state_bytes") == 256 &&
                    checkpoint.at("memory").at("device_kv_bytes") == 8192 &&
                    checkpoint.at("memory").at("host_kv_bytes") == 4096,
                "mixed State/KV residency with both KV replicas");
        require(result.at("generation").at("decode_tokens") == 120 &&
                    result.at("generation").at("decode_seconds") == 2,
                "live execution counters, not completion averages");
        require(serve::context_monitor_page().starts_with("<!doctype html>"), "embedded HTML");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cout << "OK\n";
}
