#include "serve/context_monitor.h"
#include "context_monitor_page.h"

namespace ninfer::serve {
std::string_view context_monitor_page() { return kContextMonitorPage; }

nlohmann::json context_memory_json(const ContextMemory& m) {
    return {{"device_state_bytes", m.device_state_bytes},
            {"host_state_bytes", m.host_state_bytes},
            {"device_kv_bytes", m.device_kv_bytes},
            {"host_kv_bytes", m.host_kv_bytes}};
}

nlohmann::json context_monitor_json(const ContextSnapshot& snapshot, bool ready,
                                    std::uint32_t concurrency) {
    const auto& m = snapshot.memory;
    const auto& r = snapshot.runtime;
    auto slots    = nlohmann::json::array();
    for (std::size_t i = 0; i < snapshot.slots.size(); ++i) {
        const auto& s    = snapshot.slots[i];
        auto checkpoints = nlohmann::json::array();
        for (const auto& c : s.checkpoints) {
            checkpoints.push_back({{"frontier", c.frontier},
                                   {"session_digest", c.session_digest},
                                   {"memory", context_memory_json(c.memory)}});
        }
        slots.push_back({{"id", i},
                         {"processing", s.processing},
                         {"retained", s.retained},
                         {"request_id", s.request_id},
                         {"lane", s.lane},
                         {"prefilling", s.prefilling},
                         {"suspended", s.suspended},
                         {"prompt_tokens", s.prompt_tokens},
                         {"cached_tokens", s.cached_tokens},
                         {"mapped_tokens", s.mapped_tokens},
                         {"reserved_tokens", s.reserved_tokens},
                         {"host_backing_bytes", s.host_backing_bytes},
                         {"host_backing_used_bytes", s.host_backing_used_bytes},
                         {"memory", context_memory_json(s.memory)},
                         {"session_digest", s.session_digest},
                         {"checkpoints", checkpoints}});
    }
    return {{"ready", ready},
            {"max_concurrency", concurrency},
            {"slots", slots},
            {"pools",
             {{"main_capacity_pages", m.device_main_kv_capacity_pages},
              {"main_occupied_pages", m.device_main_kv_occupied_pages},
              {"main_reserved_pages", m.device_main_kv_reserved_pages},
              {"main_page_bytes", m.device_main_kv_page_bytes},
              {"main_capacity_tokens", m.kv_capacity},
              {"backend_capacity_pages", m.device_backend_kv_capacity_pages},
              {"backend_occupied_pages", m.device_backend_kv_occupied_pages},
              {"backend_page_bytes", m.device_backend_kv_page_bytes},
              {"device_state_capacity", m.device_state_capacity_slots},
              {"device_state_used", r.device_state_occupied_slots},
              {"host_state_capacity", m.host_state_capacity_slots},
              {"host_state_used", m.host_state_occupied_slots},
              {"host_state_image_bytes", m.checkpoint_state_image_bytes},
              {"host_kv_capacity_bytes", m.host_kv_capacity_bytes},
              {"host_kv_occupied_bytes", m.host_kv_occupied_bytes}}},
            {"generation",
             {{"decode_tokens", r.committed_decode_tokens},
              {"decode_seconds", r.decode_seconds_total},
              {"running", r.running_requests},
              {"waiting", r.waiting_requests},
              {"prefilling", r.prefilling_requests},
              {"suspensions", r.generation_suspensions},
              {"resumes", r.generation_resumes},
              {"to_ram_bytes", r.state_d2h_bytes + r.main_kv_d2h_bytes + r.backend_kv_d2h_bytes},
              {"to_vram_bytes", r.state_h2d_bytes + r.main_kv_h2d_bytes + r.backend_kv_h2d_bytes},
              {"suspension_transfer_bytes", r.generation_transfer_bytes}}}};
}
} // namespace ninfer::serve
