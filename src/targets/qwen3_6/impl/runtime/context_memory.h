#pragma once

#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include <ninfer/types.h>
#include <algorithm>
#include <span>

namespace ninfer::targets::qwen3_6::detail {
// Physical reporting only: no transfers, reservations, or retention policy. A replica becomes
// visible only when its store publishes it; multiple references inside one row count once.
inline ContextMemory resident_state_memory(const StateImageStore& store,
                                           const StateImageDevicePool& physical,
                                           std::span<const StateImageHandle> states) {
    ContextMemory out;
    const auto& layout = physical.host_layout();
    // Device components have no per-image Host alignment padding.
    const auto device_state_bytes = state_image_transfer_work(layout).payload_bytes;
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (!store.valid(states[i]) ||
            std::find(states.begin(), states.begin() + i, states[i]) != states.begin() + i) {
            continue;
        }
        const auto residency = store.residency(states[i]);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            out.device_state_bytes += device_state_bytes;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            out.host_state_bytes += layout.image_bytes;
        }
    }
    return out;
}

inline void add_resident_kv_memory(ContextMemory& out, const KVAddressSpaceStore& addresses,
                                   const LogicalKVPageStore& pages, const HostKVExtentStore* host,
                                   KVAddressSpaceHandle address, std::uint32_t tokens) {
    const auto& pool       = pages.physical_pool();
    std::size_t page_bytes = 0;
    for (std::size_t i = 0; i < pool.plane_count(); ++i) {
        page_bytes += pool.plane(i).bytes() / pool.capacity_pages();
    }
    const auto count = std::min(addresses.mapped_pages(address),
                                (tokens == 0 ? 0U : 1U + (tokens - 1U) / kPagedKVPageSize));
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto page = addresses.logical_page(address, i);
        if (pages.device_resident(page)) { out.device_kv_bytes += page_bytes; }
        if (pages.host_resident(page)) {
            out.host_kv_bytes += host->view(pages.host_replica(page).extent).layout().page_stride;
        }
    }
}
} // namespace ninfer::targets::qwen3_6::detail
