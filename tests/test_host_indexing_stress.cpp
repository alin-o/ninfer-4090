// CPU-only tests for the host-side index and bookkeeping primitives that the qwen3.6 decode, MTP
// and DFlash programs rely on:
//
//   * host KV page layout and transfer-work planning, against hand-computed layouts
//   * Tensor slice/permute/view/contiguity, against element values read through the logical index
//   * the draft-tensor contract used by the MTP round: dim 0 is the fast axis, so one proposal
//     step of a {batch, drafts} tensor is a contiguous {batch} view
//   * cyclic KV cache layout planning, against hand-computed layouts
//   * the pinned Host KV arena: constructor contracts, views, stale handles, fragmentation and
//     20,000 randomized operations checked against an independent shadow model
//
// The arena sections need pinned Host memory and report a skip note without it.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "core/transfer_work.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace {

namespace ni = ninfer;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Function>
void expect_throw(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    expect(false, message);
}

void note(const std::string& message) { std::cerr << "NOTE: " << message << '\n'; }

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

std::uint32_t rnd(std::uint64_t& state, std::uint32_t modulus) {
    return static_cast<std::uint32_t>(splitmix64(state) % modulus);
}

constexpr std::size_t kAlign = 256;

std::size_t align256(std::size_t value) { return (value + (kAlign - 1)) & ~(kAlign - 1); }

// ---------------------------------------------------------------------------
// Section 1: host KV page layout planning.
// ---------------------------------------------------------------------------

void test_page_layout_planning() {
    // Hand-computed anchors: 64-token page, PageMajor, BF16 K 16x4 + FP16 V 16x4.
    //   plane 0: head = 16*64*2 = 2048, page = 4*2048 = 8192, offset 0
    //   plane 1: head = 2048, page = 8192, offset 8192
    //   stride = align256(16384) = 16384
    const ni::KVPageGeometry anchor0{
        64, ni::PagedKVPlaneOrder::PageMajor,
        {{ni::DType::BF16, 16, 4}, {ni::DType::FP16, 16, 4}}};
    const ni::HostKVPageLayout a0 = ni::plan_host_kv_page_layout(anchor0);
    expect(a0.planes.size() == 2, "anchor0 plane count");
    expect(a0.planes[0].offset == 0 && a0.planes[0].page_payload_bytes == 8192 &&
               a0.planes[0].head_payload_bytes == 2048,
           "anchor0 plane0");
    expect(a0.planes[1].offset == 8192 && a0.planes[1].page_payload_bytes == 8192,
           "anchor0 plane1");
    expect(a0.page_stride == 16384, "anchor0 stride");

    // 16-token page, HeadMajor: FP32 8x2 (1024B, offset 0) + U8 16x4 (1024B,
    // offset 1024); stride align256(2048) = 2048.
    const ni::KVPageGeometry anchor1{
        16, ni::PagedKVPlaneOrder::HeadMajor,
        {{ni::DType::FP32, 8, 2}, {ni::DType::U8, 16, 4}}};
    const ni::HostKVPageLayout a1 = ni::plan_host_kv_page_layout(anchor1);
    expect(a1.planes[0].offset == 0 && a1.planes[0].page_payload_bytes == 1024,
           "anchor1 plane0");
    expect(a1.planes[1].offset == 1024 && a1.planes[1].page_payload_bytes == 1024,
           "anchor1 plane1");
    expect(a1.page_stride == 2048, "anchor1 stride");

    // 4-token page, single I32 plane: 16B payload, stride align256(16) = 256.
    const ni::KVPageGeometry anchor2{
        4, ni::PagedKVPlaneOrder::PageMajor, {{ni::DType::I32, 1, 1}}};
    const ni::HostKVPageLayout a2 = ni::plan_host_kv_page_layout(anchor2);
    expect(a2.planes[0].page_payload_bytes == 16 && a2.page_stride == 256,
           "anchor2 stride");

    // Deterministic rejection cases.
    ni::KVPageGeometry zero_tokens   = anchor2;
    zero_tokens.page_tokens          = 0;
    expect_throw([&] { (void)ni::plan_host_kv_page_layout(zero_tokens); },
                 "zero page_tokens rejected");
    ni::KVPageGeometry no_planes     = anchor2;
    no_planes.planes                .clear();
    expect_throw([&] { (void)ni::plan_host_kv_page_layout(no_planes); },
                 "empty plane list rejected");
    ni::KVPageGeometry zero_leading  = anchor2;
    zero_leading.planes[0].leading_extent = 0;
    expect_throw([&] { (void)ni::plan_host_kv_page_layout(zero_leading); },
                 "zero leading extent rejected");
    ni::KVPageGeometry negative_head = anchor2;
    negative_head.planes[0].head_extent = -1;
    expect_throw([&] { (void)ni::plan_host_kv_page_layout(negative_head); },
                 "negative head extent rejected");

    // Overflow: UINT32_MAX tokens with a wide FP32 plane pushes the page
    // payload past size_t on 64-bit.
    ni::KVPageGeometry overflow{
        std::numeric_limits<std::uint32_t>::max(), ni::PagedKVPlaneOrder::PageMajor,
        {{ni::DType::FP32, std::numeric_limits<std::int32_t>::max(),
          std::numeric_limits<std::int32_t>::max()}}};
    expect_throw([&] { (void)ni::plan_host_kv_page_layout(overflow); },
                 "page payload overflow rejected");
}

// ---------------------------------------------------------------------------
// Section 2: host transfer and device copy work planning.
// ---------------------------------------------------------------------------

void test_transfer_planning() {
    const ni::KVPageGeometry geom0{
        64, ni::PagedKVPlaneOrder::PageMajor,
        {{ni::DType::BF16, 16, 4}, {ni::DType::FP16, 16, 4}}};
    const ni::KVPageGeometry geom1{
        16, ni::PagedKVPlaneOrder::HeadMajor,
        {{ni::DType::FP32, 8, 2}, {ni::DType::U8, 16, 4}}};
    const ni::HostKVPageLayout page0 = ni::plan_host_kv_page_layout(geom0);
    const ni::HostKVPageLayout page1 = ni::plan_host_kv_page_layout(geom1);

    // pages == 0 short-circuits with zero work.
    const ni::TransferWork z0 = ni::plan_host_kv_transfer_work(page0, 0, 0);
    expect(z0.payload_bytes == 0 && z0.copy_operations == 0, "zero-page host transfer");
    const ni::TransferWork zd = ni::plan_device_kv_copy_work(page0, 0);
    expect(zd.payload_bytes == 0 && zd.copy_operations == 0, "zero-page device copy");

    // PageMajor: 2 planes -> 2 copies per run.  Payload 16384/page.
    const ni::TransferWork h0 = ni::plan_host_kv_transfer_work(page0, 10, 3);
    expect(h0.payload_bytes == 163840 && h0.copy_operations == 6, "PageMajor transfer math");
    // HeadMajor: ops per run = sum(head_extent) = 2 + 4 = 6.
    const ni::TransferWork h1 = ni::plan_host_kv_transfer_work(page1, 10, 3);
    expect(h1.payload_bytes == 20480 && h1.copy_operations == 18, "HeadMajor transfer math");
    const ni::TransferWork d0 = ni::plan_device_kv_copy_work(page0, 1000);
    expect(d0.payload_bytes == 16384000 && d0.copy_operations == 2000, "device copy math");

    // Rejection battery.
    expect_throw([&] { (void)ni::plan_host_kv_transfer_work(page0, 8, 0); },
                 "zero runs rejected");
    expect_throw([&] { (void)ni::plan_host_kv_transfer_work(page0, 8, 9); },
                 "runs > pages rejected");
    {
        ni::HostKVPageLayout tampered = page0;
        tampered.planes.clear();
        expect_throw([&] { (void)ni::plan_host_kv_transfer_work(tampered, 8, 1); },
                     "empty planes rejected");
        expect_throw([&] { (void)ni::plan_device_kv_copy_work(tampered, 8); },
                     "empty planes rejected (device)");
    }
    {
        ni::HostKVPageLayout mismatched = page0;
        mismatched.geometry.planes.pop_back();
        expect_throw([&] { (void)ni::plan_host_kv_transfer_work(mismatched, 8, 1); },
                     "plane/geometry size mismatch rejected (host)");
        expect_throw([&] { (void)ni::plan_device_kv_copy_work(mismatched, 8); },
                     "plane/geometry size mismatch rejected (device)");
    }
}

// ---------------------------------------------------------------------------
// Section 3: Tensor index arithmetic (slice / permute / view / contiguity).
//
// Independent byte-stride reference model: the production Tensor stores
// element counts ne[4] and byte strides nb[4] for a fixed 4-slot layout.
// The model below re-derives the same state transitions with its own
// arithmetic so a divergence in any stride bookkeeping step is visible.
// ---------------------------------------------------------------------------

struct TensorModel {
    std::size_t data_off = 0; // bytes from the base buffer
    std::size_t elem     = 1; // dtype_size
    std::int32_t ne[4]   = {1, 1, 1, 1};
    std::int64_t nb[4]   = {1, 1, 1, 1};

    void contiguous() {
        nb[0] = static_cast<std::int64_t>(elem);
        for (int i = 1; i < 4; ++i) { nb[i] = nb[i - 1] * ne[i - 1]; }
    }

    // Mirrors Tensor::is_contiguous: unit strides for ne==1 dims are ignored.
    bool is_contiguous() const {
        std::int64_t expected = static_cast<std::int64_t>(elem);
        for (int i = 0; i < 4; ++i) {
            if (ne[i] <= 0) { return false; }
            if (ne[i] != 1 && nb[i] != expected) { return false; }
            if (i < 3 && ne[i] != 1) {
                if (expected > std::numeric_limits<std::int64_t>::max() / ne[i]) {
                    return false;
                }
                expected *= ne[i];
            }
        }
        return true;
    }

    std::int64_t numel() const {
        std::int64_t n = 1;
        for (int i = 0; i < 4; ++i) { n *= ne[i]; }
        return n;
    }

    TensorModel slice(int dim, std::int32_t start) const {
        TensorModel m  = *this;
        m.data_off    += static_cast<std::size_t>(start) * static_cast<std::size_t>(nb[dim]);
        m.ne[dim]     = 1;
        return m;
    }
    // The production slice keeps len elements; the model keeps the same
    // stride and only needs ne updated.
    TensorModel slice(int dim, std::int32_t start, std::int32_t len) const {
        TensorModel m  = *this;
        m.data_off    += static_cast<std::size_t>(start) * static_cast<std::size_t>(nb[dim]);
        m.ne[dim]     = len;
        return m;
    }

    TensorModel permute(const std::array<int, 4>& order) const {
        TensorModel m;
        m.data_off = data_off;
        m.elem     = elem;
        for (int i = 0; i < 4; ++i) {
            m.ne[i] = ne[order[i]];
            m.nb[i] = nb[order[i]];
        }
        return m;
    }

    TensorModel view(std::initializer_list<std::int32_t> shape) const {
        TensorModel m;
        m.data_off = data_off;
        m.elem     = elem;
        std::size_t i = 0;
        for (int d : shape) { m.ne[i++] = d; }
        m.contiguous();
        return m;
    }
};

std::size_t pattern32(std::size_t i) {
    return (i * 2654435761ULL) ^ (i >> 13) ^ 0x9e3779b9ULL;
}

std::vector<std::int64_t> divisors_of(std::int64_t n) {
    std::vector<std::int64_t> out;
    for (std::int64_t i = 1; i * i <= n; ++i) {
        if (n % i == 0) {
            out.push_back(i);
            if (i != n / i) { out.push_back(n / i); }
        }
    }
    return out;
}

std::array<std::int32_t, 4> random_view_shape(std::int64_t n, int rank,
                                              std::uint64_t& state) {
    std::array<std::int32_t, 4> shape{1, 1, 1, 1};
    std::int64_t remaining = n;
    for (int i = 0; i < rank; ++i) {
        if (i == rank - 1) {
            shape[i] = static_cast<std::int32_t>(remaining);
            break;
        }
        const auto divs = divisors_of(remaining);
        shape[i]       = static_cast<std::int32_t>(divs[rnd(state, static_cast<std::uint32_t>(divs.size()))]);
        remaining     /= shape[i];
    }
    return shape;
}

ni::Tensor view_shape(const ni::Tensor& t, int rank, const std::array<std::int32_t, 4>& s) {
    switch (rank) {
    case 1: return t.view({s[0]});
    case 2: return t.view({s[0], s[1]});
    case 3: return t.view({s[0], s[1], s[2]});
    default: return t.view({s[0], s[1], s[2], s[3]});
    }
}

struct ChainState {
    const std::int32_t* base = nullptr; // pattern-filled I32 buffer
    bool values_checkable   = true;
};

// Structural comparison plus sampled element-value verification against the
// pattern buffer.  Pointer identity of `data` pins the accumulated slice
// offsets against the chain's original storage base (valid for both I32 and
// structure-only BF16 chains); stride equality pins every permute/view
// transition.  `base` is the typed pattern buffer and may be null when only
// structure is checked.
void check_tensor_state(const ni::Tensor& real, const TensorModel& want,
                        const void* buffer, const std::int32_t* base,
                        bool values_checkable, std::uint64_t& state) {
    for (int i = 0; i < 4; ++i) {
        expect(real.ne[i] == want.ne[i], "tensor element count mismatch after op chain");
        expect(real.nb[i] == want.nb[i], "tensor byte stride mismatch after op chain");
    }
    expect(reinterpret_cast<const std::byte*>(real.data) ==
               reinterpret_cast<const std::byte*>(buffer) + want.data_off,
           "tensor data pointer mismatch after op chain");
    expect(real.is_contiguous() == want.is_contiguous(), "contiguity verdict mismatch");
    if (!values_checkable) { return; }
    const std::int64_t total = want.numel();
    if (total <= 0 || total > 16384) { return; }
    for (int sample = 0; sample < 16; ++sample) {
        std::int32_t coord[4] = {0, 0, 0, 0};
        std::int64_t element  = 0;
        std::int64_t address  = 0;
        for (int i = 0; i < 4; ++i) {
            if (want.ne[i] > 1) { coord[i] = static_cast<std::int32_t>(rnd(state, want.ne[i])); }
            element += coord[i] * want.nb[i];
            address += coord[i] * real.nb[i];
        }
        const std::int32_t expected =
            base[static_cast<std::size_t>(want.data_off + element) / 4];
        const std::int32_t actual =
            reinterpret_cast<const std::int32_t*>(reinterpret_cast<const std::byte*>(real.data) +
                                                   address)[0];
        expect(actual == expected, "sampled element value mismatch after op chain");
    }
}

void tensor_random_chains() {
    std::uint64_t state = 0x5EED0003ULL;
    for (int c = 0; c < 300; ++c) {
        const int rank      = 1 + static_cast<int>(rnd(state, 4));
        const bool bf16     = c % 10 == 0; // structure-only chains
        std::array<std::int32_t, 4> dims{1, 1, 1, 1};
        for (int i = 0; i < rank; ++i) { dims[i] = 1 + static_cast<std::int32_t>(rnd(state, 8)); }
        std::int64_t numel = 1;
        for (int i = 0; i < 4; ++i) { numel *= dims[i]; }

        std::vector<std::int32_t> storage;
        std::int32_t* base       = nullptr;
        void* buffer             = nullptr;
        if (bf16) {
            std::vector<unsigned char> bytes(static_cast<std::size_t>(numel) * 2, 0xAB);
            buffer = bytes.data();
            (void)bytes;
        } else {
            storage.resize(static_cast<std::size_t>(numel));
            for (std::size_t i = 0; i < storage.size(); ++i) { storage[i] = static_cast<std::int32_t>(pattern32(i)); }
            base   = storage.data();
            buffer = base;
        }
        ni::Tensor real(buffer, bf16 ? ni::DType::BF16 : ni::DType::I32,
                        {dims[0], dims[1], dims[2], dims[3]});
        TensorModel model;
        model.elem   = bf16 ? 2 : 4;
        for (int i = 0; i < 4; ++i) { model.ne[i] = dims[i]; }
        model.contiguous();
        check_tensor_state(real, model, buffer, base, !bf16, state);

        const int steps = 8 + static_cast<int>(rnd(state, 18));
        for (int s = 0; s < steps; ++s) {
            const int kind = static_cast<int>(rnd(state, 20));
            if (kind < 9) {
                // Valid slice on a random dimension.
                const int dim      = static_cast<int>(rnd(state, 4));
                const std::int32_t start =
                    model.ne[dim] > 1 ? static_cast<std::int32_t>(rnd(state, model.ne[dim])) : 0;
                const std::int32_t len =
                    1 + static_cast<std::int32_t>(rnd(state, model.ne[dim] - start));
                real  = real.slice(dim, start, len);
                model = model.slice(dim, start, len);
                check_tensor_state(real, model, buffer, base, !bf16, state);
            } else if (kind == 9) {
                // Out-of-range slice probes must throw and leave state intact.
                const int probe = static_cast<int>(rnd(state, 4));
                bool threw      = false;
                if (probe == 0) {
                    try { (void)real.slice(4, 0, 1); } catch (const std::exception&) { threw = true; }
                } else if (probe == 1) {
                    const int dim = static_cast<int>(rnd(state, 4));
                    try { (void)real.slice(dim, model.ne[dim] + 1, 1); }
                    catch (const std::exception&) { threw = true; }
                } else if (probe == 2) {
                    const int dim = static_cast<int>(rnd(state, 4));
                    try { (void)real.slice(dim, 0, 0); } catch (const std::exception&) { threw = true; }
                } else {
                    const int dim = static_cast<int>(rnd(state, 4));
                    try { (void)real.slice(dim, 0, model.ne[dim] + 1); }
                    catch (const std::exception&) { threw = true; }
                }
                expect(threw, "out-of-range slice accepted");
                check_tensor_state(real, model, buffer, base, !bf16, state);
            } else if (kind < 14) {
                // Valid permute: random bijection over the four slots.
                std::array<int, 4> order{0, 1, 2, 3};
                for (int i = 3; i > 0; --i) {
                    const int j = static_cast<int>(rnd(state, static_cast<std::uint32_t>(i) + 1));
                    std::swap(order[i], order[j]);
                }
                real  = real.permute({order[0], order[1], order[2], order[3]});
                model = model.permute(order);
                check_tensor_state(real, model, buffer, base, !bf16, state);
            } else if (kind < 15) {
                // Malformed permutations (duplicate or missing dimension).
                const std::array<int, 4> bad_a{1, 1, 2, 3};
                const std::array<int, 4> bad_b{0, 1, 2, 2};
                const std::array<int, 4>& bad = rnd(state, 2) == 0 ? bad_a : bad_b;
                bool threw = false;
                try { (void)real.permute({bad[0], bad[1], bad[2], bad[3]}); }
                catch (const std::exception&) { threw = true; }
                expect(threw, "invalid permutation accepted");
                check_tensor_state(real, model, buffer, base, !bf16, state);
            } else if (kind < 18) {
                // View: legal exactly when the tensor is contiguous.
                const bool contiguous = model.is_contiguous();
                if (contiguous) {
                    const int rank_v = 1 + static_cast<int>(rnd(state, 4));
                    const auto shape = random_view_shape(model.numel(), rank_v, state);
                    real  = view_shape(real, rank_v, shape);
                    model = model.view({shape[0], shape[1], shape[2], shape[3]});
                    check_tensor_state(real, model, buffer, base, !bf16, state);
                } else {
                    bool threw = false;
                    try {
                        const auto shape = random_view_shape(model.numel(), 2, state);
                        (void)view_shape(real, 2, shape);
                    } catch (const std::exception&) { threw = true; }
                    expect(threw, "view accepted a non-contiguous tensor");
                    check_tensor_state(real, model, buffer, base, !bf16, state);
                }
            } else {
                // Element-count-mismatched view: always rejected.
                bool threw = false;
                try { (void)real.view({2, 3, 5}); } catch (const std::exception&) { threw = true; }
                // numel may legitimately be 30; in that rare case the view is
                // legal, so only require rejection when counts differ.
                if (!threw && model.numel() != 30) { expect(false, "numel-mismatched view accepted"); }
                check_tensor_state(real, model, buffer, base, !bf16, state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Section 4: MTP draft-tensor contract.
//
// round_state.cpp binds the MTP egress `next_drafts` buffer as a fresh
// contiguous {batch, drafts} tensor and the ingress `current_drafts` as
// {drafts, batch}.  Tensor::set_contiguous_strides makes dim 0 the FAST axis,
// so a {batch, drafts} tensor stores next_drafts[b][s] at flat offset
// s * batch + b (batch is the contiguous axis) and {drafts, batch} stores
// current_drafts[s][b] at s + b * drafts (drafts is the contiguous axis).
//
// mtp_impl.h consumes next_drafts two ways:
//   next_drafts.slice(0, 0, batch)                    // dim-0 batch slice (line ~118)
//   next_drafts.slice(1, step, 1).view({batch_size})  // per-step column (lines ~167/171/173)
//
// Because batch is the fast axis of {batch, drafts}, the dim-1 (draft) slice
// slice(1, step, 1) keeps a unit stride on the retained batch axis and is
// genuinely contiguous; the .view({batch_size}) therefore reads
// next_drafts[b][step] for every request b -- the correct per-request column,
// not a misread.  (A {drafts, batch} egress layout would instead make that
// slice non-contiguous, which is exactly why the egress buffer is stored
// batch-contiguous.)  A single-request row, by contrast, is a {1, drafts}
// sub-slice whose kept draft-axis stride (drafts * elem) breaks contiguity
// for batch > 1, so a {drafts} view on it throws.  On the ingress side the
// host hands the kernel the whole {drafts, batch} tensor
// (current_drafts.slice(1, 0, batch)); the kernel indexes [step][b] through
// the strides.  The checks below pin these host-observable contracts:
// contiguity verdicts, the accumulated slice offsets, and the per-element
// values read back through the documented stride arithmetic.
// ---------------------------------------------------------------------------

void test_mtp_draft_tensor_contracts() {
    // Logical [r][c] of a 2-D tensor via successive unit slices.  Exercises the
    // exact host-side slice-offset arithmetic (start * nb[dim]) the kernels use.
    const auto read_elem = [](const ni::Tensor& t, std::int32_t r, std::int32_t c) {
        return *static_cast<const std::int32_t*>(t.slice(0, r, 1).slice(1, c, 1).data);
    };

    // F1: egress next_drafts, {batch=4, drafts=5}, value = b*100 + s.
    {
        const int B = 4, K = 5;
        std::vector<std::int32_t> buf(B * K);
        // Column-major {B,K}: next_drafts[b][s] sits at s*B + b.
        for (int s = 0; s < K; ++s) {
            for (int b = 0; b < B; ++b) { buf[s * B + b] = b * 100 + s; }
        }
        ni::Tensor base(buf.data(), ni::DType::I32, {B, K});
        TensorModel model;
        model.elem   = 4;
        model.ne[0]  = B;
        model.ne[1]  = K;
        model.contiguous();
        std::uint64_t sstate = 0x5EED0004ULL;
        check_tensor_state(base, model, buf.data(), buf.data(), true, sstate);
        for (int s = 0; s < K; ++s) {
            for (int b = 0; b < B; ++b) {
                expect(read_elem(base, b, s) == b * 100 + s,
                       "egress {batch,drafts} layout maps [b][s] correctly");
            }
        }

        // line ~118: dim-0 batch slice (full) stays contiguous.
        ni::Tensor batched = base.slice(0, 0, B);
        expect(batched.is_contiguous(), "{batch,drafts} dim-0 slice stays contiguous");

        // lines ~167/171/173: per-step dim-1 draft column slice + view({batch}).
        for (int step = 0; step < K; ++step) {
            ni::Tensor column = batched.slice(1, step, 1);
            expect(column.is_contiguous(),
                   "{batch,drafts} per-step draft column is contiguous (batch is fast)");
            ni::Tensor viewed = column.view({B}); // contiguous, so no throw
            const auto* v     = static_cast<const std::int32_t*>(viewed.data);
            for (int b = 0; b < B; ++b) {
                const std::int32_t want = b * 100 + step; // next_drafts[b][step]
                expect(v[b] == want, "egress draft column view reads the correct column");
                expect(v[b] == read_elem(base, b, step),
                       "egress draft column view matches the layout's [b][step]");
            }
            // The 2-D {1, batch} spelling (lines ~175-181) reads the same column.
            ni::Tensor viewed_2d = column.view({1, B});
            const auto* v2       = static_cast<const std::int32_t*>(viewed_2d.data);
            for (int b = 0; b < B; ++b) {
                expect(v2[b] == b * 100 + step, "egress {1,batch} column view is correct");
            }
        }

        // A single-request row of a multi-batch tensor is non-contiguous for
        // batch > 1 (kept draft-axis stride is K*elem), so a {K} view throws.
        // Production always reads the full batch, never one row; this pins the
        // layout's one-sided property.
        const ni::Tensor row0 = base.slice(0, 0, 1);
        expect(!row0.is_contiguous(),
               "single-request {1,drafts} row is non-contiguous for batch>1");
        expect_throw([&] { (void)row0.view({K}); },
                     "row view over multi-batch {batch,drafts} storage throws");

        // Full-page {batch,1,drafts} view is legal and preserves logical order.
        const ni::Tensor flat = base.view({B, 1, K});
        expect(flat.is_contiguous(), "{batch,1,drafts} view of a contiguous base is legal");
        expect(flat.numel() == base.numel(), "{batch,1,drafts} view preserves element count");
        expect(*static_cast<const std::int32_t*>(flat.slice(0, 0, 1).slice(2, 2, 1).data) == 2,
               "flattened [0][0][2] equals base [0][2]");
        expect(*static_cast<const std::int32_t*>(flat.slice(0, B - 1, 1).slice(2, K - 1, 1).data) ==
                   (B - 1) * 100 + (K - 1),
               "flattened [B-1][0][K-1] equals base [B-1][K-1]");
    }

    // F2: ingress current_drafts, {drafts=5, batch=4}, value = s*100 + b.
    {
        const int K = 5, B = 4;
        std::vector<std::int32_t> buf(K * B);
        // Column-major {K,B}: current_drafts[s][b] sits at s + b*K (drafts fast).
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < K; ++s) { buf[s + b * K] = s * 100 + b; }
        }
        ni::Tensor base(buf.data(), ni::DType::I32, {K, B});
        TensorModel model;
        model.elem   = 4;
        model.ne[0]  = K;
        model.ne[1]  = B;
        model.contiguous();
        std::uint64_t sstate = 0x5EED0005ULL;
        check_tensor_state(base, model, buf.data(), buf.data(), true, sstate);
        for (int s = 0; s < K; ++s) {
            for (int b = 0; b < B; ++b) {
                expect(read_elem(base, s, b) == s * 100 + b,
                       "ingress {drafts,batch} layout maps [s][b] correctly");
            }
        }

        // line ~94: dim-1 batch slice (full) is the whole tensor the kernel gets.
        ni::Tensor batched = base.slice(1, 0, B);
        expect(batched.is_contiguous(), "{drafts,batch} full dim-1 slice stays contiguous");

        // The kernel indexes [step][b] through the strides; pin the host-side
        // per-element values on the full tensor the host passes it.
        for (int s = 0; s < K; ++s) {
            for (int b = 0; b < B; ++b) {
                expect(read_elem(batched, s, b) == s * 100 + b,
                       "ingress [step][b] element matches the layout");
            }
        }
    }

    // F3: degenerate shapes the production code runs (batch=1 or drafts=1).
    {
        // {1, K}: single request, multi-step.
        std::vector<std::int32_t> buf(5);
        for (int s = 0; s < 5; ++s) { buf[s] = s; }
        const ni::Tensor base(buf.data(), ni::DType::I32, {1, 5});
        for (int s = 0; s < 5; ++s) {
            const ni::Tensor column = base.slice(0, 0, 1).slice(1, s, 1);
            expect(column.is_contiguous(), "{1,K} column slice is contiguous");
            const ni::Tensor v = column.view({1});
            expect(*static_cast<const std::int32_t*>(v.data) == s, "{1,K} column view value");
        }
        // {B, 1}: multi request, single step.
        std::vector<std::int32_t> buf2(4);
        for (int b = 0; b < 4; ++b) { buf2[b] = b * 100; }
        const ni::Tensor base2(buf2.data(), ni::DType::I32, {4, 1});
        const ni::Tensor batched = base2.slice(0, 0, 4);
        const ni::Tensor column  = batched.slice(1, 0, 1);
        expect(column.is_contiguous(), "{B,1} column slice is contiguous");
        const ni::Tensor v2       = column.view({4});
        const auto* v2_data       = static_cast<const std::int32_t*>(v2.data);
        bool v2_ok                = true;
        for (int b = 0; b < 4; ++b) { v2_ok = v2_ok && v2_data[b] == b * 100; }
        expect(v2_ok, "{B,1} column view preserves per-batch values");
    }
}

void test_tensor_indexing() {
    tensor_random_chains();
    test_mtp_draft_tensor_contracts();
}

// ---------------------------------------------------------------------------
// Section 5: cyclic KV cache layout planning.
//
// Per-layer K (BF16) and V (FP16) tensors of shape {head_dim, padded_capacity, heads,
// lane_capacity}; capacity is padded to a multiple of 128 and every region starts at a 256-byte
// aligned offset. The expected offsets below are computed by hand.
// ---------------------------------------------------------------------------

void test_cyclic_kv_planning() {
    // Hand-computed anchors.
    {
        ni::LayoutBuilder builder;
        const auto g = ni::plan_cyclic_kv_cache(builder, 1, 1, 1, 8, 1);
        // capacity 1 pads to 128: 8*128*1*1*2 = 2048 B per K/V region.
        expect(g.padded_capacity == 128 && g.k[0].region.offset == 0 &&
                   g.k[0].region.bytes == 2048 && g.v[0].region.offset == 2048 &&
                   g.payload_bytes() == 4096 && builder.finish(256) == 4096,
               "cyclic anchor (1,1,1,8,1)");
    }
    {
        ni::LayoutBuilder builder;
        const auto g = ni::plan_cyclic_kv_cache(builder, 1, 128, 1, 8, 1);
        expect(g.capacity == 128 && g.padded_capacity == 128 &&
                   g.k[0].region.offset == 0 && g.v[0].region.offset == 2048 &&
                   builder.finish(256) == 4096,
               "cyclic anchor (1,128,1,8,1) shares the padded-128 layout");
    }
    {
        ni::LayoutBuilder builder;
        const auto g = ni::plan_cyclic_kv_cache(builder, 2, 129, 2, 16, 1);
        // padded 256: 16*256*2*1*2 = 16384 B per region, layer pitch 32768.
        expect(g.padded_capacity == 256 && g.k[0].region.offset == 0 &&
                   g.v[0].region.offset == 16384 && g.k[1].region.offset == 32768 &&
                   g.v[1].region.offset == 49152 && g.payload_bytes() == 65536 &&
                   builder.finish(256) == 65536,
               "cyclic anchor (2,129,2,16,1) layer pitch");
    }

    // Rejected geometries: no layers, no capacity, capacity beyond int32 after padding, and
    // non-positive heads, head_dim or lane capacity.
    struct Invalid {
        std::uint32_t layers, capacity;
        std::int32_t heads, hd, lane;
    };
    const Invalid invalid[] = {
        {0, 128, 1, 8, 1},         {1, 0, 1, 8, 1},       {1, 0x80000000u, 1, 8, 1},
        {1, 0x7fffffffu, 1, 8, 1}, {1, 128, 0, 8, 1},     {1, 128, -1, 8, 1},
        {1, 128, 1, 0, 1},         {1, 128, 1, -1, 1},    {1, 128, 1, 8, 0},
        {1, 128, 1, 8, -1},
    };
    for (const Invalid& c : invalid) {
        ni::LayoutBuilder builder;
        expect_throw(
            [&] {
                (void)ni::plan_cyclic_kv_cache(builder, c.layers, c.capacity, c.heads, c.hd,
                                               c.lane);
            },
            "cyclic plan rejected invalid geometry");
    }
}

// ---------------------------------------------------------------------------
// Section 6: host KV arena bookkeeping.
//
// The arena is the host-side memory manager behind the host KV transfer
// plumbing used by the target decode/MTP/DFlash programs.  Everything here
// runs on pinned host memory with CPU-visible pages; the layouts below match
// the production page geometries used for host-side KV staging.
// ---------------------------------------------------------------------------

ni::HostKVPageLayout make_l0() {
    // 64-token page, PageMajor: BF16 16x4 + FP16 16x4 -> stride 16384.
    return ni::plan_host_kv_page_layout(ni::KVPageGeometry{
        64, ni::PagedKVPlaneOrder::PageMajor,
        {{ni::DType::BF16, 16, 4}, {ni::DType::FP16, 16, 4}}});
}

ni::HostKVPageLayout make_l1() {
    // 16-token page, HeadMajor: FP32 8x2 + U8 16x4 -> stride 2048.
    return ni::plan_host_kv_page_layout(ni::KVPageGeometry{
        16, ni::PagedKVPlaneOrder::HeadMajor,
        {{ni::DType::FP32, 8, 2}, {ni::DType::U8, 16, 4}}});
}

ni::HostKVPageLayout make_l2() {
    // 4-token page, single I32 plane: 16 B payload -> stride 256.
    return ni::plan_host_kv_page_layout(ni::KVPageGeometry{
        4, ni::PagedKVPlaneOrder::PageMajor, {{ni::DType::I32, 1, 1}}});
}

void arena_ctor_contracts() {
    {
        ni::HostKVPageLayout bad_stride = make_l2();
        bad_stride.page_stride += 256; // disagrees with its own planes
        std::vector<ni::HostKVPageLayout> layouts{bad_stride};
        expect_throw([&] { (void)ni::HostKVArena(0, layouts); },
                     "ctor rejects a layout whose stride disagrees with its planes");
        ni::HostKVPageLayout bad_offset = make_l0();
        bad_offset.planes[0].offset += 256;
        std::vector<ni::HostKVPageLayout> layouts2{bad_offset};
        expect_throw([&] { (void)ni::HostKVArena(0, layouts2); },
                     "ctor rejects a tampered plane offset");
        std::vector<ni::HostKVPageLayout> dup{make_l2(), make_l2()};
        expect_throw([&] { (void)ni::HostKVArena(0, dup); },
                     "ctor rejects duplicate page layouts");
        std::vector<ni::HostKVPageLayout> none;
        expect_throw([&] { (void)ni::HostKVArena(4096, none); },
                     "ctor rejects a non-empty arena with no layouts");
    }

    // Capacity-zero arena: layouts register but nothing can be served, and the
    // release-query asymmetry (plan: empty/empty is negative, can-query:
    // empty/empty is positive) is intentional.
    std::vector<ni::HostKVPageLayout> layouts{make_l2()};
    ni::HostKVArena arena(0, layouts);
    expect(arena.capacity_bytes() == 0 && arena.free_bytes() == 0, "capacity reported as zero");
    const ni::KVPageGeometry known{4, ni::PagedKVPlaneOrder::PageMajor, {{ni::DType::I32, 1, 1}}};
    expect(arena.layout_for(known) != nullptr, "known geometry resolves");
    const ni::KVPageGeometry unknown{32, ni::PagedKVPlaneOrder::PageMajor, {{ni::DType::U8, 1, 1}}};
    expect(arena.layout_for(unknown) == nullptr, "unknown geometry resolves to null");
    expect(!arena.can_allocate(layouts[0], 1), "zero arena refuses can_allocate");
    expect(!arena.allocate(layouts[0], 1).has_value(), "zero arena refuses allocate");
    expect(!arena.allocate_at(layouts[0], 1, 0).has_value(), "zero arena refuses allocate_at");
    expect(!arena.allocate_fragmented(layouts[0], 1).has_value(),
           "zero arena refuses fragmented allocation");
    std::vector<ni::HostKVAllocationHandle> no_releases;
    std::vector<ni::HostKVAllocationRequest> no_targets;
    expect(!arena.plan_after_releases(no_releases, no_targets).has_value(),
           "plan_after_releases is negative on the empty case");
    std::vector<ni::HostKVSuballocationRelease> no_sub;
    expect(arena.can_allocate_after_suballocation_releases(no_sub, no_targets),
           "can_allocate_after_suballocation_releases is positive on the empty case");
}

void arena_view_mechanics() {
    std::vector<ni::HostKVPageLayout> layouts{make_l0()};
    ni::HostKVArena* raw = nullptr;
    try {
        raw = new ni::HostKVArena(65536, layouts);
    } catch (const std::exception& e) {
        note(std::string("arena-views skipped: ") + e.what());
        return;
    }
    const std::unique_ptr<ni::HostKVArena> arena(raw);
    auto allocation = arena->allocate(layouts[0], 4);
    if (!allocation) {
        note("arena-views skipped: full-capacity allocation failed");
        return;
    }
    ni::HostKVAllocation& a = *allocation;

    const ni::HostKVAllocationView wv = arena->writable_view(a);
    expect(wv.valid() && wv.data() != nullptr && wv.page_count() == 4,
           "writable view of a full allocation");
    expect(wv.layout().page_stride == 16384, "writable view layout stride");
    const std::size_t total = std::size_t(4) * wv.layout().page_stride;
    auto* base = wv.data();
    for (std::size_t i = 0; i < total; ++i) {
        base[i] = static_cast<std::byte>(pattern32(i) & 0xFFU);
    }

    const ni::HostKVAllocationConstView cv = arena->view(a);
    expect(cv.valid() && cv.data() == base && cv.page_count() == 4,
           "const view shares the writable view's storage");
    bool sampled = true;
    for (std::size_t i = 0; i < total; i += 97) {
        sampled = sampled && cv.data()[i] == static_cast<std::byte>(pattern32(i) & 0xFFU);
    }
    expect(sampled, "const view samples match the pattern");

    const ni::HostKVAllocationView mid = wv.subview(1, 3);
    expect(mid.data() == base + wv.layout().page_stride && mid.page_count() == 3,
           "subview(1,3) advances by the page stride");
    bool mid_ok = true;
    for (std::size_t i = 0; i < std::size_t(3) * 16384; i += 97) {
        mid_ok = mid_ok && mid.data()[i] == static_cast<std::byte>(pattern32(16384 + i) & 0xFFU);
    }
    expect(mid_ok, "subview contents are the parent pattern offset by one page");
    const ni::HostKVAllocationView full = wv.subview(0, 4);
    expect(full.data() == base && full.page_count() == 4, "subview(0,4) keeps the base pointer");
    expect(cv.subview(1, 3).data() == base + 16384, "const subview advances the same way");
    expect_throw([&] { (void)wv.subview(4, 0); }, "zero-count subview rejected");
    expect_throw([&] { (void)wv.subview(5, 1); }, "subview begin past the allocation rejected");
    expect_throw([&] { (void)wv.subview(3, 2); }, "subview count past the allocation rejected");
    expect_throw([&] { (void)wv.subview(0, 5); }, "subview beyond the page count rejected");
}

void arena_stale_handles() {
    std::vector<ni::HostKVPageLayout> layouts{make_l0(), make_l2()};
    ni::HostKVArena* raw = nullptr;
    try {
        raw = new ni::HostKVArena(65536, layouts);
    } catch (const std::exception& e) {
        note(std::string("arena-stale skipped: ") + e.what());
        return;
    }
    const std::unique_ptr<ni::HostKVArena> arena(raw);

    auto a = arena->allocate(layouts[0], 1);
    auto b = arena->allocate(layouts[1], 2);
    expect(a && b, "stale-setup allocations");
    if (!a || !b) { return; }
    expect(arena->allocation_offset(a->handle()) == 0 &&
               arena->allocation_offset(b->handle()) == layouts[0].page_stride,
           "first-fit placements");

    const ni::HostKVAllocationHandle ha0 = a->handle();
    const ni::HostKVAllocationHandle hb0 = b->handle();
    expect(hb0 == b->handle(), "handle identity is value-based while live");
    const ni::HostKVAllocationConstView cv = arena->view(*b);
    expect(cv.valid() && cv.data() != nullptr && cv.page_count() == 2,
           "const view live before its allocation is released");

    // Release a; every path that references it must go inert.
    expect(a->release(), "first release succeeds");
    expect(!a->valid() && a->page_count() == 0, "released allocation is inert");
    expect(!a->release(), "second release is a no-op");
    expect(!a->handle().valid(), "released allocation returns a default handle");
    expect(ha0.valid(), "the captured handle still names its owner after release");
    expect_throw([&] { (void)arena->writable_view(*a); }, "writable_view rejects a stale allocation");
    expect_throw([&] { (void)arena->view(*a); }, "view rejects a stale allocation");
    expect_throw([&] { (void)arena->allocation_offset(ha0); },
                 "allocation_offset rejects a stale handle");
    std::vector<ni::HostKVAllocationHandle> stale_only{ha0};
    std::vector<ni::HostKVAllocationRequest> one_target{{&layouts[1], 1}};
    expect(!arena->plan_after_releases(
               std::span<const ni::HostKVAllocationHandle>(stale_only.data(), stale_only.size()),
               std::span<const ni::HostKVAllocationRequest>(one_target.data(), one_target.size()))
               .has_value(),
           "plan rejects a stale release handle");
    std::vector<ni::HostKVSuballocationRelease> stale_sub{{ha0, 0, 1}};
    expect(!arena->can_allocate_after_suballocation_releases(
               std::span<const ni::HostKVSuballocationRelease>(stale_sub.data(), stale_sub.size()),
               std::span<const ni::HostKVAllocationRequest>(one_target.data(), one_target.size())),
           "suballocation query rejects a stale handle");
    std::vector<ni::HostKVAllocationHandle> none_h;
    std::vector<ni::HostKVAllocationRequest> none_r;
    std::vector<ni::HostKVSuballocationRelease> none_s;
    expect(!arena->plan_after_releases(
               std::span<const ni::HostKVAllocationHandle>(none_h.data(), none_h.size()),
               std::span<const ni::HostKVAllocationRequest>(none_r.data(), none_r.size()))
               .has_value(),
           "plan empty/empty asymmetry (negative)");
    expect(arena->can_allocate_after_suballocation_releases(
               std::span<const ni::HostKVSuballocationRelease>(none_s.data(), none_s.size()),
               std::span<const ni::HostKVAllocationRequest>(none_r.data(), none_r.size())),
           "plan empty/empty asymmetry (positive)");
    expect_throw([&] {
        [[maybe_unused]] const auto lr = arena->split(std::move(*a), 1);
    },
                 "split rejects a stale allocation");

    // Release b; the earlier view must observe the staleness.
    expect(b->release(), "second release succeeds");
    expect(!cv.valid(), "const view goes stale with its allocation");
    expect_throw([&] { (void)cv.layout(); }, "stale view layout access throws");
    expect_throw([&] { (void)cv.subview(0, 1); }, "stale view subview throws");
}

void arena_fragmentation() {
    std::vector<ni::HostKVPageLayout> layouts{make_l2()};
    ni::HostKVArena* raw = nullptr;
    try {
        raw = new ni::HostKVArena(1280, layouts);
    } catch (const std::exception& e) {
        note(std::string("arena-fragmentation skipped: ") + e.what());
        return;
    }
    const std::unique_ptr<ni::HostKVArena> arena(raw);

    auto a = arena->allocate(layouts[0], 1);
    expect(a && arena->allocation_offset(a->handle()) == 0, "first page at offset 0");
    if (!a) { return; }
    auto p1 = arena->allocate_at(layouts[0], 1, 512);
    auto p2 = arena->allocate_at(layouts[0], 1, 768);
    expect(p1 && arena->allocation_offset(p1->handle()) == 512, "pinned placement at 512");
    expect(p2 && arena->allocation_offset(p2->handle()) == 768, "pinned placement at 768");
    if (!p1 || !p2) { return; }
    expect(arena->free_bytes() == 512, "two 256-byte holes remain");
    expect(!arena->can_allocate(layouts[0], 2), "512 contiguous pages do not fit");
    expect(arena->can_allocate(layouts[0], 1), "one page still fits");
    expect(!arena->allocate(layouts[0], 2).has_value(),
           "contiguous allocation fails on the fragmented arena");

    auto frag = arena->allocate_fragmented(layouts[0], 2);
    expect(frag && frag->size() == 2, "fragmented allocation covers both pages");
    if (!frag) { return; }
    if (frag->size() == 2) {
        expect(arena->allocation_offset((*frag)[0].handle()) == 256 &&
                   arena->allocation_offset((*frag)[1].handle()) == 1024,
               "fragmented regions land on the holes in extent order");
        expect((*frag)[0].page_count() == 1 && (*frag)[1].page_count() == 1,
               "fragmented regions are one page each");
    }
    expect(arena->free_bytes() == 0, "fragmented allocation consumes all free space");
    expect(!arena->can_allocate(layouts[0], 1),
           "descriptor pool exhaustion refuses further pages");

    for (auto& region : *frag) { expect(region.release(), "fragmented region release"); }
    expect(p2->release() && p1->release() && a->release(), "releases coalesce");
    expect(arena->free_bytes() == 1280, "releases coalesce the arena back to one extent");
    expect(!arena->allocate(layouts[0], 6).has_value(), "allocation past capacity refused");
    auto big = arena->allocate(layouts[0], 5);
    expect(big && arena->allocation_offset(big->handle()) == 0 && arena->free_bytes() == 0,
           "exact-capacity allocation");
    if (!big) { return; }
    expect(!arena->allocate(layouts[0], 1).has_value(), "exhausted arena refuses further pages");
    expect(big->release(), "exact-capacity release");

    // Pinned-placement rejection battery on the fully free arena.
    expect(!arena->allocate_at(layouts[0], 1, 257).has_value(), "unaligned offset refused");
    expect(!arena->allocate_at(layouts[0], 1, 1280).has_value(), "offset at capacity refused");
    expect(!arena->allocate_at(layouts[0], 1, 1281).has_value(), "offset past capacity refused");
    expect(!arena->allocate_at(layouts[0], 0, 0).has_value(), "zero-page placement refused");
    expect(arena->free_bytes() == 1280, "rejected placements leave the arena intact");
}

// ===========================================================================
// Section: host KV arena bookkeeping stress.
//
// A ShadowModel mirrors the arena's free-extent set and the occupied byte
// count.  Every operation is applied to the real arena AND the shadow, and
// the bookkeeping tracks the *real* outcome (never the prediction), so the
// shadow stays a faithful mirror of truth.  Three oracle-independent
// invariants are then checked hard every step:
//
//   * arena.occupied_bytes() == the bookkept occupied count
//   * the shadow is self-consistent: 256-byte aligned, in bounds, disjoint,
//     and summing to exactly the arena's free bytes
//   * tracked live regions union the shadow free extents tile [0, capacity)
//
// A separate splitmix64 first-fit oracle predicts verdicts and offsets for
// the well-defined operations (can_allocate / allocate / allocate_at /
// split / fragmented / recipe / sub-release) and is asserted while `trust`
// holds; the first divergence disables further oracle comparisons (the
// invariants remain hard) so a subtle oracle quirk cannot cascade into false
// failures, while a genuine arena defect (aliasing, leaked or miscounted
// bytes, misaligned or overlapping extents, broken occupied accounting,
// stale-handle leakage) still trips the invariant teeth.  Spot sampling of a
// live region's pinned bytes against its per-entry pattern catches
// cross-entry aliasing even when the byte accounting alone cannot.
// ===========================================================================

struct StressFreeExtent {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

class ShadowModel {
public:
    explicit ShadowModel(std::size_t capacity) : capacity_(capacity) {
        extents_.push_back({0, capacity});
    }

    [[nodiscard]] const std::vector<StressFreeExtent>& extents() const { return extents_; }
    [[nodiscard]] std::size_t capacity() const { return capacity_; }

    void normalize() {
        std::sort(extents_.begin(), extents_.end(),
                  [](const StressFreeExtent& a, const StressFreeExtent& b) {
                      return a.offset < b.offset;
                  });
        std::vector<StressFreeExtent> out;
        out.reserve(extents_.size());
        for (auto& e : extents_) {
            if (e.bytes == 0) { continue; }
            if (!out.empty() && out.back().offset + out.back().bytes == e.offset) {
                out.back().bytes += e.bytes;
            } else {
                out.push_back(e);
            }
        }
        extents_ = std::move(out);
    }

    void insert(std::size_t offset, std::size_t bytes) {
        if (bytes == 0) { return; }
        extents_.push_back({offset, bytes});
        normalize();
    }

    // Remove [offset, offset + bytes) from the extents.  The caller guarantees
    // the range is covered by a single extent; touching is merged on normalize.
    void carve(std::size_t offset, std::size_t bytes) {
        if (bytes == 0) { return; }
        const std::size_t end = offset + bytes;
        std::vector<StressFreeExtent> out;
        out.reserve(extents_.size());
        for (const auto& e : extents_) {
            if (e.offset + e.bytes <= offset || e.offset >= end) {
                out.push_back(e);
            } else if (e.offset < offset && e.offset + e.bytes > end) {
                out.push_back({e.offset, offset - e.offset});
                out.push_back({end, e.offset + e.bytes - end});
            } else if (e.offset < offset) {
                out.push_back({e.offset, offset - e.offset});
            } else {
                out.push_back({end, e.offset + e.bytes - end});
            }
        }
        extents_ = std::move(out);
        normalize();
    }

    // Offset of the first (offset-ordered) extent large enough, else SIZE_MAX.
    std::size_t first_fit(std::size_t bytes) const {
        for (const auto& e : extents_) {
            if (e.bytes >= bytes) { return e.offset; }
        }
        return std::numeric_limits<std::size_t>::max();
    }

    // Largest extent; ties resolve to the smallest offset (matching
    // std::max_element over a byte-descending comparator).
    const StressFreeExtent* largest() const {
        const StressFreeExtent* best = nullptr;
        for (const auto& e : extents_) {
            if (best == nullptr || e.bytes > best->bytes) { best = &e; }
        }
        return best;
    }

    // Check the mirror against itself.  expected_free is the arena free bytes;
    // violations are hard failures.
    void self_check(std::size_t expected_free) const {
        std::size_t total    = 0;
        std::size_t prev_end = 0;
        for (const auto& e : extents_) {
            if (e.offset % 256 != 0) { expect(false, "stress: shadow extent not 256-aligned"); }
            if (e.offset >= capacity_ || e.offset + e.bytes > capacity_) {
                expect(false, "stress: shadow extent out of bounds");
            }
            if (e.offset < prev_end) { expect(false, "stress: shadow extents overlap"); }
            total    += e.bytes;
            prev_end = std::max(prev_end, e.offset + e.bytes);
        }
        if (total != expected_free) { expect(false, "stress: shadow free bytes mismatch"); }
    }

    // True if the union of the live byte ranges and the shadow free extents
    // tiles [0, capacity) exactly.
    bool tiles_capacity(const std::vector<std::pair<std::size_t, std::size_t>>& live_ranges) const {
        std::vector<StressFreeExtent> all = extents_;
        for (const auto& [off, bytes] : live_ranges) { all.push_back({off, bytes}); }
        std::sort(all.begin(), all.end(),
                  [](const StressFreeExtent& a, const StressFreeExtent& b) {
                      return a.offset < b.offset;
                  });
        std::size_t cursor = 0;
        for (const auto& e : all) {
            if (e.bytes == 0) { continue; }
            if (e.offset > cursor) { return false; }
            const std::size_t end = e.offset + e.bytes;
            if (end > capacity_) { return false; }
            if (end > cursor) { cursor = end; }
        }
        return cursor == capacity_;
    }

private:
    std::size_t capacity_ = 0;
    std::vector<StressFreeExtent> extents_;
};

struct StressEntry {
    ni::HostKVAllocation alloc;  // move-only, owns a live descriptor
    std::size_t offset = 0;
    std::uint32_t pages = 0;
    int layout         = 0;  // index into the stress layout table
    std::uint8_t tag   = 0;  // per-entry pattern tag
    std::size_t shift  = 0;  // pattern position base for this region
};

void arena_stress_impl() {
    constexpr std::size_t kCapacity  = 655360;
    constexpr std::size_t kDescMax   = 2560;  // == kCapacity / smallest stride (256)
    constexpr std::uint32_t kOps     = 20000;

    std::vector<ni::HostKVPageLayout> layouts{make_l0(), make_l1(), make_l2()};
    const std::size_t stride_of[3] = {layouts[0].page_stride, layouts[1].page_stride,
                                      layouts[2].page_stride};
    expect(stride_of[0] == 16384 && stride_of[1] == 2048 && stride_of[2] == 256,
           "stress: layout strides as expected");

    ni::HostKVArena* raw = nullptr;
    try {
        raw = new ni::HostKVArena(kCapacity, layouts);
    } catch (const std::exception& e) {
        note(std::string("arena-stress skipped: ") + e.what());
        return;
    }
    const std::unique_ptr<ni::HostKVArena> arena(raw);
    expect(arena->capacity_bytes() == kCapacity && arena->free_bytes() == kCapacity &&
               arena->occupied_bytes() == 0,
           "stress: fresh arena reports full capacity free");

    ShadowModel shadow(kCapacity);
    std::vector<StressEntry> live;
    std::vector<ni::HostKVAllocationHandle> stale_pool;
    std::size_t occupied = 0;
    bool trust          = true;
    std::uint64_t state = 0x5EED0006ULL;
    std::uint8_t tag_counter = 0;

    const auto pattern_byte = [](std::size_t pos, std::uint8_t tag) {
        return static_cast<std::byte>(((pos >> 8) ^ tag) & 0xFFU);
    };
    const auto make_shift = [](std::uint8_t tag, std::size_t offset) {
        return static_cast<std::size_t>(tag) * 16384 + offset;
    };
    auto push_stale = [&](ni::HostKVAllocationHandle h) {
        if (stale_pool.size() >= 64) { stale_pool.clear(); }
        stale_pool.push_back(h);
    };

    // Fill + track a freshly allocated region (new bytes carved from free).
    auto add_entry = [&](std::size_t offset, std::uint32_t pages, int layout_idx,
                         std::uint8_t tag, std::size_t shift, ni::HostKVAllocation alloc) {
        const std::size_t total = std::size_t(pages) * stride_of[layout_idx];
        auto wv  = arena->writable_view(alloc);
        std::byte* base = wv.data();
        for (std::size_t i = 0; i < total; ++i) { base[i] = pattern_byte(shift + i, tag); }
        shadow.carve(offset, total);
        StressEntry e;
        e.offset = offset; e.pages = pages; e.layout = layout_idx; e.tag = tag;
        e.shift = shift;
        e.alloc = std::move(alloc);
        live.push_back(std::move(e));
        occupied += total;
    };

    // Release a tracked entry: free its range, drop it from live, park the
    // (now stale) handle for later injection.
    auto release_entry = [&](std::size_t idx) {
        StressEntry& E = live[idx];
        const std::size_t bytes = std::size_t(E.pages) * stride_of[E.layout];
        const ni::HostKVAllocationHandle h = E.alloc.handle();
        expect(E.alloc.release(), "stress: release of a live entry succeeds");
        shadow.insert(E.offset, bytes);
        occupied -= bytes;
        push_stale(h);
        live[idx] = std::move(live.back());
        live.pop_back();
    };

    auto check_occupied = [&]() {
        if (occupied != arena->occupied_bytes()) {
            expect(false, "stress: occupied bytes disagree with the arena");
        }
        shadow.self_check(arena->free_bytes());
    };

    auto check_partition = [&]() {
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(live.size());
        for (const auto& e : live) { ranges.emplace_back(e.offset, std::size_t(e.pages) * stride_of[e.layout]); }
        if (!shadow.tiles_capacity(ranges)) { expect(false, "stress: live + free do not tile the arena"); }
    };

    auto sample_entries = [&]() {
        if (live.empty()) { return; }
        const int n = std::min<int>(3, static_cast<int>(live.size()));
        for (int k = 0; k < n; ++k) {
            const std::size_t idx = rnd(state, static_cast<std::uint32_t>(live.size()));
            const StressEntry& E  = live[idx];
            const std::size_t total = std::size_t(E.pages) * stride_of[E.layout];
            const auto cv = arena->view(E.alloc);
            expect(cv.valid(), "stress: sampled live entry still has a valid view");
            if (!cv.valid()) { continue; }
            const std::byte* base = cv.data();
            bool ok = true;
            for (std::size_t i = 0; i < total; i += 511) {
                ok = ok && base[i] == pattern_byte(E.shift + i, E.tag);
            }
            ok = ok && base[total - 1] == pattern_byte(E.shift + total - 1, E.tag);
            if (!ok) { expect(false, "stress: sampled region pattern drifted (aliasing)"); }
        }
    };

    auto find_idx = [&](ni::HostKVAllocationHandle h) -> long {
        for (std::size_t i = 0; i < live.size(); ++i) {
            if (live[i].alloc.handle() == h) { return static_cast<long>(i);
            }
        }
        return -1;
    };

    // ------------------------------------------------------------------
    // (a) Deterministic contracts on the fresh, fully-free arena.
    //     Splits and recipe edge cases the random dispatch cannot reach
    //     reliably are pinned here with exact expectations.
    // ------------------------------------------------------------------
    {
        const std::size_t l2s = stride_of[2];

        // 1) split(page_offset = 0) -> out_of_range; the by-value param is
        //    untouched and releases when its owner is reset.
        {
            auto parent = arena->allocate(layouts[2], 4);
            expect(parent.has_value(), "pre: parent(L2x4) for split(0)");
            if (parent) {
                expect_throw([&] {
                [[maybe_unused]] const auto lr = arena->split(std::move(*parent), 0);
            },
                             "split(page_offset=0) rejected (out_of_range)");
            }
            // split takes an rvalue reference: a rejected split must leave the input
            // allocation untouched and still owned, so it is released here, not by split.
            if (parent) {
                expect(arena->occupied_bytes() != 0, "split(0) rejection leaves the input allocation owned");
                parent.reset();
            }
            // split takes an rvalue reference: a rejected split leaves the input allocation
            // untouched and owned, so it is released here rather than by split.
            if (parent) {
                expect(arena->occupied_bytes() != 0, "split(0) rejection leaves the input allocation owned");
                parent.reset();
            }
            expect(arena->occupied_bytes() == 0, "split(0) leaves the arena fully free");
            check_occupied();
            check_partition();
        }

        // 2) split(page_offset = pages) -> out_of_range as well.
        {
            auto parent = arena->allocate(layouts[2], 4);
            expect(parent.has_value(), "pre: parent(L2x4) for split(pages)");
            if (parent) {
                expect_throw([&] {
                [[maybe_unused]] const auto lr = arena->split(std::move(*parent), 4);
            },
                             "split(page_offset=pages) rejected (out_of_range)");
            }
            // split takes an rvalue reference: a rejected split must leave the input
            // allocation untouched and still owned, so it is released here, not by split.
            if (parent) {
                expect(arena->occupied_bytes() != 0, "split(pages) rejection leaves the input allocation owned");
                parent.reset();
            }
            // split takes an rvalue reference: a rejected split leaves the input allocation
            // untouched and owned, so it is released here rather than by split.
            if (parent) {
                expect(arena->occupied_bytes() != 0, "split(pages) rejection leaves the input allocation owned");
                parent.reset();
            }
            expect(arena->occupied_bytes() == 0, "split(pages) leaves the arena fully free");
            check_occupied();
            check_partition();
        }

        // 3) A successful split: L2x4 -> left(L2x2 @0) + right(L2x2 @512).
        {
            std::uint8_t tag = tag_counter++;
            auto parent = arena->allocate(layouts[2], 4);
            expect(parent.has_value(), "pre: parent(L2x4) for split(2)");
            if (parent) {
                const std::size_t po = arena->allocation_offset(parent->handle());
                auto pr = arena->split(std::move(*parent), 2);
                const ni::HostKVAllocationHandle lh = pr.first.handle();
                const ni::HostKVAllocationHandle rh = pr.second.handle();
                add_entry(po, 2, 2, tag, make_shift(tag, po), std::move(pr.first));
                const std::size_t ro = po + 2 * l2s;
                add_entry(ro, 2, 2, tag, make_shift(tag, ro), std::move(pr.second));
                expect(arena->allocation_offset(lh) == po, "split left offset");
                expect(arena->allocation_offset(rh) == ro, "split right offset");
            }
            check_occupied();
            check_partition();
        }

        // 4) revision-bump: any arena mutation between plan and apply invalidates
        //    the recipe (apply returns false and leaves the arena unchanged).
        {
            std::vector<ni::HostKVAllocationHandle> handles;
            for (int k = 0; k < 3; ++k) {
                const std::uint8_t tag = tag_counter++;
                auto a = arena->allocate(layouts[2], 1);
                if (!a) { break; }
                const std::size_t off = arena->allocation_offset(a->handle());
                add_entry(off, 1, 2, tag, make_shift(tag, off), std::move(*a));
                handles.push_back(live.back().alloc.handle());
            }
            expect(handles.size() == 3, "revision: three parents allocated");
            if (handles.size() == 3) {
                std::vector<ni::HostKVAllocationHandle> rel{handles[0]};
                std::vector<ni::HostKVAllocationRequest> treq{{&layouts[2], 1}};
                auto plan = arena->plan_after_releases(
                    std::span<const ni::HostKVAllocationHandle>(rel.data(), rel.size()),
                    std::span<const ni::HostKVAllocationRequest>(treq.data(), treq.size()));
                expect(plan.has_value(), "revision: plan ok before mutation");
                const long ci = find_idx(handles[2]);
                expect(ci >= 0, "revision: C still live");
                if (ci >= 0) { release_entry(static_cast<std::size_t>(ci)); }  // bumps revision
                std::vector<ni::HostKVAllocation*> rptrs;
                const long ai = find_idx(handles[0]);
                expect(ai >= 0, "revision: A still live after releasing C");
                if (ai >= 0) { rptrs.push_back(&live[static_cast<std::size_t>(ai)].alloc); }
                std::vector<ni::HostKVAllocation> tallocs(1);
                bool applied = false;
                if (plan && !rptrs.empty()) {
                    applied = arena->apply_recipe(std::move(*plan),
                        std::span<ni::HostKVAllocation* const>(rptrs.data(), rptrs.size()),
                        std::span<ni::HostKVAllocation>(tallocs.data(), tallocs.size()));
                }
                expect(!applied, "revision bump invalidates the recipe (apply=false)");
            }
            check_occupied();
            check_partition();
        }

        // 5) pre-filled target: a non-fresh output makes apply fail (no mutation).
        {
            std::uint8_t tagA = tag_counter++;
            auto a = arena->allocate(layouts[2], 1);
            expect(a.has_value(), "prefill: allocate A");
            ni::HostKVAllocationHandle hA;
            if (a) {
                const std::size_t off = arena->allocation_offset(a->handle());
                hA = a->handle();
                add_entry(off, 1, 2, tagA, make_shift(tagA, off), std::move(*a));
            }
            std::uint8_t tagP = tag_counter++;
            auto pre = arena->allocate(layouts[2], 1);
            expect(pre.has_value(), "prefill: allocate pre");
            std::size_t pre_off = 0, pre_bytes = 0;
            if (pre) {
                pre_off = arena->allocation_offset(pre->handle());
                pre_bytes = stride_of[2];
                shadow.carve(pre_off, pre_bytes);
                occupied += pre_bytes;
            }
            check_occupied();  // `pre` is shadow-carved but intentionally untracked in `live`
            std::vector<ni::HostKVAllocationHandle> rel{hA};
            std::vector<ni::HostKVAllocationRequest> treq{{&layouts[2], 1}};
            auto plan = arena->plan_after_releases(
                std::span<const ni::HostKVAllocationHandle>(rel.data(), rel.size()),
                std::span<const ni::HostKVAllocationRequest>(treq.data(), treq.size()));
            expect(plan.has_value(), "prefill: plan ok");
            std::vector<ni::HostKVAllocation*> rptrs;
            const long ai = find_idx(hA);
            if (ai >= 0) { rptrs.push_back(&live[static_cast<std::size_t>(ai)].alloc); }
            std::vector<ni::HostKVAllocation> tallocs(1);
            if (pre) { tallocs[0] = std::move(*pre); }  // make the target non-fresh
            const bool applied = arena->apply_recipe(std::move(*plan),
                std::span<ni::HostKVAllocation* const>(rptrs.data(), rptrs.size()),
                std::span<ni::HostKVAllocation>(tallocs.data(), tallocs.size()));
            expect(!applied, "prefilled (non-fresh) target makes apply fail");
            if (tallocs[0].valid()) { tallocs[0].release(); }  // free pre's bytes
            shadow.insert(pre_off, pre_bytes);
            occupied -= pre_bytes;
            check_occupied();
            check_partition();
        }
    }

    // ------------------------------------------------------------------
    // (b) 20000-op random dispatch.
    // ------------------------------------------------------------------
    const auto new_tag = [&]() { return tag_counter++; };
    const auto one_extent_contains = [&](std::size_t off, std::size_t bytes) {
        for (const auto& e : shadow.extents()) {
            if (e.offset <= off && off - e.offset <= e.bytes &&
                bytes <= e.bytes - (off - e.offset)) { return true; }
        }
        return false;
    };
    const auto sub_release_oracle = [&](const StressEntry& E,
                                        const std::vector<ni::HostKVSuballocationRelease>& rels,
                                        const std::vector<ni::HostKVAllocationRequest>& treq) {
        // Mirrors can_allocate_after_suballocation_releases for one valid
        // allocation (handle h) released by `rels` subranges.
        const std::uint32_t P = E.pages;
        const std::size_t stride = stride_of[E.layout];
        std::vector<std::pair<std::uint32_t, std::uint32_t>> intervals;
        intervals.reserve(rels.size());
        for (const auto& r : rels) {
            intervals.emplace_back(r.begin_page, r.begin_page + r.page_count);
        }
        std::sort(intervals.begin(), intervals.end());
        std::uint32_t cursor = 0;
        std::size_t runs = 0;
        for (const auto& [b, e] : intervals) { if (b > cursor) { ++runs; } cursor = e; }
        if (cursor < P) { ++runs; }
        std::size_t available = kDescMax - live.size();
        if (runs == 0) { ++available; }
        std::size_t required = treq.size();
        if (runs > 1) { required += runs - 1; }
        if (required > available) { return false; }
        ShadowModel sim(shadow);
        for (const auto& r : rels) {
            const std::size_t roff = E.offset + static_cast<std::size_t>(r.begin_page) * stride;
            const std::size_t rbytes = static_cast<std::size_t>(r.page_count) * stride;
            sim.insert(roff, rbytes);
        }
        for (const auto& req : treq) {
            const std::size_t bytes = req.layout->page_stride * req.pages;
            if (sim.first_fit(bytes) == std::numeric_limits<std::size_t>::max()) { return false; }
            const std::size_t off = sim.first_fit(bytes);
            sim.carve(off, bytes);
        }
        return true;
    };

    const auto step = [&](std::uint32_t r) {
        if (r < 30) {
            const int li = static_cast<int>(rnd(state, 3));
            const std::uint32_t pages =
                li == 0 ? 1u + rnd(state, 2) : li == 1 ? 1u + rnd(state, 4) : 1u + rnd(state, 8);
            const std::size_t bytes = stride_of[li] * pages;
            const bool oracle_can =
                (live.size() < kDescMax) &&
                shadow.first_fit(bytes) != std::numeric_limits<std::size_t>::max();
            if (trust) {
                expect(arena->can_allocate(layouts[li], pages) == oracle_can,
                       "allocate: can_allocate oracle mismatch");
            }
            auto opt = arena->allocate(layouts[li], pages);
            const bool real_ok = opt.has_value();
            if (trust) { expect(real_ok == oracle_can, "allocate: verdict oracle mismatch"); }
            if (real_ok) {
                const std::size_t off = arena->allocation_offset(opt->handle());
                if (trust) { expect(off == shadow.first_fit(bytes), "allocate: offset oracle mismatch"); }
                const std::uint8_t tg = new_tag();
                add_entry(off, pages, li, tg, make_shift(tg, off), std::move(*opt));
            }
            return;
        }
        if (r < 50) {
            if (live.empty()) { return; }
            release_entry(rnd(state, static_cast<std::uint32_t>(live.size())));
            return;
        }
        if (r < 62) {
            const int li = static_cast<int>(rnd(state, 3));
            const std::uint32_t pages = 1u + rnd(state, 2);
            const std::size_t bytes = stride_of[li] * pages;
            const std::uint32_t kind = rnd(state, 4);
            const auto& exts = shadow.extents();
            std::size_t off = 0;
            if (kind == 0) {
                if (exts.empty()) { off = kCapacity + 256u; }
                else {
                    const StressFreeExtent& ex =
                        exts[rnd(state, static_cast<std::uint32_t>(exts.size()))];
                    const std::size_t slots = ex.bytes / 256u;
                    off = ex.offset +
                          (slots == 0 ? 0u : 256u * rnd(state, static_cast<std::uint32_t>(slots)));
                }
            } else if (kind == 1) {
                off = kCapacity + 256u;
            } else if (kind == 2) {
                off = 256u * rnd(state, static_cast<std::uint32_t>(kCapacity / 256u) + 1u);
            } else {
                off = exts.empty() ? (kCapacity + 256u) : (exts[0].offset + 1u);  // unaligned
            }
            const bool oracle_at = (off % 256u == 0u) && (live.size() < kDescMax) &&
                (off <= kCapacity) && (bytes <= kCapacity - off) && one_extent_contains(off, bytes);
            auto opt = arena->allocate_at(layouts[li], pages, off);
            const bool real_ok = opt.has_value();
            if (trust) { expect(real_ok == oracle_at, "allocate_at: verdict oracle mismatch"); }
            if (real_ok) {
                const std::size_t real_off = arena->allocation_offset(opt->handle());
                if (trust) { expect(real_off == off, "allocate_at: placed at requested offset"); }
                const std::uint8_t tg = new_tag();
                add_entry(off, pages, li, tg, make_shift(tg, off), std::move(*opt));
            }
            return;
        }
        if (r < 72) {
            const int li = static_cast<int>(rnd(state, 3));
            const std::uint32_t pages = 1u + rnd(state, 8);
            const std::size_t stride = stride_of[li];
            ShadowModel sim(shadow);
            bool sim_fail = false;
            std::vector<std::pair<std::size_t, std::uint32_t>> regions;
            {
                std::uint32_t remaining = pages;
                std::size_t used = 0;
                while (remaining != 0) {
                    const StressFreeExtent* lg = sim.largest();
                    if (lg == nullptr) { sim_fail = true; break; }
                    const std::uint32_t count = static_cast<std::uint32_t>(
                        std::min<std::size_t>(remaining, lg->bytes / stride));
                    if (count == 0) { sim_fail = true; break; }
                    if (live.size() + used + 1 > kDescMax) { sim_fail = true; break; }
                    const std::size_t region_offset = lg->offset;  // carve() rebuilds extents_
                    sim.carve(region_offset, static_cast<std::size_t>(count) * stride);
                    regions.emplace_back(region_offset, count);
                    used += count;
                    remaining -= count;
                }
            }
            const bool oracle_ok = !sim_fail;
            auto got = arena->allocate_fragmented(layouts[li], pages);
            const bool real_ok = got.has_value();
            if (trust) { expect(real_ok == oracle_ok, "fragmented: verdict oracle mismatch"); }
            if (real_ok) {
                const std::size_t n = got->size();
                for (std::size_t j = 0; j < n; ++j) {
                    const std::size_t off = arena->allocation_offset((*got)[j].handle());
                    const std::uint32_t pg = (*got)[j].page_count();
                    if (trust && j < regions.size()) {
                        expect(off == regions[j].first && pg == regions[j].second,
                               "fragmented: region offset/pages oracle mismatch");
                    }
                    const std::uint8_t tg = new_tag();
                    add_entry(off, pg, li, tg, make_shift(tg, off), std::move((*got)[j]));
                }
            }
            return;
        }
        if (r < 82) {
            if (live.size() >= kDescMax) { return; }  // split would logic_error; skip
            std::vector<std::size_t> cand;
            cand.reserve(live.size());
            for (std::size_t i = 0; i < live.size(); ++i) {
                if (live[i].pages > 1) { cand.push_back(i); }
            }
            if (cand.empty()) { return; }
            const std::size_t idx = cand[rnd(state, static_cast<std::uint32_t>(cand.size()))];
            const std::uint32_t P = live[idx].pages;
            const std::size_t O = live[idx].offset;
            const int LI = live[idx].layout;
            const std::uint8_t T = live[idx].tag;
            const std::size_t S = live[idx].shift;
            const std::size_t stride = stride_of[LI];
            const ni::HostKVAllocationHandle parent_h = live[idx].alloc.handle();
            const std::uint32_t off = 1u + rnd(state, static_cast<std::uint32_t>(P - 1));
            auto pr = arena->split(std::move(live[idx].alloc), off);
            push_stale(parent_h);
            live[idx].alloc = std::move(pr.first);
            live[idx].offset = O;
            live[idx].pages = off;
            live[idx].layout = LI;
            live[idx].tag = T;
            live[idx].shift = S;
            StressEntry right;
            right.alloc = std::move(pr.second);
            right.offset = O + static_cast<std::size_t>(off) * stride;
            right.pages = P - off;
            right.layout = LI;
            right.tag = T;
            right.shift = S + static_cast<std::size_t>(off) * stride;
            live.push_back(std::move(right));
            return;
        }
        if (r < 86) {
            if (live.empty()) { return; }
            const std::size_t max_rel = (live.size() >= 2) ? (1u + rnd(state, 2)) : 1u;
            std::vector<std::size_t> idx_all(live.size());
            for (std::size_t i = 0; i < live.size(); ++i) { idx_all[i] = i; }
            for (std::size_t i = 0; i < max_rel; ++i) {
                const std::size_t j = i + rnd(state, static_cast<std::uint32_t>(live.size() - i));
                std::swap(idx_all[i], idx_all[j]);
            }
            std::vector<std::size_t> rel_idx(idx_all.begin(),
                                             idx_all.begin() + static_cast<std::ptrdiff_t>(max_rel));
            const std::size_t nrel = rel_idx.size();
            const std::size_t ntar = 1 + rnd(state, 4);

            if (!stale_pool.empty() && rnd(state, 6) == 0) {
                const auto stale =
                    stale_pool[rnd(state, static_cast<std::uint32_t>(stale_pool.size()))];
                std::vector<ni::HostKVAllocationHandle> rel{stale};
                std::vector<ni::HostKVAllocationRequest> treq{{&layouts[rnd(state, 3)], 1}};
                auto plan = arena->plan_after_releases(
                    std::span<const ni::HostKVAllocationHandle>(rel.data(), rel.size()),
                    std::span<const ni::HostKVAllocationRequest>(treq.data(), treq.size()));
                expect(!plan.has_value(), "recipe: stale release handle -> nullopt");
                return;
            }

            std::vector<ni::HostKVAllocationHandle> rel;
            for (std::size_t i : rel_idx) { rel.push_back(live[i].alloc.handle()); }
            std::vector<ni::HostKVAllocationRequest> treq;
            for (std::size_t t = 0; t < ntar; ++t) {
                const int li = static_cast<int>(rnd(state, 3));
                const std::uint32_t pages = 1u + rnd(state, 4);
                treq.push_back({&layouts[li], pages});
            }
            ShadowModel sim(shadow);
            for (std::size_t i : rel_idx) {
                const StressEntry& E = live[i];
                sim.insert(E.offset, static_cast<std::size_t>(E.pages) * stride_of[E.layout]);
            }
            bool ok = ntar <= (kDescMax - live.size()) + nrel;
            if (ok) {
                for (const auto& req : treq) {
                    const std::size_t bytes = req.layout->page_stride * req.pages;
                    if (sim.first_fit(bytes) == std::numeric_limits<std::size_t>::max()) {
                        ok = false;
                        break;
                    }
                    const std::size_t off = sim.first_fit(bytes);
                    sim.carve(off, bytes);
                }
            }
            auto plan = arena->plan_after_releases(
                std::span<const ni::HostKVAllocationHandle>(rel.data(), rel.size()),
                std::span<const ni::HostKVAllocationRequest>(treq.data(), treq.size()));
            if (trust) { expect(plan.has_value() == ok, "recipe: plan verdict oracle mismatch"); }
            if (!plan.has_value()) { return; }

            std::vector<ni::HostKVAllocation*> rptrs;
            for (std::size_t i : rel_idx) { rptrs.push_back(&live[i].alloc); }
            std::vector<ni::HostKVAllocation> tallocs(ntar);
            const bool applied = arena->apply_recipe(std::move(*plan),
                std::span<ni::HostKVAllocation* const>(rptrs.data(), rptrs.size()),
                std::span<ni::HostKVAllocation>(tallocs.data(), tallocs.size()));
            expect(applied, "recipe: clean apply succeeds");
            if (!applied) { return; }

            for (std::size_t k = 0; k < rel_idx.size(); ++k) {
                const StressEntry& E = live[rel_idx[k]];
                const std::size_t bytes = static_cast<std::size_t>(E.pages) * stride_of[E.layout];
                shadow.insert(E.offset, bytes);
                occupied -= bytes;
                push_stale(rel[k]);
            }
            std::vector<char> released(live.size(), 0);
            for (std::size_t i : rel_idx) { released[i] = 1; }
            std::vector<StressEntry> kept;
            kept.reserve(live.size());
            for (std::size_t i = 0; i < live.size(); ++i) {
                if (!released[i]) { kept.push_back(std::move(live[i])); }
            }
            live = std::move(kept);
            for (std::size_t t = 0; t < ntar; ++t) {
                const std::size_t off = arena->allocation_offset(tallocs[t].handle());
                const std::uint32_t pg = tallocs[t].page_count();
                int li = 0;
                for (int k = 0; k < 3; ++k) {
                    if (&layouts[k] == treq[t].layout) { li = k; }
                }
                const std::uint8_t tg = new_tag();
                add_entry(off, pg, li, tg, make_shift(tg, off), std::move(tallocs[t]));
            }
            return;
        }

        // r in [86, 100): suballocation-release query (const, no mutation).
        if (live.empty()) { return; }
        const std::size_t idx = rnd(state, static_cast<std::uint32_t>(live.size()));
        const StressEntry& E = live[idx];
        const std::uint32_t P = E.pages;
        const std::uint32_t tcount = 1 + rnd(state, 2);
        std::vector<ni::HostKVAllocationRequest> treq;
        for (std::uint32_t t = 0; t < tcount; ++t) {
            const int li = static_cast<int>(rnd(state, 3));
            const std::uint32_t pages = 1u + rnd(state, 4);
            treq.push_back({&layouts[li], pages});
        }
        const ni::HostKVAllocationHandle h = E.alloc.handle();
        std::vector<ni::HostKVSuballocationRelease> rels;
        int mode = static_cast<int>(rnd(state, 5));
        bool oracle = false;
        if (mode == 0) {
            if (stale_pool.empty()) { mode = 4; }
            else {
                const auto sh =
                    stale_pool[rnd(state, static_cast<std::uint32_t>(stale_pool.size()))];
                rels.push_back({sh, 0, 0});
                oracle = false;
            }
        }
        if (mode == 1) {
            rels.push_back({h, 0, 0});
            oracle = false;
        } else if (mode == 2) {
            rels.push_back({h, P, 1});
            oracle = false;
        } else if (mode == 3) {
            if (P < 2) { mode = 4; }
            else {
                const std::uint32_t b = rnd(state, P);
                const std::uint32_t c = 1u + rnd(state, static_cast<std::uint32_t>(P - b));
                rels.push_back({h, b, c});
                rels.push_back({h, b, c});
                oracle = false;
            }
        }
        if (mode == 4) {
            bool good = false;
            for (int attempt = 0; attempt < 8 && !good; ++attempt) {
                rels.clear();
                std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
                const int n = (P >= 2 && rnd(state, 2) == 0) ? 2 : 1;
                bool ok = true;
                for (int k = 0; k < n; ++k) {
                    const std::uint32_t b = rnd(state, P);
                    const std::uint32_t maxc = P - b;
                    const std::uint32_t c = 1u + rnd(state, static_cast<std::uint32_t>(maxc));
                    const std::uint32_t ne = b + c;
                    for (const auto& rr : ranges) {
                        const std::uint32_t re = rr.first + rr.second;
                        if (b < re && rr.first < ne) { ok = false; break; }
                    }
                    if (!ok) { break; }
                    ranges.emplace_back(b, c);
                    rels.push_back({h, b, c});
                }
                good = ok;
                if (!ok) { rels.clear(); }
            }
            if (!good) { rels.clear(); rels.push_back({h, 0, P}); }
            oracle = sub_release_oracle(E, rels, treq);
        }
        const bool real = arena->can_allocate_after_suballocation_releases(
            std::span<const ni::HostKVSuballocationRelease>(rels.data(), rels.size()),
            std::span<const ni::HostKVAllocationRequest>(treq.data(), treq.size()));
        if (trust) { expect(real == oracle, "sub-release: verdict oracle mismatch"); }
    };

    for (std::size_t op = 0; op < kOps; ++op) {
        const std::uint32_t r = rnd(state, 100);
        step(r);
        check_occupied();
        if ((op & 255) == 255) { sample_entries(); }
        if ((op & 511) == 511) { check_partition(); }
    }

    // ------------------------------------------------------------------
    // (c) Drain: release every live entry, verify the arena is fully free.
    // ------------------------------------------------------------------
    while (!live.empty()) {
        StressEntry& E = live.back();
        const std::size_t bytes = static_cast<std::size_t>(E.pages) * stride_of[E.layout];
        const ni::HostKVAllocationHandle h = E.alloc.handle();
        expect(E.alloc.release(), "stress: drain release succeeds");
        shadow.insert(E.offset, bytes);
        occupied -= bytes;
        push_stale(h);
        live.pop_back();
    }
    expect(occupied == 0, "stress: drained occupied is 0");
    expect(arena->occupied_bytes() == 0, "stress: drained arena occupied is 0");
    expect(arena->free_bytes() == kCapacity, "stress: drained arena fully free");
    {
        const auto& exts = shadow.extents();
        expect(exts.size() == 1 && exts[0].offset == 0 && exts[0].bytes == kCapacity,
               "stress: drained shadow is one full-capacity extent");
    }
    check_partition();
}

void arena_stress() {
    try {
        arena_stress_impl();
    } catch (const std::exception& e) {
        note(std::string("arena-stress skipped: ") + e.what());
    }
}

} // namespace

struct Section { const char* name; void (*fn)(); };

int main() {
    const Section sections[] = {
        {"page-layout-planning", &test_page_layout_planning},
        {"transfer-planning", &test_transfer_planning},
        {"tensor-indexing", &test_tensor_indexing},
        {"cyclic-planning", &test_cyclic_kv_planning},
        {"arena-contracts", &arena_ctor_contracts},
        {"arena-views", &arena_view_mechanics},
        {"arena-stale", &arena_stale_handles},
        {"arena-fragmentation", &arena_fragmentation},
        {"arena-stress", &arena_stress},
    };
    int total = failures;
    for (const auto& s : sections) {
        const int before = failures;
        try { s.fn(); }
        catch (const std::exception& e) {
            ++failures;
            std::cerr << "FAIL: " << s.name << " threw: " << e.what() << '\n';
        }
        const int after = failures;
        if (after == before) { std::cout << "ok   " << s.name << '\n'; }
        else { std::cout << "fail " << s.name << " (+" << (after - before) << ")\n"; }
        total = after;
    }
    std::cout << "host-indexing-stress: " << (total == 0 ? "PASS" : "FAIL")
              << " (failures: " << total << ")\n";
    return total == 0 ? 0 : 1;
}
