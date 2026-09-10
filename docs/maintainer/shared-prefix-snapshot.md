# Complete shared-prefix snapshot format

`NINFSHR1` is the target-private persistence envelope for one complete immutable
`SharedPrefixHandle`. It is distinct from the `NINFSES1` private-continuation format: adding this
format does not change the `NINFSES1` version 3 writer, reader, bytes, or slot restore semantics.
The implementation and codec live in `Program`; Engine selects publication boundaries and
ResourceManager owns the logical shared catalog. A shared import never borrows, addresses, or
evicts a private continuation slot.

## Version 1 envelope

All integers and trivially-copyable records use the serving host representation. The supported
deployment is x86-64 little-endian; cross-endian transport is not supported. Fields are packed in
the following order without implicit envelope padding:

| Field | Encoding |
|---|---|
| magic | 8 bytes, ASCII `NINFSHR1` |
| version | `uint32`, currently 1 |
| total size | `uint64`, entire envelope and payload |
| payload size | `uint64`, bytes after this 60-byte header |
| payload checksum | 32-byte SHA-256 of every byte after the header |
| model binding | `uint32` byte count, then exact target/model/weights binding bytes |
| execution config | KV dtype, quant group and flags; speculative backend and draft window; page size; StateImage byte size; Main/backend plane counts and Host page strides; max context; token domain; proposal head; identity schema and identity tag |
| boundary | Main frontier, backend frontier, rope delta, tail-hidden validity, exact `PrefillWork`, and Main/backend page counts |
| provenance | evidence bits, structural origin bits, durability-ordered role, SSD eligibility, and optional cumulative first-volatile-token cutoff |
| exact identity | 32-byte SHA-256, two `uint64` shortlist-digest lanes, `uint64` byte count, then the identity payload |
| physical payload | one terminal StateImage, Main KV pages in logical order, then MTP KV pages in logical order |

The exact identity payload contains counted vectors for token IDs, token types, all three position
axes, media identity records, and rewrite execution frontiers. Version 1 deliberately requires an
empty media record set: media serving and in-memory prefix reuse remain supported, but media-aware
durable persistence is not yet defined. The identity digest binds semantic prefix identity; the
envelope checksum additionally binds State and every KV byte.

The StateImage is captured at the shared execution boundary. It is not reconstructed from a later
completion State. Main KV coverage ends exactly at the shared frontier. With MTP, backend coverage
ends at `frontier - 1`; without a backend it is empty. DFlash shared persistence is rejected.

## Compatibility and durability

Import requires exact equality for:

- target, model ID, and weights ID;
- tokenizer/rendered token identity, token types, three-axis positions, and rewrite frontiers;
- max context, token domain, position/layout geometry, KV format flags and strides;
- StateImage geometry; and
- speculative backend, draft window, proposal head, and derived identity tag.

Structural role and origin are provenance, not semantic identity. Coincident or repeated imports
therefore coalesce into one exact physical owner and merge cumulative provenance. A durable export
or import must be SSD eligible, carry classified structural origins and Engine-structural evidence,
have a non-transient role, and end strictly before its cumulative volatility cutoff. Volatile,
unclassified, media-bearing, or DFlash candidates are rejected.

## Validation, transfer, and adoption

The reader validates magic, version, exact envelope lengths, and a context/State/KV
geometry-derived maximum size before hashing. It then checks the payload checksum, bounded field
counts, configuration, boundary geometry, exact identity digest, token domain, and complete
consumption before any Device or Host State/KV reservation.

Export reserves the complete two-image assembly footprint before submitting a transfer. Existing
Host State/KV replicas copy directly to the immutable assembly buffer; they are not uploaded merely
to download them again. Device-only ranges remain pinned until their transfer event settles.
Imported State and KV retain complete immutable Host backing. Adoption may upload KV to establish
the Program's normal address-space representation, but publication is sealed and revalidated
before ResourceManager installs the logical owner.

ResourceManager first coalesces an exact resident identity. Otherwise it proves a vacant shared
catalog cell, asks Program to adopt the physical owner, revalidates the returned summary, and then
publishes it transactionally. Cancellation or capacity failure releases any unpublished owner;
checksum and compatibility failures allocate nothing. Valid existing shared and private owners and
their accounting remain unchanged.

Filesystem indexing, crash-consistent publication, and SSD eviction/replacement policy are outside
this format contract. Durable catalog code consumes the stable Program codec and Engine import
boundary rather than reimplementing target layout parsing.
