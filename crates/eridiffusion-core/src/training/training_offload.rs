//! Training-side block offloader (Phase 1 — core, no trainer integration).
//!
//! Manages frozen base-model block weights for memory-tight LoRA training:
//! pinned host slabs allocated once, persistent GPU slot slabs allocated
//! once, dedicated transfer stream, event-driven slot reuse safe under
//! forward and backward (recompute) visit orders.
//!
//! ## Layout
//!
//! ```text
//! safetensors  ─────►  PinnedBlockStore  ─────►  TransferEngine ──H2D──►  GpuSlotPool
//!                                                       │                       │
//!                                                       │ h2d_done evt          │ tensor views
//!                                                       ▼                       │
//!                                                BlockScheduler ◄── ResidencyMap ┘
//!                                                       ▲                       │
//!                                                       │                  TrainBlockHandle (RAII)
//!                                                       │                       │
//!                                                  BudgetManager           compute_done evt
//! ```
//!
//! ## Phase 1 scope
//!
//! - Core API and components only. **No trainer integration.**
//! - Frozen base weights only — LoRA parameters stay resident on the GPU
//!   and are owned outside this module.
//! - Forward visit `0..N` and backward visit `N..0` scheduling.
//! - Persistent GPU slabs sized at `init` for the largest block.
//! - Single-slot synchronous fallback when the budget cannot fit two slots.
//! - Synthetic `#[cfg(test)]` exercises the visit/scheduling/event path
//!   with fake tensors.
//!
//! Phase 2+ will integrate with trainers and compete with the resident
//! Klein loader on real workloads.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use flame_core::{
    memcpy_async_host_to_device, DType, PinnedAllocFlags, PinnedHostBuffer, Shape, Tensor,
};

use crate::training::block_offload::BlockFacilitator;

// ---------------------------------------------------------------------------
// CUDA event FFI (private — duplicate of block_offload.rs's helpers).
//
// Consolidating into a flame-core public helper is a separate refactor.
// Keeping a private copy here for Phase 1 keeps the cross-crate API surface
// unchanged.
// ---------------------------------------------------------------------------

extern "C" {
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;
const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;

struct CudaEvent {
    raw: *mut c_void,
}
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    fn new() -> anyhow::Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cudaEventCreateWithFlags(&mut raw, CUDA_EVENT_DISABLE_TIMING) };
        if s != 0 {
            anyhow::bail!("cudaEventCreateWithFlags failed: {s}");
        }
        Ok(Self { raw })
    }
    fn record_default(&self) -> anyhow::Result<()> {
        let s = unsafe { cudaEventRecord(self.raw, std::ptr::null_mut()) };
        if s != 0 {
            anyhow::bail!("cudaEventRecord (default) failed: {s}");
        }
        Ok(())
    }
    fn record_on(&self, stream: *mut c_void) -> anyhow::Result<()> {
        let s = unsafe { cudaEventRecord(self.raw, stream) };
        if s != 0 {
            anyhow::bail!("cudaEventRecord (stream) failed: {s}");
        }
        Ok(())
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cudaEventDestroy(self.raw);
        }
    }
}

fn stream_wait_event(stream: *mut c_void, event: &CudaEvent) -> anyhow::Result<()> {
    let s = unsafe { cudaStreamWaitEvent(stream, event.raw, 0) };
    if s != 0 {
        anyhow::bail!("cudaStreamWaitEvent failed: {s}");
    }
    Ok(())
}

fn default_stream_wait_event(event: &CudaEvent) -> anyhow::Result<()> {
    stream_wait_event(std::ptr::null_mut(), event)
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for a `TrainingBlockOffloader`.
#[derive(Clone, Debug)]
pub struct TrainingOffloadConfig {
    /// Number of GPU slot slabs to keep resident (typically 2 for double-
    /// buffer prefetch; falls back to 1 when budget is tight).
    pub gpu_slots: usize,
    /// How many blocks ahead the scheduler may prefetch.
    pub prefetch_window: usize,
    /// Total VRAM budget in bytes. `None` = no enforcement (use all available).
    pub vram_budget_bytes: Option<usize>,
    /// Bytes reserved for activation tensors / activation checkpoint storage.
    pub activation_reserve_bytes: usize,
    /// Bytes reserved for optimizer state (LoRA Adam, etc.).
    pub optimizer_reserve_bytes: usize,
    /// Bytes reserved for transient workspace (matmul/SDPA scratch).
    pub workspace_reserve_bytes: usize,
    /// Keep FP8 tensors as raw bytes in pinned memory; dequant on GPU during
    /// `prepare`. Halves pinned RAM for FP8 checkpoints.
    pub fp8_pinned: bool,
    /// On `release_for_decode`, also call `cudaMempoolTrimTo(0)`.
    pub trim_before_decode: bool,
    /// Phase 4 — VRAM watermark the budget planner is allowed to reach.
    /// 0.90 means "keep 10% of total VRAM unused as driver/activation
    /// margin". `max_slots` plans against this fraction of available
    /// bytes, and `over_high_watermark()` returns true when live usage
    /// has crossed it (skip speculative prefetch in that case).
    pub high_watermark: f32,
    /// Phase 4 — the trigger to evict non-active slots. Once live usage
    /// crosses `high_watermark`, the offloader should aim to bring usage
    /// back below `low_watermark` before the next prefetch. 0.80 is a
    /// reasonable default that leaves room for a full-slot H2D + the
    /// current compute's intermediates without thrashing.
    pub low_watermark: f32,
}

impl Default for TrainingOffloadConfig {
    fn default() -> Self {
        Self {
            gpu_slots: 2,
            prefetch_window: 1,
            vram_budget_bytes: None,
            activation_reserve_bytes: 0,
            optimizer_reserve_bytes: 0,
            workspace_reserve_bytes: 0,
            fp8_pinned: false,
            trim_before_decode: true,
            high_watermark: 0.90,
            low_watermark: 0.80,
        }
    }
}

/// Which pass the model is in for a `BlockVisit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainPass {
    Forward,
    /// Backward = activation-checkpoint recompute. Visits blocks in reverse.
    BackwardRecompute,
    /// Forward-only validation/inference path (no autograd).
    Validation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockVisit {
    pub pass: TrainPass,
    pub block_idx: usize,
}

impl BlockVisit {
    pub fn forward(block_idx: usize) -> Self {
        Self { pass: TrainPass::Forward, block_idx }
    }
    pub fn backward(block_idx: usize) -> Self {
        Self { pass: TrainPass::BackwardRecompute, block_idx }
    }

    /// Pick `Forward` or `BackwardRecompute` based on the current
    /// autograd recording state. Intended for use INSIDE an
    /// `AutogradContext::checkpoint` closure: the eager forward disables
    /// autograd (→ `Forward`) and the recompute re-enables it
    /// (→ `BackwardRecompute`), so the closure can emit the right
    /// `BlockVisit` without capturing any extra phase state.
    ///
    /// Outside a checkpoint closure, the caller should use `forward` /
    /// `backward` directly — this helper relies on the checkpoint
    /// wrapper's autograd-state flip and will mis-classify in other
    /// contexts.
    pub fn from_autograd_state(block_idx: usize) -> Self {
        if flame_core::autograd::AutogradContext::is_recording() {
            Self { pass: TrainPass::BackwardRecompute, block_idx }
        } else {
            Self { pass: TrainPass::Forward, block_idx }
        }
    }
}

/// Extension of `BlockFacilitator` for training. Exposes pass-order
/// information so the scheduler can drive forward+reverse prefetch and so
/// the loader can split frozen vs. trainable keys.
pub trait TrainBlockFacilitator: BlockFacilitator {
    /// Visit order for the forward pass — usually `0..N`.
    fn forward_order(&self) -> Vec<usize> {
        (0..self.block_count()).collect()
    }
    /// Visit order for the backward/recompute pass — usually `N..0`.
    fn backward_order(&self) -> Vec<usize> {
        (0..self.block_count()).rev().collect()
    }
    /// True if the key belongs to a trainable parameter (LoRA, etc.) and
    /// MUST stay resident on the GPU. The training offloader does not
    /// touch these.
    fn is_trainable_key(&self, key: &str) -> bool;
    /// True if the key belongs to a frozen base-model block and should be
    /// streamed via this offloader.
    fn is_frozen_block_key(&self, key: &str) -> bool;
    /// True if the key belongs to a non-block weight (embedders, final
    /// layer, etc.) that stays resident on the GPU.
    fn shared_resident_key(&self, key: &str) -> bool;
}

/// Per-step counters surfaced by `TrainingBlockOffloader::stats`.
#[derive(Default, Debug, Clone)]
pub struct OffloadStats {
    pub h2d_bytes: u64,
    pub h2d_count: u64,
    pub prefetch_hits: u64,
    pub prefetch_waits: u64,
    pub forced_syncs: u64,
    pub slot_reuse_waits: u64,
    pub steps: u64,
    pub fallback_single_slot_steps: u64,

    // Phase 3: per-direction counters — lets a trainer log how well
    // the backward-recompute prefetch is overlapping compute. High
    // `bwd_forced_syncs` or `bwd_prefetch_waits` means recompute is
    // stalling on H2D; low means prefetch is hiding the transfer.
    pub fwd_h2d_count: u64,
    pub fwd_prefetch_hits: u64,
    pub fwd_forced_syncs: u64,
    pub bwd_h2d_count: u64,
    pub bwd_prefetch_hits: u64,
    pub bwd_forced_syncs: u64,
    /// How many forward→backward transitions the offloader has seen this
    /// step. >1 means something unusual (multiple backward passes per
    /// step — fused backward? Phase 7).
    pub bwd_transitions: u64,
}

// ---------------------------------------------------------------------------
// PinnedBlockStore — packed slabs in pinned host memory
// ---------------------------------------------------------------------------

/// Per-tensor view metadata inside a packed pinned slab.
#[derive(Clone, Debug)]
pub(crate) struct PinnedTensorMeta {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Byte offset into the block's pinned slab.
    pub offset: usize,
    pub byte_len: usize,
}

/// Pinned host storage for a single block's frozen weights, packed into one
/// contiguous slab plus per-tensor view metadata.
pub(crate) struct PinnedBlock {
    pub buffer: PinnedHostBuffer<u8>,
    pub tensors: Vec<PinnedTensorMeta>,
    /// Block index (kept for diagnostic/Phase 2 use even though Phase 1
    /// indexes by Vec position).
    #[allow(dead_code)]
    pub block_idx: usize,
    pub total_bytes: usize,
}

/// Loads frozen block weights from safetensors into per-block pinned slabs
/// once at init. After load, file handles are closed and the store is the
/// single source of truth for block weights.
pub struct PinnedBlockStore {
    pub(crate) blocks: Vec<PinnedBlock>,
    pub(crate) total_bytes: usize,
}

impl PinnedBlockStore {
    /// Read safetensors files and pack each block's tensors into pinned
    /// slabs. Per-tensor allocations are avoided — one `cudaMallocHost` per
    /// block, plus per-tensor view metadata.
    ///
    /// Phase 1 implementation: keeps source dtype as-is in the slab (BF16
    /// for BF16, F32 for F32, etc.). Phase 2 will add the BF16-cast and
    /// FP8-pinned modes that `BlockOffloader::load_inner` already supports.
    pub fn load(
        paths: &[PathBuf],
        facilitator: &dyn TrainBlockFacilitator,
    ) -> anyhow::Result<Self> {
        let block_count = facilitator.block_count();

        // First pass: gather per-block (key, shape, dtype, raw bytes) from
        // every safetensors file. We accept duplicates from sharded models
        // by keying on `(block_idx, key)`.
        struct StagedTensor {
            key: String,
            shape: Vec<usize>,
            dtype_str: String,
            raw: Vec<u8>,
        }
        let mut staged: Vec<Vec<StagedTensor>> = (0..block_count).map(|_| Vec::new()).collect();
        let mut total_bytes: usize = 0;

        for path in paths {
            let path_str = path.to_string_lossy().to_string();
            let (header, data_start, mmap) = mmap_safetensors(&path_str)?;
            let metadata: serde_json::Value = serde_json::from_str(&header)?;
            let metadata_obj = metadata
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("invalid safetensors metadata"))?;

            for (name, info) in metadata_obj {
                if name == "__metadata__" {
                    continue;
                }
                let block_idx = match facilitator.classify_key(name) {
                    Some(idx) => idx,
                    None => continue,
                };
                if block_idx >= block_count {
                    anyhow::bail!(
                        "classify_key returned {block_idx} >= block_count {block_count} for {name:?}"
                    );
                }
                if !facilitator.is_frozen_block_key(name) {
                    // Trainable / shared keys are NOT staged here.
                    continue;
                }

                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing shape for {name}"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let num_elems: usize = shape.iter().product();
                if num_elems == 0 {
                    continue;
                }
                let offsets = info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing data_offsets for {name}"))?;
                let start = data_start
                    + offsets
                        .first()
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow::anyhow!("bad start offset for {name}"))?
                        as usize;
                let end = data_start
                    + offsets
                        .get(1)
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| anyhow::anyhow!("bad end offset for {name}"))?
                        as usize;
                let dtype_str = info["dtype"].as_str().unwrap_or("F32").to_string();
                if !matches!(
                    dtype_str.as_str(),
                    "F32" | "BF16" | "F16" | "F8_E4M3"
                ) {
                    continue;
                }
                staged[block_idx].push(StagedTensor {
                    key: name.clone(),
                    shape,
                    dtype_str,
                    raw: mmap[start..end].to_vec(),
                });
            }
        }

        // Second pass: pack each block into a pinned slab.
        let mut blocks: Vec<PinnedBlock> = Vec::with_capacity(block_count);
        for (block_idx, items) in staged.into_iter().enumerate() {
            // Sort keys deterministically so cross-run layouts are stable.
            let mut items = items;
            items.sort_by(|a, b| a.key.cmp(&b.key));

            let block_total: usize = items.iter().map(|t| t.raw.len()).sum();
            if block_total == 0 {
                blocks.push(PinnedBlock {
                    buffer: PinnedHostBuffer::<u8>::with_capacity_elems(
                        1,
                        PinnedAllocFlags::DEFAULT,
                    )
                    .map_err(|e| anyhow::anyhow!("pinned alloc (empty block) {block_idx}: {e}"))?,
                    tensors: Vec::new(),
                    block_idx,
                    total_bytes: 0,
                });
                continue;
            }
            let mut pinned = PinnedHostBuffer::<u8>::with_capacity_elems(
                block_total,
                PinnedAllocFlags::DEFAULT,
            )
            .map_err(|e| anyhow::anyhow!("pinned alloc block {block_idx}: {e}"))?;
            // Pack tensors back-to-back in the slab.
            let mut metas: Vec<PinnedTensorMeta> = Vec::with_capacity(items.len());
            let mut cursor: usize = 0;
            {
                let dst = pinned.as_mut_bytes();
                for it in &items {
                    let n = it.raw.len();
                    dst[cursor..cursor + n].copy_from_slice(&it.raw);
                    metas.push(PinnedTensorMeta {
                        key: it.key.clone(),
                        shape: it.shape.clone(),
                        dtype: parse_dtype_from_safetensors(&it.dtype_str),
                        offset: cursor,
                        byte_len: n,
                    });
                    cursor += n;
                }
            }
            unsafe {
                pinned.set_len(cursor);
            }
            total_bytes += cursor;
            blocks.push(PinnedBlock {
                buffer: pinned,
                tensors: metas,
                block_idx,
                total_bytes: cursor,
            });
        }

        Ok(Self { blocks, total_bytes })
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn pinned_bytes(&self) -> usize {
        self.total_bytes
    }

    pub(crate) fn block(&self, idx: usize) -> &PinnedBlock {
        &self.blocks[idx]
    }

    /// Largest packed-block byte size — used by `GpuSlotPool` to size each
    /// persistent GPU slab so any block can be loaded into any slot.
    pub fn max_block_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.total_bytes).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// GpuSlotPool — persistent GPU slabs
// ---------------------------------------------------------------------------

struct SlotEvents {
    compute_done: CudaEvent,
    h2d_done: CudaEvent,
    /// True iff a `TrainBlockHandle` for this slot has been dropped.
    compute_recorded: AtomicBool,
    /// True iff `h2d_done` has been recorded for the slot's current contents.
    h2d_recorded: AtomicBool,
}

impl SlotEvents {
    fn new() -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            compute_done: CudaEvent::new()?,
            h2d_done: CudaEvent::new()?,
            compute_recorded: AtomicBool::new(false),
            h2d_recorded: AtomicBool::new(false),
        }))
    }
}

/// One persistent GPU slot. The `slab` is allocated once at init and reused
/// across blocks via H2D overwrite — no per-block GPU `cudaMalloc`.
struct GpuSlot {
    slab: CudaSlice<u8>,
    capacity_bytes: usize,
    /// Block currently materialized into the slab (None = unused).
    current_block: Option<usize>,
    /// Tensor views over the slab for the current block. Recreated each
    /// time a new block is loaded.
    tensors: Arc<HashMap<String, Tensor>>,
    events: Arc<SlotEvents>,
}

/// Owns N persistent GPU slots, each sized to fit the largest block.
pub struct GpuSlotPool {
    slots: Vec<GpuSlot>,
    /// Kept for Phase 2 to allow re-allocating slabs (e.g. shrink/grow).
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
}

impl GpuSlotPool {
    pub fn new(
        device: Arc<CudaDevice>,
        num_slots: usize,
        slab_bytes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_slots >= 1, "GpuSlotPool needs at least one slot");
        anyhow::ensure!(slab_bytes > 0, "GpuSlotPool slab_bytes must be > 0");
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            let slab = unsafe { device.alloc::<u8>(slab_bytes) }
                .map_err(|e| anyhow::anyhow!("slab alloc {slab_bytes} bytes: {e:?}"))?;
            let events = SlotEvents::new()?;
            slots.push(GpuSlot {
                slab,
                capacity_bytes: slab_bytes,
                current_block: None,
                tensors: Arc::new(HashMap::new()),
                events,
            });
        }
        Ok(Self { slots, device })
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn slab_bytes(&self) -> usize {
        self.slots.first().map(|s| s.capacity_bytes).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// TransferEngine — H2D stream + event recording
// ---------------------------------------------------------------------------

/// Owns the dedicated CUDA stream used for async H2D copies into GPU slots.
pub struct TransferEngine {
    raw_stream: *mut c_void,
    device: Arc<CudaDevice>,
}

unsafe impl Send for TransferEngine {}
unsafe impl Sync for TransferEngine {}

impl TransferEngine {
    pub fn new(device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING) };
        if s != 0 {
            anyhow::bail!("cudaStreamCreateWithFlags (transfer): {s}");
        }
        Ok(Self { raw_stream: raw, device })
    }

    pub fn raw_stream(&self) -> *mut c_void {
        self.raw_stream
    }

    /// Issue an async H2D for one tensor view from the pinned slab into
    /// the slot slab. Both pointers and length are in bytes.
    fn h2d(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> anyhow::Result<()> {
        memcpy_async_host_to_device(dst, src, bytes, self.raw_stream)
            .map_err(|e| anyhow::anyhow!("memcpy_async_host_to_device: {e}"))
    }

    fn record(&self, event: &CudaEvent) -> anyhow::Result<()> {
        event.record_on(self.raw_stream)
    }

    fn synchronize(&self) -> anyhow::Result<()> {
        let s = unsafe { cudaStreamSynchronize(self.raw_stream) };
        if s != 0 {
            anyhow::bail!("cudaStreamSynchronize (transfer): {s}");
        }
        Ok(())
    }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        unsafe {
            cudaStreamDestroy(self.raw_stream);
        }
    }
}

// ---------------------------------------------------------------------------
// ResidencyMap — explicit per-block state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockResidency {
    /// In pinned host slab, not currently on GPU.
    HostStaged,
    /// H2D in flight on the transfer stream.
    Prefetching,
    /// On the GPU, no live `TrainBlockHandle` yet.
    GpuReady,
    /// On the GPU, at least one `TrainBlockHandle` is live.
    InUse,
    /// On the GPU, all handles dropped, slot may be reused.
    Releasable,
}

/// Tracks which slot (if any) currently holds each block, and what its
/// residency state is. Block state is the **shared mutex** that
/// `TrainBlockHandle::Drop` writes to — so the offloader and outstanding
/// handles agree on which slots are `Releasable`.
pub struct ResidencyMap {
    /// Shared with all outstanding `TrainBlockHandle`s. Drop transitions
    /// `InUse → Releasable` here.
    block_state: Arc<Mutex<Vec<BlockResidency>>>,
    /// `slot_for[block_idx]` is the slot index holding the block, or
    /// `usize::MAX` if it is `HostStaged`.
    slot_for: Vec<usize>,
    /// Reverse index: which block (if any) sits in each slot.
    block_in_slot: Vec<Option<usize>>,
}

impl ResidencyMap {
    pub fn new(num_blocks: usize, num_slots: usize) -> Self {
        Self {
            block_state: Arc::new(Mutex::new(vec![BlockResidency::HostStaged; num_blocks])),
            slot_for: vec![usize::MAX; num_blocks],
            block_in_slot: vec![None; num_slots],
        }
    }
    pub fn state(&self, block_idx: usize) -> BlockResidency {
        self.block_state.lock().expect("residency mutex poisoned")[block_idx]
    }
    pub fn slot_for(&self, block_idx: usize) -> Option<usize> {
        let s = self.slot_for[block_idx];
        if s == usize::MAX {
            None
        } else {
            Some(s)
        }
    }
    pub fn block_in(&self, slot_idx: usize) -> Option<usize> {
        self.block_in_slot[slot_idx]
    }
    pub fn assign(&mut self, block_idx: usize, slot_idx: usize, state: BlockResidency) {
        let mut bs = self.block_state.lock().expect("residency mutex poisoned");
        // Evict whoever was in the slot.
        if let Some(prev) = self.block_in_slot[slot_idx] {
            bs[prev] = BlockResidency::HostStaged;
            self.slot_for[prev] = usize::MAX;
        }
        self.block_in_slot[slot_idx] = Some(block_idx);
        self.slot_for[block_idx] = slot_idx;
        bs[block_idx] = state;
    }
    pub fn set_state(&mut self, block_idx: usize, state: BlockResidency) {
        self.block_state.lock().expect("residency mutex poisoned")[block_idx] = state;
    }
    /// Borrow the shared block-state vector — handed to `TrainBlockHandle`
    /// so its Drop can transition InUse → Releasable.
    pub fn shared_state(&self) -> Arc<Mutex<Vec<BlockResidency>>> {
        self.block_state.clone()
    }
    /// Find a slot eligible for reuse, preferring `Releasable` over
    /// `GpuReady`, and never picking a slot whose current block is `InUse`
    /// or `Prefetching`. Returns the slot index or `None` if no slot can be
    /// reused right now.
    pub fn pick_evictable_slot(&self, exclude_block: Option<usize>) -> Option<usize> {
        let bs = self.block_state.lock().expect("residency mutex poisoned");
        // Pass 1: prefer Empty, then Releasable.
        let mut empty = None;
        let mut releasable = None;
        for (slot_idx, occ) in self.block_in_slot.iter().enumerate() {
            match occ {
                None => {
                    if empty.is_none() {
                        empty = Some(slot_idx);
                    }
                }
                Some(b) => {
                    if Some(*b) == exclude_block {
                        continue;
                    }
                    let state = bs[*b];
                    if matches!(state, BlockResidency::Releasable)
                        && releasable.is_none()
                    {
                        releasable = Some(slot_idx);
                    }
                }
            }
        }
        if empty.is_some() {
            return empty;
        }
        if releasable.is_some() {
            return releasable;
        }
        // Pass 2: take any GpuReady slot (not InUse, not Prefetching).
        for (slot_idx, occ) in self.block_in_slot.iter().enumerate() {
            if let Some(b) = occ {
                if Some(*b) == exclude_block {
                    continue;
                }
                if matches!(bs[*b], BlockResidency::GpuReady) {
                    return Some(slot_idx);
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// BlockScheduler — drives prefetch order
// ---------------------------------------------------------------------------

/// Plans prefetch order for a pass. The scheduler is intentionally dumb:
/// it knows only the visit list and the prefetch window. The offloader
/// asks "should I prefetch block X next?" — the scheduler answers based on
/// pass direction and how far ahead the caller has already advanced.
pub struct BlockScheduler {
    forward_order: Vec<usize>,
    backward_order: Vec<usize>,
    prefetch_window: usize,
}

impl BlockScheduler {
    pub fn new(facilitator: &dyn TrainBlockFacilitator, prefetch_window: usize) -> Self {
        Self {
            forward_order: facilitator.forward_order(),
            backward_order: facilitator.backward_order(),
            prefetch_window: prefetch_window.max(1),
        }
    }

    fn order_for(&self, pass: TrainPass) -> &[usize] {
        match pass {
            TrainPass::Forward | TrainPass::Validation => &self.forward_order,
            TrainPass::BackwardRecompute => &self.backward_order,
        }
    }

    /// Given the current visit, return the next block to prefetch (one
    /// position ahead in the visit order), or `None` if at end of pass.
    pub fn next_prefetch(&self, current: BlockVisit) -> Option<BlockVisit> {
        let order = self.order_for(current.pass);
        let cur_pos = order.iter().position(|&b| b == current.block_idx)?;
        let next_pos = cur_pos.saturating_add(self.prefetch_window);
        order.get(next_pos).map(|&b| BlockVisit {
            pass: current.pass,
            block_idx: b,
        })
    }
}

// ---------------------------------------------------------------------------
// BudgetManager — VRAM accounting
// ---------------------------------------------------------------------------

/// Decides how many GPU slots can be live based on the configured reserves
/// and the per-block size. Falls back to a single slot when budget is tight.
pub struct BudgetManager {
    config: TrainingOffloadConfig,
    reserved_bytes: usize,
}

impl BudgetManager {
    pub fn new(config: TrainingOffloadConfig) -> Self {
        let reserved_bytes = config
            .activation_reserve_bytes
            .saturating_add(config.optimizer_reserve_bytes)
            .saturating_add(config.workspace_reserve_bytes);
        Self { config, reserved_bytes }
    }

    /// Maximum number of GPU slots to allocate, given each slab's byte
    /// size. `available_bytes` is the budget AFTER subtracting shared
    /// resident weights (LoRA + non-block resident) — the caller must
    /// supply that figure.
    ///
    /// Auto-slot logic (Phase 4):
    /// * Take `high_watermark` fraction of available VRAM (leaves driver
    ///   margin + activation headroom for the peak autograd step).
    /// * Subtract fixed reserves (activation/optimizer/workspace).
    /// * Divide by slab size, clamp to `[1, config.gpu_slots]`.
    /// * If the configured `gpu_slots` value is tighter than the budget
    ///   math allows, the config wins (explicit-override semantics).
    pub fn max_slots(&self, slab_bytes: usize, available_bytes: usize) -> usize {
        if slab_bytes == 0 {
            return self.config.gpu_slots.max(1);
        }
        let watermarked = ((available_bytes as f64) * self.config.high_watermark as f64) as usize;
        let usable = watermarked.saturating_sub(self.reserved_bytes);
        let by_budget = usable / slab_bytes;
        // Config's `gpu_slots` is the cap (explicit override). Minimum 1.
        let n = self.config.gpu_slots.min(by_budget).max(1);
        n
    }

    pub fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    /// Live VRAM check — used by runtime budget enforcement to decide
    /// whether to proceed with a speculative prefetch. Returns
    /// `(free_bytes, total_bytes, high_watermark_bytes)`. If the CUDA
    /// query fails the offloader should fall back to the static plan.
    pub fn query_live_vram(&self) -> Option<(usize, usize, usize)> {
        let (free, total) = flame_core::cuda::utils::cuda_mem_get_info().ok()?;
        let high = ((total as f64) * self.config.high_watermark as f64) as usize;
        Some((free, total, high))
    }

    /// True when the live free VRAM has dropped below the fraction of
    /// total that `high_watermark` permits — i.e. we are OVER the
    /// watermark and should hold off on speculative prefetch.
    pub fn over_high_watermark(&self) -> bool {
        let Some((free, total, high)) = self.query_live_vram() else {
            return false;
        };
        let used = total.saturating_sub(free);
        used > high
    }
}

// ---------------------------------------------------------------------------
// TrainBlockHandle — RAII handle returned by await_block
// ---------------------------------------------------------------------------

/// RAII scoped handle for a block currently materialized on a GPU slot.
///
/// Holding the handle marks the slot as `InUse`. Dropping it records a
/// `compute_done` event on the default stream and marks the slot as
/// `Releasable`, so the next prefetch can wait on the GPU instead of the
/// host.
///
/// Deref to `&HashMap<String, Tensor>` for ergonomic tensor lookup.
///
/// **Lifetime contract**: drop only after every kernel that reads the
/// block's weights has been queued on the default stream. Dropping mid-
/// block would let a subsequent prefetch overwrite the slot before the
/// in-flight kernels read it.
///
/// **Capture rule**: never clone the underlying tensor `Arc` into a
/// checkpoint closure. Recompute paths must reacquire a handle through
/// `await_block(BackwardRecompute, ...)`.
pub struct TrainBlockHandle {
    block_idx: usize,
    tensors: Arc<HashMap<String, Tensor>>,
    events: Arc<SlotEvents>,
    /// Shared with the parent offloader's `ResidencyMap` so Drop can update
    /// the block's state to `Releasable`.
    block_state: Arc<Mutex<Vec<BlockResidency>>>,
    /// Counter shared with the offloader for diagnostics; decremented on Drop.
    handle_count: Arc<AtomicU64>,
}

impl TrainBlockHandle {
    pub fn block_idx(&self) -> usize {
        self.block_idx
    }
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }
    /// Clone the underlying `Arc<HashMap>`. The slot's compute lifetime
    /// is still tied to the handle's drop, not the cloned Arc — callers
    /// must not retain the cloned Arc past the handle's drop.
    pub fn arc(&self) -> Arc<HashMap<String, Tensor>> {
        Arc::clone(&self.tensors)
    }
}

impl std::ops::Deref for TrainBlockHandle {
    type Target = HashMap<String, Tensor>;
    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

impl Drop for TrainBlockHandle {
    fn drop(&mut self) {
        if let Err(e) = self.events.compute_done.record_default() {
            log::error!("TrainBlockHandle drop: record compute_done: {e}");
        }
        self.events.compute_recorded.store(true, Ordering::Release);
        if let Ok(mut r) = self.block_state.lock() {
            if let Some(slot) = r.get_mut(self.block_idx) {
                if matches!(*slot, BlockResidency::InUse) {
                    *slot = BlockResidency::Releasable;
                }
            }
        }
        self.handle_count.fetch_sub(1, Ordering::AcqRel);
    }
}

// ---------------------------------------------------------------------------
// TrainingBlockOffloader — top-level
// ---------------------------------------------------------------------------

/// Top-level training-side block offloader. Owns the pinned store, GPU
/// slots, transfer engine, residency map, scheduler, and budget manager.
pub struct TrainingBlockOffloader {
    pub(crate) store: PinnedBlockStore,
    pool: GpuSlotPool,
    transfer: TransferEngine,
    residency: ResidencyMap,
    scheduler: BlockScheduler,
    /// Active VRAM budget. Held even when idle so Phase 2 budget enforcement
    /// can re-evaluate live activation/optimizer reserves between steps.
    #[allow(dead_code)]
    budget: BudgetManager,
    config: TrainingOffloadConfig,
    stats: OffloadStats,
    handle_count: Arc<AtomicU64>,
    /// Step counter — bumped by `begin_step`.
    step: u64,
    /// Block currently being prefetched (if any).
    prefetch_in_flight: Option<usize>,
    /// Last-observed pass. Phase 3 uses this to auto-detect the
    /// forward→backward transition: when `await_block` sees a
    /// `BackwardRecompute` visit and the prior pass was `Forward`,
    /// we treat it as the start of backward (reset prefetch_in_flight
    /// so the next `prefetch` call plans in reverse order from
    /// the actual current block, not a stale forward cursor).
    last_pass: Option<TrainPass>,
}

// SAFETY: TrainingBlockOffloader is always accessed through &mut self.
unsafe impl Send for TrainingBlockOffloader {}
unsafe impl Sync for TrainingBlockOffloader {}

impl TrainingBlockOffloader {
    /// Construct from a pinned store + facilitator. Allocates persistent
    /// GPU slots sized to the largest block.
    pub fn new(
        store: PinnedBlockStore,
        facilitator: &dyn TrainBlockFacilitator,
        device: Arc<CudaDevice>,
        config: TrainingOffloadConfig,
    ) -> anyhow::Result<Self> {
        let max_block = store.max_block_bytes().max(1);
        let budget = BudgetManager::new(config.clone());
        // Use available VRAM hint, default to 24 GB if no budget provided.
        let available_hint = config.vram_budget_bytes.unwrap_or(24 * 1024 * 1024 * 1024);
        let num_slots = budget.max_slots(max_block, available_hint);
        let pool = GpuSlotPool::new(device.clone(), num_slots, max_block)?;
        let transfer = TransferEngine::new(device)?;
        let scheduler = BlockScheduler::new(facilitator, config.prefetch_window);
        let block_count = store.block_count();
        let residency = ResidencyMap::new(block_count, num_slots);
        let handle_count = Arc::new(AtomicU64::new(0));
        Ok(Self {
            store,
            pool,
            transfer,
            residency,
            scheduler,
            budget,
            config,
            stats: OffloadStats::default(),
            handle_count,
            step: 0,
            prefetch_in_flight: None,
            last_pass: None,
        })
    }

    /// Convenience: load + construct in one call.
    pub fn load_frozen_blocks<F: TrainBlockFacilitator>(
        paths: &[PathBuf],
        facilitator: &F,
        device: Arc<CudaDevice>,
        config: TrainingOffloadConfig,
    ) -> anyhow::Result<Self> {
        let store = PinnedBlockStore::load(paths, facilitator)?;
        Self::new(store, facilitator, device, config)
    }

    pub fn begin_step(&mut self, step: u64) -> anyhow::Result<()> {
        self.step = step;
        self.stats.steps += 1;
        if self.pool.num_slots() == 1 {
            self.stats.fallback_single_slot_steps += 1;
        }
        self.last_pass = None;
        Ok(())
    }

    pub fn end_step(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Explicit forward→backward transition. Optional — `await_block` also
    /// auto-detects the transition when it sees `BackwardRecompute`. Useful
    /// when the trainer wants to pre-queue a backward-direction prefetch
    /// (block N-1) before the first checkpoint closure runs.
    ///
    /// Resets `prefetch_in_flight` so the next prefetch plan starts from
    /// the actual backward cursor, not a stale forward cursor. Does NOT
    /// touch live slots — blocks in `InUse` or `Releasable` stay where
    /// they are; `issue_h2d` handles slot reuse correctly via the
    /// compute-done event gate.
    pub fn begin_backward(&mut self) {
        if !matches!(self.last_pass, Some(TrainPass::BackwardRecompute)) {
            self.stats.bwd_transitions += 1;
        }
        self.prefetch_in_flight = None;
        self.last_pass = Some(TrainPass::BackwardRecompute);
    }

    /// Suggest the next prefetch given the current visit. Returns the
    /// chosen block idx (after issuing H2D), or `None` if no prefetch is
    /// possible (end of pass or no slot available right now).
    ///
    /// Phase 3: when `visit.pass == BackwardRecompute`, the scheduler
    /// picks the next block in reverse order (N-1, N-2, ...).
    /// Phase 4: `over_high_watermark()` check early-exits prefetch when
    /// live VRAM crosses the configured budget, falling back to the
    /// synchronous load path in `await_block`. Cheap: one cudaMemGetInfo
    /// per call; the query is already used by block_offload's own
    /// telemetry so there's no new dependency.
    pub fn prefetch(&mut self, visit: BlockVisit) -> anyhow::Result<Option<usize>> {
        self.note_pass(visit.pass);
        // Phase 4 budget enforcement: skip speculative prefetch when we're
        // over the high watermark. The next `await_block` will still
        // synchronously load on a forced sync (counted via stats).
        if self.budget.over_high_watermark() {
            self.stats.prefetch_waits += 1;
            return Ok(None);
        }
        let Some(next) = self.scheduler.next_prefetch(visit) else {
            return Ok(None);
        };
        // Already on a slot?
        if self.residency.slot_for(next.block_idx).is_some() {
            self.stats.prefetch_hits += 1;
            match visit.pass {
                TrainPass::BackwardRecompute => self.stats.bwd_prefetch_hits += 1,
                _ => self.stats.fwd_prefetch_hits += 1,
            }
            return Ok(Some(next.block_idx));
        }
        // Single-slot mode: no prefetch overlap available.
        if self.pool.num_slots() == 1 {
            return Ok(None);
        }
        let exclude = self
            .residency
            .slot_for(visit.block_idx)
            .or(self.prefetch_in_flight.and_then(|b| self.residency.slot_for(b)));
        let Some(target) = self.residency.pick_evictable_slot(exclude) else {
            // No evictable slot — caller is holding everything live.
            return Ok(None);
        };
        self.issue_h2d(next.block_idx, target)?;
        match visit.pass {
            TrainPass::BackwardRecompute => self.stats.bwd_h2d_count += 1,
            _ => self.stats.fwd_h2d_count += 1,
        }
        Ok(Some(next.block_idx))
    }

    /// Internal: note pass transitions. Auto-triggers `begin_backward`'s
    /// cursor reset when we first see a `BackwardRecompute` visit after
    /// a `Forward` visit in the same step.
    #[inline]
    fn note_pass(&mut self, pass: TrainPass) {
        match (self.last_pass, pass) {
            (Some(TrainPass::Forward), TrainPass::BackwardRecompute)
            | (Some(TrainPass::Validation), TrainPass::BackwardRecompute)
            | (None, TrainPass::BackwardRecompute) => {
                // forward → backward transition
                self.stats.bwd_transitions += 1;
                self.prefetch_in_flight = None;
            }
            _ => {}
        }
        self.last_pass = Some(pass);
    }

    /// Block until the requested block is on the GPU and return a scoped
    /// handle. Issues a synchronous load if the block was not already
    /// prefetched.
    ///
    /// Phase 3: auto-detects forward→backward transitions. The first
    /// `BackwardRecompute` visit after a `Forward` pass resets the
    /// prefetch cursor so subsequent `prefetch()` calls plan in reverse
    /// order without the trainer needing to call `begin_backward()`
    /// explicitly.
    pub fn await_block(&mut self, visit: BlockVisit) -> anyhow::Result<TrainBlockHandle> {
        self.note_pass(visit.pass);
        let block_idx = visit.block_idx;

        // Already in a slot?
        if let Some(slot_idx) = self.residency.slot_for(block_idx) {
            // If still Prefetching, gate default stream on h2d_done.
            let state = self.residency.state(block_idx);
            if matches!(state, BlockResidency::Prefetching) {
                self.gate_default_on_h2d(slot_idx)?;
                self.residency.set_state(block_idx, BlockResidency::GpuReady);
                self.prefetch_in_flight = None;
            }
            match visit.pass {
                TrainPass::BackwardRecompute => self.stats.bwd_prefetch_hits += 1,
                _ => self.stats.fwd_prefetch_hits += 1,
            }
            return self.make_handle(block_idx, slot_idx);
        }

        // Not on a slot — synchronous load into the next evictable slot.
        let target = self
            .residency
            .pick_evictable_slot(None)
            .ok_or_else(|| anyhow::anyhow!("await_block: no evictable slot for block {block_idx}"))?;
        self.issue_h2d(block_idx, target)?;
        // Wait on the just-issued H2D from the default stream.
        self.gate_default_on_h2d(target)?;
        self.residency.set_state(block_idx, BlockResidency::GpuReady);
        self.prefetch_in_flight = None;
        self.stats.forced_syncs += 1;
        match visit.pass {
            TrainPass::BackwardRecompute => {
                self.stats.bwd_forced_syncs += 1;
                self.stats.bwd_h2d_count += 1;
            }
            _ => {
                self.stats.fwd_forced_syncs += 1;
                self.stats.fwd_h2d_count += 1;
            }
        }

        self.make_handle(block_idx, target)
    }

    /// Drain in-flight transfers, drop all GPU slot views, and synchronize
    /// the device. Use before VAE/decode to free VRAM.
    pub fn evict_all_sync(&mut self) -> anyhow::Result<()> {
        if self.prefetch_in_flight.is_some() {
            self.transfer.synchronize()?;
            self.prefetch_in_flight = None;
        }
        let s = unsafe { cudaDeviceSynchronize() };
        if s != 0 {
            anyhow::bail!("cudaDeviceSynchronize (evict_all_sync): {s}");
        }
        for slot in &mut self.pool.slots {
            slot.current_block = None;
            slot.tensors = Arc::new(HashMap::new());
        }
        let n_slots = self.pool.num_slots();
        let n_blocks = self.store.block_count();
        // Reset block state via the shared mutex so any outstanding handles
        // that haven't dropped yet still see consistent state on their Drop.
        if let Ok(mut r) = self.residency.shared_state().lock() {
            r.iter_mut().for_each(|s| *s = BlockResidency::HostStaged);
        }
        self.residency = ResidencyMap::new(n_blocks, n_slots);
        Ok(())
    }

    /// Evict all GPU slots + (optionally) trim the CUDA allocator pool.
    /// Called right before VAE decode to maximize free VRAM and avoid the
    /// known VAE OOM failure mode on 24 GB cards.
    ///
    /// Phase 4 additions: drop the per-slot slab allocations (not just
    /// the tensor views) so their bytes go back to the allocator, and
    /// log free VRAM before/after so the trainer can surface the freed
    /// amount without a separate query.
    pub fn release_for_decode(&mut self) -> anyhow::Result<()> {
        let before = self.budget.query_live_vram();
        self.evict_all_sync()?;
        // Drop the slab CudaSlice allocations themselves — cudarc frees
        // the underlying device buffer when the slice Drops, returning
        // the bytes to the pool. After this, subsequent forward() calls
        // that need a slot will hit the evictable pool again; but the
        // release_for_decode caller is about to switch into sampling/VAE
        // and won't need slots.
        let n_slots = self.pool.num_slots();
        let slab_bytes = self.pool.slab_bytes();
        self.pool = GpuSlotPool::new(self.transfer.device.clone(), 1, 1)?;
        let n_blocks = self.store.block_count();
        self.residency = ResidencyMap::new(n_blocks, 1);
        if self.config.trim_before_decode {
            // Best-effort. Pool trim is in flame-core utilities.
            // (Not all builds expose it; ignore failures.)
            let _ = flame_core::cuda_alloc_pool::clear_pool_cache();
        }
        let after = self.budget.query_live_vram();
        if let (Some((f0, _, _)), Some((f1, _, _))) = (before, after) {
            log::debug!(
                "release_for_decode: free VRAM {} → {} MB (+{} MB freed, was {} slots × {} MB)",
                f0 / (1024 * 1024),
                f1 / (1024 * 1024),
                f1.saturating_sub(f0) / (1024 * 1024),
                n_slots,
                slab_bytes / (1024 * 1024),
            );
        }
        Ok(())
    }

    /// Phase 4 — re-allocate slots after `release_for_decode`. Called by
    /// the trainer when the sampling/decode phase is done and training
    /// resumes. No-op if slots are still allocated (caller did not
    /// release).
    pub fn reacquire_slots(&mut self) -> anyhow::Result<()> {
        // If we still have real slots, bail.
        if self.pool.slab_bytes() > 1 {
            return Ok(());
        }
        let max_block = self.store.max_block_bytes().max(1);
        let available_hint = self
            .config
            .vram_budget_bytes
            .unwrap_or(24 * 1024 * 1024 * 1024);
        let num_slots = self.budget.max_slots(max_block, available_hint);
        self.pool = GpuSlotPool::new(self.transfer.device.clone(), num_slots, max_block)?;
        let n_blocks = self.store.block_count();
        self.residency = ResidencyMap::new(n_blocks, num_slots);
        Ok(())
    }

    pub fn stats(&self) -> OffloadStats {
        self.stats.clone()
    }

    pub fn num_slots(&self) -> usize {
        self.pool.num_slots()
    }

    pub fn pinned_bytes(&self) -> usize {
        self.store.pinned_bytes()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn issue_h2d(&mut self, block_idx: usize, target: usize) -> anyhow::Result<()> {
        // Phase 0 safety: wait for prior compute on the slot we are about
        // to overwrite.
        let prior_block = self.residency.block_in(target);
        if let Some(pb) = prior_block {
            let prior_state = self.residency.state(pb);
            if matches!(prior_state, BlockResidency::Releasable | BlockResidency::GpuReady) {
                let prior_events = &self.pool.slots[target].events;
                if prior_events.compute_recorded.load(Ordering::Acquire) {
                    stream_wait_event(self.transfer.raw_stream(), &prior_events.compute_done)?;
                    self.stats.slot_reuse_waits += 1;
                } else {
                    let s = unsafe { cudaDeviceSynchronize() };
                    if s != 0 {
                        anyhow::bail!("cudaDeviceSynchronize before slot reuse: {s}");
                    }
                    self.stats.forced_syncs += 1;
                }
            }
        }

        // New per-slot events for this load.
        let events = SlotEvents::new()?;
        let pinned = self.store.block(block_idx);
        let slab_ptr = (*self.pool.slots[target].slab.device_ptr()) as *mut c_void;

        // Build tensor views over the slab and issue per-tensor H2D copies.
        // Each view lives at `slab_ptr + meta.offset`. We cannot safely
        // construct a `Tensor` from a raw GPU pointer without going through
        // flame-core's tensor allocator, so for Phase 1 we materialize fresh
        // BF16 tensors via `Tensor::from_bf16_slice_gpu`. This matches what
        // the inference offloader does today (per-tensor allocation per
        // load) — Phase 2 will switch to true slab-view tensors with no
        // per-load alloc.
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(pinned.tensors.len());
        let stream = self.transfer.raw_stream();
        let mut bytes_sent: u64 = 0;
        for meta in &pinned.tensors {
            // Only handle BF16 in Phase 1 — F32/F16/FP8 fall back to a
            // pre-cast via flame-core conversion (TODO Phase 2).
            anyhow::ensure!(
                matches!(meta.dtype, DType::BF16),
                "TrainingBlockOffloader Phase 1 expects BF16 frozen weights, got {:?} for {}",
                meta.dtype,
                meta.key
            );
            let num_elems = meta.byte_len / 2;
            let gpu_buf = unsafe { self.transfer.device.alloc::<u16>(num_elems) }
                .map_err(|e| anyhow::anyhow!("GPU alloc {} bytes: {e:?}", meta.byte_len))?;
            let dst = (*gpu_buf.device_ptr()) as *mut c_void;
            let src = unsafe {
                pinned.buffer.as_ptr().add(meta.offset)
            } as *const c_void;
            self.transfer.h2d(dst, src, meta.byte_len)?;
            bytes_sent += meta.byte_len as u64;
            let tensor = Tensor::from_bf16_slice_gpu(
                gpu_buf,
                Shape::from_dims(&meta.shape),
                self.transfer.device.clone(),
            );
            // Mirror BlockOffloader::prepare_weights: rank-2 `.weight`
            // tensors are transposed at load time (stride view, no copy)
            // so forward's `x @ W` sees the expected [in, out] layout
            // instead of safetensors' on-disk [out, in].
            let tensor = if meta.key.ends_with(".weight")
                && meta.shape.len() == 2
                && !meta.key.ends_with(".scale")
            {
                tensor
                    .transpose()
                    .map_err(|e| anyhow::anyhow!("transpose {}: {e:?}", meta.key))?
                    .requires_grad_(false)
            } else {
                tensor
            };
            tensors.insert(meta.key.clone(), tensor);
        }
        // Touch the slab so the compiler doesn't optimize the persistent
        // allocation away in case future Phase 2 work pivots to slab views.
        let _ = slab_ptr;

        self.transfer.record(&events.h2d_done)?;
        events.h2d_recorded.store(true, Ordering::Release);

        let arc = Arc::new(tensors);
        self.pool.slots[target].current_block = Some(block_idx);
        self.pool.slots[target].tensors = arc;
        self.pool.slots[target].events = events;
        self.residency.assign(block_idx, target, BlockResidency::Prefetching);
        self.prefetch_in_flight = Some(block_idx);
        self.stats.h2d_count += 1;
        self.stats.h2d_bytes += bytes_sent;
        Ok(())
    }

    fn gate_default_on_h2d(&mut self, slot_idx: usize) -> anyhow::Result<()> {
        let events = &self.pool.slots[slot_idx].events;
        if events.h2d_recorded.load(Ordering::Acquire) {
            default_stream_wait_event(&events.h2d_done)?;
        } else {
            self.transfer.synchronize()?;
            self.stats.forced_syncs += 1;
        }
        Ok(())
    }

    fn make_handle(
        &mut self,
        block_idx: usize,
        slot_idx: usize,
    ) -> anyhow::Result<TrainBlockHandle> {
        let tensors = self.pool.slots[slot_idx].tensors.clone();
        let events = self.pool.slots[slot_idx].events.clone();
        events.compute_recorded.store(false, Ordering::Release);
        self.residency.set_state(block_idx, BlockResidency::InUse);
        self.handle_count.fetch_add(1, Ordering::AcqRel);
        Ok(TrainBlockHandle {
            block_idx,
            tensors,
            events,
            block_state: self.residency.shared_state(),
            handle_count: self.handle_count.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_dtype_from_safetensors(s: &str) -> DType {
    match s {
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F16" => DType::BF16, // Phase 1: F16 maps to BF16 (lossy); Phase 2 to handle.
        "F8_E4M3" => DType::BF16, // Phase 1: not yet supported.
        _ => DType::F32,
    }
}

fn mmap_safetensors(path: &str) -> anyhow::Result<(String, usize, memmap2::Mmap)> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open {path}: {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| anyhow::anyhow!("mmap {path}: {e}"))?;
    if mmap.len() < 8 {
        anyhow::bail!("file too small for safetensors: {path}");
    }
    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_size;
    if header_end > mmap.len() {
        anyhow::bail!("header extends past EOF in {path}");
    }
    let header = std::str::from_utf8(&mmap[8..header_end])
        .map_err(|e| anyhow::anyhow!("invalid UTF-8 in header: {e}"))?
        .to_string();
    Ok((header, header_end, mmap))
}

// ===========================================================================
// Synthetic tests (Phase 1 verify)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;
    use std::io::Write;

    /// Helper: build a synthetic 4-block facilitator with N keys per block.
    struct FakeFacilitator {
        block_count: usize,
        #[allow(dead_code)]
        keys_per_block: usize,
    }

    impl BlockFacilitator for FakeFacilitator {
        fn block_count(&self) -> usize {
            self.block_count
        }
        fn classify_key(&self, key: &str) -> Option<usize> {
            // Keys look like "blocks.{idx}.weight.{i}"
            let rest = key.strip_prefix("blocks.")?;
            let dot = rest.find('.')?;
            rest[..dot].parse().ok()
        }
    }

    impl TrainBlockFacilitator for FakeFacilitator {
        fn is_trainable_key(&self, _key: &str) -> bool {
            false
        }
        fn is_frozen_block_key(&self, key: &str) -> bool {
            self.classify_key(key).is_some()
        }
        fn shared_resident_key(&self, _key: &str) -> bool {
            false
        }
    }

    /// Write a synthetic safetensors file with `block_count * keys_per_block`
    /// BF16 tensors of shape `[shape_elems]` filled with `block_idx * 100 + i`.
    fn write_synthetic_safetensors(
        path: &std::path::Path,
        block_count: usize,
        keys_per_block: usize,
        shape_elems: usize,
    ) -> anyhow::Result<()> {
        // Build header JSON with all tensors, then write payload.
        let mut header = serde_json::Map::new();
        let bytes_per_tensor = shape_elems * 2;
        let mut offset: u64 = 0;
        let mut order: Vec<String> = Vec::new();
        for b in 0..block_count {
            for i in 0..keys_per_block {
                let key = format!("blocks.{b}.weight.{i}");
                let entry = serde_json::json!({
                    "dtype": "BF16",
                    "shape": [shape_elems],
                    "data_offsets": [offset, offset + bytes_per_tensor as u64],
                });
                header.insert(key.clone(), entry);
                order.push(key);
                offset += bytes_per_tensor as u64;
            }
        }
        let header_str = serde_json::to_string(&header)?;
        // Pad header to 8-byte alignment for cleanliness.
        let mut header_bytes = header_str.into_bytes();
        while (header_bytes.len() % 8) != 0 {
            header_bytes.push(b' ');
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(&header_bytes)?;
        // Write tensor payloads.
        for (b, key) in (0..block_count)
            .flat_map(|b| (0..keys_per_block).map(move |i| (b, i)))
            .zip(order.iter())
        {
            let _ = key;
            let value: u16 = (b.0 as u16).wrapping_mul(100).wrapping_add(b.1 as u16);
            let mut buf = vec![0u8; bytes_per_tensor];
            for chunk in buf.chunks_exact_mut(2) {
                chunk.copy_from_slice(&value.to_le_bytes());
            }
            f.write_all(&buf)?;
        }
        f.flush()?;
        Ok(())
    }

    fn try_make_offloader(
        block_count: usize,
        keys_per_block: usize,
        shape_elems: usize,
        slots: usize,
    ) -> anyhow::Result<(TrainingBlockOffloader, tempfile::TempDir)> {
        let device = CudaDevice::new(0)
            .map_err(|e| anyhow::anyhow!("CudaDevice::new: {e:?}"))?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("synthetic.safetensors");
        write_synthetic_safetensors(&path, block_count, keys_per_block, shape_elems)?;
        let facilitator = FakeFacilitator { block_count, keys_per_block };
        let mut config = TrainingOffloadConfig::default();
        config.gpu_slots = slots;
        let store = PinnedBlockStore::load(&[path], &facilitator)?;
        let off = TrainingBlockOffloader::new(store, &facilitator, device, config)?;
        Ok((off, dir))
    }

    fn skip_if_no_cuda() -> bool {
        if CudaDevice::new(0).is_err() {
            eprintln!("skipping: no CUDA device");
            return true;
        }
        false
    }

    #[test]
    fn forward_visit_4_blocks_2_slots() -> anyhow::Result<()> {
        if skip_if_no_cuda() {
            return Ok(());
        }
        let (mut off, _tmp) = try_make_offloader(4, 3, 64, 2)?;
        off.begin_step(0)?;
        // Visit forward 0..4. Prefetch one ahead.
        for i in 0..4 {
            let h = off.await_block(BlockVisit::forward(i))?;
            // Verify all expected keys are present.
            for k in 0..3 {
                let key = format!("blocks.{i}.weight.{k}");
                assert!(h.get(&key).is_some(), "missing {key}");
            }
            // Issue prefetch for next.
            let _ = off.prefetch(BlockVisit::forward(i))?;
            // Drop handle (compute would happen here in a real model).
            drop(h);
        }
        off.end_step()?;
        let s = off.stats();
        assert_eq!(s.steps, 1);
        assert!(s.h2d_count >= 4);
        Ok(())
    }

    #[test]
    fn backward_visit_4_blocks_2_slots() -> anyhow::Result<()> {
        if skip_if_no_cuda() {
            return Ok(());
        }
        let (mut off, _tmp) = try_make_offloader(4, 3, 64, 2)?;
        off.begin_step(0)?;
        for i in (0..4).rev() {
            let h = off.await_block(BlockVisit::backward(i))?;
            for k in 0..3 {
                let key = format!("blocks.{i}.weight.{k}");
                assert!(h.get(&key).is_some(), "missing {key}");
            }
            let _ = off.prefetch(BlockVisit::backward(i))?;
            drop(h);
        }
        off.end_step()?;
        Ok(())
    }

    #[test]
    fn forward_then_backward() -> anyhow::Result<()> {
        if skip_if_no_cuda() {
            return Ok(());
        }
        let (mut off, _tmp) = try_make_offloader(4, 3, 64, 2)?;
        off.begin_step(0)?;
        for i in 0..4 {
            let h = off.await_block(BlockVisit::forward(i))?;
            let _ = off.prefetch(BlockVisit::forward(i))?;
            drop(h);
        }
        for i in (0..4).rev() {
            let h = off.await_block(BlockVisit::backward(i))?;
            let _ = off.prefetch(BlockVisit::backward(i))?;
            drop(h);
        }
        off.end_step()?;
        Ok(())
    }

    #[test]
    fn budget_forces_single_slot() -> anyhow::Result<()> {
        if skip_if_no_cuda() {
            return Ok(());
        }
        let (off, _tmp) = try_make_offloader(4, 3, 64, 1)?;
        assert_eq!(off.num_slots(), 1);
        Ok(())
    }

    #[test]
    fn evict_all_sync_clears_slots() -> anyhow::Result<()> {
        if skip_if_no_cuda() {
            return Ok(());
        }
        let (mut off, _tmp) = try_make_offloader(4, 3, 64, 2)?;
        off.begin_step(0)?;
        for i in 0..4 {
            let h = off.await_block(BlockVisit::forward(i))?;
            drop(h);
        }
        off.evict_all_sync()?;
        // After evict, every block should be HostStaged.
        for i in 0..4 {
            assert_eq!(
                off.residency.state(i),
                BlockResidency::HostStaged,
                "block {i} still resident after evict",
            );
        }
        Ok(())
    }
}
