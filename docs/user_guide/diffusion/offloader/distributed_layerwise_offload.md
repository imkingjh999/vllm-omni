# Distributed Layerwise Offloading

Distributed layerwise offloading (DLO) extends block streaming to multi-device
deployments. With AllGather enabled, each rank stores roughly `1 / dp_size` of
the host weights and reconstructs each layer at runtime. Without AllGather,
each rank streams a complete block independently. Compatible TP1 deployments
can share checkpoint-backed host pages among processes on the same node;
otherwise DLO streams the ordinary loader's rank-local tensors.

See the [DLO feature design](../../../design/feature/offloader/distributed_layerwise_offload.md)
for the implementation contract and compatibility matrix.

## Execution model

DLO overlaps three operations with a fixed two-block device buffer:

```text
Compute stream:  [Layer N]          [Layer N+1]        [Layer N+2]
H2D stream:      [H2D shard N+1]    [H2D shard N+2]
AllGather:       [AG N+1]           [AG N+2]
Slots:           slot 0: Layer N    slot 1: Layer N+1
```

AllGather communicates only request-independent weight shards, so data-
parallel ranks may process different requests concurrently.

## Usage

```bash
# Four ranks with sharded host weights and AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4

# Standard-loader rank-local weights, without DLO AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4 \
  --dlo-no-use-allgather

# Sequence parallel deployment
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --usp 4
```

```python
from vllm_omni import Omni

omni = Omni(
    model="/path/to/model",
    enable_distributed_layerwise_offload=True,
    dlo_use_allgather=True,
)
```

## Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--data-parallel-size N` | DP ranks and AllGather weight-sharding group | `1` |
| `--dlo-use-allgather` | Shard host weights and reconstruct with AllGather | `true` |
| `--dlo-no-use-allgather` | Stream complete rank-local blocks without a DLO weight collective | `false` |
| `--dlo-resident-layers N` | Keep N leading main-DiT blocks on device; requires no-AllGather and model-declared resident paths | `0` |
| `--host-weight-runtime-mode {disabled,preferred,required}` | Reuse exact final-layout host weights for eligible no-AllGather DLO | `disabled` |
| `--host-weight-runtime-root PATH` | Node-local Host Weight Runtime store root; required when HWR is enabled | unset |

## Host-weight loading

The diffusion loader chooses host storage before DLO is enabled. It first
attempts to build a complete, validated direct-checkpoint mmap plan. If names,
coverage, shape, dtype, topology, or loader-callback compatibility cannot be
proven, it runs the ordinary model loader instead. DLO consumes that result and
does not make a second checkpoint-compatibility decision.

The shared-mmap optimization in this phase is supported only with TP1. TP
greater than one falls back before model mutation to ordinary TP-aware loading.
DLO may still consume those TP-local tensors, but this is a compatibility path:
it does not share checkpoint-backed runtime weights across DP replicas and
provides no shared-mmap host-memory guarantee.

The mmap plan skips only dedicated DiT weight sources. Other component sources,
such as a text encoder loaded through the shared diffusion loader, continue to
use their ordinary component loader. A checkpoint source that mixes DiT and
non-DiT weights falls back completely rather than leaving an unplanned
component uninitialized.

With direct checkpoint mmap, the loader:

1. saves non-persistent buffers such as RoPE frequencies;
2. moves the normally created transformer to the meta device;
3. loads checkpoint tensors as mmap views backed by the shared OS page cache;
4. applies any loader-owned bounded layout adapters while packing blocks;
5. restores saved non-persistent buffers; and
6. preserves `post_load_weights()` and `validate_loaded_weights()` lifecycle
   hooks.

For AllGather with a group larger than one, each process copies only its
persistent shard and then releases the source mapping. For no-AllGather, each
process keeps the mapping open and packs complete blocks through two bounded
pinned staging slots. Processes mapping the same files on one node share the
immutable pages; no-AllGather still performs a complete-block H2D copy in each
process.

When the effective DLO group size is one, `dlo_use_allgather=True` does not
perform a collective and uses the same rank-local transfer behavior.

### Final-layout Host Weight Runtime

HWR is an opt-in startup optimization for models that declare the final-layout
BF16 restore contract. Use it only with no-AllGather DLO:

```bash
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --host-weight-runtime-mode preferred \
  --host-weight-runtime-root /var/cache/vllm-omni/hwr
```

On a cold start, the canonical loader remains authoritative and publishes a
validated final-layout artifact for later workers. A warm start restores the
DiT final tensors without ordinary DiT materialization, then DLO streams them
through the same two bounded host staging slots. The artifact identity includes
the TP rank/size and SP layout, so TP1, TP2 rank-local shards, and distinct SP
layouts do not alias one another. `required` fails if an exact artifact cannot
be acquired; `preferred` falls back to canonical loading and keeps publication
failure separate from the serving startup result.

For local canonical checkpoints, the first eligible worker may hash source
shards to establish immutable identity. HWR caches those digests in the same
node-local storage domain and validates file metadata before reuse, so later
workers normally avoid repeating that read. Cold BF16 publication also hashes
ordered payloads as they are written and overlaps payload durability work with
later shards; this changes startup work only, not artifact contents or runtime
H2D behavior.

When HWR is disabled, DLO is disabled, or AllGather is enabled, the loader does
not resolve HWR sources or construct its store.

## Declarative topology

Models may declare an `OffloadPlan` instead of embedding offload logic:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"context_encoder": "layers"},
    )
```

When no plan exists, discovery falls back to
`_layerwise_offload_blocks_attrs` and then heuristic attribute lookup.

## Data-parallel concurrency

With `data_parallel_size > 1` and AllGather enabled, the scheduler can process
up to `dp_size` requests per denoising step. Every concurrent request must set
the same explicit `num_inference_steps`; `None` is rejected because every rank
must enter each collective.

## Limitations

- Direct checkpoint mmap currently requires TP1. TP greater than one is
  outside the Phase A shared-mmap support scope and falls back before model
  mutation to the ordinary TP-aware loader. DLO can stream that runtime layout,
  and eligible no-AllGather configurations may use HWR final-layout artifacts,
  but direct checkpoint mmap provides no shared-mmap host-memory guarantee for
  TP greater than one.
- HSDP plus AllGather is rejected to avoid double sharding. HSDP without
  AllGather has limited end-to-end validation.
- Per-tensor online FP8 linears use the ordinary loader and can run with either
  DLO transfer path. With AllGather, every rank temporarily materializes the
  complete FP8 model in host memory before DLO retains only its shard. Other
  online quantization methods require no-AllGather until their runtime layouts
  are validated.
- Resident leading layers require `--dlo-no-use-allgather` and a model
  `OffloadPlan` that declares eligible `resident_dit_paths`.
- DP concurrency requires an explicit, identical inference-step count.

Sharing quantized or otherwise unvalidated transformed layouts through a
normalized HWR producer is a follow-up design in
[RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195), not part of
the current BF16 final-layout path.

See the [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
for an end-to-end example.
