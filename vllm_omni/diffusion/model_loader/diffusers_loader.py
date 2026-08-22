# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import dataclasses
import glob
import hashlib
import os
import re
import time
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import nn
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    maybe_download_from_modelscope,
    multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.transformers_utils.repo_utils import file_exists
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig, apply_hsdp_to_model
from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    get_checkpoint_adapter,
)
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    TensorBinding,
    build_checkpoint_mmap_plan,
    has_online_quantization,
)
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.registry import initialize_model


# download_gguf was removed from upstream vLLM (commit 6635279d8).
# Inlined from the last upstream version before the GGUF plugin migration.
def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    allow_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]
    folder = download_weights_from_hf(
        model_name_or_path=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )
    local_files: list[str] = []
    for pattern in allow_patterns:
        glob_pattern = os.path.join(folder, pattern)
        local_files.extend(glob.glob(glob_pattern))
    if not local_files:
        raise ValueError(f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}")
    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]


logger = init_logger(__name__)


class _HWRCommitError(RuntimeError):
    """A committed warm restore made the current model disposable."""


def _natural_sort_key(filepath: str) -> list:
    """Natural sort key for filenames with numeric components, e.g.
    model-00001-of-00005.safetensors -> ['model-', 1, '-of-', 5, '.safetensors']."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(filepath))]


DIFFUSION_MODEL_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
TRANSFORMER_WEIGHTS_INDEX = "model.safetensors.index.json"
INDEX_FILES = [DIFFUSION_MODEL_WEIGHTS_INDEX, TRANSFORMER_WEIGHTS_INDEX]


def _resolve_custom_pipeline_cls(custom_pipeline_name: str | type | None) -> type:
    """Resolve a custom pipeline reference to a class.

    Accepts either a fully qualified name string (resolved via import) or an
    already-imported class object (returned as-is).
    """
    if custom_pipeline_name is None:
        raise ValueError("custom_pipeline_name is required for load_format='custom_pipeline'")
    if isinstance(custom_pipeline_name, str):
        return resolve_obj_by_qualname(custom_pipeline_name)
    if isinstance(custom_pipeline_name, type):
        return custom_pipeline_name
    raise TypeError(
        f"custom_pipeline_name must be a qualified name string or a class, got {type(custom_pipeline_name).__name__}"
    )


class DiffusersPipelineLoader:
    """Model loader that can load diffusers pipeline components from disk."""

    @dataclasses.dataclass
    class ComponentSource:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        subfolder: str | None
        """The subfolder inside the model repo."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig, od_config: OmniDiffusionConfig):
        self.load_config = load_config
        self.od_config = od_config
        self.quant_config = od_config.quantization_config
        self.parallel_config = od_config.parallel_config
        self.host_weight_plan: HostWeightPlan | None = None
        self._hwr_state: dict[str, object] | None = None
        self._last_load_request: dict[str, object] | None = None
        self._force_canonical_load = False

    def take_host_weight_plan(self) -> HostWeightPlan | None:
        """Transfer the loader-produced plan to the offload backend."""
        plan = self.host_weight_plan
        self.host_weight_plan = None
        return plan

    def _prepare_weights(
        self,
        model_name_or_path: Path | str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[Path | str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        possible_index_files = [
            f"{subfolder}/{index_file}" if subfolder is not None else index_file for index_file in INDEX_FILES
        ]
        available_index_file = [
            f for f in possible_index_files if file_exists(model_name_or_path, f, revision=revision)
        ]
        if len(available_index_file) > 1:
            raise ValueError(
                f"Multiple index files found in {model_name_or_path} with subfolder {subfolder}: {available_index_file}"
            )
        index_file = available_index_file[0] if available_index_file else ""

        # only hf is supported currently
        if load_format == "auto":
            load_format = "hf"

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                # Decide by actual files rather than pattern name (patterns may include subfolders).
                use_safetensors = any(f.endswith(".safetensors") for f in hf_weights_files)
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self,
        source: "ComponentSource",
        model: nn.Module | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        _, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )

        use_multithread = (
            use_safetensors
            and getattr(self.od_config, "enable_multithread_weight_load", False)
            and self.load_config.safetensors_load_strategy != "torchao"
        )
        if use_multithread:
            num_threads = getattr(self.od_config, "num_weight_load_threads", 4)
            # Keep deterministic shard order before passing to vLLM helper.
            sorted_hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)
            weights_iterator = multi_thread_safetensors_weights_iterator(
                sorted_hf_weights_files,
                self.load_config.use_tqdm_on_load,
                max_workers=num_threads,
            )
        else:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.safetensors_load_strategy,
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        prefixed_weights_iterator = ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
        if model is not None:
            checkpoint_adapter = self._get_checkpoint_adapter(model, source, use_safetensors)
            if checkpoint_adapter is not None:
                return checkpoint_adapter.adapt(prefixed_weights_iterator)
        return prefixed_weights_iterator

    def _get_source_quant_config(self, source: "ComponentSource") -> object | None:
        quant_config = self.quant_config
        if hasattr(quant_config, "resolve"):
            return quant_config.resolve(source.prefix.rstrip("."))
        return quant_config

    def _get_checkpoint_adapter(
        self,
        model: nn.Module,
        source: "ComponentSource",
        use_safetensors: bool,
    ):
        return get_checkpoint_adapter(
            model=model,
            source=source,
            quant_config=self._get_source_quant_config(source),
            use_safetensors=use_safetensors,
        )

    def get_all_weights(
        self,
        model: nn.Module,
        sources: Sequence["ComponentSource"] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if sources is None:
            sources = self._get_weight_sources(model)
        for source in sources:
            yield from self._get_weights_iterator(source, model=model)

    @staticmethod
    def _stream_online_quant_weights_to_cpu(
        model: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Offload each online-quantized layer as soon as it is complete.

        Upstream vLLM's online layerwise loader materializes and quantizes a
        layer synchronously while consuming ``weights``.  Generator execution
        resumes after that weight has been consumed, which gives us a safe
        point to move completed layers to CPU before the next layer is loaded.
        This bounds accelerator residency during CPU-offloaded model startup
        instead of retaining the entire quantized model until loading ends.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            get_layerwise_info,
        )

        pending = {
            module
            for module in model.modules()
            if getattr(getattr(module, "quant_method", None), "uses_meta_device", False)
            and get_layerwise_info(module).can_load()
        }
        offloaded = 0

        def offload_completed() -> None:
            nonlocal offloaded
            for module in tuple(pending):
                if get_layerwise_info(module).can_load():
                    continue
                module.to("cpu")
                pending.remove(module)
                offloaded += 1

        for weight in weights:
            # This runs after the consumer has handled the previous yield.
            offload_completed()
            yield weight
        offload_completed()

        # Quantization workspaces and the old accelerator-side parameter
        # storages are now reusable cache blocks.  Release them before the
        # remaining (unquantized) model tensors are copied to CPU; otherwise
        # the cached quantization footprint and final offload overlap in the
        # process-level startup peak.
        if offloaded:
            torch.accelerator.empty_cache()

        logger.info(
            "Stream-offloaded %d online-quantized layers to CPU during weight loading",
            offloaded,
        )

    def _get_weight_sources(self, model: nn.Module) -> tuple["ComponentSource", ...]:
        return tuple(
            cast(
                Iterable[DiffusersPipelineLoader.ComponentSource],
                getattr(model, "weights_sources", ()),
            )
        )

    def _get_expected_parameter_names(self, model: nn.Module) -> set[str]:
        """Return parameter names that should be covered by strict load checks."""
        all_parameter_names = {name for name, _ in model.named_parameters()}
        sources = self._get_weight_sources(model)

        # Keep strict behavior if no source metadata exists.
        if not sources:
            return all_parameter_names

        # Empty prefix means "root" source, i.e. entire model should be covered.
        if any(source.prefix == "" for source in sources):
            return all_parameter_names

        source_prefixes = tuple(source.prefix for source in sources if source.prefix)
        if not source_prefixes:
            return all_parameter_names
        return {name for name in all_parameter_names if name.startswith(source_prefixes)}

    @staticmethod
    def _identity_value(value: object) -> object:
        """Convert config objects into deterministic identity metadata."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dict):
            return {str(key): DiffusersPipelineLoader._identity_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [DiffusersPipelineLoader._identity_value(item) for item in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return DiffusersPipelineLoader._identity_value(to_dict())
            except TypeError:
                pass
        if dataclasses.is_dataclass(value):
            return DiffusersPipelineLoader._identity_value(dataclasses.asdict(value))
        return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"

    @staticmethod
    def _identity_fingerprint(value: object) -> str:
        from vllm_omni.host_weight_runtime import CanonicalJson

        return hashlib.sha256(CanonicalJson.from_value(value).encoded).hexdigest()

    @staticmethod
    def _snapshot_final_layout_tensors(model: nn.Module, names: Iterable[str]) -> dict[str, tuple[int, str]]:
        snapshot: dict[str, tuple[int, str]] = {}
        for name in names:
            parent_path, _, leaf_name = name.rpartition(".")
            parent = model.get_submodule(parent_path)
            tensor = parent._parameters.get(leaf_name)
            if tensor is None:
                tensor = parent._buffers.get(leaf_name)
            if tensor is None or tensor.is_meta:
                raise RuntimeError(f"cannot snapshot missing or meta final-layout tensor {name!r}")
            value = tensor.detach()
            if value.device.type != "cpu":
                value = value.cpu()
            value = value.contiguous()
            digest = hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()
            snapshot[name] = (tensor.untyped_storage().data_ptr(), digest)
        return snapshot

    @classmethod
    def _assert_final_layout_tensors_unchanged(
        cls,
        model: nn.Module,
        snapshot: dict[str, tuple[int, str]],
    ) -> None:
        current = cls._snapshot_final_layout_tensors(model, snapshot)
        changed = [name for name in snapshot if current[name] != snapshot[name]]
        if changed:
            raise RuntimeError(
                "shared warm finalization changed restored final-layout tensor bytes or backing pointers: "
                f"{changed[:5]}"
            )

    def _hwr_eligibility_mode(
        self,
        model: nn.Module,
        modules: object,
        *,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
    ) -> object | None:
        """Return the enabled HWR mode only after all zero-interaction gates."""
        from vllm_omni.host_weight_runtime import RuntimeMode

        raw_mode = getattr(self.od_config, "host_weight_runtime_mode", "disabled")
        try:
            mode = RuntimeMode(raw_mode)
        except ValueError as exc:
            raise ValueError("host_weight_runtime_mode must be disabled, preferred, or required") from exc

        # These gates intentionally precede HWR imports, source preparation,
        # identity construction, and store creation.
        if mode is RuntimeMode.DISABLED or not dist_offload or use_allgather:
            return None

        parallel = self.parallel_config
        reasons: list[str] = []
        if load_format != "default":
            reasons.append("load_format must be 'default'")
        if bool(getattr(parallel, "use_hsdp", False)):
            reasons.append("HSDP layouts are not supported by the final-layout BF16 consumer")
        if self.quant_config is not None:
            reasons.append("quantized layouts require a representation-specific HWR producer")
        if getattr(self.od_config, "lora_path", None):
            reasons.append("adapted weights are not reusable base-model artifacts")

        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        if not dit_modules:
            reasons.append("no DiT modules were discovered")
        elif any(
            getattr(dit, "host_weight_restore_contract", None) is None
            or not callable(getattr(dit, "validate_restored_host_weights", None))
            for _, dit in dit_modules
        ):
            reasons.append("every owned DiT must declare the final-layout restore contract")

        if reasons:
            message = "; ".join(reasons)
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        return mode

    def _prepare_hwr_sources(
        self,
        model: nn.Module,
        modules: object,
        sources: Sequence["ComponentSource"],
    ) -> tuple[object, ...]:
        from vllm_omni.diffusion.model_loader.host_weights import (
            ImplementationIdentity,
            PreparedWeightSource,
            WeightSourceKind,
        )

        dit_prefixes = tuple(f"{name}." for name in getattr(modules, "dit_names", ()))
        selected_sources = tuple(
            source
            for source in sources
            if not source.prefix or any(prefix.startswith(source.prefix) for prefix in dit_prefixes)
        )
        if not selected_sources:
            raise ValueError("final-layout HWR requires canonical weight sources covering every DiT")

        prepared: list[PreparedWeightSource] = []
        for source in selected_sources:
            resolved_root, weight_files, use_safetensors = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
            adapter = self._get_checkpoint_adapter(model, source, use_safetensors)
            adapter_identity = None
            if adapter is not None:
                adapter_name = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
                adapter_identity = ImplementationIdentity(
                    implementation_id=adapter_name,
                    version="1",
                    fingerprint=self._identity_fingerprint({"adapter": adapter_name}),
                )
            source_kind = (
                WeightSourceKind.LOCAL_PATH
                if os.path.isdir(os.fspath(source.model_or_path))
                else WeightSourceKind.HUGGING_FACE_HUB
            )
            prepared.append(
                PreparedWeightSource(
                    model_or_path=os.fspath(source.model_or_path),
                    subfolder=source.subfolder,
                    requested_revision=source.revision,
                    prefix=source.prefix,
                    resolved_root=Path(os.fspath(resolved_root)),
                    weight_files=tuple(Path(os.fspath(path)) for path in weight_files),
                    use_safetensors=use_safetensors,
                    checkpoint_adapter=adapter_identity,
                    source_kind=source_kind,
                )
            )
        return tuple(prepared)

    def _build_hwr_context(
        self,
        model: nn.Module,
        modules: object,
        *,
        load_format: str,
        sources: Sequence["ComponentSource"],
    ) -> object:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        from vllm_omni.diffusion.model_loader.host_weights import (
            FINAL_LAYOUT_BF16_POLICY,
            FinalLayoutLoaderIdentity,
            FinalLayoutParallelIdentity,
            FinalLayoutRequest,
            ImplementationIdentity,
            build_final_layout_identity,
        )

        parallel = self.parallel_config
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
        try:
            tp_rank = int(get_tensor_model_parallel_rank()) if tp_size > 1 else 0
        except Exception:
            tp_rank = int(getattr(parallel, "tensor_parallel_rank", 0))
        sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
        loader_config = {
            "dtype": str(self.od_config.dtype),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_config": self._identity_value(getattr(self.od_config, "tf_model_config", None)),
            "load_format": load_format,
            "quantization": self._identity_value(self.quant_config),
        }
        contracts = [
            self._identity_value(getattr(dit, "host_weight_restore_contract"))
            for dit in getattr(modules, "dits", ())
        ]
        loader_identity = FinalLayoutLoaderIdentity(
            implementation=ImplementationIdentity(
                implementation_id="vllm-omni.diffusion.diffusers-loader",
                version="final-layout-v1",
                fingerprint=self._identity_fingerprint(
                    {
                        "loader": "diffusers-loader-final-layout-v1",
                        "pipeline": f"{type(model).__module__}.{type(model).__qualname__}",
                    }
                ),
            ),
            model_config_fingerprint=self._identity_fingerprint(loader_config),
            weight_transform_fingerprint=self._identity_fingerprint(
                {
                    "contracts": contracts,
                    "transform": "diffusion-final-layout-loader-transforms-v1",
                }
            ),
        )
        semantic_parallel = FinalLayoutParallelIdentity(
            tensor_parallel_size=tp_size,
            tensor_parallel_rank=tp_rank,
            sequence_parallel_size=sp_size,
            ulysses_degree=int(getattr(parallel, "ulysses_degree", 1)),
            ring_degree=int(getattr(parallel, "ring_degree", 1)),
            allgather_degree=int(getattr(parallel, "allgather_degree", 1)),
            ulysses_mode=str(getattr(parallel, "ulysses_mode", "strict")),
            pipeline_parallel_size=int(getattr(parallel, "pipeline_parallel_size", 1)),
            cfg_parallel_size=int(getattr(parallel, "cfg_parallel_size", 1)),
            use_hsdp=bool(getattr(parallel, "use_hsdp", False)),
            enable_expert_parallel=bool(getattr(parallel, "enable_expert_parallel", False)),
        )
        model_id = str(getattr(self.od_config, "model", "") or "")
        if not model_id:
            raise ValueError("final-layout HWR requires a canonical model identifier")
        request = FinalLayoutRequest(
            model_id=model_id,
            loader=loader_identity,
            parallel=semantic_parallel,
            load_format=load_format,
        )
        prepared_sources = self._prepare_hwr_sources(model, modules, sources)
        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        return build_final_layout_identity(
            model,
            dit_modules=dit_modules,
            prepared_sources=prepared_sources,
            request=request,
            policy=FINAL_LAYOUT_BF16_POLICY,
        )

    def _resolve_hwr(
        self,
        model: nn.Module,
        modules: object,
        *,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
        sources: Sequence["ComponentSource"],
    ) -> dict[str, object] | None:
        """Resolve an eligible no-AllGather final-layout HWR transaction."""
        from vllm_omni.diffusion.model_loader.host_weights import FinalLayoutTensorRestorer
        from vllm_omni.host_weight_runtime import (
            HostWeightLeaseCarrier,
            HostWeightRuntime,
            HostWeightRuntimeConfig,
            ProductionPolicy,
            ResolutionOutcome,
            RuntimeMode,
            StorageDomainPolicy,
        )
        from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

        mode = self._hwr_eligibility_mode(
            model,
            modules,
            dist_offload=dist_offload,
            use_allgather=use_allgather,
            load_format=load_format,
        )
        if mode is None:
            return None
        assert isinstance(mode, RuntimeMode)
        expected_prefixes = frozenset(f"{name}." for name in getattr(modules, "dit_names", ()))
        available_prefixes = frozenset(getattr(source, "prefix", "") for source in sources)
        if not expected_prefixes <= available_prefixes:
            message = "final-layout HWR requires one dedicated source prefix per owned DiT"
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        overlapping_prefixes = frozenset(
            prefix
            for prefix in available_prefixes
            if any(dit_prefix.startswith(prefix) for dit_prefix in expected_prefixes)
        )
        if not overlapping_prefixes <= expected_prefixes:
            message = "final-layout HWR requires dedicated DiT sources and rejects mixed component sources"
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        context = self._build_hwr_context(model, modules, load_format=load_format, sources=sources)
        root_value = getattr(self.od_config, "host_weight_runtime_root", None)
        if not isinstance(root_value, str) or not root_value.strip():
            raise ValueError("enabled Host Weight Runtime requires host_weight_runtime_root")
        root = Path(root_value).expanduser()
        runtime_config = HostWeightRuntimeConfig(
            mode=mode,
            domain=StorageDomainPolicy(root=root, storage_class=detect_storage_class(root)),
            production=ProductionPolicy(allow_local_build=False, allow_post_load_publish=True),
        )
        runtime = HostWeightRuntime.from_config(runtime_config)
        resolution = runtime.resolve(context.identity)
        state: dict[str, object] = {
            "mode": mode,
            "context": context,
            "runtime": runtime,
            "resolution": resolution,
            "outcome": resolution.report.outcome,
        }
        if resolution.report.outcome is ResolutionOutcome.FAILED:
            failure = next(
                (attempt.failure for attempt in reversed(resolution.report.attempts) if attempt.failure is not None),
                None,
            )
            detail = failure.message if failure is not None else "resolution failed without a typed detail"
            raise RuntimeError(f"Host Weight Runtime resolution failed: {detail}")
        if resolution.report.outcome is not ResolutionOutcome.LOCAL_HIT:
            return state

        lease = resolution.lease
        if lease is None:
            raise RuntimeError("Host Weight Runtime returned LOCAL_HIT without a lease")
        restorer = FinalLayoutTensorRestorer(context)
        try:
            restore_plan = restorer.plan_restore(model, lease)
        except Exception:
            lease.close()
            if mode is RuntimeMode.REQUIRED:
                raise
            logger.warning("HWR warm restore planning failed; falling back to canonical loading", exc_info=True)
            state["outcome"] = ResolutionOutcome.CANONICAL_FALLBACK
            return state

        try:
            restore_plan.commit()
        except Exception as exc:
            lease.close()
            raise _HWRCommitError(
                "Host Weight Runtime restore commit failed; the partially restored model must be discarded"
            ) from exc

        try:
            carrier = HostWeightLeaseCarrier(lease)
            warm_snapshot = self._snapshot_final_layout_tensors(model, context.tensor_names)
        except Exception as exc:
            lease.close()
            raise _HWRCommitError(
                "Host Weight Runtime committed restore could not establish its startup ownership boundary"
            ) from exc
        source_metadata = context.identity.source.metadata.to_value()
        target_bindings = source_metadata.get("target_bindings") if isinstance(source_metadata, dict) else None
        planned_prefixes = frozenset(
            binding["source_prefix"]
            for binding in (target_bindings or ())
            if isinstance(binding, dict) and isinstance(binding.get("source_prefix"), str)
        )
        if not planned_prefixes:
            lease.close()
            raise _HWRCommitError("Host Weight Runtime identity did not record owned canonical source prefixes")
        state["plan"] = HostWeightPlan(
            backing_kind="host_weight_runtime",
            bindings={name: TensorBinding(name, "") for name in context.tensor_names},
            planned_source_prefixes=planned_prefixes,
            lease_carrier=carrier,
            runtime_mode=mode.value,
        )
        state["warm_snapshot"] = warm_snapshot
        return state

    def load_fresh_canonical_model(self) -> nn.Module:
        """Reload a disposable startup model without HWR or checkpoint mmap.

        This is used only after a warm restore has committed and a pre-service
        transport setup failure has made the restored model disposable.  The
        fresh construction is intentionally independent of the failed model's
        parameter storage and of the HWR resolution/publication transaction.
        """
        if self._last_load_request is None:
            raise RuntimeError("cannot construct a fresh canonical model before the initial load")
        request = dict(self._last_load_request)
        self._force_canonical_load = True
        try:
            return self.load_model(**cast(dict[str, object], request))
        finally:
            self._force_canonical_load = False

    def _publish_hwr_after_load(self, model: nn.Module, modules: object, state: dict[str, object] | None) -> None:
        from vllm_omni.diffusion.model_loader.host_weights import FinalLayoutBF16Producer
        from vllm_omni.host_weight_runtime import PostLoadPublicationOutcome, ResolutionOutcome

        if state is None or state.get("outcome") is not ResolutionOutcome.CANONICAL_FALLBACK:
            return
        context = state["context"]
        runtime = state["runtime"]
        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        try:
            producer = FinalLayoutBF16Producer(context, model, dit_modules)
            report = runtime.publish_after_load(context.identity, producer=producer)
        except Exception:
            # Publication is an independent post-load operation. The current
            # canonical model is already valid even when producer construction
            # or publication fails.
            logger.warning("Host Weight Runtime post-load publication failed", exc_info=True)
            return
        state["publication"] = report
        if report.outcome is PostLoadPublicationOutcome.FAILED:
            logger.warning("Host Weight Runtime post-load publication failed: %s", report.failure)

    def load_model(
        self,
        load_device: str,
        load_format: str | None = "default",
        custom_pipeline_name: str | type[nn.Module] | None = None,
        device: torch.device | None = None,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        self.host_weight_plan = None
        self._hwr_state = None
        self._last_load_request = {
            "load_device": load_device,
            "load_format": load_format,
            "custom_pipeline_name": custom_pipeline_name,
            "device": device,
        }
        if load_format is None:
            load_format = "default"
        # CPU offload + quantization: for offline-quantized models (e.g., AutoRound MXFP8),
        # weights are already quantized in the checkpoint — load directly on CPU.
        # For online quantization, load on device so quantization can run on accelerator,
        # then move back to CPU afterward.
        offload_after_quant = False
        if load_device == "cpu" and self.quant_config is not None and device is not None:
            quant_cfg = self.quant_config
            is_offline = getattr(quant_cfg, "data_type", None) == "mx_fp" or getattr(
                quant_cfg, "is_checkpoint_quantized", False
            )
            if not is_offline:
                load_device = device.type
                offload_after_quant = True
                logger.info(
                    "Online quantization with CPU offload, using %s for weight loading (will offload back to CPU)",
                    load_device,
                )
            else:
                logger.info("Offline-quantized model with CPU offload, loading weights directly on CPU")

        target_device = torch.device(load_device)
        with set_default_torch_dtype(self.od_config.dtype):
            if self.parallel_config.use_hsdp:
                model = self._load_model_with_hsdp(
                    target_device=device, load_format=load_format, custom_pipeline_name=custom_pipeline_name
                )
            else:
                model = self._init_from_load_format(load_format, target_device, custom_pipeline_name, is_hsdp=False)

                _dist_offload = getattr(self.od_config, "enable_distributed_layerwise_offload", False)
                _use_ag = getattr(self.od_config, "dlo_use_allgather", True)
                _has_online_quant = self._has_online_quant(model)
                _tp_size = int(getattr(self.parallel_config, "tensor_parallel_size", 1))
                _use_hsdp = bool(getattr(self.parallel_config, "use_hsdp", False))
                _dp_size = int(getattr(self.parallel_config, "data_parallel_size", 1))
                _sp_size = int(getattr(self.parallel_config, "sequence_parallel_size", 1))
                _dlo_group_size = _dp_size if _dp_size > 1 else _sp_size

                plan_result = None
                weight_sources = self._get_weight_sources(model)
                hwr_state = None
                if not self._force_canonical_load:
                    try:
                        hwr_state = self._resolve_hwr(
                            model,
                            ModuleDiscovery.discover(model),
                            dist_offload=_dist_offload,
                            use_allgather=_use_ag,
                            load_format=load_format,
                            sources=weight_sources,
                        )
                    except _HWRCommitError:
                        from vllm_omni.host_weight_runtime import RuntimeMode

                        mode = RuntimeMode(getattr(self.od_config, "host_weight_runtime_mode", "disabled"))
                        if mode is not RuntimeMode.PREFERRED:
                            raise
                        logger.warning(
                            "HWR restore commit failed; discarding the model and retrying a fresh canonical load",
                            exc_info=True,
                        )
                        del model
                        return self.load_fresh_canonical_model()
                self._hwr_state = hwr_state
                hwr_active = hwr_state is not None
                if hwr_active and hwr_state is not None:
                    self.host_weight_plan = cast(HostWeightPlan | None, hwr_state.get("plan"))
                if _dist_offload and not hwr_active and not self._force_canonical_load:
                    modules = ModuleDiscovery.discover(model)
                    plan_result = build_checkpoint_mmap_plan(
                        model,
                        dit_modules=tuple(zip(modules.dit_names, modules.dits)),
                        sources=weight_sources,
                        model_path=str(getattr(self.od_config, "model", "")) or None,
                        tensor_parallel_size=_tp_size,
                        use_hsdp=_use_hsdp,
                        online_quantization=_has_online_quant,
                    )
                    self.host_weight_plan = plan_result.plan

                _skip_load = self.host_weight_plan is not None

                if _skip_load:
                    logger.info(
                        "DLO host-weight plan active (%s, %s): skipping ordinary materialization for %s",
                        "AllGather" if _use_ag and _dlo_group_size > 1 else "rank-local",
                        self.host_weight_plan.backing_kind,
                        sorted(self.host_weight_plan.planned_source_prefixes) or "legacy DiT sources",
                    )
                    ordinary_sources = tuple(
                        source
                        for source in weight_sources
                        if source.prefix not in self.host_weight_plan.planned_source_prefixes
                    )
                    if ordinary_sources:
                        logger.info(
                            "Loading %d component weight source(s) outside the DLO host-weight plan",
                            len(ordinary_sources),
                        )
                        self.load_weights(
                            model,
                            sources=ordinary_sources,
                            planned_weights=self.host_weight_plan.bindings,
                        )
                else:
                    if _dist_offload and _use_ag and _has_online_quant:
                        unsupported_methods = self._unsupported_dlo_allgather_online_quant_methods(model)
                        if unsupported_methods:
                            raise ValueError(
                                "DLO+AllGather supports online quantization only for "
                                "per-tensor FP8 linears; unsupported online methods: "
                                f"{', '.join(unsupported_methods)}. Please use "
                                "--dlo-no-use-allgather or disable online quantization."
                            )
                        logger.info(
                            "Online per-tensor FP8 with DLO+AllGather: using the "
                            "ordinary loader before sharding finalized weights and scales"
                        )
                    if _dist_offload and plan_result is not None:
                        logger.info(
                            "DLO direct checkpoint mmap unavailable; using ordinary loader: %s",
                            plan_result.fallback_reason,
                        )
                    logger.debug("Loading weights on %s ...", load_device)
                    if offload_after_quant:
                        marked = self._request_offload_after_quant(model)
                        if marked:
                            logger.info(
                                "Online quantization will return each of %d layers to CPU as it is quantized",
                                marked,
                            )
                    if load_format == "diffusers":
                        cast(DiffusersAdapterPipeline, model).load_weights()
                    else:
                        if offload_after_quant:
                            self.load_weights(model, stream_online_quant_to_cpu=True)
                        else:
                            self.load_weights(model)
                    self._process_weights_after_loading(model, target_device)

                # A warm final-layout hit has already completed all
                # byte-changing work through the restorer.  Shared runtime
                # finalization happens once at the end for both cold and warm
                # paths; the warm path never re-enters the ordinary
                # materialization/finalization pipeline.

            if offload_after_quant:
                model.to("cpu")
                logger.info("Quantization complete, offloaded model back to CPU")

        try:
            self._apply_skip_softmax_calibration(model)
            model = model.eval()
            if self._hwr_state is not None:
                warm_snapshot = self._hwr_state.get("warm_snapshot")
                if warm_snapshot is not None:
                    self._assert_final_layout_tensors_unchanged(model, cast(dict[str, tuple[int, str]], warm_snapshot))
                self._publish_hwr_after_load(model, ModuleDiscovery.discover(model), self._hwr_state)
        except Exception:
            hwr_plan = self._hwr_state.get("plan") if self._hwr_state is not None else None
            if isinstance(hwr_plan, HostWeightPlan):
                carrier = hwr_plan.lease_carrier
                if carrier is not None:
                    carrier.close()
                from vllm_omni.host_weight_runtime import RuntimeMode

                mode = RuntimeMode(getattr(self.od_config, "host_weight_runtime_mode", "disabled"))
                if mode is RuntimeMode.PREFERRED:
                    logger.warning(
                        "HWR warm finalization failed; discarding the model and retrying a fresh canonical load",
                        exc_info=True,
                    )
                    del model
                    return self.load_fresh_canonical_model()
            raise
        return model

    @staticmethod
    def _request_offload_after_quant(model: nn.Module) -> int:
        """Ask online-quant layers to return to host memory once quantized.

        The weights only visit the accelerator so the quant kernels can run on
        them; ``load_model`` sends the model back to the host afterwards either
        way. Without this the whole transformer accumulates on device until that
        final move, which for MiniMax H3 is a ~43 GiB peak that no longer fits
        beside a resident TP-sharded text encoder — even though layer-wise
        offload means none of it is supposed to be resident at inference time.

        Only quant methods that advertise ``supports_offload_after_quant`` are
        asked, since the implementation has to know when a layer is finished.
        Deferring materialization to the ``meta`` device does not imply that.
        """
        marked = 0
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if getattr(quant_method, "supports_offload_after_quant", False):
                quant_method.enable_offload_after_quant()
                marked += 1
        return marked

    @staticmethod
    def _has_online_quant(model: nn.Module) -> bool:
        """Whether any layer uses an online-quant method that defers weight
        materialization onto the ``meta`` device (upstream vLLM
        ``uses_meta_device=True``, e.g. online FP8)."""
        return has_online_quantization(model)

    @staticmethod
    def _unsupported_dlo_allgather_online_quant_methods(model: nn.Module) -> tuple[str, ...]:
        """Return unsupported online-quant methods for DLO AllGather.

        Per-tensor online FP8 is safe after the ordinary loader has finalized
        its weight and scale parameters. DLO shards those runtime tensors by
        dtype and reconstructs their recorded shapes and strides before the
        kernel consumes them. Other online methods may create different scale,
        packing, or aliasing layouts and remain fail-closed until validated.
        """
        from vllm.model_executor.layers.quantization.online.fp8 import (
            Fp8PerTensorOnlineLinearMethod,
        )

        unsupported: set[str] = set()
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if not getattr(quant_method, "uses_meta_device", False):
                continue
            if not isinstance(quant_method, Fp8PerTensorOnlineLinearMethod):
                unsupported.add(type(quant_method).__name__)
        return tuple(sorted(unsupported))

    def _apply_skip_softmax_calibration(self, model: nn.Module) -> None:
        from vllm_omni.diffusion.attention.backends.trtllm_calibration import (
            apply_skip_softmax_calibration,
        )

        cfg = getattr(self.od_config, "diffusion_attention_config", None)
        apply_skip_softmax_calibration(cfg, model)

    def _process_weights_after_loading(self, model: nn.Module, target_device: torch.device) -> None:
        """Process weights after loading for quantization methods.

        This handles vLLM's quantization methods that need to process weights
        after loading (e.g., FP8 online quantization from BF16/FP16 weights).
        """
        # Newer upstream vLLM online-quant methods (uses_meta_device=True) create
        # weights on the ``meta`` device and materialize them just-in-time as each
        # layer's weights finish loading (via the layerwise online-process loader).
        # Any "straggler" layers whose weights were not fully materialized during
        # load (padded / partially-loaded layers) remain on ``meta``. Upstream's
        # base_loader calls finalize_layerwise_processing() to materialize them;
        # the diffusion loader must mirror that, otherwise the module.to() below
        # raises "Cannot copy out of meta tensor; no data!". This whole meta-device
        # handling is gated on online quant actually being in use, so that the
        # proven code path for everything else (in particular FSDP/HSDP-sharded
        # params, whose per-parameter .data cannot be cross-device reassigned) is
        # left untouched. Import lazily so older vLLM (no meta-device quant) is
        # unaffected.
        has_online_quant = self._has_online_quant(model)
        if has_online_quant:
            from vllm.model_executor.model_loader.reload.layerwise import (
                finalize_layerwise_processing,
            )

            # model_config is only dereferenced by finalize for vLLM Attention /
            # MLAAttention layers; diffusion DiT models use their own attention and
            # have none, so passing None is safe here.
            finalize_layerwise_processing(model, model_config=None)

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None or not isinstance(quant_method, QuantizeMethodBase):
                continue

            # Layers finished during loading would only be staged onto the target
            # device for a process call that immediately returns. That round trip
            # is wasted work in general, and undoes the point of the offload for
            # layers that already went back to the host.
            if getattr(module, "_already_called_process_weights_after_loading", False):
                continue

            if has_online_quant:
                # finalize_layerwise_processing() and the synchronous online
                # loader already processed these layers.  Avoid moving their
                # quantized CPU weights back to the accelerator merely to call
                # an idempotent no-op; doing so rebuilds a large CUDA allocator
                # cache and defeats streaming CPU offload's startup-memory bound.
                if getattr(module, "_already_called_process_weights_after_loading", False):
                    continue

                # Online quant may leave straggler params on the ``meta`` device.
                # Move only real (non-meta) params onto the target device for
                # processing and restore them afterward, mirroring upstream vLLM's
                # device_loading_context — a blanket module.to(target_device) would
                # raise NotImplementedError on meta params. Online quant initializes
                # on the accelerator, so params are normally already on the target
                # device and this loop is a no-op move; the point is to skip meta.
                original_devices: dict[str, torch.device] = {}
                for name, param in module.named_parameters():
                    if param.device.type != "meta" and param.device != target_device:
                        original_devices[name] = param.device
                        param.data = param.data.to(target_device)

                quant_method.process_weights_after_loading(module)

                # Restore pre-existing params to their original device; leave any
                # newly created (e.g. quantized) params on the target device.
                for name, param in module.named_parameters():
                    if name in original_devices:
                        param.data = param.data.to(original_devices[name])
            else:
                # No meta params possible here. Preserve the original FSDP/HSDP-aware
                # whole-module move (module.to()), which correctly handles sharded
                # DTensor params that per-parameter .data reassignment cannot.
                module_device = next(module.parameters(), None)
                if module_device is not None:
                    module_device = module_device.device
                needs_device_move = module_device != target_device

                if needs_device_move:
                    module.to(target_device)

                quant_method.process_weights_after_loading(module)

                if needs_device_move:
                    module.to(module_device)

    def load_weights(
        self,
        model: nn.Module,
        *,
        stream_online_quant_to_cpu: bool = False,
        sources: Sequence["ComponentSource"] | None = None,
        planned_weights: Iterable[str] = (),
    ) -> None:
        weights_to_load = self._get_expected_parameter_names(model)
        weights = self.get_all_weights(model) if sources is None else self.get_all_weights(model, sources=sources)
        if stream_online_quant_to_cpu:
            weights = self._stream_online_quant_weights_to_cpu(model, weights)
        loaded_weights = model.load_weights(weights)
        if loaded_weights is not None:
            loaded_weights = set(loaded_weights).union(planned_weights)

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # TODO(Isotr0py): Enable weights loading check after decoupling
        # all components' weights loading (AutoModel.from_pretrained etc).
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            # NOTE: if the model is quantized, ignore not_loaded check for scale
            # weights. ModelOpt FP8 carries a per-tensor `weight_scale` and a
            # static activation `input_scale`, which the quant method may
            # fold/track differently than plain parameters.
            weights_scale_not_loaded = {
                name for name in weights_not_loaded if name.endswith(("weight_scale", "input_scale"))
            }
            weights_not_loaded = weights_not_loaded - weights_scale_not_loaded
            if weights_not_loaded:
                self._check_unloaded_weights(weights_not_loaded)
            if weights_scale_not_loaded:
                logger.warning(
                    f"Following weight_scale weights were not initialized from checkpoint: {weights_scale_not_loaded}"
                )

    @staticmethod
    def _is_expected_quantized_weight(name: str) -> bool:
        """Return True if *name* is a quantization-specific parameter.

        Quantization methods (GPTQ, AWQ, FP8, Autoround, etc.) create extra
        parameters that have no counterpart in an unquantized checkpoint.
        These are expected to be absent and should not trigger a load error.
        """
        # Weight suffixes that quantization methods register in the model but
        # are not present in unquantized checkpoints.
        _QUANTIZED_WEIGHT_SUFFIXES = (
            # GPTQ / AWQ / AutoRound – g_idx is optional (not all checkpoints include it)
            ".g_idx",
            # FP8
            ".weight_scale",
            ".weight_scale_inv",
            ".input_scale",
            # INT8  (weight_scale already covered above)
        )
        return name.endswith(_QUANTIZED_WEIGHT_SUFFIXES)

    def _check_unloaded_weights(
        self,
        weights_not_loaded: set[str],
    ) -> None:
        """Validate unloaded weights, tolerating expected quantization artifacts.

        For quantized models, weights matching known quant-specific suffixes
        are logged as a warning.  Any *other* missing weight raises
        ``ValueError`` regardless of quantization.
        """
        if self.quant_config is None:
            raise ValueError(
                "The quantization config is None, and the following weights "
                f"were not initialized from checkpoint: {weights_not_loaded}"
            )

        expected_missing = {w for w in weights_not_loaded if self._is_expected_quantized_weight(w)}
        unexpected_missing = weights_not_loaded - expected_missing

        if expected_missing:
            logger.warning(
                "Following weights were not initialized from checkpoint (expected for quantized models): %s",
                expected_missing,
            )
        if unexpected_missing:
            raise ValueError(f"Following weights were not initialized from checkpoint: {unexpected_missing}")

    def _init_from_load_format(
        self,
        load_format: str,
        target_device: torch.device,
        custom_pipeline_name: str | type[nn.Module] | None = None,
        is_hsdp: bool = False,
    ) -> nn.Module:
        """Initialize the model from a specified load format."""
        if load_format == "custom_pipeline":
            # NOTE: Custom pipelines call HuggingFace `from_pretrained(...).to(device)`
            # internally. If we construct them under `with target_device:` (CUDA),
            # safetensors takes a direct-to-GPU fast path that calls `cudaMalloc`
            # via the driver API and BYPASSES PyTorch's caching allocator.
            # That makes those bytes invisible to CuMemAllocator, so `sleep()`
            # cannot offload/unmap them and GPU memory stays pinned.
            #
            # Fix: build the custom pipeline on CPU first (no default device
            # context), then explicitly move it to the target device. The
            # subsequent `.to(target_device)` issues `torch.empty(..., device=cuda)`
            # + `copy_`, which goes through the caching allocator and is fully
            # tracked by CuMemAllocator.
            model_cls = _resolve_custom_pipeline_cls(custom_pipeline_name)
            with set_current_diffusion_config(self.od_config):
                model = model_cls(od_config=self.od_config)
            # HSDP normally defers GPU placement to apply_hsdp_to_model to keep peak
            # load-time memory on CPU. Online quantization (e.g. fp8) runs CUDA-only
            # kernels inside load_weights via the layerwise loader, so when a quant
            # config is set we initialize on the accelerator like the non-HSDP path;
            # apply_hsdp_to_model shards GPU-resident params equally well.
            hsdp_defer_to_cpu = is_hsdp and self.quant_config is None
            if not hsdp_defer_to_cpu and target_device.type != "cpu":
                model.to(target_device)
        else:
            hsdp_defer_to_cpu = is_hsdp and self.quant_config is None
            device_ctx = contextlib.nullcontext() if hsdp_defer_to_cpu else target_device
            with device_ctx:
                if load_format == "default":
                    model = initialize_model(self.od_config)
                elif load_format == "diffusers":
                    model = DiffusersAdapterPipeline(od_config=self.od_config, device=target_device)
                else:
                    raise ValueError(f"Unknown load_format: {load_format}")
        return model

    def _load_model_with_hsdp(
        self,
        target_device: torch.device,
        load_format: str = "default",
        custom_pipeline_name: str | type[nn.Module] | None = None,
    ) -> nn.Module:
        """Load model with HSDP sharding for inference.

        The pipeline contains multiple components (text_encoder, VAE, transformer).
        Only the transformer is sharded with HSDP. Other components are loaded normally.

        Approach: Load weights first using model's load_weights (handles QKV fusion etc.),
        then apply HSDP sharding to redistribute weights across GPUs.
        """
        hsdp_config = HSDPInferenceConfig(
            enabled=True,
            hsdp_replicate_size=self.parallel_config.hsdp_replicate_size,
            hsdp_shard_size=self.parallel_config.hsdp_shard_size,
            param_dtype=self.od_config.dtype,
        )

        # Initialize model WITHOUT device context (weights start on CPU).
        # Unlike the non-HSDP path which uses `with target_device:` to create weights
        # directly on GPU, HSDP needs weights on CPU first so they can be redistributed
        # across GPUs by apply_hsdp_to_model. The model's load_weights handles weight
        # mapping (QKV fusion, etc.).
        if load_format == "diffusers":
            raise ValueError("HSDP is not supported with the diffusers adapter load format")
        model = self._init_from_load_format(load_format, target_device, custom_pipeline_name, is_hsdp=True)
        self.load_weights(model)

        # Quantization methods must finish while parameters are ordinary local
        # tensors. Some post-load transforms use operations (for example,
        # torch.unique in ModelOpt NVFP4) that do not support DTensor inputs.
        self._process_weights_after_loading(model, target_device)

        # Discover pipeline components (DiT, encoders, VAEs) via
        # ModuleDiscovery, which consults SupportsComponentDiscovery
        # when available and falls back to well-known attribute names.
        # This supports nested pipelines (e.g. LTX2DistilledPipeline
        # where the transformer lives at "pipe.transformer").
        discovered_modules = ModuleDiscovery.discover(model)

        # Shard only the outermost DiTs. A pipeline may list a DiT and one of its
        # submodules as separate DiTs (e.g. Cosmos3's transformer and the nested
        # transformer.language_model) for offload's independent rings; for HSDP an
        # inner DiT is already covered by its ancestor's _hsdp_shard_conditions, so
        # sharding it again would double-wrap blocks and require the inner stack to
        # declare its own conditions.
        outer_dit_names, outer_dits = discovered_modules.outermost_dits()

        # Online FP8 quantization (Fp8OnlineLinearMethod) leaves layer weights
        # as non-contiguous transpose views (qweight.t()) so the Cutlass kernel
        # gets a column-major B. FSDP2 fully_shard rejects non-contiguous params.
        # Rewrite affected layers in-place to row-major contiguous storage and
        # shift the .t() to GEMM-call time. Layers using other quant methods or
        # already-contiguous weights are left untouched.
        if self.quant_config is not None:
            from vllm_omni.diffusion.quantization.hsdp_fp8 import (
                prepare_fp8_layers_for_fsdp,
            )

            for trans in outer_dits:
                prepare_fp8_layers_for_fsdp(trans)

        if not outer_dits:
            raise ValueError("No DiT modules discovered for HSDP sharding")

        # Apply HSDP sharding to each outermost DiT transformer
        for name, trans in zip(outer_dit_names, outer_dits):
            logger.debug("Applying HSDP to %s", name)
            apply_hsdp_to_model(trans, hsdp_config, target_device=target_device)

        # HSDP only shards transformer modules. All other runtime modules must
        # be placed on the execution device explicitly after sharding.
        modules_to_move: list[nn.Module] = []
        if discovered_modules.vaes is not None:
            modules_to_move.extend(discovered_modules.vaes)
        if discovered_modules.encoders is not None:
            modules_to_move.extend(discovered_modules.encoders)
        if discovered_modules.resident_modules is not None:
            modules_to_move.extend(discovered_modules.resident_modules)

        for module in modules_to_move:
            module.to(target_device)

        return model
