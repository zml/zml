# Copyright 2024 ZML
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import builtins
import enum
import inspect
import logging
import re

import torch

log = logging.getLogger(__name__)

class ActivationCollector:
    """Wrap a given torch.nn.Module and collect all its intermediary activations.

    Usage:

    collector = zml_utils.ActivationCollector(model, **collection_config)
    model_output, activations = collector(model_input)
    zml_utils.save_with_confirmation("activations.pt", activations)

    Args:
        * max_layers: stop collecting activations after this many indexed transformer layers
        * stop_after_first_step: if a layer is called twice (typically for generative model), stop immediately
        * blacklist_regexes: skip layers matching any of the regexes
        * gather_sharded_tensors: gather tensor-parallel inputs/outputs into logical full tensors
        * merge_rank_local_tensors: merge rank-local modules, such as MoE experts, across distributed ranks
        * rank_local_regexes: tensor keys that exist on only one rank and need an object merge
    """

    class CollectionOver(Exception):
        pass

    def __init__(
        self,
        model,
        *,
        max_layers: int = -1,
        stop_after_first_step: bool = False,
        blacklist_regexes: list[str] | None = None,
        gather_sharded_tensors: bool = True,
        merge_rank_local_tensors: bool = True,
        rank_local_regexes: list[str] | None = None,
    ):
        self.model = model
        self.max_layers = max_layers
        self.stop_after_first_step = stop_after_first_step
        self.blacklist_regexes = blacklist_regexes or []
        self.gather_sharded_tensors = gather_sharded_tensors
        self.merge_rank_local_tensors = merge_rank_local_tensors
        self.rank_local_regexes = rank_local_regexes or [r".*\.experts\.\d+\."]
        self.count = 0
        mods = list(named_modules(model))
        self.modules = {id(module): module for name, module in mods}
        self.ins = {id(module): (name, None) for name, module in mods}
        self.outs = {id(module): (name, None, None, None) for name, module in mods}
        self.has_indexed_layers = any(_indexed_layer_key(name) is not None for name, module in mods)
        self.seen_layer_keys: set[str] = set()
        self.seen_module_ids: set[int] = set()

    def __call__(self, *args, **kwargs):
        """Call the wrapped model with the given arguments.

        Return the model output and the activations.
        """
        self.count = 0
        self.seen_layer_keys.clear()
        self.seen_module_ids.clear()
        hook_pre = torch.nn.modules.module.register_module_forward_pre_hook(
            self.log_activation_pre_forward
        )
        hook = torch.nn.modules.module.register_module_forward_hook(
            self.log_activation_hook
        )

        try:
            res = self.model(*args, **kwargs)
        except ActivationCollector.CollectionOver:
            res = None
        finally:
            hook_pre.remove()
            hook.remove()

        tensors = {}

        for name, outputs, inputs, out_kv_cache in self.outs.values():
            if name == "":
                # Skip the softmax output
                continue
            if self._is_blacklisted(name):
                continue
            if (outputs, inputs) == (None, None):
                # print(f"no inputs/outputs for {name}")
                continue
            for idx, inp in enumerate(inputs or []):
                if inp is not None:
                    tensors[f"{name}.in.{idx}"] = inp

            for idx, out in enumerate(outputs or []):
                if out is not None:
                    tensors[f"{name}.out.{idx}"] = out

            for kv_name, in_kv_cache in self.ins.values():
                if name == kv_name and in_kv_cache is not None:
                    tensors[f"{name}.in.kv_cache"] = in_kv_cache
                    break

            if out_kv_cache is not None:
                tensors[f"{name}.out.kv_cache"] = out_kv_cache

        tensors = self._merge_rank_local_tensors(tensors)

        for k, v in tensors.items():
            print(k, "->", v.shape)

        return res, tensors

    def log_activation_pre_forward(self, module, input) -> None:
        entry = self.ins.get(id(module))

        if entry is None:
            return

        name, prev_kv = entry
        kv_cache = None
        if hasattr(module, "kv_cache"):
            kv_cache = self._clone_tensor(module.kv_cache)

        self.ins[id(module)] = (name, kv_cache)

    def log_activation_hook(self, module, input, out) -> None:
        entry = self.outs.get(id(module))
        if entry is None:
            return

        name, prev_out, prev_in, prev_kv = entry
        if self._is_blacklisted(name):
            return

        if self.stop_after_first_step and prev_out is not None:
            print(f"stopping collection cause {name} was already recorded or stop_after_first_step was set to `True`")
            raise ActivationCollector.CollectionOver()

        if prev_out is None:
            self._count_layer(id(module), name)

        outs = [
            self._clone_activation(module, "out", idx, tensor)
            for idx, tensor in enumerate(_flatten(out))
        ]
        inputs = [
            self._clone_activation(module, "in", idx, tensor)
            for idx, tensor in enumerate(_flatten(input))
        ]

        kwargs = _forward_kwargs()
        extra_inputs = [
            self._clone_activation(module, "in", len(inputs) + idx, tensor)
            for idx, tensor in enumerate(_flatten(kwargs))
        ]

        kv_cache = None
        if hasattr(module, "kv_cache"):
            kv_cache = self._clone_tensor(module.kv_cache)

        self.outs[id(module)] = (name, outs, inputs + extra_inputs, kv_cache)

    def _count_layer(self, module_id: int, name: str) -> None:
        if self.max_layers <= 0:
            return

        if self.has_indexed_layers:
            layer_key = _indexed_layer_key(name)
            if layer_key is None or layer_key in self.seen_layer_keys:
                return
            self.seen_layer_keys.add(layer_key)
            self.count = len(self.seen_layer_keys)
        else:
            if module_id in self.seen_module_ids:
                return
            self.seen_module_ids.add(module_id)
            self.count = len(self.seen_module_ids)

        if 0 < self.max_layers < self.count:
            print(f"stopping collection cause we got {self.count} activations already")
            raise ActivationCollector.CollectionOver()

    def _clone_activation(self, module, direction: str, idx: int, tensor: torch.Tensor) -> torch.Tensor:
        if self.gather_sharded_tensors:
            tensor = _maybe_gather_sharded_tensor(module, direction, idx, tensor)
        return self._clone_tensor(tensor)

    def _clone_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().cpu().clone()

    def _is_blacklisted(self, name: str) -> bool:
        return any(re.match(pattern, name) for pattern in self.blacklist_regexes)

    def _merge_rank_local_tensors(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.merge_rank_local_tensors or not _dist_initialized():
            return tensors

        rank_local_tensors = {
            key: tensor
            for key, tensor in tensors.items()
            if self._is_rank_local_key(key)
        }
        world_size = torch.distributed.get_world_size()
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, rank_local_tensors)

        merged = dict(tensors)
        for rank, rank_tensors in enumerate(gathered):
            for key, tensor in rank_tensors.items():
                if key not in merged:
                    merged[key] = tensor
                elif merged[key].shape != tensor.shape:
                    merged[f"{key}.rank{rank}"] = tensor
        return merged

    def _is_rank_local_key(self, key: str) -> bool:
        return any(re.match(pattern, key) for pattern in self.rank_local_regexes)


def save_with_confirmation(filename: str, tensors: dict):
    """Regular torch.save with a CLI confirmation."""
    sizes = [(v.numel() * v.dtype.itemsize, k) for k, v in tensors.items()]
    sizes.sort()
    disk_size = sum(s for s, k in sizes)

    GB = 1024**3
    print(f"About to write {disk_size/ GB:.3f}GB at {filename}. Biggest tensors:")
    print(sizes[-20:])
    print("Enter `c` to continue, `q` to quit.")

    breakpoint()
    torch.save(tensors, filename)


def _flatten(out):
    if out is None:
        return []
    elif isinstance(out, torch.Tensor):
        outs = [out]
    elif isinstance(out, tuple):
        outs = []
        for x in out:
            outs.extend(_flatten(x))
    elif isinstance(out, dict):
        outs = []
        for x in out.values():
            outs.extend(_flatten(x))
    else:
        outs = []
    return outs


def _forward_kwargs() -> dict:
    try:
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            frame = frame.f_back.f_back
        if frame is not None:
            kwargs = frame.f_locals.get("kwargs", {})
            if isinstance(kwargs, dict):
                return kwargs
    except Exception:
        return {}
    return {}


def _dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _indexed_layer_key(name: str) -> str | None:
    match = re.search(r"(?:^|\.)(?:layers|layer|blocks|block)\.(\d+)(?:\.|$)", name)
    if match is None:
        return None
    return match.group(0).strip(".")


def _maybe_gather_sharded_tensor(module, direction: str, idx: int, tensor: torch.Tensor) -> torch.Tensor:
    if not _dist_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    shard_dim = _tensor_parallel_shard_dim(module, direction, idx)
    if shard_dim is None:
        return tensor

    return _all_gather_tensor_dim(tensor, shard_dim)


def _tensor_parallel_shard_dim(module, direction: str, idx: int) -> int | None:
    class_name = module.__class__.__name__

    if direction == "out" and (class_name == "ColumnParallelLinear" or hasattr(module, "part_out_features")):
        return -1

    if direction == "in" and idx == 0 and (class_name == "RowParallelLinear" or hasattr(module, "part_in_features")):
        return -1

    return None


def _all_gather_tensor_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor

    dim = dim % tensor.dim()
    local = tensor.contiguous()
    original_device = local.device

    if original_device.type == "cpu" and "nccl" in str(torch.distributed.get_backend()).lower():
        if not torch.cuda.is_available():
            return tensor
        local = local.cuda()

    chunks = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(chunks, local)
    gathered = torch.cat(chunks, dim=dim)
    if gathered.device != original_device:
        gathered = gathered.to(original_device)
    return gathered


def named_modules(model):
    if hasattr(model, "named_modules"):
        return model.named_modules()

    else:
        root_modules = [
            (k, v) for k, v in model.__dict__.items() if isinstance(v, torch.nn.Module)
        ]
        for root, mod in root_modules:
            for k, v in mod.named_modules():
                if k:
                    yield f"{root}.{k}", v
                else:
                    yield root, v


def read_layer_config(model: torch.nn.Module) -> dict:
    layer_config = {}

    def _append_node_config(node, prefix: str) -> None:
        for k, v in node.__dict__.items():
            # Skip special members. In particular all children module and tensors
            # will be hidden in special dicts `_parameters` and `_modules`
            if k.startswith("_"):
                continue
            # All modules have a "training" flag
            if k in ("training", "init_fn"):
                continue
            if v is None:
                continue

            if not is_basic_type(v):
                log.warning(f"Skipping layer config {k}={v!r}")
                continue
            layer_config[prefix + k] = v

    _append_node_config(model, "")
    for name, node in find_children(model, torch.nn.Module):
        _append_node_config(node, name + ".")

    return layer_config


def find_children(model: torch.nn.Module, t: type, layer_filter: str = "") -> list:
    queue = list(model._modules.items())
    modules = []
    while queue:
        name, node = queue.pop()
        if node is None:
            continue
        if layer_filter and not re.match(layer_filter, name):
            continue
        if isinstance(node, t):
            modules.append((name, node))
        for child_name, child_node in node._modules.items():
            queue.append((".".join((name, child_name)), child_node))

    return modules


def is_basic_type(value) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, enum.Enum):
        return True
    if isinstance(value, tuple) and len(value) == 1:
        return True
    if isinstance(value, str) and len(value) < 8:
        return True
    return False


def pdb_persistent(name, fn, *args, **kwargs):
    """Cache that can survive through a PDB restart.

    Useful when debugging to avoid reloading models all the time.
    """
    import sys

    pdb = sys.modules.get("pdb", None)
    if pdb is None:
        return fn(*args, **kwargs)

    if not hasattr(pdb, "__cache__"):
        setattr(pdb, "__cache__", {})

    cache = getattr(pdb, "__cache__")
    entry = cache.get(name)
    if entry is not None:
        return entry

    res = fn(*args, **kwargs)
    cache[name] = res
    return res
