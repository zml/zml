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
import functools
import inspect
import logging
import re

import torch
from torch import Tensor

log = logging.getLogger(__name__)

builtin_open = builtins.open


def log_and_open(file, *args, **kwargs):
    print("open:", file, args, kwargs)
    return builtin_open(file, *args, **kwargs)


builtins.open = log_and_open


class CollectionOver(Exception):
    pass

class ActivationCollector:
    """Wrap a given torch.nn.Module and collect all its intermediary activations.

    Usage:

    collector = zml_utils.ActivationCollector(model, **collection_config)
    model_output, activations = collector(model_input)
    zml_utils.save_with_confirmation("activations.pt", activations)

    Args:
        * max_layers: stop collecting activations after this many collected
        * stop_after_first_step: if a layer is called twice (typically for generative model), stop immediately
        * blacklist_regexes: skeep layers matching any of the regexes
    """

    def __init__(
        self,
        model,
        *,
        max_layers: int = -1,
        stop_after_first_step: bool = False,
        blacklist_regexes: list[str] = [r".*\.(\d\d+)\.", r".*\.[1-9]\."],
    ):
        self.model = model
        self.max_layers = max_layers
        self.stop_after_first_step = stop_after_first_step
        self.blacklist_regexes = blacklist_regexes
        self.count = 0
        mods = named_modules(model)
        self.outs = {id(module): (name, None, None) for name, module in mods}

    def __call__(self, *args, **kwargs):
        """Call the wrapped model with the given arguments.

        Return the model output and the activations.
        """
        self.count = 0
        hook = torch.nn.modules.module.register_module_forward_hook(
            self.log_activation_hook
        )
        # modeling_llama.apply_rotary_pos_emb = my_rot
        try:
            res = self.model(*args, **kwargs)
        except ActivationCollector.CollectionOver:
            res = None
        finally:
            # modeling_llama.apply_rotary_pos_emb = rot
            hook.remove()

        tensors = {}
        # print(module_outs)
        for name, outputs, inputs in self.outs.values():
            # Only save first layer for a smaller file.
            for blacklist in self.blacklist_regexes:
                if re.match(blacklist, name):
                    # print("skipping:", name)
                    continue

            if name == "":
                # Skip the softmax output
                continue
            if (outputs, inputs) == (None, None):
                # print(f"no inputs/outputs for {name}")
                continue
            for idx, inp in enumerate(inputs):
                tensors[f"{name}.in.{idx}"] = inp

            for idx, out in enumerate(outputs):
                tensors[f"{name}.out.{idx}"] = out

        for k, v in tensors.items():
            if k.endswith(".out.1"):
                print(k, "->", v.shape)

        return res, tensors

    def log_activation_hook(self, module, input, out) -> None:
        name, prev_out, prev_in = self.outs.get(id(module), (None, None, None))

        if self.stop_after_first_step and prev_out is not None:
            print(f"stopping collection cause {name} was already recorded")
            raise ActivationCollector.CollectionOver()
            return

        if prev_out is None:
            self.count += 1

        if name is None:
            print("err: unknown module", module.__class__)
            breakpoint()
            return

        assert out is not None
        outs = [o.detach().cpu() for o in _flatten(out)]
        inputs = [i.detach().cpu() for i in _flatten(input)]

        kwargs = inspect.stack()[1].frame.f_locals["kwargs"]
        extra_inputs = [i.detach().cpu() for i in _flatten(kwargs)]

        self.outs[id(module)] = (name, outs, inputs + extra_inputs)
        if 0 < self.max_layers < self.count:
            print(f"stopping collection cause we got {self.count} activations already")
            raise ActivationCollector.CollectionOver()


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
