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
import logging
import re
from typing import Optional

import torch

log = logging.getLogger(__name__)

class ActivationCollector:
    """Wrap a torch module and collect intermediate activations.

    Usage:

    collector = zml_utils.ActivationCollector(model, **collection_config)
    model_output, activations = collector(model_input)
    zml_utils.save_with_confirmation("activations.pt", activations)

    Args:
        * max_layers: stop collecting after this many captured module calls.
        * stop_after_first_step: if a module is called twice, keep first capture only.
        * include_regexes: optional module-name regex allowlist for pass-based tracing.
        * leaf_modules_only: when True, capture only modules with no child modules.
        * capture_inputs: when True, also store positional and keyword inputs.
        * max_capture_bytes: in-memory capture budget; disables further captures when reached.
        * blacklist_regexes: legacy field kept for compatibility.
    """

    class CollectionOver(Exception):
        pass

    def __init__(
        self,
        model,
        *,
        max_layers: int = -1,
        stop_after_first_step: bool = False,
        blacklist_regexes: list[str] = [r".*\.(\d\d+)\.", r".*\.[1-9]\."],
        include_regexes: Optional[list[str]] = None,
        leaf_modules_only: bool = True,
        capture_inputs: bool = False,
        max_capture_bytes: int = 2 * 1024**3,
    ):
        self.model = model
        self.max_layers = max_layers
        self.stop_after_first_step = stop_after_first_step
        self.blacklist_regexes = blacklist_regexes
        self.include_regexes = include_regexes or []
        self.leaf_modules_only = leaf_modules_only
        self.capture_inputs = capture_inputs
        self.max_capture_bytes = max_capture_bytes
        self.count = 0
        self.capture_bytes = 0
        self.capture_disabled = False
        self.outs = {}
        self.modules = []
        self._seen = 0

        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"ActivationCollector expects a torch.nn.Module, got {type(model)}")
        mods = list(model.named_modules())
        print("ActivationCollector __init__: registered modules count:", len(mods))
        for name, mod in mods:
            if self.include_regexes and not any(re.match(pat, name) for pat in self.include_regexes):
                continue
            if self.leaf_modules_only and any(True for _ in mod.children()):
                continue
            self.modules.append(mod)
            self.outs[id(mod)] = (name, None, None)
        print("ActivationCollector __init__: traced modules count:", len(self.modules))


        # mods = list(named_modules(model))
        # print("mods names:", [name for name, _ in mods])
        # self.outs = {id(module): (name, None, None) for name, module in mods}
        # print("ActivationCollector __init__: registered modules count:", len(self.outs))

    def __call__(self, *args, **kwargs):
        """Call the wrapped model with the given arguments.

        Return the model output and the activations.
        """
        self.count = 0
        self.capture_bytes = 0
        self.capture_disabled = False
        self._seen = 0
        hooks = [
            module.register_forward_hook(self.log_activation_hook, with_kwargs=True)
            for module in self.modules
        ]

        try:
            res = self.model(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()

        # tensors = {}

        # for name, outputs, inputs in self.outs.values():
        #     # Only save first layer for a smaller file.
        #     for blacklist in self.blacklist_regexes:
        #         if re.match(blacklist, name):
        #             continue

        #     if name == "":
        #         # Skip the softmax output
        #         continue
        #     if (outputs, inputs) == (None, None):
        #         print(f"no inputs/outputs for {name}")
        #         continue
        #     for idx, inp in enumerate(inputs):
        #         tensors[f"{name}.in.{idx}"] = inp

        #     for idx, out in enumerate(outputs):
        #         tensors[f"{name}.out.{idx}"] = out

        # for k, v in tensors.items():
        #     print(k, "->", v.shape)

        activations = {}
        for _, (name, out, inp) in self.outs.items():
            if out is not None:
                activations[name] = {
                    "input": inp,
                    "output": out,
                }

        # return res, tensors
        return res, activations

    def log_activation_hook(self, module, input, kwargs, out) -> None:
        # Debug
        # print("hook:", type(module), "known:", id(module) in self.outs)

        # name, prev_out, prev_in = self.outs.get(id(module), (None, None, None))

        # if self.stop_after_first_step and prev_out is not None:
        #     print(f"stopping collection cause {name} was already recorded or stop_after_first_step was set to `True`")
        #     raise ActivationCollector.CollectionOver()

        # if prev_out is None:
        #     self.count += 1

        # if name is None:
        #     return

        # assert out is not None
        # outs = [o.detach().cpu() for o in _flatten(out)]
        # inputs = [i.detach().cpu() for i in _flatten(input)]

        # kwargs = inspect.stack()[1].frame.f_locals["kwargs"]
        # extra_inputs = [i.detach().cpu() for i in _flatten(kwargs)]

        # self.outs[id(module)] = (name, outs, inputs + extra_inputs)
        # if 0 < self.max_layers < self.count:
        #     print(f"stopping collection cause we got {self.count} activations already")
        #     raise ActivationCollector.CollectionOver()
        
        if self.capture_disabled:
            return

        entry = self.outs.get(id(module))
        if entry is None:
            return

        self._seen += 1
        if self._seen % 200 == 0:
            print(f"ActivationCollector progress: seen {self._seen} module calls")

        name, prev_out, prev_in = entry

        if self.stop_after_first_step and prev_out is not None:
            return

        if out is None:
            return

        outs = [o.detach().cpu() for o in _flatten(out)]
        captured_inputs = []
        if self.capture_inputs:
            captured_inputs = [i.detach().cpu() for i in _flatten(input)]
            captured_inputs.extend(i.detach().cpu() for i in _flatten(kwargs))

        new_bytes = 0
        for t in outs:
            new_bytes += t.numel() * t.element_size()
        for t in captured_inputs:
            new_bytes += t.numel() * t.element_size()

        if self.max_capture_bytes > 0 and self.capture_bytes + new_bytes > self.max_capture_bytes:
            self.capture_disabled = True
            print(
                "ActivationCollector: reached capture budget "
                f"({self.capture_bytes / 1024**3:.2f} GiB), disabling further captures"
            )
            return

        self.capture_bytes += new_bytes
        self.outs[id(module)] = (name, outs, captured_inputs)

        self.count += 1
        if self.count % 100 == 0:
            print(
                "ActivationCollector progress: captured "
                f"{self.count} activations, stored {self.capture_bytes / 1024**3:.2f} GiB"
            )
        if 0 < self.max_layers < self.count:
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
        print("-------------- hasattr(model, \"named_modules\") -------------- ")
        return model.named_modules()

    else:
        print("ELSE -------------- hasattr(model, \"named_modules\") -------------- ")
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
