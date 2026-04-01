import torch
import transformers
from safetensors.torch import save_file


def flatten_tensors(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, tuple):
        out = []
        for item in value:
            out.extend(flatten_tensors(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_tensors(item))
        return out
    return []


def collect_activations(model, root_name, **model_kwargs):
    module_data = {
        id(module): {
            "name": root_name if name == "" else f"{root_name}.{name}",
            "inputs": None,
            "outputs": None,
        }
        for name, module in model.named_modules()
    }
    pending_inputs = {}
    handles = []

    def pre_hook(module, args, kwargs):
        entry = module_data.get(id(module))
        if entry is None or entry["outputs"] is not None:
            return
        inps = [t.detach().cpu() for t in flatten_tensors(args)]
        inps.extend([t.detach().cpu() for t in flatten_tensors(kwargs)])
        pending_inputs[id(module)] = inps

    def hook(module, args, kwargs, out):
        entry = module_data.get(id(module))
        if entry is None or entry["outputs"] is not None:
            return
        outs = [t.detach().cpu() for t in flatten_tensors(out)]
        inps = pending_inputs.pop(id(module), None)
        if inps is None:
            inps = [t.detach().cpu() for t in flatten_tensors(args)]
            inps.extend([t.detach().cpu() for t in flatten_tensors(kwargs)])
        entry["inputs"] = inps
        entry["outputs"] = outs

    try:
        for _, module in model.named_modules():
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(hook, with_kwargs=True))

        with torch.no_grad():
            output = model(**model_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    tensors = {}
    captured_modules = 0
    for entry in module_data.values():
        name = entry["name"]
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        if inputs is None or outputs is None:
            continue
        captured_modules += 1
        for idx, inp in enumerate(inputs):
            tensors[f"{name}.in.{idx}"] = inp
        for idx, out in enumerate(outputs):
            tensors[f"{name}.out.{idx}"] = out
    print(f"Captured activations from {captured_modules} modules")
    return output, tensors


def _count_video_runs(mm_token_type_ids: torch.Tensor) -> int:
    vals = mm_token_type_ids[0].tolist()
    runs = 0
    in_video = False
    for v in vals:
        if v == 2:
            if not in_video:
                runs += 1
                in_video = True
        else:
            in_video = False
    return runs


def _repair_video_grid_mismatch(model_inputs: dict) -> dict:
    mm_token_type_ids = model_inputs.get("mm_token_type_ids")
    video_grid_thw = model_inputs.get("video_grid_thw")
    if not isinstance(mm_token_type_ids, torch.Tensor) or not isinstance(video_grid_thw, torch.Tensor):
        return model_inputs
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
        return model_inputs

    runs = _count_video_runs(mm_token_type_ids)
    rows = int(video_grid_thw.shape[0])
    if runs <= rows or rows != 1:
        return model_inputs

    t, h, w = video_grid_thw[0].tolist()
    if int(t) != runs:
        return model_inputs

    expanded = torch.stack(
        [torch.tensor([1, h, w], dtype=video_grid_thw.dtype, device=video_grid_thw.device) for _ in range(runs)],
        dim=0,
    )
    fixed = dict(model_inputs)
    fixed["video_grid_thw"] = expanded
    return fixed


def main() -> None:
    model_path = "/var/models/Qwen/Qwen3.5-0.8B"
    video_path = "../video_test/data/goat.mp4"
    prompt = "What is happening in this video?"
    out_path = "../video_test/data/video_activations.safetensors"
    num_frames = 8
    video_size = {
        "shortest_edge": 128 * 128,
        "longest_edge": 256 * 256,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=num_frames,
        fps=None,
        size=video_size,
    )
    model_inputs = {
        k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in model_inputs.items()
    }
    model_inputs = _repair_video_grid_mismatch(model_inputs)

    visual_inputs = {
        "hidden_states": model_inputs["pixel_values_videos"],
        "grid_thw": model_inputs["video_grid_thw"],
    }
    output, activations = collect_activations(
        model.model.visual,
        "model.visual",
        **visual_inputs,
    )

    if isinstance(output, tuple):
        print(tuple(x.shape for x in output if isinstance(x, torch.Tensor)))
    elif output is not None and hasattr(output, "shape"):
        print(output.shape)
    else:
        print(output)

    activations = {k: v.contiguous() for k, v in activations.items()}
    save_file(activations, out_path)
    print(f"Saved {len(activations)} activations to {out_path}")


if __name__ == "__main__":
    main()
