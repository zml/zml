import torch
import transformers
from PIL import Image
from safetensors.torch import save_file

# Each image dimension must be a multiple of this value (patch_size * spatial_merge_size).
IMAGE_DIM_DIV_FACTOR = 32


def nearest_multiple(value: int, multiple: int) -> int:
    floor = (value // multiple) * multiple
    ceil = ((value + multiple - 1) // multiple) * multiple
    floor = max(multiple, floor)
    ceil = max(multiple, ceil)
    if value - floor < ceil - value:
        return floor
    return ceil

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


def collect_activations(model, **model_kwargs):
    module_data = {
        id(module): {"name": name, "inputs": None, "outputs": None}
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
        if name == "" or inputs is None or outputs is None:
            continue
        captured_modules += 1
        for idx, inp in enumerate(inputs):
            tensors[f"{name}.in.{idx}"] = inp
        for idx, out in enumerate(outputs):
            tensors[f"{name}.out.{idx}"] = out
    print(f"Captured activations from {captured_modules} modules")
    return output, tensors


model_path = "/var/models/Qwen/Qwen3.5-0.8B"
image_path = "../image_test/data/girafe.png"
prompt = "What is in this picture?\n"
out_path = "../image_test/data/image_activations.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Preprocess image with the official processor path.
processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
image = Image.open(image_path).convert("RGB")
target_width = nearest_multiple(image.width, IMAGE_DIM_DIV_FACTOR)
target_height = nearest_multiple(image.height, IMAGE_DIM_DIV_FACTOR)
image = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
processed = processor.image_processor(images=[image], return_tensors="pt")
pixel_values = processed["pixel_values"].to(device=device)
image_grid_thw = processed["image_grid_thw"].to(device=device)

# 2) Build multimodal inputs similarly to image_test wiring.
tokenizer = processor.tokenizer
model = transformers.AutoModelForImageTextToText.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
).to(device)
model.eval()

prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device=device, dtype=torch.int64)
image_token_count = int(
    (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item()
    // (model.config.vision_config.spatial_merge_size**2)
)
text_before_len = prompt_ids.shape[0]
image_tokens = torch.full(
    (image_token_count,),
    fill_value=model.config.image_token_id,
    dtype=torch.int64,
    device=device,
)
input_ids = torch.cat([prompt_ids, image_tokens], dim=0).unsqueeze(0)
attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
mm_token_type_ids[:, text_before_len : text_before_len + image_token_count] = 1

# 3) Collect activations and save.
output, activations = collect_activations(
    model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    mm_token_type_ids=mm_token_type_ids,
)

if output is not None and hasattr(output, "logits"):
    print(output.logits.shape)
else:
    print(output)

activations = {k: v.contiguous() for k, v in activations.items()}
save_file(activations, out_path)
print(f"Saved {len(activations)} activations to {out_path}")
