from pathlib import Path
import torch

for path in sorted(Path("trace_run").glob("*.pt")):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n== {path.name} ==")

    if torch.is_tensor(obj):
        print(f"tensor shape={tuple(obj.shape)} dtype={obj.dtype}")

    elif isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                print(f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"{k}: list len={len(v)}")
                if len(v) > 0:
                    first = v[0]
                    if torch.is_tensor(first):
                        print(f"  first: tensor shape={tuple(first.shape)} dtype={first.dtype}")
                    else:
                        print(f"  first: type={type(first)}")
            else:
                print(f"{k}: {type(v)}")

    elif isinstance(obj, list):
        print(f"list len={len(obj)}")
        if len(obj) > 0:
            first = obj[0]
            print(f"first type={type(first)}")
            if isinstance(first, dict):
                print(f"first keys={list(first.keys())}")
                for k, v in first.items():
                    if torch.is_tensor(v):
                        print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
                    else:
                        print(f"  {k}: {type(v)}")
            elif torch.is_tensor(first):
                print(f"first tensor shape={tuple(first.shape)} dtype={first.dtype}")

    else:
        print(type(obj))
