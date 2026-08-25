#!/usr/bin/env python3
"""Build the versioned Kimi K3 routed-expert cache used by //examples/llm.

The cache is deliberately component-major. Runtime can therefore stream six
large extents from each of four canonical 224-expert files, while an eight-GPU
run consumes each canonical range as two contiguous 112-expert halves.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

CACHE_NAME = ".zml-kimi-k3-experts-v1"
PARTIAL_NAME = CACHE_NAME + ".partial"
SCHEMA_VERSION = 1
EXPERTS = 896
PARTS = 4
EXPERTS_PER_PART = EXPERTS // PARTS
FIRST_LAYER = 1
END_LAYER = 93
EXPECTED_PAYLOAD = 1_446_456_066_048
EXPECTED_PART_PAYLOAD = 361_614_016_512
READ_GROUP_EXPERTS = 8
COMPONENTS = (
    ("w1", "weight_packed"),
    ("w1", "weight_scale"),
    ("w2", "weight_packed"),
    ("w2", "weight_scale"),
    ("w3", "weight_packed"),
    ("w3", "weight_scale"),
)
COMPONENT_SHAPES = (
    (EXPERTS_PER_PART, 3072, 1792),
    (EXPERTS_PER_PART, 3072, 112),
    (EXPERTS_PER_PART, 3584, 1536),
    (EXPERTS_PER_PART, 3584, 96),
    (EXPERTS_PER_PART, 3072, 1792),
    (EXPERTS_PER_PART, 3072, 112),
)


def tensor_shape_manifest() -> list[dict]:
    return [
        {
            "projection": projection,
            "component": component,
            "dtype": "U8",
            "shape": list(shape),
        }
        for (projection, component), shape in zip(
            COMPONENTS, COMPONENT_SHAPES
        )
    ]


def sha256_file(path: Path, chunk_size: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as src:
        while chunk := src.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb", buffering=0) as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"short safetensors prefix: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        header = source.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"short safetensors header: {path}")
    return json.loads(header), 8 + header_length


def encode_safetensors_header(tensors: dict) -> bytes:
    encoded = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def write_all_at(fd: int, data: bytes | bytearray | memoryview, offset: int) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise OSError("short positional write")
        offset += written
        view = view[written:]


class SourceIndex:
    def __init__(self, model: Path):
        index_path = model / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as source:
            self.weight_map = json.load(source)["weight_map"]
        self.model = model
        self.headers: dict[str, tuple[dict, int]] = {}
        self.fds: dict[str, int] = {}

    def tensor(self, name: str) -> dict:
        shard = self.weight_map.get(name)
        if shard is None:
            raise ValueError(f"missing source tensor: {name}")
        if shard not in self.headers:
            self.headers[shard] = read_safetensors_header(self.model / shard)
        header, data_start = self.headers[shard]
        metadata = header.get(name)
        if metadata is None:
            raise ValueError(f"index/header mismatch for {name}")
        begin, end = metadata["data_offsets"]
        return {
            "path": self.model / shard,
            "shard": shard,
            "offset": data_start + begin,
            "nbytes": end - begin,
            "shape": metadata["shape"],
            "dtype": metadata["dtype"],
        }

    def read(self, tensor: dict) -> bytes:
        shard = tensor["shard"]
        fd = self.fds.get(shard)
        if fd is None:
            fd = os.open(tensor["path"], os.O_RDONLY)
            self.fds[shard] = fd
        data = os.pread(fd, tensor["nbytes"], tensor["offset"])
        if len(data) != tensor["nbytes"]:
            raise ValueError(f"short source read: {tensor['path']}")
        return data

    def read_extent(self, first: dict, size: int) -> bytes:
        shard = first["shard"]
        fd = self.fds.get(shard)
        if fd is None:
            fd = os.open(first["path"], os.O_RDONLY)
            self.fds[shard] = fd
        data = os.pread(fd, size, first["offset"])
        if len(data) != size:
            raise ValueError(f"short source extent: {first['path']}")
        return data

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def source_name(layer: int, expert: int, projection: str, component: str) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        f"experts.{expert}.{projection}.{component}"
    )


def cache_name(layer: int, part: int, projection: str, component: str) -> str:
    return f"layers.{layer}.part.{part}.{projection}.{component}"


def cache_contract(fingerprint: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "expert_count": EXPERTS,
        "canonical_parts": PARTS,
        "experts_per_part": EXPERTS_PER_PART,
        "first_layer": FIRST_LAYER,
        "end_layer": END_LAYER,
        "expert_payload_bytes": EXPECTED_PAYLOAD,
        **fingerprint,
    }


def build_layout(source: SourceIndex) -> tuple[list[dict], dict[str, str]]:
    layouts: list[dict] = []
    weight_map: dict[str, str] = {}
    total_payload = 0
    for part in range(PARTS):
        tensors: dict[str, dict] = {}
        payload_offset = 0
        first_expert = part * EXPERTS_PER_PART
        for layer in range(FIRST_LAYER, END_LAYER):
            for projection, component in COMPONENTS:
                exemplar = source.tensor(
                    source_name(layer, first_expert, projection, component)
                )
                shape = [EXPERTS_PER_PART, *exemplar["shape"]]
                nbytes = exemplar["nbytes"] * EXPERTS_PER_PART
                name = cache_name(layer, part, projection, component)
                tensors[name] = {
                    "dtype": exemplar["dtype"],
                    "shape": shape,
                    "data_offsets": [payload_offset, payload_offset + nbytes],
                }
                weight_map[name] = f"experts-{part:05d}-of-{PARTS:05d}.safetensors"
                payload_offset += nbytes
        if payload_offset != EXPECTED_PART_PAYLOAD:
            raise ValueError(
                f"part {part} payload {payload_offset} != {EXPECTED_PART_PAYLOAD}"
            )
        header = encode_safetensors_header(tensors)
        layouts.append(
            {
                "part": part,
                "first_expert": first_expert,
                "end_expert": first_expert + EXPERTS_PER_PART,
                "filename": f"experts-{part:05d}-of-{PARTS:05d}.safetensors",
                "tensors": tensors,
                "header": header,
                "data_start": len(header),
                "payload_bytes": payload_offset,
            }
        )
        total_payload += payload_offset
    if total_payload != EXPECTED_PAYLOAD:
        raise ValueError(f"total payload {total_payload} != {EXPECTED_PAYLOAD}")
    return layouts, weight_map


def initialize_partial(
    partial: Path,
    layouts: list[dict],
    weight_map: dict[str, str],
    fingerprint: dict,
) -> dict:
    partial.mkdir()
    for layout in layouts:
        path = partial / layout["filename"]
        with path.open("w+b", buffering=0) as output:
            output.write(layout["header"])
            output.truncate(layout["data_start"] + layout["payload_bytes"])
    atomic_json(
        partial / "model.safetensors.index.json",
        {
            "metadata": {"total_size": EXPECTED_PAYLOAD},
            "weight_map": weight_map,
        },
    )
    journal = {**cache_contract(fingerprint), "completed": []}
    atomic_json(partial / "journal.json", journal)
    return journal


def validate_completed_cache(cache: Path, fingerprint: dict) -> bool:
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    expected = cache_contract(fingerprint)
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    if manifest.get("tensor_shapes") != tensor_shape_manifest():
        return False
    files = manifest.get("files", [])
    if len(files) != PARTS:
        return False
    for part, entry in enumerate(files):
        expected_name = f"experts-{part:05d}-of-{PARTS:05d}.safetensors"
        checksum = entry.get("sha256", "")
        if (
            entry.get("name") != expected_name
            or entry.get("first_expert") != part * EXPERTS_PER_PART
            or entry.get("end_expert") != (part + 1) * EXPERTS_PER_PART
            or entry.get("payload_bytes") != EXPECTED_PART_PAYLOAD
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            return False
        path = cache / expected_name
        if not path.is_file() or path.stat().st_size != entry.get("file_size"):
            return False
        if sha256_file(path) != checksum:
            return False
    return True


def load_or_initialize(
    model: Path,
    partial: Path,
    layouts: list[dict],
    weight_map: dict[str, str],
    fingerprint: dict,
) -> dict:
    if not partial.exists():
        return initialize_partial(partial, layouts, weight_map, fingerprint)
    journal_path = partial / "journal.json"
    if not journal_path.is_file():
        raise ValueError(f"conflicting partial cache without journal: {partial}")
    with journal_path.open("r", encoding="utf-8") as source:
        journal = json.load(source)
    for key, value in cache_contract(fingerprint).items():
        if journal.get(key) != value:
            raise ValueError(f"partial cache fingerprint mismatch: {key}")
    for layout in layouts:
        path = partial / layout["filename"]
        expected_size = layout["data_start"] + layout["payload_bytes"]
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"partial cache file mismatch: {path}")
    return journal


def layer_descriptors(
    source: SourceIndex, layer: int, first_expert: int, end_expert: int
) -> list[list[dict]]:
    experts = []
    for expert in range(first_expert, end_expert):
        records = [
            source.tensor(source_name(layer, expert, projection, component))
            for projection, component in COMPONENTS
        ]
        if len({record["shard"] for record in records}) != 1:
            raise ValueError(f"expert {expert} is split across safetensors shards")
        for previous, current in zip(records, records[1:]):
            if previous["offset"] + previous["nbytes"] != current["offset"]:
                raise ValueError(f"expert {expert} components are not contiguous")
        experts.append({"expert": expert, "records": records})
    experts.sort(
        key=lambda item: (
            item["records"][0]["shard"],
            item["records"][0]["offset"],
        )
    )
    return experts


def pack_unit(
    source: SourceIndex,
    output_fd: int,
    layout: dict,
    layer: int,
) -> int:
    experts = layer_descriptors(
        source, layer, layout["first_expert"], layout["end_expert"]
    )
    transferred = 0
    group_start = 0
    while group_start < len(experts):
        first = experts[group_start]["records"][0]
        shard = first["shard"]
        group_end = group_start + 1
        previous = experts[group_start]["records"][-1]
        while group_end < len(experts) and group_end - group_start < READ_GROUP_EXPERTS:
            candidate = experts[group_end]["records"][0]
            if (
                candidate["shard"] != shard
                or previous["offset"] + previous["nbytes"] != candidate["offset"]
            ):
                break
            previous = experts[group_end]["records"][-1]
            group_end += 1
        group = experts[group_start:group_end]
        last = group[-1]["records"][-1]
        extent_size = last["offset"] + last["nbytes"] - first["offset"]
        extent = source.read_extent(first, extent_size)
        transferred += extent_size
        for item in group:
            expert = item["expert"]
            for component_index, (projection, component) in enumerate(COMPONENTS):
                record = item["records"][component_index]
                relative = record["offset"] - first["offset"]
                target = layout["tensors"][
                    cache_name(layer, layout["part"], projection, component)
                ]
                per_expert = record["nbytes"]
                target_offset = (
                    layout["data_start"]
                    + target["data_offsets"][0]
                    + (expert - layout["first_expert"]) * per_expert
                )
                write_all_at(
                    output_fd,
                    extent[relative : relative + per_expert],
                    target_offset,
                )
        group_start = group_end
    return transferred


def run(model: Path) -> None:
    model = model.resolve()
    final = model / CACHE_NAME
    partial = model / PARTIAL_NAME
    fingerprint = {
        "source_index_sha256": sha256_file(
            model / "model.safetensors.index.json"
        ),
        "source_config_sha256": sha256_file(model / "config.json"),
    }
    if final.exists():
        if validate_completed_cache(final, fingerprint):
            print(f"Kimi K3 expert cache already valid: {final}")
            return
        raise ValueError(
            f"present Kimi K3 expert cache is invalid and will not be overwritten: {final}"
        )

    source = SourceIndex(model)
    output_fds: dict[int, int] = {}
    try:
        layouts, weight_map = build_layout(source)
        journal = load_or_initialize(
            model, partial, layouts, weight_map, fingerprint
        )
        completed = set(journal["completed"])
        for layout in layouts:
            output_fds[layout["part"]] = os.open(
                partial / layout["filename"], os.O_RDWR
            )

        started = time.monotonic()
        total_written = 0
        for layer in range(FIRST_LAYER, END_LAYER):
            for layout in layouts:
                unit = f"{layer}:{layout['part']}"
                if unit in completed:
                    continue
                transferred = pack_unit(
                    source, output_fds[layout["part"]], layout, layer
                )
                os.fsync(output_fds[layout["part"]])
                completed.add(unit)
                journal["completed"] = sorted(
                    completed,
                    key=lambda item: tuple(map(int, item.split(":"))),
                )
                atomic_json(partial / "journal.json", journal)
                total_written += transferred
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"layer={layer}/92 part={layout['part'] + 1}/4 "
                    f"read={transferred / (1 << 30):.2f}GiB "
                    f"rate={total_written / elapsed / (1 << 30):.2f}GiB/s",
                    flush=True,
                )

        for fd in output_fds.values():
            os.fsync(fd)
            os.close(fd)
        output_fds.clear()

        files = []
        for layout in layouts:
            path = partial / layout["filename"]
            print(f"checksumming {path.name}", flush=True)
            files.append(
                {
                    "name": path.name,
                    "first_expert": layout["first_expert"],
                    "end_expert": layout["end_expert"],
                    "payload_bytes": layout["payload_bytes"],
                    "file_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "expert_count": EXPERTS,
            "canonical_parts": PARTS,
            "experts_per_part": EXPERTS_PER_PART,
            "first_layer": FIRST_LAYER,
            "end_layer": END_LAYER,
            "expert_payload_bytes": EXPECTED_PAYLOAD,
            **fingerprint,
            "tensor_shapes": tensor_shape_manifest(),
            "files": files,
        }
        atomic_json(partial / "manifest.json", manifest)
        os.replace(partial, final)
        print(f"Published Kimi K3 expert cache: {final}")
    finally:
        for fd in output_fds.values():
            os.close(fd)
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.model)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"prepack_experts: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
