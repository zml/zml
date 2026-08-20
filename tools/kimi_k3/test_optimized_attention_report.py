import json
from types import SimpleNamespace

from tools.kimi_k3.summarize_optimized_attention_report import KDA_CASES, MLA_CASES, summarize


def write(path, text):
    path.write_text(text)
    return path


def test_summarize_optimized_attention(tmp_path):
    kda_lines = []
    for name in sorted(KDA_CASES):
        sequence = 64 if name == "production_prefill64" else 1
        optimized = 90 if name.startswith("production_") else 0
        reference = 100 if name.startswith("production_") else 0
        kda_lines.append(
            f"KIMI_K3_KDA_OPT_PASS case={name} sequence={sequence} optimized_compile_us=1 "
            f"reference_compile_us=1 optimized_execute_us={optimized} reference_execute_us={reference}"
        )
    kda_lines.extend(("max_absolute_error: 1e-8", "finite=true", "KIMI_K3_KDA_OPT_ALL_PASS"))
    mla_lines = []
    for name in sorted(MLA_CASES):
        capacity = int(name.split("capacity", 1)[1].split("_", 1)[0])
        valid = int(name.rsplit("valid", 1)[1])
        execute = 500 if capacity == 4096 else 400 if capacity == 64 and valid == 64 else 0
        if execute:
            mla_lines.append(f"KIMI_K3_MLA_OPT_BENCH case={name} warmups=2 repetitions=7 mean_execute_us={execute}")
        mla_lines.append(
            f"KIMI_K3_MLA_OPT_PASS case={name} capacity={capacity} valid_tokens={valid} "
            f"compile_us=1 execute_us={execute} cache_values_per_token=576 expanded_kv=false"
        )
    mla_lines.extend(("finite=true", "KIMI_K3_MLA_OPT_ALL_PASS"))
    kda_manifest = {
        "cpu_inference_fallback": False, "checkpoint_downloaded": False,
        "cases": [{"name": name} for name in KDA_CASES],
    }
    mla_manifest = {
        "cpu_inference_fallback": False, "checkpoint_downloaded": False,
        "device": "NVIDIA H100 80GB HBM3",
        "tensor_file_sha256": "70595a6921f668872f7f963bab7a33b2fa5c6dfa4c0ace25c418d17d2e6c939a",
        "cases": [{"name": name} for name in MLA_CASES],
    }
    memory = {"memory": {"compression_ratio": 53.333, "latent_bytes_per_token_layer_bf16": 1152}}
    trace = {"traceEvents": [{"name": "cuda kernel"}]}
    args = SimpleNamespace(
        official_kda_log=write(tmp_path / "official.log", "KIMI_K3_KDA_PREFILL_ALL_PASS"),
        kda_log=write(tmp_path / "kda.log", "\n".join(kda_lines)),
        mla_cache_log=write(tmp_path / "cache.log", "KIMI_K3_MLA_SESSION_CACHE_PASS reset=1\nKIMI_K3_MLA_CACHE_ALL_PASS"),
        mla_log=write(tmp_path / "mla.log", "\n".join(mla_lines)),
        layer_family_log=write(tmp_path / "layer.log", "KIMI_K3_LAYER_FAMILY_ALL_PASS"),
        kda_manifest=write(tmp_path / "kda.json", json.dumps(kda_manifest)),
        mla_manifest=write(tmp_path / "mla.json", json.dumps(mla_manifest)),
        memory_baseline=write(tmp_path / "memory.json", json.dumps(memory)),
        kda_trace=write(tmp_path / "kda-trace.json", json.dumps({"traceEvents": trace["traceEvents"] + [{"name": "kimi_k3.kda.optimized_case"}]})),
        mla_trace=write(tmp_path / "mla-trace.json", json.dumps({"traceEvents": trace["traceEvents"] + [{"name": "kimi_k3.mla.optimized_case"}]})),
    )
    report = summarize(args)
    assert report["status"] == "PASS"
    assert report["kda"]["cases"]["production_prefill64"]["speedup"] > 1
    assert report["mla"]["benchmarks_us"]["capacity4096_valid4096"] == 500
