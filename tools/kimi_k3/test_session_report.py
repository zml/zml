from summarize_session_report import summarize


def test_summarize_session_report() -> None:
    rows = [
        (0, 1, 50462),
        (1, 42, 131246),
        (2, 32000, 29932),
        (3, 160000, 95385),
    ]
    prefix_lines = []
    for repeat in range(2):
        prefix_lines.extend(
            f"token_index={index} input={token} greedy={greedy} load_us={100 + index} "
            f"execute_us={10 + index} total_us={120 + index}"
            for index, token, greedy in rows
        )
        prefix_lines.append(
            f"KIMI_K3_SESSION_PASS repeat={repeat} tokens=4 greedy=95385 "
            "compile_us=500 session_us=600 backend=cuda"
        )
    prefix_lines.append(
        "KIMI_K3_SESSION_ALL_PASS reset_deterministic=true official_prefix_checked=true"
    )
    report = summarize(
        "\n".join(prefix_lines),
        "KIMI_K3_SESSION_DECODE_PASS streamed=50462 next=103700 history_tokens=2 capacity=2",
        "KIMI_K3_TOKENIZER_ALL_PASS",
        {
            "authority": "official",
            "source": {"tiktoken.model": "a" * 64},
            "ordinary": [{}] * 7,
            "fuzz": [{}] * 512,
            "structural": [{}] * 6,
            "first_turn": [{}] * 3,
            "continuation": [{}] * 3,
        },
        {
            "token_ids": [1, 42, 32000, 160000],
            "prefill_greedy_token": 95385,
            "timing": {"repeat": {"wall_ms": 44_231.1}},
        },
        {
            "raw_token_ids": [1, 42, 32000, 160000],
            "official_greedy_token": 95385,
            "tokenizer_cases": {
                "ordinary": 7,
                "fuzz": 512,
                "structural": 6,
                "first_turn": 3,
                "continuation": 3,
            },
            "moonshot_revision": "c" * 40,
            "prefix_fixture_sha256": "d" * 64,
            "prefix_fixture_semantic_sha256": "e" * 64,
            "tokenizer_reference_sha256": "f" * 64,
            "tokenizer_json_sha256": "0" * 64,
            "layer_selection": [0, 1, 2, 3],
        },
        "1" * 40,
        "KIMI_K3_SESSION_FULL_COMPILE_PASS layers=93 source_slots=8 "
        "kda_boundary=true mla_boundary=false compile_us=700 backend=cuda",
        {"required_shards": 96, "present_shards": 5, "missing_shards": 91},
    )
    assert report["status"] == "PASS"
    assert report["full_schedule_readiness"]["status"] == "PASS"
    assert report["full_schedule_readiness"]["runtime_numeric_validation"] == "blocked_by_missing_local_shards"
    assert report["tokenizer"]["exact_corpus_parity"]
    assert report["zml_session"]["reset_deterministic"]
    assert report["zml_session"]["official_final_greedy_exact"]
    assert report["zml_session"]["streaming_decode"]["next"] == 103700
    assert report["performance"]["mean_synchronized_execute_us_per_token"] == 12
