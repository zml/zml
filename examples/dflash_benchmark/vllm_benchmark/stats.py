"""Dependency-free benchmark statistics helpers for DFlash-vs-baseline runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


TokenIds = Sequence[int]


def _as_int_list(values: Sequence[int] | None) -> list[int]:
    return [int(value) for value in values or ()]


def _duration_ms(elapsed_s: float) -> float:
    return float(elapsed_s) * 1000.0


def calc_tpot_ms(elapsed_s: float, tokens: int) -> float:
    if tokens == 0:
        return 0.0
    return _duration_ms(elapsed_s) / float(tokens)


def calc_tokens_per_second(elapsed_s: float, tokens: int) -> float:
    if tokens == 0 or elapsed_s == 0:
        return 0.0
    return float(tokens) / float(elapsed_s)


def normalize_histogram(counts: Sequence[int]) -> list[float]:
    total = sum(int(count) for count in counts)
    if total == 0:
        return [0.0 for _ in counts]
    rates = [float(count) / float(total) for count in counts]
    # Keep the JSON output stable while making non-empty rates sum exactly to 1.
    rates[-1] += 1.0 - sum(rates)
    return rates


@dataclass(frozen=True)
class QualityComparison:
    exact_match: bool = False
    compared_tokens: int = 0
    first_mismatch_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "compared_tokens": self.compared_tokens,
            "first_mismatch_index": self.first_mismatch_index,
        }


@dataclass(frozen=True)
class MethodResult:
    token_ids: TokenIds = field(default_factory=tuple)
    decoded_tokens: int = 0
    generated_text: str = ""
    elapsed_s: float = 0.0
    stopped_on_eos: bool = False
    acceptance_lengths: Sequence[int] = field(default_factory=tuple)
    committed_lengths: Sequence[int] = field(default_factory=tuple)
    acceptance_length_histogram: Sequence[int] = field(default_factory=tuple)

    def token_count(self) -> int:
        return int(self.decoded_tokens) if self.decoded_tokens else len(self.token_ids)

    def tpot_ms(self) -> float:
        return calc_tpot_ms(self.elapsed_s, self.token_count())

    def tokens_per_second(self) -> float:
        return calc_tokens_per_second(self.elapsed_s, self.token_count())

    def tau(self) -> float:
        steps = 0
        total = 0
        if self.committed_lengths:
            steps = len(self.committed_lengths)
            total = sum(int(length) for length in self.committed_lengths)
        elif self.acceptance_lengths:
            steps = len(self.acceptance_lengths)
            total = sum(int(length) + 1 for length in self.acceptance_lengths)
        else:
            for accepted_tokens, count in enumerate(self.acceptance_length_histogram):
                steps += int(count)
                total += int(count) * (accepted_tokens + 1)
        return 0.0 if steps == 0 else float(total) / float(steps)

    def histogram_rates(self) -> list[float]:
        return normalize_histogram(self.acceptance_length_histogram)

    def to_dict(self, include_acceptance: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "decoded_tokens": self.token_count(),
            "elapsed_s": float(self.elapsed_s),
            "elapsed_ms": _duration_ms(self.elapsed_s),
            "tpot_ms": self.tpot_ms(),
            "tps": self.tokens_per_second(),
            "stopped_on_eos": self.stopped_on_eos,
            "token_ids": _as_int_list(self.token_ids),
            "generated_text": self.generated_text,
        }
        if include_acceptance:
            result.update(
                {
                    "tau": self.tau(),
                    "acceptance_lengths": _as_int_list(self.acceptance_lengths),
                    "committed_lengths": _as_int_list(self.committed_lengths),
                    "acceptance_length_histogram": _as_int_list(
                        self.acceptance_length_histogram
                    ),
                    "acceptance_length_histogram_rates": self.histogram_rates(),
                }
            )
        return result


@dataclass(frozen=True)
class SampleResult:
    id: str
    dataset: str
    prompt_tokens: int
    baseline: MethodResult
    dflash: MethodResult
    quality: QualityComparison | None = None

    def comparison(self) -> QualityComparison:
        if self.quality is not None:
            return self.quality
        return compare_token_ids(self.baseline.token_ids, self.dflash.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "prompt_tokens": int(self.prompt_tokens),
            "baseline": self.baseline.to_dict(),
            "dflash": self.dflash.to_dict(include_acceptance=True),
            "quality": self.comparison().to_dict(),
        }


@dataclass(frozen=True)
class Summary:
    sample_count: int
    dataset: str
    baseline_tokens: int
    dflash_tokens: int
    baseline_elapsed_s: float
    dflash_elapsed_s: float
    baseline_tpot_ms: float
    baseline_tps: float
    dflash_tpot_ms: float
    dflash_tps: float
    speedup: float
    tau: float
    acceptance_length_histogram: list[int]
    acceptance_length_histogram_rates: list[float]
    exact_match_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "dataset": self.dataset,
            "baseline_tokens": self.baseline_tokens,
            "dflash_tokens": self.dflash_tokens,
            "baseline_elapsed_s": self.baseline_elapsed_s,
            "dflash_elapsed_s": self.dflash_elapsed_s,
            "baseline_tpot_ms": self.baseline_tpot_ms,
            "baseline_tps": self.baseline_tps,
            "dflash_tpot_ms": self.dflash_tpot_ms,
            "dflash_tps": self.dflash_tps,
            "speedup": self.speedup,
            "tau": self.tau,
            "exact_match_count": self.exact_match_count,
            "acceptance_length_histogram": list(self.acceptance_length_histogram),
            "acceptance_length_histogram_rates": list(
                self.acceptance_length_histogram_rates
            ),
        }


def compare_token_ids(baseline: TokenIds, dflash: TokenIds) -> QualityComparison:
    compared = min(len(baseline), len(dflash))
    for index, (baseline_token, dflash_token) in enumerate(
        zip(baseline[:compared], dflash[:compared])
    ):
        if baseline_token != dflash_token:
            return QualityComparison(
                exact_match=False,
                compared_tokens=compared,
                first_mismatch_index=index,
            )
    exact_match = len(baseline) == len(dflash)
    return QualityComparison(
        exact_match=exact_match,
        compared_tokens=compared,
        first_mismatch_index=None if exact_match else compared,
    )


def compute_summary(samples: Sequence[SampleResult]) -> Summary:
    baseline_tokens = sum(sample.baseline.token_count() for sample in samples)
    dflash_tokens = sum(sample.dflash.token_count() for sample in samples)
    baseline_elapsed_s = sum(float(sample.baseline.elapsed_s) for sample in samples)
    dflash_elapsed_s = sum(float(sample.dflash.elapsed_s) for sample in samples)
    exact_match_count = sum(1 for sample in samples if sample.comparison().exact_match)

    max_acceptance_len = 0
    acceptance_step_count = 0
    acceptance_total = 0
    for sample in samples:
        method = sample.dflash
        if method.acceptance_length_histogram:
            max_acceptance_len = max(
                max_acceptance_len, len(method.acceptance_length_histogram) - 1
            )
        if method.acceptance_lengths:
            max_acceptance_len = max(
                max_acceptance_len,
                max(int(length) for length in method.acceptance_lengths),
            )

        if method.committed_lengths:
            acceptance_step_count += len(method.committed_lengths)
            acceptance_total += sum(int(length) for length in method.committed_lengths)
        elif method.acceptance_lengths:
            acceptance_step_count += len(method.acceptance_lengths)
            acceptance_total += sum(int(length) + 1 for length in method.acceptance_lengths)
        else:
            for accepted_tokens, count in enumerate(method.acceptance_length_histogram):
                if count == 0:
                    continue
                max_acceptance_len = max(max_acceptance_len, accepted_tokens)
                acceptance_step_count += int(count)
                acceptance_total += int(count) * (accepted_tokens + 1)

    histogram = [0 for _ in range(max_acceptance_len + 1)]
    for sample in samples:
        method = sample.dflash
        if method.acceptance_lengths:
            for accepted_tokens in method.acceptance_lengths:
                histogram[int(accepted_tokens)] += 1
        else:
            for accepted_tokens, count in enumerate(method.acceptance_length_histogram):
                histogram[accepted_tokens] += int(count)

    baseline_tpot_ms = calc_tpot_ms(baseline_elapsed_s, baseline_tokens)
    dflash_tpot_ms = calc_tpot_ms(dflash_elapsed_s, dflash_tokens)
    return Summary(
        sample_count=len(samples),
        dataset=samples[0].dataset if samples else "",
        baseline_tokens=baseline_tokens,
        dflash_tokens=dflash_tokens,
        baseline_elapsed_s=baseline_elapsed_s,
        dflash_elapsed_s=dflash_elapsed_s,
        baseline_tpot_ms=baseline_tpot_ms,
        baseline_tps=calc_tokens_per_second(baseline_elapsed_s, baseline_tokens),
        dflash_tpot_ms=dflash_tpot_ms,
        dflash_tps=calc_tokens_per_second(dflash_elapsed_s, dflash_tokens),
        speedup=0.0 if dflash_tpot_ms == 0 else baseline_tpot_ms / dflash_tpot_ms,
        tau=(
            0.0
            if acceptance_step_count == 0
            else float(acceptance_total) / float(acceptance_step_count)
        ),
        acceptance_length_histogram=histogram,
        acceptance_length_histogram_rates=normalize_histogram(histogram),
        exact_match_count=exact_match_count,
    )


def report_lines(summary: Summary) -> list[str]:
    lines = [
        "============================================================",
        "RESULTS",
        "============================================================",
        f"Dataset:        {summary.dataset}",
        f"Samples:        {summary.sample_count}",
        (
            "Baseline TPOT:  "
            f"{summary.baseline_tpot_ms:.2f} ms ({summary.baseline_tps:.1f} TPS)"
        ),
        (
            "DFlash TPOT:    "
            f"{summary.dflash_tpot_ms:.2f} ms ({summary.dflash_tps:.1f} TPS)"
        ),
        f"Speedup:        {summary.speedup:.2f}x",
        f"Tau:            {summary.tau:.2f}",
        "",
        "Valid draft tokens histogram:",
    ]
    if summary.acceptance_length_histogram_rates:
        for valid_tokens, rate in enumerate(summary.acceptance_length_histogram_rates):
            lines.append(f"  {valid_tokens:>2}: {rate:.3f} {_bar(rate, 50)}")
    else:
        lines.append("  (no DFlash acceptance data)")
    lines.extend(
        [
            "",
            (
                "Output quality: "
                f"{summary.exact_match_count}/{summary.sample_count} "
                "samples match baseline exactly"
            ),
        ]
    )
    return lines


def format_report(summary: Summary) -> str:
    return "\n".join(report_lines(summary)) + "\n"


def benchmark_to_dict(
    samples: Sequence[SampleResult],
    summary: Summary | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "config": dict(config) if config is not None else None,
        "samples": [sample.to_dict() for sample in samples],
        "summary": (summary or compute_summary(samples)).to_dict(),
    }


def _bar(rate: float, width: int) -> str:
    if rate <= 0:
        return ""
    scaled = round(rate * float(width))
    return "x" * min(width, max(1, int(scaled)))

