#!/usr/bin/env python3
"""Minimal OD-distance -> link observations -> fare recovery experiment.

Place this file beside fare_model.py and run it there.  The script deliberately
keeps the model small: no hub boost, RFID only, no false-positive generation.

OD demand uses commercial distance d_ij:
    pi_ij = exp(-gamma * d_ij) / sum_ab exp(-gamma * d_ab)

Observation opportunities use link count m_ij (not distance):
    X_l ~ Bernoulli(p), l in Gamma_ij,  m_ij = |Gamma_ij|
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib が必要です。現在のPython環境で "
        "python3 -m pip install matplotlib を実行してください。"
    ) from exc

try:
    import fare_model
except ModuleNotFoundError as exc:
    raise SystemExit(
        "fare_model.py が見つかりません。このスクリプトを "
        "bibo_link_inference_model.py / fare_model.py / od_model.py と同じフォルダに置いてください。"
    ) from exc


DEFAULT_GAMMAS = (0.18, 0.0, -0.10)
DEFAULT_RATES = (0.65, 0.70)
GAMMA_LABELS = {0.18: "short_trip", 0.0: "distance_neutral", -0.10: "long_trip"}


@dataclass(frozen=True)
class Hypothesis:
    origin: str
    destination: str
    distance_km: float
    links: tuple[tuple[str, str], ...]
    fare_yen: float

    @property
    def trip_links(self) -> int:
        return len(self.links)


def _first_attr(names: Iterable[str]):
    for name in names:
        if hasattr(fare_model, name):
            return getattr(fare_model, name)
    return None


def load_station_order() -> list[str]:
    value = _first_attr(("STATIONS", "STATION_ORDER", "INNER_STATIONS", "YAMANOTE_STATIONS"))
    if isinstance(value, dict):
        value = list(value.keys())
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if isinstance(value[0], str):
            return list(value)
        if hasattr(value[0], "name"):
            return [x.name for x in value]
    raise RuntimeError(
        "fare_model.py から駅順を取得できません。STATIONS / STATION_ORDER / "
        "INNER_STATIONS / YAMANOTE_STATIONS のいずれかを駅名リストとして公開してください。"
    )


def _call_path_summary(origin: str, destination: str) -> dict:
    fn = getattr(fare_model, "path_summary", None)
    if fn is None:
        return {}
    for kwargs in ({"direction": "shortest"}, {}):
        try:
            result = fn(origin, destination, **kwargs)
            if isinstance(result, dict):
                return result
        except TypeError:
            continue
    return {}


def distance_between(origin: str, destination: str) -> float:
    summary = _call_path_summary(origin, destination)
    for key in ("distance_km", "distance", "path_distance_km"):
        if key in summary:
            return float(summary[key])
    fn = getattr(fare_model, "path_distance_km", None)
    if fn:
        try:
            return float(fn(origin, destination, direction="shortest"))
        except TypeError:
            return float(fn(origin, destination))
    raise RuntimeError("fare_model.py に path_summary() または path_distance_km() が必要です。")


def fare_between(origin: str, destination: str) -> float:
    summary = _call_path_summary(origin, destination)
    for key in ("ic_fare_yen", "fare_yen", "fare", "ic_fare"):
        if key in summary:
            return float(summary[key])
    fn = getattr(fare_model, "fare_between", None)
    if fn:
        attempts = (
            {"payment": "ic", "direction": "shortest"},
            {"payment": "ic"},
            {"direction": "shortest"},
            {},
        )
        for kwargs in attempts:
            try:
                return float(fn(origin, destination, **kwargs))
            except TypeError:
                continue
    raise RuntimeError("fare_model.py に path_summary() の運賃値または fare_between() が必要です。")


def shortest_path(stations: Sequence[str], origin: str, destination: str) -> list[str]:
    n = len(stations)
    i, j = stations.index(origin), stations.index(destination)
    forward_steps = (j - i) % n
    backward_steps = (i - j) % n
    forward = [stations[(i + k) % n] for k in range(forward_steps + 1)]
    backward = [stations[(i - k) % n] for k in range(backward_steps + 1)]
    df = sum(distance_between(forward[k], forward[k + 1]) for k in range(len(forward) - 1))
    db = sum(distance_between(backward[k], backward[k + 1]) for k in range(len(backward) - 1))
    return forward if (df, forward_steps) <= (db, backward_steps) else backward


def canonical_link(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def build_hypotheses(stations: Sequence[str]) -> list[Hypothesis]:
    result: list[Hypothesis] = []
    for origin in stations:
        for destination in stations:
            if origin == destination:
                continue
            path = shortest_path(stations, origin, destination)
            links = tuple(canonical_link(path[k], path[k + 1]) for k in range(len(path) - 1))
            result.append(Hypothesis(origin, destination, distance_between(origin, destination), links,
                                     fare_between(origin, destination)))
    return result


def od_probabilities(hypotheses: Sequence[Hypothesis], gamma: float) -> list[float]:
    raw = [math.exp(-gamma * h.distance_km) for h in hypotheses]
    total = sum(raw)
    return [x / total for x in raw]


def weighted_choice(rng: random.Random, cumulative: Sequence[float]) -> int:
    x = rng.random()
    lo, hi = 0, len(cumulative) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if x <= cumulative[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def is_subsequence(observed: Sequence[tuple[str, str]], path: Sequence[tuple[str, str]]) -> bool:
    it = iter(path)
    return all(any(x == y for y in it) for x in observed)


def infer_map(observed: tuple[tuple[str, str], ...], hypotheses: Sequence[Hypothesis], priors: Sequence[float],
              p: float, threshold: float) -> Hypothesis | None:
    # No false positives are generated. Candidates unable to explain the ordered
    # observations receive zero likelihood. Missing path links cost (1-p).
    scores: list[float] = []
    c = len(observed)
    for h, prior in zip(hypotheses, priors):
        if not is_subsequence(observed, h.links):
            scores.append(float("-inf"))
            continue
        scores.append(math.log(prior) + c * math.log(p) + (h.trip_links - c) * math.log1p(-p))
    best = max(scores)
    if not math.isfinite(best):
        return None
    denom = sum(math.exp(x - best) for x in scores if math.isfinite(x))
    confidence = 1.0 / denom
    return hypotheses[scores.index(best)] if confidence >= threshold else None


def run_seed(seed: int, hypotheses: Sequence[Hypothesis], priors: Sequence[float], p: float,
             sample_size: int, threshold: float) -> tuple[dict, dict[int, dict]]:
    rng = random.Random(seed)
    cumulative, acc = [], 0.0
    for x in priors:
        acc += x
        cumulative.append(acc)
    cache: dict[tuple[tuple[str, str], ...], Hypothesis | None] = {}
    total_true = total_collected = 0.0
    buckets: dict[int, dict[str, float]] = defaultdict(lambda: {"count": 0, "true": 0.0, "collected": 0.0})
    for _ in range(sample_size):
        truth = hypotheses[weighted_choice(rng, cumulative)]
        observed = tuple(link for link in truth.links if rng.random() < p)
        if observed not in cache:
            cache[observed] = infer_map(observed, hypotheses, priors, p, threshold)
        inferred = cache[observed]
        collected = 0.0 if inferred is None else inferred.fare_yen
        total_true += truth.fare_yen
        total_collected += collected
        b = buckets[truth.trip_links]
        b["count"] += 1
        b["true"] += truth.fare_yen
        b["collected"] += collected
    overall = {"true_fare": total_true, "collected_fare": total_collected,
               "recovery_rate": total_collected / total_true if total_true else math.nan}
    return overall, buckets


def mean_ci(values: Sequence[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if values else math.nan
    return mean, sd, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gammas", nargs="+", type=float, default=list(DEFAULT_GAMMAS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=3300)
    parser.add_argument("--base-seed", type=int, default=20260804)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=Path("simple_od_recovery_outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stations = load_station_order()
    hypotheses = build_hypotheses(stations)
    overall_rows: list[dict] = []
    bucket_rows: list[dict] = []
    for gamma in args.gammas:
        priors = od_probabilities(hypotheses, gamma)
        label = GAMMA_LABELS.get(gamma, f"gamma_{gamma:g}")
        for p in args.rates:
            per_seed_overall, per_bucket = [], defaultdict(list)
            per_bucket_count = defaultdict(list)
            for k in range(args.seed_count):
                overall, buckets = run_seed(args.base_seed + k, hypotheses, priors, p,
                                            args.sample_size, args.confidence_threshold)
                per_seed_overall.append(overall["recovery_rate"])
                for m, b in buckets.items():
                    per_bucket[m].append(b["collected"] / b["true"] if b["true"] else math.nan)
                    per_bucket_count[m].append(b["count"])
            mean, sd, low, high = mean_ci(per_seed_overall)
            overall_rows.append({"od_label": label, "gamma": gamma, "rfid_rate": p,
                                 "seed_count": args.seed_count, "sample_size_per_seed": args.sample_size,
                                 "mean_recovery_rate": mean, "sd_recovery_rate": sd,
                                 "ci95_low": low, "ci95_high": high})
            for m in sorted(per_bucket):
                values = [x for x in per_bucket[m] if math.isfinite(x)]
                mean, sd, low, high = mean_ci(values)
                bucket_rows.append({"od_label": label, "gamma": gamma, "rfid_rate": p,
                                    "trip_links_m_ij": m, "valid_seed_count": len(values),
                                    "mean_sample_count_per_seed": statistics.fmean(per_bucket_count[m]),
                                    "mean_recovery_rate": mean, "sd_recovery_rate": sd,
                                    "ci95_low": low, "ci95_high": high})

    def write_csv(name: str, rows: list[dict]) -> None:
        with (args.output_dir / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv("overall_recovery.csv", overall_rows)
    write_csv("recovery_by_trip_length.csv", bucket_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"output_dir": str(args.output_dir)}, f, ensure_ascii=False, indent=2)

    labels = [f"{r['od_label']}\np={r['rfid_rate']:.2f}" for r in overall_rows]
    means = [100 * r["mean_recovery_rate"] for r in overall_rows]
    errors = [100 * 1.96 * r["sd_recovery_rate"] / math.sqrt(r["seed_count"]) for r in overall_rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(means)), means, yerr=errors, capsize=4, color="#4472C4")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Total fare recovery rate (%)")
    ax.set_title("OD distance distribution and total fare recovery (mean ± 95% CI)")
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.output_dir / "overall_recovery_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(args.gammas), figsize=(6 * len(args.gammas), 5), sharey=True)
    if len(args.gammas) == 1: axes = [axes]
    for ax, gamma in zip(axes, args.gammas):
        label = GAMMA_LABELS.get(gamma, f"gamma_{gamma:g}")
        for p in args.rates:
            rows = [r for r in bucket_rows if r["gamma"] == gamma and r["rfid_rate"] == p]
            ax.plot([r["trip_links_m_ij"] for r in rows], [100*r["mean_recovery_rate"] for r in rows],
                    marker="o", label=f"p={p:.2f}")
        ax.set_title(f"{label} (gamma={gamma:g})"); ax.set_xlabel("Observation opportunities m_ij (links)")
        ax.grid(alpha=.25); ax.legend()
    axes[0].set_ylabel("Fare recovery rate (%)")
    fig.suptitle("Recovery by trip length: d_ij defines OD weight; m_ij defines observations")
    fig.tight_layout(); fig.savefig(args.output_dir / "recovery_by_trip_length_comparison.png", dpi=180); plt.close(fig)
    print(f"Done: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
