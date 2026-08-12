#!/usr/bin/env python3
"""Evaluate OD accuracy and fare-band misclassification for nagoya_bibo.py.

Run from the same directory as nagoya_bibo.py.  Results distinguish:
  1) correct OD,
  2) wrong OD but same fare,
  3) wrong fare (undercharge / overcharge).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from nagoya_bibo import (
    build_lines,
    census_prior,
    gravity_prior,
    hypotheses,
    load_profile,
    posterior,
    section_fitted_prior,
)


def stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mu = mean(values) if values else float("nan")
    sd = stdev(values) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n else float("nan")
    return {"mean": mu, "sd": sd, "ci95_low": mu-half, "ci95_high": mu+half}


def empty_counter() -> dict[str, int]:
    return {
        "count": 0,
        "od_correct": 0,
        "same_fare_wrong_od": 0,
        "wrong_fare": 0,
        "undercharge": 0,
        "overcharge": 0,
        "unresolved": 0,
    }


def add(counter: dict[str, int], true_idx: int, pred_idx: int,
        confidence: float, threshold: float, hs: list[dict]) -> None:
    counter["count"] += 1
    if confidence < threshold:
        counter["unresolved"] += 1
        # The current fare-recovery model treats unresolved as zero-yen collected.
        counter["wrong_fare"] += 1
        counter["undercharge"] += 1
        return
    true_h, pred_h = hs[true_idx], hs[pred_idx]
    if true_h["o"] == pred_h["o"] and true_h["d"] == pred_h["d"]:
        counter["od_correct"] += 1
    elif true_h["fare"] == pred_h["fare"]:
        counter["same_fare_wrong_od"] += 1
    else:
        counter["wrong_fare"] += 1
        if pred_h["fare"] < true_h["fare"]:
            counter["undercharge"] += 1
        else:
            counter["overcharge"] += 1


def rates(counter: dict[str, int]) -> dict[str, float]:
    n = counter["count"]
    if n == 0:
        return {k: float("nan") for k in (
            "od_accuracy", "same_fare_wrong_od_rate", "wrong_fare_rate",
            "undercharge_rate", "overcharge_rate", "unresolved_rate",
            "fare_correct_rate")}
    return {
        "od_accuracy": counter["od_correct"] / n,
        "same_fare_wrong_od_rate": counter["same_fare_wrong_od"] / n,
        "wrong_fare_rate": counter["wrong_fare"] / n,
        "undercharge_rate": counter["undercharge"] / n,
        "overcharge_rate": counter["overcharge"] / n,
        "unresolved_rate": counter["unresolved"] / n,
        "fare_correct_rate": (counter["od_correct"] + counter["same_fare_wrong_od"]) / n,
    }


def one_seed(hs: list[dict], prior: np.ndarray, p: float, sample_size: int,
             seed: int, threshold: float) -> tuple[dict, dict[int, dict]]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(hs), size=sample_size, p=prior)
    overall = empty_counter()
    buckets: dict[int, dict] = {}
    cache: dict[tuple[int, ...], tuple[int, float]] = {}
    for raw_idx in chosen:
        idx = int(raw_idx)
        h = hs[idx]
        observed = tuple(e for e in h["edges"] if rng.random() < p)
        if observed not in cache:
            cache[observed] = posterior(observed, hs, prior, p)
        pred_idx, confidence = cache[observed]
        add(overall, idx, pred_idx, confidence, threshold, hs)
        bucket = buckets.setdefault(h["m"], empty_counter())
        add(bucket, idx, pred_idx, confidence, threshold, hs)
    return overall, buckets


METRICS = (
    "od_accuracy", "same_fare_wrong_od_rate", "wrong_fare_rate",
    "undercharge_rate", "overcharge_rate", "unresolved_rate", "fare_correct_rate",
)


def aggregate(seed_rows: list[dict], id_fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in seed_rows:
        groups.setdefault(tuple(row[k] for k in id_fields), []).append(row)
    output = []
    for key, rows in groups.items():
        result = dict(zip(id_fields, key))
        result["valid_seed_count"] = len(rows)
        result["mean_sample_count"] = mean(r["sample_count"] for r in rows)
        for metric in METRICS:
            s = stats([r[metric] for r in rows])
            result[f"mean_{metric}"] = s["mean"]
            result[f"sd_{metric}"] = s["sd"]
            result[f"ci95_low_{metric}"] = s["ci95_low"]
            result[f"ci95_high_{metric}"] = s["ci95_high"]
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trip_summary(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        print("WARNING: matplotlib is not installed; CSV output is complete, plots were skipped.")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for line in sorted({r["line"] for r in rows}):
        line_rows = [r for r in rows if r["line"] == line]
        for model in sorted({r["od_model"] for r in line_rows}):
            fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
            for p in sorted({float(r["p"]) for r in line_rows}):
                z = sorted(
                    [r for r in line_rows if r["od_model"] == model and float(r["p"]) == p],
                    key=lambda r: int(r["trip_links"]),
                )
                x = [int(r["trip_links"]) for r in z]
                axes[0].plot(x, [100*r["mean_od_accuracy"] for r in z], marker="o", label=f"OD correct, p={p:g}")
                axes[0].plot(x, [100*r["mean_fare_correct_rate"] for r in z], marker="s", linestyle="--", label=f"Fare correct, p={p:g}")
                axes[1].plot(x, [100*r["mean_same_fare_wrong_od_rate"] for r in z], marker="o", label=f"Wrong OD / same fare, p={p:g}")
                axes[1].plot(x, [100*r["mean_wrong_fare_rate"] for r in z], marker="s", linestyle="--", label=f"Wrong fare, p={p:g}")
            axes[0].set_ylabel("Accuracy (%)")
            axes[1].set_ylabel("Error composition (%)")
            axes[1].set_xlabel("Trip length (links)")
            axes[0].set_title(f"{line} / {model}: OD accuracy versus fare accuracy")
            for ax in axes:
                ax.set_ylim(0, 103); ax.grid(alpha=.25); ax.legend(ncol=2)
            fig.tight_layout()
            fig.savefig(output_dir / f"{line}_{model}_od_and_fare_accuracy.png", dpi=170)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="+", default=["higashiyama", "tsurumai", "sakuradori", "meijo_meiko"])
    ap.add_argument("--od-models", nargs="+", choices=["census_od", "gravity", "section_fit"], default=["census_od", "gravity", "section_fit"])
    ap.add_argument("--detection-rates", nargs="+", type=float, default=[0.65, 0.70])
    ap.add_argument("--gamma", type=float, default=0.18)
    ap.add_argument("--seed-count", type=int, default=150)
    ap.add_argument("--sample-size", type=int, default=3300)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--output-dir", default="od_accuracy_outputs")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = build_lines()
    overall_seed_rows: list[dict] = []
    trip_seed_rows: list[dict] = []

    for line_key in args.lines:
        line = lines[line_key]
        hs = hypotheses(line)
        base = gravity_prior(hs, args.gamma)
        for model in args.od_models:
            if model == "section_fit":
                prior, _ = section_fitted_prior(
                    hs, base, load_profile(line.key, len(line.edges)), len(line.edges)
                )
            elif model == "census_od":
                prior = census_prior(line, hs)
            else:
                prior = base
            for p in args.detection_rates:
                for seed in range(args.seed_count):
                    overall, buckets = one_seed(
                        hs, prior, p, args.sample_size, seed, args.confidence_threshold
                    )
                    row = {"line": line_key, "od_model": model, "p": p, "seed": seed,
                           "sample_count": overall["count"], **rates(overall)}
                    overall_seed_rows.append(row)
                    for m, counter in buckets.items():
                        trip_seed_rows.append({
                            "line": line_key, "od_model": model, "p": p,
                            "trip_links": m, "seed": seed,
                            "sample_count": counter["count"], **rates(counter),
                        })

    overall_summary = aggregate(overall_seed_rows, ["line", "od_model", "p"])
    trip_summary = aggregate(trip_seed_rows, ["line", "od_model", "p", "trip_links"])
    write_csv(output_dir / "overall_seed_results.csv", overall_seed_rows)
    write_csv(output_dir / "overall_accuracy_summary.csv", overall_summary)
    write_csv(output_dir / "trip_length_seed_results.csv", trip_seed_rows)
    write_csv(output_dir / "trip_length_accuracy_summary.csv", trip_summary)
    plot_trip_summary(trip_summary, output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir.resolve()}")


if __name__ == "__main__":
    main()
