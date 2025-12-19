#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# Root directory that contains the experiment folders with trace files.
ROOT = Path("/Users/adhityapolavaram/Desktop/experiments")
CSV_DIR = ROOT

# Communication ops we want to surface.
COMM_KEYS = {
    "all-reduce",
    "all_gather",
    "all-gather",
    "all-to-all",
    "reduce-scatter",
    "collective-permute",
    "collective-permute-start",
    "collective-permute-done",
    "send",
    "recv",
}


def is_comm_event(event: dict) -> bool:
    """Return True if the event looks like a communication kernel."""
    name = str(event.get("name", "")).lower()
    category = str(event.get("args", {}).get("hlo_category", "")).lower()
    return any(key in name for key in COMM_KEYS) or category in COMM_KEYS


def comm_type(event: dict) -> str:
    """Return the communication type label for aggregation."""
    name = str(event.get("name", "")).lower()
    category = str(event.get("args", {}).get("hlo_category", "")).lower()
    if category in COMM_KEYS:
        return category
    for key in COMM_KEYS:
        if key in name:
            return key
    return "unknown"


def duration_us(event: dict) -> float:
    """Get duration in microseconds (uses device_duration_ps when available)."""
    args = event.get("args", {})
    if "device_duration_ps" in args:
        return float(args["device_duration_ps"]) / 1e6
    if "dur" in event:
        return float(event["dur"])
    raise KeyError("No duration field found")


def iter_trace_events(path: Path):
    """Yield trace events from a .trace.json.gz file."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    # Some traces wrap events under traceEvents, others may be the array itself.
    events = data.get("traceEvents", data)
    for evt in events:
        yield evt


def parse_batch_tp(path: Path) -> tuple[str | None, str | None]:
    """Extract batch size and TP from the model directory name."""
    batch = None
    tp = None
    for part in path.parts:
        if part.startswith("MODEL_"):
            for token in part.split(","):
                if token.startswith("BATCH_"):
                    batch = token.split("_", 1)[1]
                if token.startswith("TP_"):
                    tp = token.split("_", 1)[1]
    return batch, tp


def parse_model(path: Path) -> str | None:
    """Extract the model identifier from the path part that starts with MODEL_."""
    for part in path.parts:
        if part.startswith("MODEL_"):
            token = part.split(",", 1)[0]  # keep only model piece before first comma
            return token.removeprefix("MODEL_")
    return None


def sanitize_filename(text: str) -> str:
    """Make a filesystem-friendly token (lowercase, hyphen separated)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def main() -> None:
    trace_paths = sorted(ROOT.rglob("*.trace.json.gz"))
    if not trace_paths:
        raise SystemExit(f"No trace files found under {ROOT}")

    # comm_type -> {(batch, tp): (total_us, count)}
    totals_by_comm = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    # (model, comm_type) -> {(batch, tp): (total_us, count)}
    totals_by_model_comm = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    # processed_data[model][tp][batch][comm_type] = {"total_us": float, "count": int}
    processed_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    batches = set()
    tps = set()

    for trace_path in trace_paths:
        print(f"\n=== {trace_path} ===")
        events = list(iter_trace_events(trace_path))
        comm_events = [e for e in events if e.get("ph") == "X" and is_comm_event(e)]

        if not comm_events:
            print("No communication kernels found.")
            continue

        totals_us = defaultdict(float)
        counts = defaultdict(int)
        for evt in comm_events:
            totals_us[comm_type(evt)] += duration_us(evt)
            counts[comm_type(evt)] += 1

        batch, tp = parse_batch_tp(trace_path)
        model = parse_model(trace_path)
        batches.add(batch)
        tps.add(tp)
        for ctype, total in sorted(totals_us.items(), key=lambda kv: kv[1], reverse=True):
            cnt = counts[ctype]
            print(f"{ctype}  |  total={total:.3f} us  |  count={cnt}  |  batch={batch}  |  tp={tp}")
            cell = totals_by_comm[ctype][(batch, tp)]
            cell[0] += total
            cell[1] += cnt
            model_key = model or "unknown_model"
            mcell = totals_by_model_comm[(model_key, ctype)][(batch, tp)]
            mcell[0] += total
            mcell[1] += cnt
            processed_data[model_key][tp][batch][ctype] = {
                "total_us": total,
                "count": cnt,
            }

        grand_total = sum(totals_us.values())
        print(f"TOTAL communication time: {grand_total:.3f} us across {len(comm_events)} events")

    if not totals_by_comm:
        return

    def sort_key(val: str | None) -> tuple[int, str | None]:
        if val is None:
            return (1, "")
        try:
            return (0, f"{int(val):010d}")
        except ValueError:
            return (0, val)

    sorted_batches = sorted(batches, key=sort_key)
    sorted_tps = sorted(tps, key=sort_key)

    # Per-model tables for each communication type.
    for (model, ctype), grid in totals_by_model_comm.items():
        model_token = sanitize_filename(model or "unknown_model")
        comm_token = ctype.replace("_", "-")
        out_path = CSV_DIR / f"{model_token}-communication-results-{comm_token}.csv"

        # Determine batches/tps present for this model to avoid empty columns/rows.
        grid_batches = {b for (b, _t) in grid.keys()}
        grid_tps = {t for (_b, t) in grid.keys()}
        sorted_batches_local = sorted(grid_batches, key=sort_key)
        sorted_tps_local = sorted(grid_tps, key=sort_key)

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch \\ tp"] + sorted_tps_local)
            for batch in sorted_batches_local:
                row = [batch]
                for tp in sorted_tps_local:
                    total_us, count = grid.get((batch, tp), (0.0, 0))
                    row.append(f"{total_us:.3f}/{count}")
                writer.writerow(row)
        print(f"Saved {ctype} table for model {model} to {out_path}")

    # Flatten processed_data into a long-form DataFrame.
    if processed_data:
        rows = []
        for model, tp_map in processed_data.items():
            for tp, batch_map in tp_map.items():
                for batch, comm_map in batch_map.items():
                    for ctype, metrics in comm_map.items():
                        rows.append(
                            {
                                "model": model,
                                "tp": tp,
                                "batch": batch,
                                "comm_type": ctype,
                                "total_us": metrics["total_us"],
                                "count": metrics["count"],
                            }
                        )
        df = pd.DataFrame(rows)
        out_parquet = CSV_DIR / "communication_results.parquet"
        out_csv = CSV_DIR / "communication_results_flat.csv"
        df.to_parquet(out_parquet, index=False)
        df.to_csv(out_csv, index=False)
        print(f"Saved processed_data DataFrame to {out_parquet} and {out_csv}")


if __name__ == "__main__":
    main()

