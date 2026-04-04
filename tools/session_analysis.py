#!/usr/bin/env python3
"""
Post-Session Analysis Tool
Joins OpenCode session log with CodeCarbon monitor CSV to derive per-inference emissions.

Usage:
    python session_analysis.py <opencode_session_log.json> <codecarbon_emissions_detail.csv>

Output:
    - Formatted table to stdout
    - JSON summary to build_session_summary.json
"""

import json
import csv
import sys
from datetime import datetime, timedelta
from collections import defaultdict


def parse_csv_timestamp(ts_str):
    """Parse CodeCarbon CSV timestamp to datetime."""
    try:
        return datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(ts_str.strip()[:19], "%Y-%m-%d %H:%M:%S")


def load_codecarbon_csv(filepath):
    """Load CodeCarbon emissions CSV into a list of dicts."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_ts"] = parse_csv_timestamp(row["timestamp"])
            rows.append(row)
    return rows


def load_opencode_log(filepath):
    """
    Load OpenCode session log and extract inference events.
    Returns list of dicts with timestamp, duration, tokens_out, etc.
    """
    with open(filepath, "r") as f:
        log = json.load(f)

    print(f"OpenCode log top-level keys: {list(log.keys())}")

    messages = log.get("messages", [])
    if len(messages) > 0:
        print(f"First message keys: {list(messages[0].keys())}")

    events = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        if "tool_calls" in msg and msg["tool_calls"]:
            events.append(
                {
                    "timestamp": msg.get("timestamp", ""),
                    "tool_calls": msg["tool_calls"],
                    "tokens_out": msg.get("tokens_out", 0),
                    "duration_s": msg.get("duration_s", 0),
                }
            )
        elif "generation_id" in msg:
            events.append(
                {
                    "timestamp": msg.get("timestamp", ""),
                    "tokens_out": msg.get("tokens_out", 0),
                    "duration_s": msg.get("duration_s", 0),
                }
            )

    return events


def join_emissions(events, cc_rows):
    """
    For each event, find overlapping CodeCarbon rows and sum emissions.
    """
    result = []

    for event in events:
        ts_str = event.get("timestamp", "")
        duration_s = event.get("duration_s", 0)

        if not ts_str:
            continue

        event_ts = parse_csv_timestamp(ts_str)
        end_ts = event_ts + timedelta(seconds=duration_s)

        overlapping = []
        for row in cc_rows:
            rts = row["_ts"]
            if event_ts <= rts <= end_ts:
                overlapping.append(row)

        if not overlapping:
            continue

        total_emissions = sum(float(r.get("emissions", 0)) for r in overlapping)
        total_energy = sum(float(r.get("energy_consumed", 0)) for r in overlapping)
        avg_gpu = sum(float(r.get("gpu_power", 0)) for r in overlapping) / len(
            overlapping
        )

        result.append(
            {
                "timestamp": ts_str,
                "duration_s": duration_s or len(overlapping),
                "tokens_out": event.get("tokens_out", 0),
                "emissions_g": total_emissions * 1000,
                "energy_kwh": total_energy,
                "avg_gpu_w": avg_gpu,
            }
        )

    return result


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    opencode_log = sys.argv[1]
    cc_csv = sys.argv[2]

    print("Loading OpenCode session log...")
    events = load_opencode_log(opencode_log)
    print(f"Found {len(events)} inference events")

    print("Loading CodeCarbon CSV...")
    cc_rows = load_codecarbon_csv(cc_csv)
    print(f"Found {len(cc_rows)} CodeCarbon rows")

    joined = join_emissions(events, cc_rows)
    if not joined:
        print("No overlapping events found. Falling back to session totals.")
        total_emissions = sum(float(r.get("emissions", 0)) for r in cc_rows)
        total_energy = sum(float(r.get("energy_consumed", 0)) for r in cc_rows)
        avg_gpu = (
            sum(float(r.get("gpu_power", 0)) for r in cc_rows) / len(cc_rows)
            if cc_rows
            else 0
        )
        joined = [
            {
                "timestamp": "session-total",
                "duration_s": len(cc_rows),
                "tokens_out": sum(e.get("tokens_out", 0) for e in events),
                "emissions_g": total_emissions * 1000,
                "energy_kwh": total_energy,
                "avg_gpu_w": avg_gpu,
            }
        ]

    total_emissions = sum(r["emissions_g"] for r in joined)
    total_energy = sum(r["energy_kwh"] for r in joined)
    total_tokens = sum(r["tokens_out"] for r in joined)
    avg_gpu = sum(r["avg_gpu_w"] for r in joined) / len(joined) if joined else 0

    print("\nPer-Inference Breakdown")
    print("-" * 70)
    print(
        f"{'Call':<6} {'Duration':<12} {'Tokens Out':<12} {'CO2eq (g)':<12} {'Avg GPU (W)':<10}"
    )
    print("-" * 70)

    for i, row in enumerate(joined, 1):
        print(
            f"{i:<6} {row['duration_s']:<12.1f}s {row['tokens_out']:<12} "
            f"{row['emissions_g']:<12.4f} {row['avg_gpu_w']:<10.1f}"
        )

    print("\nSummary")
    print("-" * 70)
    print(f"Total calls:              {len(joined)}")
    print(f"Total emissions:          {total_emissions:.2f} g CO2eq")
    print(f"Total energy:             {total_energy:.4f} kWh")
    print(
        f"Avg per call:             {total_emissions / len(joined):.2f} g CO2eq"
        if joined
        else ""
    )
    print(
        f"Avg per 1K output tokens: {total_emissions / (total_tokens / 1000):.2f} g CO2eq"
        if total_tokens
        else ""
    )
    print(f"Avg GPU draw:             {avg_gpu:.1f} W")

    summary = {
        "total_calls": len(joined),
        "total_emissions_g": total_emissions,
        "total_energy_kwh": total_energy,
        "total_tokens_out": total_tokens,
        "avg_emissions_per_call_g": total_emissions / len(joined) if joined else 0,
        "avg_emissions_per_1k_tokens_g": total_emissions / (total_tokens / 1000)
        if total_tokens
        else 0,
        "avg_gpu_draw_w": avg_gpu,
        "per_call": joined,
    }

    output_path = "build_session_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
