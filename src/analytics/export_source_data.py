"""Compiles and exports CFO environment simulation trajectories to CSV.

This script aggregates raw cash histories from all 50 samples across all methods
for Gemini 3.5 Flash and Grok 4.1. It exports two CSV files:
1. cfo_raw_trajectories.csv: Raw trajectories for all samples, methods, months.
2. cfo_average_trajectories.csv: Aggregated mean and SEM per month, method, LLM.
"""

import json
import os
from typing import cast
import numpy as np
import pandas as pd

# Base directories
_BASE_RESULTS_DIR = os.environ.get(
    "PG_RESULTS_DIR",
    "results/cfo_generative_pg_memory_comparison"
)
_CFO_DIR = os.path.join(_BASE_RESULTS_DIR, "cfo")

# Define configurations
_CONFIGS = {
    "gemini-3.5-flash": {
        "dir_name": "gemini-3.5-flash",
        "methods": [
            {
                "name": "Pure ReAct Baseline",
                "rel_path": "react/comp_cfo/baseline",
            },
            {
                "name": "Generative PG (Val)",
                "rel_path": "react/comp_cfo/generative_pg_val/unknown_samples",
            },
            {
                "name": "Memory Retrieval",
                "rel_path": "react/comp_cfo/memory_retrieval/unknown_samples",
            },
            {
                "name": "Memory Summarization",
                "rel_path": (
                    "react/comp_cfo/memory_summarization/unknown_samples"
                ),
            },
            {
                "name": "Train Samples",
                "rel_path": "react/comp_cfo/train_samples",
            },
        ],
    },
    "grok-4.1-fast-non-reasoning": {
        "dir_name": "grok-4.1-fast-non-reasoning",
        "methods": [
            {
                "name": "Pure ReAct Baseline",
                "rel_path": "react/comp_cfo_grok/baseline",
            },
            {
                "name": "Scratch Onetime",
                "rel_path": (
                    "react/comp_cfo_grok/scratch_onetime/50_samples"
                ),
            },
            {
                "name": "Memory Retrieval",
                "rel_path": (
                    "react/comp_cfo_grok/memory_retrieval/unknown_samples"
                ),
            },
            {
                "name": "Memory Summarization",
                "rel_path": (
                    "react/comp_cfo_grok/memory_summarization/unknown_samples"
                ),
            },
            {
                "name": "Train Samples",
                "rel_path": "react/comp_cfo_grok/train_samples",
            },
        ],
    },
}


def compile_data():
    """Compiles and exports raw and aggregated cash trajectories to CSV."""
    max_months = 132
    raw_rows = []

    print("Compiling raw trajectory data...")

    for llm_name, llm_config in _CONFIGS.items():
        for method in llm_config["methods"]:
            method_name = method["name"]
            rel_path = method["rel_path"]

            for i in range(1, 51):
                sample_name = f"sample_{i:03d}"
                filepath = os.path.join(
                    _CFO_DIR,
                    llm_name,
                    rel_path,
                    "samples",
                    sample_name,
                    "financial_history.json",
                )

                if not os.path.exists(filepath):
                    continue

                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    existing_months = {}
                    last_recorded_month = -1

                    for entry in data:
                        m = entry.get("current_month", 0)
                        if m <= max_months:
                            existing_months[m] = (
                                entry.get("cash_balance", 0.0) / 1e6
                            )
                            if m > last_recorded_month:
                                last_recorded_month = m

                    for m in range(max_months + 1):
                        if m in existing_months:
                            cash = existing_months[m]
                        elif m > last_recorded_month:
                            cash = 0.0
                        else:
                            cash = 0.0

                        raw_rows.append({
                            "llm": llm_name,
                            "method": method_name,
                            "sample_id": sample_name,
                            "month": m,
                            "cash_balance_m": cash,
                        })

                except (OSError, json.JSONDecodeError) as e:
                    print(f"Error reading {filepath}: {e}")

    df_raw = pd.DataFrame(raw_rows)

    raw_csv_path = os.path.join(_BASE_RESULTS_DIR, "cfo_raw_trajectories.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Saved raw trajectories ({len(df_raw)} rows) to: {raw_csv_path}")

    print("Generating aggregated summary data...")
    summary_rows = []

    grouped = df_raw.groupby(["llm", "method", "month"])

    for key, group in grouped:
        # Cast key to tuple to satisfy strict Pytype checking
        key_tuple = cast(tuple, key)
        llm = key_tuple[0]
        method = key_tuple[1]
        month = key_tuple[2]

        cash_vals = group["cash_balance_m"].values
        # Clip negative cash balances to 0.0 to prevent bankrupt liabilities
        # from dragging down the portfolio average (as requested by user)
        clipped_cash_vals = np.maximum(0.0, cash_vals)
        mean_val = np.mean(clipped_cash_vals)
        sem_val = (
            np.std(clipped_cash_vals) / np.sqrt(len(clipped_cash_vals))
            if len(clipped_cash_vals) > 0
            else 0.0
        )

        summary_rows.append({
            "llm": llm,
            "method": method,
            "month": month,
            "mean_cash_balance_m": mean_val,
            "sem_cash_balance_m": sem_val,
            "sample_count": len(cash_vals),
        })

    df_summary = pd.DataFrame(summary_rows)

    summary_csv_path = os.path.join(
        _BASE_RESULTS_DIR, "cfo_average_trajectories.csv"
    )
    df_summary.to_csv(summary_csv_path, index=False)
    print(
        f"Saved aggregated trajectories ({len(df_summary)} rows) to:"
        f" {summary_csv_path}"
    )


if __name__ == "__main__":
    compile_data()
