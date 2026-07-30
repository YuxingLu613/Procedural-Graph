"""Generates professional dual-panel cash trajectory and survival charts for CFO.

This script produces exactly one consolidated figure per LLM containing two
stacked subplots (sharing the X-axis) with compact, publication-ready styling:
1. No internal title and no footnote (designed to be captioned in LaTeX/Markdown).
2. Physical figure size is reduced to 8.5 x 6.5 inches with larger fonts (11pt base)
   to ensure the chart is highly compact and readable when embedded.
3. Upper Panel (70% height): Plots the 50 individual trajectories as thin spaghetti
   lines (linewidth=0.4, alpha=0.10) and the bold mean trajectory on top
   (linewidth=3.0, alpha=1.0, zorder=4).
   - Y-axis limits: [-5, 60]
   - Y-ticks: [0, 20, 40, 60] (negative ticks are omitted).
4. Lower Panel (30% height): Plots the Kaplan-Meier step-survival curves matching
   the corresponding mean lines exactly (linewidth=3.0, alpha=1.0, zorder=4).
   - Y-axis represents Survival Rate (0% - 100%).
   - Y-axis limits: [-5, 105]
   - Y-ticks: [0, 20, 40, 60, 80, 100] with '%' suffix.
5. Spine and Grid Cleanup: Removes top/right spines, sets pure white backgrounds,
   and uses light grey dotted gridlines.
6. Axis Alignment: X-ticks (numbers) are placed only in the middle.
7. Crisis lines run vertically across both panels (color='silver', linewidth=1.0).
8. Label De-collision: Deterministically stacks direct text labels at the right spine
   (cash values on top, survival percentages at the bottom) so they never overlap.
"""

from collections import Counter
import json
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Set global RC parameters for compact, high-contrast academic style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11  # Increased from 9.5 for larger, clearer print
plt.rcParams["grid.color"] = "lightgrey"
plt.rcParams["grid.linestyle"] = ":"

# Base directories
_BASE_RESULTS_DIR = os.environ.get(
    "PG_RESULTS_DIR",
    "results/cfo_generative_pg_memory_comparison"
)
_CFO_DIR = os.path.join(_BASE_RESULTS_DIR, "cfo")
_OUTPUT_DIR = _BASE_RESULTS_DIR

# Define configurations (Unified names, high-contrast colors, distinct styles)
_CONFIGS = {
    "gemini-3.5-flash": {
        "dir_name": "gemini-3.5-flash",
        "title_label": "Gemini 3.5 Flash",
        "methods": [
            {
                "name": "Baseline",
                "rel_path": "react/comp_cfo/baseline",
                "color": "#C62828",  # Dark Crimson
                "style": "--",
            },
            {
                "name": "Procedural Graph",
                "rel_path": "react/comp_cfo/generative_pg_val/unknown_samples",
                "color": "#1565C0",  # Royal Blue
                "style": "-",
            },
            {
                "name": "Memory Retrieval",
                "rel_path": "react/comp_cfo/memory_retrieval/unknown_samples",
                "color": "#2E7D32",  # Forest Green
                "style": ":",
            },
            {
                "name": "Memory Summarization",
                "rel_path": (
                    "react/comp_cfo/memory_summarization/unknown_samples"
                ),
                "color": "#EF6C00",  # Deep Orange
                "style": "-.",
            },
        ],
    },
    "grok-4.1-fast-non-reasoning": {
        "dir_name": "grok-4.1-fast-non-reasoning",
        "title_label": "Grok 4.1 Fast Non-Reasoning",
        "methods": [
            {
                "name": "Baseline",
                "rel_path": "react/comp_cfo_grok/baseline",
                "color": "#C62828",  # Dark Crimson
                "style": "--",
            },
            {
                "name": "Procedural Graph",
                "rel_path": "react/comp_cfo_grok/scratch_onetime/50_samples",
                "color": "#1565C0",  # Royal Blue
                "style": "-",
            },
            {
                "name": "Memory Retrieval",
                "rel_path": (
                    "react/comp_cfo_grok/memory_retrieval/unknown_samples"
                ),
                "color": "#2E7D32",  # Forest Green
                "style": ":",
            },
            {
                "name": "Memory Summarization",
                "rel_path": (
                    "react/comp_cfo_grok/memory_summarization/unknown_samples"
                ),
                "color": "#EF6C00",  # Deep Orange
                "style": "-.",
            },
        ],
    },
}


def load_cash_history(llm_dir, rel_path, sample):
    """Loads cash balance history from financial_history.json file."""
    filepath = os.path.join(
        _CFO_DIR, llm_dir, rel_path, "samples", sample, "financial_history.json"
    )
    if not os.path.exists(filepath):
        return [], []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        months = [entry.get("current_month", 0) for entry in data]
        cash = [entry.get("cash_balance", 0.0) / 1e6 for entry in data]  # in $M
        return months, cash
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading {filepath}: {e}")
        return [], []


def smooth_trajectory(y, window=5):
    """Applies a boxcar moving average to smooth out high-frequency noise."""
    if len(y) < window:
        return y
    box = np.ones(window) / window
    pad_size = window // 2
    y_padded = np.pad(y, pad_size, mode="edge")
    y_smooth = np.convolve(y_padded, box, mode="valid")
    return y_smooth


def draw_decollided_labels(ax, x, label_data, y_spacing):
    """Draws text labels vertically de-collided to prevent overlapping.

    Args:
      ax: Matplotlib axis to draw on.
      x: The X coordinate to place the labels.
      label_data: A list of tuples: (actual_value, label_text, color)
      y_spacing: Minimum vertical distance between labels.
    """
    # Sort descending by actual value
    sorted_data = sorted(label_data, key=lambda item: item[0], reverse=True)
    
    if not sorted_data:
        return

    # Adjust Y coordinates
    adjusted_y = [item[0] for item in sorted_data]
    for i in range(1, len(adjusted_y)):
        # If the gap between current and the one above it is less than spacing
        if adjusted_y[i-1] - adjusted_y[i] < y_spacing:
            adjusted_y[i] = adjusted_y[i-1] - y_spacing

    # Draw de-collided labels
    for i, (val, text, color) in enumerate(sorted_data):
        ax.text(
            x,
            adjusted_y[i],
            text,
            color=color,
            fontsize=10,  # Increased from 8.5 for compact look
            fontweight="bold",
            va="center",
            ha="left",
            zorder=5
        )


def main():
    samples = [f"sample_{i:03d}" for i in range(1, 51)]
    max_months = 131  # Last month index
    total_steps = max_months + 1
    total_samples_expected = 50

    for _, llm_config in _CONFIGS.items():
        llm_name = llm_config["dir_name"]
        print(f"Generating professional dual-panel chart for {llm_name}...")

        # Physical size reduced to (8.5 x 6.5) to make it compact relative to the font sizes
        fig, (ax_cash, ax_surv) = plt.subplots(
            2, 1, figsize=(8.5, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [7, 2.8]}
        )
        has_data = False

        # Accumulate label data for de-collision
        cash_labels_data = []
        surv_labels_data = []

        for method in llm_config["methods"]:
            valid_histories = []
            sample_run_metadata = []

            for sample in samples:
                months, cash = load_cash_history(llm_name, method["rel_path"], sample)
                if not months:
                    continue
                
                has_data = True
                
                # Genuine bankruptcy: ended early AND final cash went below 0.0
                is_bankrupt = (len(months) < total_steps) and (cash[-1] < 0.0)
                
                # Pad with 0.0 if ended early
                if len(months) < total_steps:
                    padded_months = list(months) + list(range(months[-1] + 1, total_steps))
                    padded_cash = list(cash) + [0.0] * (total_steps - len(months))
                else:
                    padded_months = months
                    padded_cash = cash

                # Smooth the trajectory to shave off sharp monthly spikes
                padded_cash_arr = np.array(padded_cash)
                smoothed_cash = smooth_trajectory(padded_cash_arr, window=5)

                valid_histories.append(smoothed_cash)

                # Record metadata for Kaplan-Meier survival curve
                sample_run_metadata.append({
                    "is_bankrupt": is_bankrupt,
                    "bankruptcy_month": months[-1] if is_bankrupt else total_steps
                })

                # 1. Plot individual spaghetti line (Upper Panel: alpha=0.10, linewidth=0.4)
                ax_cash.plot(
                    padded_months,
                    smoothed_cash,
                    color=method["color"],
                    linestyle="-",
                    linewidth=0.4,
                    alpha=0.10,
                    zorder=2
                )

            if not valid_histories:
                continue

            # Calculate mean cash
            valid_histories = np.array(valid_histories)
            # Clip negative cash balances to 0.0 to prevent bankruptcy liabilities
            # from dragging down the portfolio average (as requested by user)
            valid_histories = np.maximum(0.0, valid_histories)
            mean_cash = np.mean(valid_histories, axis=0)
            
            # 2. Plot bold Mean Line on the Upper Panel (zorder=4, linewidth=3.0)
            ax_cash.plot(
                list(range(total_steps)),
                mean_cash,
                color=method["color"],
                linestyle=method["style"],
                linewidth=3.0,
                alpha=1.0,
                label=method["name"],
                zorder=4
            )

            # Store cash label info for de-collision drawing
            final_cash = mean_cash[-1]
            cash_labels_data.append((
                final_cash,
                f"${final_cash:.1f}M",
                method["color"]
            ))

            # 3. Calculate Kaplan-Meier Survival Rate (Percentage 0% - 100%)
            actual_n = len(sample_run_metadata)
            survival_rates = np.zeros(total_steps)
            
            for t in range(total_steps):
                alive_count = 0
                for metadata in sample_run_metadata:
                    if not metadata["is_bankrupt"]:
                        alive_count += 1
                    elif t < metadata["bankruptcy_month"]:
                        alive_count += 1
                # Scale to 0 - 100%
                survival_rates[t] = (alive_count / actual_n) * 100.0

            # 4. Plot Step Survival Curve on the Lower Panel (linewidth=3.0, matching styles exactly)
            ax_surv.step(
                list(range(total_steps)),
                survival_rates,
                color=method["color"],
                linestyle=method["style"],
                linewidth=3.0,
                alpha=1.0,
                where="post",
                zorder=4
            )

            # Store survival percentage label info for de-collision drawing
            final_survival_pct = survival_rates[-1]
            surv_labels_data.append((
                final_survival_pct,
                f"{final_survival_pct:.0f}%",
                method["color"]
            ))

        if not has_data:
            plt.close()
            print(f"No data found for {llm_name}, skipping.")
            continue

        # --- DRAW DE-COLLIDED LABELS (Layer 5) ---
        # Spacing increased to 3.5 units for cash (due to larger fonts)
        draw_decollided_labels(ax_cash, max_months + 1.5, cash_labels_data, y_spacing=3.5)
        # Spacing increased to 5.0 units for survival percentage
        draw_decollided_labels(ax_surv, max_months + 1.5, surv_labels_data, y_spacing=5.0)

        # --- SPINE & GRID CLEANUP ---
        for ax in [ax_cash, ax_surv]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#888888")
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_color("#888888")
            ax.spines["bottom"].set_linewidth(0.8)
            
            # Gridlines: Horizontal only, dotted, lightgrey
            ax.grid(axis="y", linestyle=":", color="lightgrey", alpha=0.8, zorder=1)

        # --- AXIS ALIGNMENT (Tuning Tick Labels) ---
        ax_cash.tick_params(axis="x", labelbottom=True, colors="#444444", labelsize=10.5)
        ax_surv.tick_params(axis="x", labelbottom=False, colors="#444444")
        
        # Axis Labels (Slightly larger fonts: 11.5pt)
        ax_cash.set_ylabel("Cash Balance ($ Millions)", fontsize=11.5, fontweight="bold", color="#212121")
        ax_surv.set_ylabel("Survival Rate", fontsize=11.5, fontweight="bold", color="#212121")
        ax_surv.set_xlabel("Simulation Month", fontsize=11.5, fontweight="bold", color="#212121", labelpad=8)

        # Set strict ticks
        ax_cash.set_yticks([0, 20, 40, 60])
        ax_surv.set_yticks([0, 20, 40, 60, 80, 100])
        ax_surv.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

        # Zero baselines
        ax_cash.axhline(0, color="#444444", linewidth=1.0, linestyle="-", zorder=1)
        ax_surv.axhline(0, color="#444444", linewidth=1.0, linestyle="-", zorder=1)

        # --- CRISIS MARKERS (silver dotted lines, running continuously across both panels) ---
        crisis_months = [32, 59, 112]
        for idx, cm in enumerate(crisis_months, start=1):
            ax_cash.axvline(
                cm, color="silver", linewidth=1.0, linestyle=":", alpha=0.9, zorder=1
            )
            ax_surv.axvline(
                cm, color="silver", linewidth=1.0, linestyle=":", alpha=0.9, zorder=1
            )
            # Text label at the top of the upper panel (font size increased to 9.5pt)
            ax_cash.text(
                cm - 1.0,
                55.0,
                f"Crisis {idx}",
                color="#616161",
                fontsize=9.5,
                fontweight="bold",
                rotation=90,
                va="top",
                ha="right",
                zorder=5
            )

        # Limits: X expanded for labels
        ax_cash.set_xlim(0, 140)
        ax_cash.set_ylim(-5, 60)
        ax_surv.set_ylim(-5, 105)

        # --- SLIM LEGEND (Increased font size to 10pt) ---
        legend_elements = []
        for method in llm_config["methods"]:
            legend_elements.append(
                mlines.Line2D(
                    [],
                    [],
                    color=method["color"],
                    linestyle=method["style"],
                    linewidth=2.5,
                    label=method["name"],
                )
            )
        
        ax_cash.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            edgecolor="#D8D8D8",
            shadow=False,
            fontsize=10,
        )

        # --- NO TITLE AND NO FOOTNOTE INSIDE FIGURE (designed for LaTeX captioning) ---
        # ax_cash.set_title(...) -> Completely removed!
        # fig.text(...) -> Completely removed!

        # Adjust layout to be extremely tight and compact since there is no footnote or title
        plt.tight_layout()

        out_path = os.path.join(
            _OUTPUT_DIR, f"{llm_name}_ensemble_cash_comparison.png"
        )
        # Maintained at 300 DPI for ultra-high print resolution
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated professional compact dual-panel cash & survival chart: {out_path}")


if __name__ == "__main__":
    main()
