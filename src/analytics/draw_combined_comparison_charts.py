import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np

# Set professional plotting style (will be overridden by xkcd where applicable)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#475569'  # Slate 600
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['text.color'] = '#1E293B'  # Slate 800

# Base directories
_BASE_RESULTS_DIR = os.environ.get(
    "PG_RESULTS_DIR",
    "results/cfo_generative_pg_memory_comparison"
)
_CFO_DIR = os.path.join(_BASE_RESULTS_DIR, "cfo")

samples = [f"sample_{i:03d}" for i in range(1, 51)]
max_months = 131
total_steps = 132  # Month 0 to Month 131 is 132 points

# Define custom dash patterns to survive xkcd wiggling
dashed_style = (0, (10, 8))
dotted_style = (0, (2, 8))
dashdot_style = (0, (10, 5, 2, 5))

# Modern Color Palette (similar to the reference style)
COLOR_BASELINE = '#E63946'      # Vibrant Red
COLOR_PROCEDURAL_GRAPH = '#2563EB'  # Royal Blue
COLOR_RETRIEVAL = '#10B981'     # Emerald Green
COLOR_SUMMARIZATION = '#F97316'  # Orange
TEXT_GRAY = '#475569'           # Slate 600 for annotations

configs = {
    "anthropic-claude-sonnet-4-6": {
        "title": "Claude Sonnet 4.6",
        "methods": [
            {"name": "Baseline", "path": "react/comp_cfo_claude/baseline", "color": COLOR_BASELINE, "style": dashed_style, "width": 1.5},
            {"name": "Procedural Graph", "path": "react/comp_cfo_claude/scratch_onetime/unknown_samples", "color": COLOR_PROCEDURAL_GRAPH, "style": "-", "width": 3.0}, # Thicker main line
            {"name": "Memory Retrieval", "path": "react/comp_cfo_claude/memory_retrieval/unknown_samples", "color": COLOR_RETRIEVAL, "style": dotted_style, "width": 1.5},
            {"name": "Memory Summarization", "path": "react/comp_cfo_claude/memory_summarization/unknown_samples", "color": COLOR_SUMMARIZATION, "style": dashdot_style, "width": 1.5}
        ]
    },
    "gemini-3.1-pro-preview": {
        "title": "Gemini 3.1 Pro",
        "methods": [
            {"name": "Baseline", "path": "react/comp_cfo_gemini31pro/baseline", "color": COLOR_BASELINE, "style": dashed_style, "width": 1.5},
            {"name": "Procedural Graph", "path": "react/comp_cfo_gemini31pro/scratch_onetime/unknown_samples", "color": COLOR_PROCEDURAL_GRAPH, "style": "-", "width": 3.0},
            {"name": "Memory Retrieval", "path": "react/comp_cfo_gemini31pro/memory_retrieval/unknown_samples", "color": COLOR_RETRIEVAL, "style": dotted_style, "width": 1.5},
            {"name": "Memory Summarization", "path": "react/comp_cfo_gemini31pro/memory_summarization/unknown_samples", "color": COLOR_SUMMARIZATION, "style": dashdot_style, "width": 1.5}
        ]
    },
    "gemini-3.5-flash": {
        "title": "Gemini 3.5 Flash",
        "methods": [
            {"name": "Baseline", "path": "react/comp_cfo/baseline", "color": COLOR_BASELINE, "style": dashed_style, "width": 1.5},
            {"name": "Procedural Graph", "path": "react/comp_cfo/scratch_onetime/50_samples", "color": COLOR_PROCEDURAL_GRAPH, "style": "-", "width": 3.0},
            {"name": "Memory Retrieval", "path": "react/comp_cfo/memory_retrieval/unknown_samples", "color": COLOR_RETRIEVAL, "style": dotted_style, "width": 1.5},
            {"name": "Memory Summarization", "path": "react/comp_cfo/memory_summarization/unknown_samples", "color": COLOR_SUMMARIZATION, "style": dashdot_style, "width": 1.5}
        ]
    },
    "grok-4.1-fast-non-reasoning": {
        "title": "Grok 4.1 Fast",
        "methods": [
            {"name": "Baseline", "path": "react/comp_cfo_grok/baseline", "color": COLOR_BASELINE, "style": dashed_style, "width": 1.5},
            {"name": "Procedural Graph", "path": "react/comp_cfo_grok/scratch_onetime/50_samples", "color": COLOR_PROCEDURAL_GRAPH, "style": "-", "width": 3.0},
            {"name": "Memory Retrieval", "path": "react/comp_cfo_grok/memory_retrieval/unknown_samples", "color": COLOR_RETRIEVAL, "style": dotted_style, "width": 1.5},
            {"name": "Memory Summarization", "path": "react/comp_cfo_grok/memory_summarization/unknown_samples", "color": COLOR_SUMMARIZATION, "style": dashdot_style, "width": 1.5}
        ]
    }
}

def smooth_trajectory(y, window_size=5):
    if len(y) < window_size:
        return y
    box = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, box, mode='same')
    for i in range(window_size // 2):
        y_smooth[i] = np.mean(y[:2*i+1])
        y_smooth[-i-1] = np.mean(y[-2*i-1:])
    return y_smooth

def load_method_data(llm, path):
    trajectories = []
    bankrupt_counts = np.zeros(total_steps)
    valid_samples_count = 0
    
    for s in samples:
        filepath = os.path.join(_CFO_DIR, llm, path, "samples", s, "financial_history.json")
        if not os.path.exists(filepath):
            continue
            
        valid_samples_count += 1
        with open(filepath, "r") as f:
            data = json.load(f)
            
        raw_cash = []
        for entry in data:
            m = entry.get("current_month", 0)
            if m <= max_months:
                raw_cash.append(max(0.0, entry.get("cash_balance", 0.0) / 1e6))
                
        termination_month = len(raw_cash) if len(raw_cash) < total_steps else None
        active_length = len(raw_cash)
        if active_length < total_steps:
            padded_cash = raw_cash + [0.0] * (total_steps - active_length)
        else:
            padded_cash = raw_cash[:total_steps]
            
        if termination_month is not None and termination_month > 0:
            active_cash = padded_cash[:termination_month]
            smoothed_active = smooth_trajectory(np.array(active_cash), window_size=5)
            smoothed_cash = list(smoothed_active) + [0.0] * (total_steps - len(smoothed_active))
        else:
            smoothed_cash = list(smooth_trajectory(np.array(padded_cash), window_size=5))
            
        trajectories.append(smoothed_cash)
        if termination_month is not None:
            for m in range(termination_month, total_steps):
                bankrupt_counts[m] += 1
                
    if valid_samples_count == 0:
        return None
        
    trajectories = np.array(trajectories)
    mean_trajectory = np.mean(trajectories, axis=0)
    survival_curve = (valid_samples_count - bankrupt_counts) / valid_samples_count
    
    # Calculate 95% Confidence Interval of the Mean for Cash
    cash_std = np.std(trajectories, axis=0)
    cash_sem = cash_std / np.sqrt(valid_samples_count)
    cash_ci_lower = np.clip(mean_trajectory - 1.96 * cash_sem, 0, None)
    cash_ci_upper = mean_trajectory + 1.96 * cash_sem
    
    # Calculate 95% Confidence Interval for Survival (Binomial Proportion CI)
    # Standard error for binomial: sqrt(p*(1-p)/N)
    p = survival_curve
    surv_sem = np.sqrt(p * (1 - p) / valid_samples_count)
    surv_ci_lower = np.clip(p - 1.96 * surv_sem, 0, 1.0)
    surv_ci_upper = np.clip(p + 1.96 * surv_sem, 0, 1.0)
    
    return {
        "trajectories": trajectories,
        "mean": mean_trajectory,
        "survival": survival_curve,
        "cash_ci_lower": cash_ci_lower,
        "cash_ci_upper": cash_ci_upper,
        "surv_ci_lower": surv_ci_lower,
        "surv_ci_upper": surv_ci_upper,
        "count": valid_samples_count
    }

def solve_overlap(val_label_pairs, min_dist=0.04, min_val=0.0, max_val=1.0):
    """Adjusts y-positions of labels to prevent overlapping."""
    sorted_pairs = sorted(val_label_pairs, key=lambda x: x[0])
    if not sorted_pairs:
        return []
        
    adjusted_vals = [x[0] for x in sorted_pairs]
    
    for _ in range(100):  # limit iterations
        overlap_found = False
        for i in range(len(adjusted_vals) - 1):
            diff = adjusted_vals[i+1] - adjusted_vals[i]
            if diff < min_dist:
                overlap_found = True
                overlap_mid = (adjusted_vals[i+1] + adjusted_vals[i]) / 2
                adjusted_vals[i] = max(min_val, overlap_mid - min_dist/2)
                adjusted_vals[i+1] = min(max_val, overlap_mid + min_dist/2)
        if not overlap_found:
            break
            
    return [(adjusted_vals[i], sorted_pairs[i][1], sorted_pairs[i][2]) for i in range(len(sorted_pairs))]

def main():
    # Use the official xkcd context manager
    with plt.xkcd(scale=1, length=100, randomness=1.5):
        # Figure size
        fig = plt.figure(figsize=(15, 8.0), facecolor='white')
        
        # GridSpec: Increased hspace from 0.05 to 0.18 for more gap between Cash and Survival.
        # Adjusted Row 2 (gap row) to 0.4 to keep the overall layout balanced.
        gs = gridspec.GridSpec(5, 2, height_ratios=[2, 1.2, 0.4, 2, 1.2], hspace=0.18, wspace=0.16)
        
        # Sequence: Claude 4.6, Gemini 3.1, Gemini 3.5, Grok
        llm_mapping = {
            "anthropic-claude-sonnet-4-6": (0, 0), # Col 0, Top (Row 0 & 1)
            "gemini-3.1-pro-preview": (1, 0),      # Col 1, Top (Row 0 & 1)
            "gemini-3.5-flash": (0, 3),            # Col 0, Bottom (Row 3 & 4)
            "grok-4.1-fast-non-reasoning": (1, 3)  # Col 1, Bottom (Row 3 & 4)
        }
        
        months = np.arange(total_steps)
        legend_handles = []
        legend_labels = []
        all_axes = []
        
        for llm_key, (col, row_start) in llm_mapping.items():
            llm_config = configs[llm_key]
            
            # Create axes
            ax_cash = fig.add_subplot(gs[row_start, col])
            ax_surv = fig.add_subplot(gs[row_start + 1, col], sharex=ax_cash)
            all_axes.extend([ax_cash, ax_surv])
            
            final_survival_rates = []
            final_cash_values = []
            
            # Determine max cash for Y-limit
            if llm_key == "anthropic-claude-sonnet-4-6":
                max_cash_val = 120.0
            else:
                max_cash_val = 50.0
                
            # Plot data
            for method in llm_config["methods"]:
                res = load_method_data(llm_key, method["path"])
                if res is None:
                    continue
                    
                # 1. Faint Spaghetti Lines (individual runs) - zorder=2
                # Slightly darker (alpha=0.12, linewidth=0.4) to be visible but not overwhelming
                for traj in res["trajectories"]:
                    ax_cash.plot(months, traj, color=method["color"], alpha=0.12, linewidth=0.4, zorder=2)
                    
                # 2. Shaded 95% CI Area - zorder=3 (above spaghetti, below mean)
                # Reduced alpha from 0.08 to 0.04 to make the shading even lighter
                ax_cash.fill_between(months, res["cash_ci_lower"], res["cash_ci_upper"], 
                                     color=method["color"], alpha=0.04, zorder=3)
                
                ax_surv.fill_between(months, res["surv_ci_lower"], res["surv_ci_upper"], 
                                     step='post', color=method["color"], alpha=0.04, zorder=3)
                    
                # 3. Mean Line - zorder=4
                line, = ax_cash.plot(months, res["mean"], color=method["color"], 
                                     linestyle=method["style"], linewidth=method["width"], 
                                     label=method["name"], zorder=4)
                
                # Plot survival - zorder=4
                ax_surv.step(months, res["survival"], color=method["color"], 
                            linestyle=method["style"], linewidth=2.0, where='post', zorder=4)
                
                # Save final values for later labeling
                final_survival_rates.append((res["survival"][-1], method["name"], method["color"]))
                final_cash_values.append((res["mean"][-1], method["name"], method["color"]))
                
                # Collect legend handles from the first plot
                if llm_key == "anthropic-claude-sonnet-4-6":
                    legend_handles.append(line)
                    legend_labels.append(method["name"])
                    
            # --- Aesthetics (Reference Style) ---
            
            # Title (only on cash plot) - Larger and padded
            ax_cash.set_title(llm_config["title"], fontsize=14, fontweight='bold', pad=10, color='#1E293B')
            
            # Y-limits and ticks
            if llm_key == "anthropic-claude-sonnet-4-6":
                ax_cash.set_ylim(0, 120)
                ax_cash.set_yticks([0, 30, 60, 90, 120])
            else:
                ax_cash.set_ylim(0, 50)
                ax_cash.set_yticks([0, 10, 20, 30, 40, 50])
                
            ax_surv.set_ylim(-0.02, 1.05)
            ax_surv.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_surv.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Extend X-limit to make room for labels inside the plot
            ax_cash.set_xlim(0, max_months + 12)
            ax_surv.set_xlim(0, max_months + 12)
            
            # Spines & Ticks - Clean look (NO GRID, matching reference)
            for ax in (ax_cash, ax_surv):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#64748B')  # Slate 500
                ax.spines['left'].set_color('#64748B')
                ax.tick_params(top=False, right=False, labelsize=10, colors='#475569')
                
            # Hide X-axis ticks and tick labels for cash plots
            ax_cash.tick_params(bottom=False, labelbottom=False)
            
            # Y-labels (only on the left column) - Larger with padding
            if col == 0:
                ax_cash.set_ylabel("Cash ($M)", fontsize=12, fontweight='bold', labelpad=8, color='#1E293B')
                ax_surv.set_ylabel("Survival", fontsize=12, fontweight='bold', labelpad=8, color='#1E293B')
                
            # X-labels (only on the bottom row) - Larger
            if row_start == 3:
                ax_surv.set_xlabel("Month", fontsize=12, fontweight='bold', labelpad=8, color='#1E293B')
                ax_surv.set_xticks([0, 20, 40, 60, 80, 100, 120])
            else:
                # For the top row survival plots, we still want X-ticks because there is a gap
                ax_surv.set_xticks([0, 20, 40, 60, 80, 100, 120])
                
            # --- Direct Labeling of Final Average Cash (No Overlap) ---
            if final_cash_values:
                # For Gemini 3.5, we only label the Procedural Graph (blue) at the request of the user
                if llm_key == "gemini-3.5-flash":
                    cash_to_label = [item for item in final_cash_values if item[1] == "Procedural Graph"]
                else:
                    cash_to_label = final_cash_values
                
                # Set min_val to 3.0 (for 50 max) or 7.2 (for 120 max) to prevent wiggled X-axis from covering the text
                min_val_cash = max_cash_val * 0.06
                adjusted_cash = solve_overlap(cash_to_label, min_dist=max_cash_val * 0.08, min_val=min_val_cash, max_val=max_cash_val + 10)
                for y_pos, label, color in adjusted_cash:
                    actual_cash = next(item[0] for item in cash_to_label if item[1] == label)
                    ax_cash.text(max_months + 1.5, y_pos, f"${actual_cash:.1f}M", 
                                 color=color, fontsize=9, fontweight='bold', va='center', zorder=10)
                       
            # --- Direct Labeling of Survival Rates (No Overlap) ---
            if final_survival_rates:
                if llm_key == "gemini-3.5-flash":
                    surv_to_label = [item for item in final_survival_rates if item[1] == "Procedural Graph"]
                else:
                    surv_to_label = final_survival_rates
                
                # Set min_val to 0.05 to keep it above the X-axis
                adjusted_labels = solve_overlap(surv_to_label, min_dist=0.08, min_val=0.05, max_val=1.0)
                for y_pos, label, color in adjusted_labels:
                    actual_rate = next(item[0] for item in surv_to_label if item[1] == label)
                    ax_surv.text(max_months + 1.5, y_pos, f"{actual_rate*100:.0f}%", 
                                 color=color, fontsize=9, fontweight='bold', va='center', zorder=10)
                
            # Crisis lines - zorder=3 (above spaghetti lines at zorder=2, below mean lines at zorder=4)
            # Note: Crisis lines MUST be solid ('-') in xkcd mode to avoid ValueError.
            # Made them slightly darker slate gray to stand out without a grid.
            crisis_months = [32, 59, 112]
            for idx, cm in enumerate(crisis_months, start=1):
                ax_cash.axvline(cm, color="#CBD5E1", linewidth=1.0, linestyle="-", alpha=0.7, zorder=3) # Slate 300
                ax_surv.axvline(cm, color="#CBD5E1", linewidth=1.0, linestyle="-", alpha=0.7, zorder=3)
                # Add Crisis labels at the top of the cash plot - zorder=5
                ax_cash.text(cm - 1.5, max_cash_val * 0.9, f"Crisis {idx}", color=TEXT_GRAY, fontsize=9, 
                             fontweight="bold", rotation=90, va="top", ha="right", zorder=5)
                
        # Align all y-labels on the left column
        fig.align_ylabels([ax for ax in all_axes if ax.get_ylabel()])
                
        # Add a single common legend at the bottom of the figure (No border, matching reference)
        # Moved bbox_to_anchor Y from 0.002 to -0.005 to move it even further down
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.005), ncol=4, frameon=False, 
                   fontsize=12)
                   
        # Adjust layout to make room for the legend (rect bottom changed from 0.07 to 0.09 to push plots up)
        plt.tight_layout(rect=[0, 0.09, 1, 1])
        
        # Save the combined plot
        out_path = os.path.join(_BASE_RESULTS_DIR, "combined_ensemble_cash_comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        pdf_out_path = out_path.replace(".png", ".pdf")
        plt.savefig(pdf_out_path, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully generated combined chart: {out_path}")
        print(f"Vector PDF: {pdf_out_path}")

if __name__ == "__main__":
    main()
