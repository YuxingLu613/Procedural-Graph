import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Set professional plotting style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#111111'

# Base directories
_BASE_RESULTS_DIR = os.environ.get(
    "PG_RESULTS_DIR",
    "results/cfo_generative_pg_memory_comparison"
)
_CFO_DIR = os.path.join(_BASE_RESULTS_DIR, "cfo")

samples = [f"sample_{i:03d}" for i in range(1, 51)]
max_months = 131
total_steps = 132  # Month 0 to Month 131 is 132 points

configs = {
    "gemini-3.5-flash": {
        "title": "Gemini 3.5 Flash",
        "filename": "gemini_ensemble_cash_comparison.png",
        "methods": [
            {
                "name": "Baseline",
                "path": "react/comp_cfo/baseline",
                "color": "#D32F2F",  # Red
                "style": "--",
                "width": 2.0
            },
            {
                "name": "Procedural Graph",
                "path": "react/comp_cfo/scratch_onetime/50_samples",
                "color": "#1976D2",  # Blue
                "style": "-",
                "width": 3.0
            },
            {
                "name": "Memory Retrieval",
                "path": "react/comp_cfo/memory_retrieval/unknown_samples",
                "color": "#388E3C",  # Green
                "style": ":",
                "width": 2.0
            },
            {
                "name": "Memory Summarization",
                "path": "react/comp_cfo/memory_summarization/unknown_samples",
                "color": "#F57C00",  # Orange
                "style": "-.",
                "width": 2.0
            }
        ]
    },
    "grok-4.1-fast-non-reasoning": {
        "title": "Grok 4.1 Fast Non-Reasoning",
        "filename": "grok_ensemble_cash_comparison.png",
        "methods": [
            {
                "name": "Baseline",
                "path": "react/comp_cfo_grok/baseline",
                "color": "#D32F2F",
                "style": "--",
                "width": 2.0
            },
            {
                "name": "Procedural Graph",
                "path": "react/comp_cfo_grok/scratch_onetime/50_samples",
                "color": "#1976D2",
                "style": "-",
                "width": 3.0
            },
            {
                "name": "Memory Retrieval",
                "path": "react/comp_cfo_grok/memory_retrieval/unknown_samples",
                "color": "#388E3C",
                "style": ":",
                "width": 2.0
            },
            {
                "name": "Memory Summarization",
                "path": "react/comp_cfo_grok/memory_summarization/unknown_samples",
                "color": "#F57C00",
                "style": "-.",
                "width": 2.0
            }
        ]
    },
    "anthropic-claude-sonnet-4-6": {
        "title": "Claude 3.5 Sonnet",
        "filename": "claude_ensemble_cash_comparison.png",
        "methods": [
            {
                "name": "Baseline",
                "path": "react/comp_cfo_claude/baseline",
                "color": "#D32F2F",
                "style": "--",
                "width": 2.0
            },
            {
                "name": "Procedural Graph",
                "path": "react/comp_cfo_claude/scratch_onetime/unknown_samples",
                "color": "#1976D2",
                "style": "-",
                "width": 3.0
            },
            {
                "name": "Memory Retrieval",
                "path": "react/comp_cfo_claude/memory_retrieval/unknown_samples",
                "color": "#388E3C",
                "style": ":",
                "width": 2.0
            },
            {
                "name": "Memory Summarization",
                "path": "react/comp_cfo_claude/memory_summarization/unknown_samples",
                "color": "#F57C00",
                "style": "-.",
                "width": 2.0
            }
        ]
    },
    "gemini-3.1-pro-preview": {
        "title": "Gemini 3.1 Pro",
        "filename": "gemini31pro_ensemble_cash_comparison.png",
        "methods": [
            {
                "name": "Baseline",
                "path": "react/comp_cfo_gemini31pro/baseline",
                "color": "#D32F2F",
                "style": "--",
                "width": 2.0
            },
            {
                "name": "Procedural Graph",
                "path": "react/comp_cfo_gemini31pro/scratch_onetime/unknown_samples",
                "color": "#1976D2",
                "style": "-",
                "width": 3.0
            },
            {
                "name": "Memory Retrieval",
                "path": "react/comp_cfo_gemini31pro/memory_retrieval/unknown_samples",
                "color": "#388E3C",
                "style": ":",
                "width": 2.0
            },
            {
                "name": "Memory Summarization",
                "path": "react/comp_cfo_gemini31pro/memory_summarization/unknown_samples",
                "color": "#F57C00",
                "style": "-.",
                "width": 2.0
            }
        ]
    }
}

def smooth_trajectory(y, window_size=5):
    """Applies a moving average to smooth the trajectory, handling edges gracefully."""
    if len(y) < window_size:
        return y
    box = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, box, mode='same')
    # Correct edge effects
    for i in range(window_size // 2):
        y_smooth[i] = np.mean(y[:2*i+1])
        y_smooth[-i-1] = np.mean(y[-2*i-1:])
    return y_smooth

def load_method_data(llm, path):
    trajectories = []
    survival_timeline = np.ones(total_steps)  # Start with 1.0 (100% alive)
    bankrupt_counts = np.zeros(total_steps)
    
    valid_samples_count = 0
    
    for s in samples:
        filepath = os.path.join(_CFO_DIR, llm, path, "samples", s, "financial_history.json")
        if not os.path.exists(filepath):
            continue
            
        valid_samples_count += 1
        with open(filepath, "r") as f:
            data = json.load(f)
            
        # Extract cash balance, convert to $M, and clip negative to 0.0
        raw_cash = []
        for entry in data:
            m = entry.get("current_month", 0)
            if m <= max_months:
                raw_cash.append(max(0.0, entry.get("cash_balance", 0.0) / 1e6))
                
        # The run is dead from raw_cash length onwards if it ended early
        termination_month = len(raw_cash) if len(raw_cash) < total_steps else None
                
        # Pad with 0.0 if the run went bankrupt/ended early
        active_length = len(raw_cash)
        if active_length < total_steps:
            padded_cash = raw_cash + [0.0] * (total_steps - active_length)
        else:
            padded_cash = raw_cash[:total_steps]
            
        # Smooth only the active portion of the trajectory, then pad with 0.0
        if termination_month is not None and termination_month > 0:
            active_cash = padded_cash[:termination_month]
            smoothed_active = smooth_trajectory(np.array(active_cash), window_size=5)
            smoothed_cash = list(smoothed_active) + [0.0] * (total_steps - len(smoothed_active))
        else:
            # If it survived the whole time, smooth the entire thing
            smoothed_cash = list(smooth_trajectory(np.array(padded_cash), window_size=5))
            
        trajectories.append(smoothed_cash)
        
        # Record termination for survival curve
        if termination_month is not None:
            # From termination_month onwards, the sample is dead
            for m in range(termination_month, total_steps):
                bankrupt_counts[m] += 1
                
    if valid_samples_count == 0:
        return None
        
    trajectories = np.array(trajectories)
    mean_trajectory = np.mean(trajectories, axis=0)
    
    # Calculate survival curve (percentage of active samples at each month)
    survival_curve = (valid_samples_count - bankrupt_counts) / valid_samples_count
    
    return {
        "trajectories": trajectories,
        "mean": mean_trajectory,
        "survival": survival_curve,
        "count": valid_samples_count
    }

def solve_overlap(val_label_pairs, min_dist=0.04, min_val=0.0, max_val=1.0):
    """Adjusts y-positions of labels to prevent overlapping."""
    # Sort by value
    sorted_pairs = sorted(val_label_pairs, key=lambda x: x[0])
    if not sorted_pairs:
        return []
        
    adjusted_vals = [x[0] for x in sorted_pairs]
    
    # Iterative adjustment to ensure minimum distance
    for _ in range(100):  # limit iterations
        overlap_found = False
        for i in range(len(adjusted_vals) - 1):
            diff = adjusted_vals[i+1] - adjusted_vals[i]
            if diff < min_dist:
                overlap_found = True
                overlap_mid = (adjusted_vals[i+1] + adjusted_vals[i]) / 2
                # Spread them apart
                adjusted_vals[i] = max(min_val, overlap_mid - min_dist/2)
                adjusted_vals[i+1] = min(max_val, overlap_mid + min_dist/2)
        if not overlap_found:
            break
            
    return [(adjusted_vals[i], sorted_pairs[i][1], sorted_pairs[i][2]) for i in range(len(sorted_pairs))]

def plot_llm_comparison(llm_key, llm_config):
    print(f"Plotting charts for {llm_config['title']}...")
    
    # Create two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, 
                                   gridspec_kw={'height_ratios': [7, 3]})
    
    # Adjust spacing: tight but elegant
    plt.subplots_adjust(hspace=0.08)
    
    months = np.arange(total_steps)
    
    # Track final values for labeling
    final_survival_rates = []
    final_cash_values = []
    
    for method in llm_config["methods"]:
        res = load_method_data(llm_key, method["path"])
        if res is None:
            print(f"  Warning: No data for {method['name']}, skipping.")
            continue
            
        print(f"  Loaded {method['name']}: {res['count']} valid runs. Final Survival: {res['survival'][-1]*100:.1f}%")
        
        # --- Plot 1: Cash Trajectories (Upper Plot) ---
        # 1. Faint Spaghetti Lines (individual runs)
        for traj in res["trajectories"]:
            ax1.plot(months, traj, color=method["color"], alpha=0.10, linewidth=0.4)
            
        # 2. Mean Line
        ax1.plot(months, res["mean"], color=method["color"], linestyle=method["style"],
                 linewidth=method["width"], label=method["name"])
        
        # --- Plot 2: Survival Curves (Lower Plot) ---
        # Kaplan-Meier step function
        ax2.step(months, res["survival"], color=method["color"], linestyle=method["style"],
                 linewidth=2.2, where='post')
        
        # Save final values for later labeling
        final_survival_rates.append((res["survival"][-1], method["name"], method["color"]))
        final_cash_values.append((res["mean"][-1], method["name"], method["color"]))
        
    # --- Upper Plot (Cash Trajectories) Aesthetics ---
    if llm_key == "anthropic-claude-sonnet-4-6":
        ax1.set_ylim(0, 200)
        ax1.set_yticks([0, 50, 100, 150, 200])
    else:
        ax1.set_ylim(0, 50)
        ax1.set_yticks([0, 10, 20, 30, 40, 50])
    ax1.set_ylabel("Cash Balance ($ Millions)", fontsize=11, fontweight='bold', labelpad=8)
    
    ax1.grid(axis='y', linestyle=':', color='#CCCCCC', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color('#888888')
    ax1.spines['left'].set_color('#888888')
    ax1.tick_params(top=False, right=False)
    
    # (Legend is drawn on ax2 below to prevent it being covered by ax2 background)
               
    # --- Direct Labeling of Final Average Cash (No Overlap) ---
    if llm_key != "gemini-3.5-flash":
        adjusted_cash = solve_overlap(final_cash_values, min_dist=3.5, min_val=-5.0, max_val=60.0)
        for y_pos, label, color in adjusted_cash:
            actual_cash = next(item[0] for item in final_cash_values if item[1] == label)
            # Place label at Month 131 + 1.5 to make it look clean
            ax1.text(max_months + 1.5, y_pos, f"${actual_cash:.1f}M", 
                     color=color, fontsize=10, fontweight='bold', va='center')
               
    # --- Lower Plot (Survival Curves) Aesthetics ---
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_ylabel("Survival Rate", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_xlabel("Simulation Month", fontsize=11, fontweight='bold', labelpad=8)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax2.grid(axis='both', linestyle=':', color='#CCCCCC', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#888888')
    ax2.spines['left'].set_color('#888888')
    ax2.tick_params(top=False, right=False)
    
    # --- Direct Labeling of Survival Rates (No Overlap) ---
    if llm_key != "gemini-3.5-flash":
        adjusted_labels = solve_overlap(final_survival_rates, min_dist=0.05, min_val=0.0, max_val=1.0)
        for y_pos, label, color in adjusted_labels:
            # Get the actual final survival rate for the text
            actual_rate = next(item[0] for item in final_survival_rates if item[1] == label)
            # Place label at Month 131 + 1.5 to make it look clean
            ax2.text(max_months + 1.5, y_pos, f"{actual_rate*100:.0f}%", 
                     color=color, fontsize=10, fontweight='bold', va='center')
                 
    # --- Continuous Crisis Timeline Markers ---
    crisis_months = [32, 59, 112]
    for idx, cm in enumerate(crisis_months, start=1):
        ax1.axvline(cm, color="#9E9E9E", linewidth=1.0, linestyle=":", alpha=0.8)
        ax2.axvline(cm, color="#9E9E9E", linewidth=1.0, linestyle=":", alpha=0.8)
        # Position label near the top of the upper subplot (Y=45 for 0-50 ylim)
        ax1.text(cm - 1.2, 45, f"Crisis {idx}", color="#757575", fontsize=8.5, 
                 fontweight="bold", rotation=90, va="top", ha="right")
                 
    # --- Legend (Drawn on ax2 but showing ax1 handles to sit on top of background) ---
    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 1.10), frameon=True, 
               framealpha=0.9, edgecolor='#E0E0E0', shadow=False, fontsize=8.8,
               labelspacing=0.3, handletextpad=0.4, borderpad=0.4)
                 
    # Adjust x-axis limits slightly to accommodate the labels
    ax2.set_xlim(0, max_months + 8)
    
    # Ensure x-ticks are clear
    ax2.set_xticks([0, 20, 40, 60, 80, 100, 120])
    
    # Save the figure (both PNG and PDF vector version)
    out_path = os.path.join(_BASE_RESULTS_DIR, llm_config["filename"])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_out_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_out_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated publication-ready chart: {out_path} and vector PDF: {pdf_out_path}")

def main():
    for llm_key, llm_config in configs.items():
        plot_llm_comparison(llm_key, llm_config)

if __name__ == "__main__":
    main()
