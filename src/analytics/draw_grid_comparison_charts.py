import os
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Base directories
BASE_RESULTS_DIR = os.environ.get(
    "PG_RESULTS_DIR",
    "results/cfo_generative_pg_memory_comparison"
)
CFO_DIR = os.path.join(BASE_RESULTS_DIR, "cfo")
OUTPUT_DIR = BASE_RESULTS_DIR  # Save the 2 grid figures directly in the results directory!

# Define the configurations for Gemini and Grok
CONFIGS = {
    "gemini-3.5-flash": {
        "dir_name": "gemini-3.5-flash",
        "title_label": "Gemini 3.5 Flash",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo/scratch_onetime/50_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    },
    "grok-4.1-fast-non-reasoning": {
        "dir_name": "grok-4.1-fast-non-reasoning",
        "title_label": "Grok 4.1 Fast Non-Reasoning",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo_grok/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo_grok/scratch_onetime/50_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo_grok/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo_grok/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    },
    "anthropic-claude-sonnet-4-6": {
        "dir_name": "anthropic-claude-sonnet-4-6",
        "title_label": "Claude 3.5 Sonnet",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo_claude/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo_claude/scratch_onetime/unknown_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo_claude/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo_claude/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    },
    "gemini-3.1-pro-preview": {
        "dir_name": "gemini-3.1-pro-preview",
        "title_label": "Gemini 3.1 Pro",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo_gemini31pro/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo_gemini31pro/scratch_onetime/unknown_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo_gemini31pro/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo_gemini31pro/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    }
}

def load_cash_history(llm_dir, rel_path, sample):
    """Loads cash balance history from financial_history.json file."""
    filepath = os.path.join(
        CFO_DIR, llm_dir, rel_path, "samples", sample, "financial_history.json"
    )
    if not os.path.exists(filepath):
        return [], []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        months = [entry.get("current_month", 0) for entry in data]
        cash = [entry.get("cash_balance", 0.0) / 1e6 for entry in data]  # in $M
        return months, cash
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return [], []

def main():
    # Grid dimensions: 10 rows, 5 columns (for all 50 samples)
    rows = 10
    cols = 5
    samples = [f"sample_{i:03d}" for i in range(1, 51)]
    
    for llm_key, llm_config in CONFIGS.items():
        llm_name = llm_config["dir_name"]
        print(f"Generating massive grid chart for {llm_name}...")
        
        # Create a large figure for the 10x5 grid
        # sharex and sharey make comparison extremely clean and remove redundant labels
        fig, axes = plt.subplots(rows, cols, figsize=(24, 36), sharex=True, sharey=True)
        
        # Flatten axes array for easy 1D iteration
        axes_flat = axes.flatten()
        
        for idx, sample in enumerate(samples):
            ax = axes_flat[idx]
            
            # Plot each method on the current subplot
            for method in llm_config["methods"]:
                months, cash = load_cash_history(llm_name, method["rel_path"], sample)
                if months:
                    ax.plot(
                        months,
                        cash,
                        color=method["color"],
                        linestyle="-",
                        linewidth=1.2,
                        label=method["name"] if idx == 0 else ""  # Only label the very first subplot for the legend
                    )
            
            # Subplot styling
            ax.set_title(sample.upper(), fontsize=10, fontweight='bold', pad=2)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
            ax.axhline(5, color="red", linewidth=0.6, linestyle="--")
            
            # Limit adjustments
            ax.set_xlim(0, 132)
            
        # Add a single common legend at the top of the figure
        # This prevents duplicating the legend 50 times
        handles, labels = axes_flat[0].get_legend_handles_labels()
        # Add the red warning line to the legend
        warning_line = matplotlib.lines.Line2D([0], [0], color="red", linestyle="--", linewidth=1.0)
        handles.append(warning_line)
        labels.append("Liquidity Warning ($5M)")
        
        fig.legend(
            handles, 
            labels, 
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.98), 
            ncol=6, 
            fontsize=12, 
            frameon=True, 
            shadow=True
        )
        
        # Add common X and Y labels for the entire figure
        fig.text(0.5, 0.01, "Simulation Month", ha="center", va="center", fontsize=14, fontweight="bold")
        fig.text(0.01, 0.5, "Cash Balance ($ Millions)", ha="center", va="center", rotation="vertical", fontsize=14, fontweight="bold")
        
        fig.suptitle(
            f"CFO Liquidity Cash Comparison Grid ({llm_config['title_label']}) - All 50 Samples", 
            fontsize=18, 
            fontweight="bold",
            y=0.99
        )
        
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
        
        out_path = os.path.join(OUTPUT_DIR, f"{llm_name}_grid_cash_comparison.png")
        print(f"Saving massive grid image (this might take a moment)...")
        # Save with high DPI so zooming in is perfectly crisp
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Generated grid chart: {out_path}")

if __name__ == "__main__":
    main()
