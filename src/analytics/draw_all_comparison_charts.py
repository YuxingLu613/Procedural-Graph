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
OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "plots")

# Define the configurations for Gemini and Grok
CONFIGS = {
    "gemini-3.5-flash": {
        "dir_name": "gemini-3.5-flash",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo/scratch_onetime/50_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    },
    "grok-4.1-fast-non-reasoning": {
        "dir_name": "grok-4.1-fast-non-reasoning",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo_grok/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo_grok/scratch_onetime/50_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo_grok/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo_grok/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    },
    "anthropic-claude-sonnet-4-6": {
        "dir_name": "anthropic-claude-sonnet-4-6",
        "methods": [
            {"name": "Pure ReAct Baseline", "rel_path": "react/comp_cfo_claude/baseline", "color": "crimson", "marker": "o"},
            {"name": "Procedural Graph", "rel_path": "react/comp_cfo_claude/scratch_onetime/unknown_samples", "color": "dodgerblue", "marker": "s"},
            {"name": "Memory Retrieval", "rel_path": "react/comp_cfo_claude/memory_retrieval/unknown_samples", "color": "forestgreen", "marker": "^"},
            {"name": "Memory Summarization", "rel_path": "react/comp_cfo_claude/memory_summarization/unknown_samples", "color": "darkorange", "marker": "d"},
        ]
    },
    "gemini-3.1-pro-preview": {
        "dir_name": "gemini-3.1-pro-preview",
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
    # We will generate plots for all 50 samples
    samples = [f"sample_{i:03d}" for i in range(1, 51)]
    
    for llm_key, llm_config in CONFIGS.items():
        llm_name = llm_config["dir_name"]
        llm_out_dir = os.path.join(OUTPUT_DIR, llm_name)
        os.makedirs(llm_out_dir, exist_ok=True)
        print(f"Generating charts for {llm_name}...")
        
        for sample in samples:
            plt.figure(figsize=(11, 6))
            has_data = False
            
            for method in llm_config["methods"]:
                months, cash = load_cash_history(llm_name, method["rel_path"], sample)
                if months:
                    has_data = True
                    plt.plot(
                        months,
                        cash,
                        color=method["color"],
                        linestyle="-",
                        marker=method["marker"],
                        markersize=4,
                        linewidth=1.5,
                        label=method["name"]
                    )
            
            if not has_data:
                plt.close()
                continue
                
            plt.title(f"CFO Liquidity Cash Comparison ({llm_name}): {sample}", fontsize=12, fontweight='bold')
            plt.xlabel("Simulation Month", fontsize=10)
            plt.ylabel("Cash Balance ($ Millions)", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.axhline(0, color="black", linewidth=1.2, linestyle="-")
            plt.axhline(
                5,
                color="red",
                linewidth=1.0,
                linestyle="--",
                label="Liquidity Warning Threshold ($5M)",
            )
            
            plt.legend(loc="upper left", frameon=True, shadow=True)
            plt.tight_layout()
            
            out_path = os.path.join(llm_out_dir, f"{sample}_cash_comparison.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            
        print(f"Completed charts for {llm_name} in {llm_out_dir}")

if __name__ == "__main__":
    main()
