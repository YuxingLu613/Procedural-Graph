"""File-system based Results Compilation and tracking."""

import csv
import json
import os
import time

class ResultsCompiler:
  """Compiles local summary.csv files under results directory into a CSV and Markdown dashboard."""

  def __init__(self, results_dir: str = "results_0609"):
    self.results_dir = results_dir

  def compile_dashboard(self, output_md_path: str, output_csv_path: str) -> None:
    """Crawls results_dir to find all summary.csv files and compiles them."""
    import glob
    
    # Find all summary.csv files under results_dir recursively
    search_pattern = os.path.join(self.results_dir, "**", "summary.csv")
    summary_files = glob.glob(search_pattern, recursive=True)
    
    csv_headers = [
        "Experiment ID",
        "Samples",
        "Success Rate",
        "Avg Steps",
        "Avg Token Cost",
        "Avg Latency (s)",
        "Full Surv. %",
        "Avg. Score ($M)",
        "1st Crisis %",
        "2nd Crisis %",
        "3rd Crisis %",
        "Tools/Mo",
        "Actions",
        "Raised ($M)",
    ]
    
    csv_rows = []
    for sf in summary_files:
      # Exclude the master dashboard CSV itself if it matches the pattern
      if os.path.abspath(sf) == os.path.abspath(output_csv_path):
        continue
      try:
        with open(sf, "r", newline="", encoding="utf-8") as f:
          reader = csv.DictReader(f)
          for row in reader:
            new_row = []
            for h in csv_headers:
              new_row.append(row.get(h, ""))
            csv_rows.append(new_row)
      except Exception as e:
        print(f"⚠️ Failed to read summary file {sf}: {e}")
        
    # Sort by experiment ID to maintain deterministic order
    csv_rows.sort(key=lambda x: x[0])
    
    # Write CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(csv_headers)
      writer.writerows(csv_rows)
      
    # Write Markdown Report
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
      f.write("# Procedural Graph Experiment Summary Report\n\n")
      f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
      
      f.write("| " + " | ".join(csv_headers) + " |\n")
      f.write("|" + "|".join(["---"] * len(csv_headers)) + "|\n")
      for row in csv_rows:
        f.write("| " + " | ".join(row) + " |\n")
        
    print(f"Generated global dashboard.csv at {output_csv_path} and markdown report at {output_md_path}")

  def compile_cfo_dashboard(self, output_md_path: str) -> None:
    """Crawls results_dir to find all trajectory.json files and compiles a CFO-specific dashboard."""
    import glob

    search_pattern = os.path.join(self.results_dir, "cfo", "**", "trajectory.json")
    trajectory_files = glob.glob(search_pattern, recursive=True)

    runs = {}
    for tf in trajectory_files:
      dir_path = os.path.dirname(os.path.dirname(os.path.dirname(tf)))
      if dir_path not in runs:
        runs[dir_path] = []
      runs[dir_path].append(tf)

    compiled_results = []
    for run_dir, files in runs.items():
      cfo_dir = os.path.join(self.results_dir, "cfo")
      if not os.path.exists(cfo_dir):
        continue
      rel_path = os.path.relpath(run_dir, cfo_dir)
      parts = rel_path.split(os.sep)
      if len(parts) < 4:
        continue
      llm, method, timestamp, ref_mode = parts[0], parts[1], parts[2], parts[3]

      total_samples = 0
      survived_samples = 0
      crisis1_samples = 0
      crisis2_samples = 0
      crisis3_samples = 0

      sum_months = 0.0
      sum_score_m = 0.0
      sum_tools_per_mo = 0.0
      sum_actions = 0.0
      sum_raised_m = 0.0

      for fpath in files:
        try:
          with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            metrics = data.get("metrics", {})
            if "cfo_months" not in metrics:
              continue
            total_samples += 1

            months = metrics.get("cfo_months", 0.0)
            survived = metrics.get("cfo_survived", 0.0)
            crisis1 = metrics.get("cfo_crisis1", 0.0)
            crisis2 = metrics.get("cfo_crisis2", 0.0)
            crisis3 = metrics.get("cfo_crisis3", 0.0)

            if survived > 0:
              survived_samples += 1
            if crisis1 > 0:
              crisis1_samples += 1
            if crisis2 > 0:
              crisis2_samples += 1
            if crisis3 > 0:
              crisis3_samples += 1

            sum_months += months
            sum_score_m += metrics.get("cfo_score_m", 0.0)
            sum_tools_per_mo += metrics.get("cfo_tools_per_mo", 0.0)
            sum_actions += metrics.get("cfo_actions", 0.0)
            sum_raised_m += metrics.get("cfo_raised_m", 0.0)
        except Exception as e:
          print(f"⚠️ Failed to read trajectory {fpath}: {e}")

      if total_samples > 0:
        compiled_results.append({
            "llm": llm,
            "method": method,
            "timestamp": timestamp,
            "ref_mode": ref_mode,
            "samples": total_samples,
            "full_surv_pct": (survived_samples / total_samples) * 100.0,
            "avg_months": sum_months / total_samples,
            "avg_score_m": sum_score_m / total_samples,
            "crisis1_pct": (crisis1_samples / total_samples) * 100.0,
            "crisis2_pct": (crisis2_samples / total_samples) * 100.0,
            "crisis3_pct": (crisis3_samples / total_samples) * 100.0,
            "tools_per_mo": sum_tools_per_mo / total_samples,
            "actions": sum_actions / total_samples,
            "raised_m": sum_raised_m / total_samples,
        })

    compiled_results.sort(
        key=lambda x: (x["llm"], x["method"], x["ref_mode"], x["timestamp"])
    )

    markdown_lines = []
    markdown_lines.append("# CFO Procedural Graph Experiment Results")
    markdown_lines.append("")
    markdown_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    markdown_lines.append("")
    markdown_lines.append(
        "| Models Overall (↑) | | Multi-Crisis Survival (↑) | | | | | Agent"
        " Performance (↑) | | | |"
    )
    markdown_lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        " :---: | :---: | :---: |"
    )
    markdown_lines.append(
        "| Model & Config | Full Surv.% | Avg. Months | Avg. Mon. Score ($M) | 1st Crisis |"
        " 2nd Crisis | 3rd Crisis | Tools/Mo | Actions | Raised ($M) |"
        " Samples |"
    )

    for res in compiled_results:
      model_name = f"{res['llm']} ({res['method']}, {res['ref_mode']})"
      line = (
          f"| {model_name} "
          f"| {res['full_surv_pct']:.1f}% "
          f"| {res['avg_months']:.2f} "
          f"| ${res['avg_score_m']:.3f}M "
          f"| {res['crisis1_pct']:.1f}% "
          f"| {res['crisis2_pct']:.1f}% "
          f"| {res['crisis3_pct']:.1f}% "
          f"| {res['tools_per_mo']:.2f} "
          f"| {res['actions']:.1f} "
          f"| ${res['raised_m']:.2f}M "
          f"| {res['samples']} |"
      )
      markdown_lines.append(line)

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
      f.write("\n".join(markdown_lines) + "\n")
    print(f"Generated CFO dashboard at {output_md_path}")

  def compile_general_dashboard(self, output_md_path: str, dataset_filter: str | None = None) -> None:
    """Compiles a clean 6-column dashboard for non-financial datasets like ALFWorld."""
    import glob
    search_pattern = os.path.join(self.results_dir, "**", "summary.csv")
    summary_files = glob.glob(search_pattern, recursive=True)

    general_headers = [
        "Experiment ID", "Samples", "Success Rate", "Avg Steps", "Avg Token Cost", "Avg Latency (s)"
    ]

    rows = []
    for sf in summary_files:
      if dataset_filter and f"/{dataset_filter}/" not in sf and not sf.startswith(f"{dataset_filter}/"):
        continue
      try:
        with open(sf, "r", newline="", encoding="utf-8") as f:
          reader = csv.DictReader(f)
          for row in reader:
            rows.append([row.get(h, "") for h in general_headers])
      except Exception as e:
        print(f"⚠️ Failed to read {sf}: {e}")

    rows.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
      title = f"{dataset_filter.upper()} " if dataset_filter else "General "
      f.write(f"# {title}Procedural Graph Experiment Summary\n\n")
      f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
      f.write("| " + " | ".join(general_headers) + " |\n")
      f.write("|" + "|".join(["---"] * len(general_headers)) + "|\n")
      for r in rows:
        f.write("| " + " | ".join(r) + " |\n")
    print(f"Generated clean dashboard at {output_md_path}")

