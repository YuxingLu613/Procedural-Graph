"""Generates a big table of experiment results like Sheet2.csv."""

import csv
import json
import os
import pandas as pd
from absl import app


def main(argv):
  del argv
  script_dir = os.path.dirname(os.path.abspath(__file__))
  parquet_path = os.path.join(
      script_dir, "dataset/General/MultiChallenge/data/test_166.parquet"
  )
  if not os.path.exists(parquet_path):
    # Fallback to default relative data path
    parquet_path = os.environ.get(
        "PG_MULTICHALLENGE_PARQUET",
        "data/datasets/General/MultiChallenge/data/test_166.parquet"
    )

  if not os.path.exists(parquet_path):
    print(f"Error: Parquet file not found at any path: {parquet_path}")
    return

  df = pd.read_parquet(parquet_path)
  print(f"Loaded parquet file with {len(df)} samples.")
  sample_axes = {}
  for i, row in df.iterrows():
    sample_axes[f"sample_{i+1:03d}"] = row["axis"]

  results_dir = os.path.join(script_dir, "results_0609/multichallenge")
  if not os.path.exists(results_dir):
    results_dir = os.environ.get(
        "PG_RESULTS_0609_DIR",
        "results_0609/multichallenge"
    )
  results = []

  # Walk results directory to find trajectory.json files
  for root, dirs, files in os.walk(results_dir):
    for f in files:
      if f == "trajectory.json":
        path = os.path.join(root, f)
        rel_path = os.path.relpath(root, results_dir)
        parts = rel_path.split(os.sep)

        if len(parts) >= 4 and "samples" in parts:
          model = parts[0]
          method = parts[1]
          samples_idx = parts.index("samples")
          refinement_mode = "/".join(parts[2:samples_idx])
          sample_id = parts[samples_idx + 1]

          try:
            with open(path, "r", encoding="utf-8") as json_file:
              data = json.load(json_file)
              metrics = data.get("metrics", {})
              if "success" in metrics:
                score = float(metrics["success"])
                axis = sample_axes.get(sample_id)
                if axis:
                  results.append({
                      "model": model,
                      "method": method,
                      "refinement_mode": refinement_mode,
                      "sample_id": sample_id,
                      "score": score,
                      "axis": axis,
                  })
          except Exception as e:
            print(f"Warning: Failed to parse {path}: {e}")

  if not results:
    print("No evaluation results found.")
    return

  res_df = pd.DataFrame(results)

  # Group by model, method, refinement_mode, and axis
  grouped = (
      res_df.groupby(["model", "method", "refinement_mode", "axis"])["score"]
      .mean()
      .reset_index()
  )

  # Pivot the table to show axes as columns
  pivot_df = (
      grouped.pivot(
          index=["model", "method", "refinement_mode"],
          columns="axis",
          values="score",
      )
      .reset_index()
  )

  # Calculate overall success rate
  overall_success = (
      res_df.groupby(["model", "method", "refinement_mode"])["score"]
      .mean()
      .reset_index()
  )
  overall_success.rename(columns={"score": "Average"}, inplace=True)

  # Merge overall results back
  final_df = pd.merge(
      pivot_df, overall_success, on=["model", "method", "refinement_mode"]
  )

  # We want to format the output to match Sheet2.csv structure.
  # Columns to write:
  # LLM Model, Methods, Condition, HotPotQA Answer EM, HotPotQA Answer F1, HotPotQA Support EM,
  # HotPotQA Support F1, HotPotQA Joint EM, HotPotQA Joint F1,
  # MultiChallenge Instruction Retention, MultiChallenge Inference Memory, MultiChallenge Reliable Version Editing,
  # MultiChallenge Self-Coherence, MultiChallenge Average, CodeAct Accuracy, SweBench-Lite % Resolved

  # Mapping of our columns:
  axis_mapping = {
      "ADVERSARIAL_INSTRUCTION_FOLLOWING": "MultiChallenge Instruction Retention",
      "INSTRUCTION_RETENTION": "MultiChallenge Instruction Retention",
      "AMBIGUITY_AND_CLARIFICATION": "MultiChallenge Inference Memory",
      "INFERENCE_MEMORY": "MultiChallenge Inference Memory",
      "COMPLEX_INSTRUCTION_FOLLOWING": "MultiChallenge Reliable Version Editing",
      "RELIABLE_VERSION_EDITING": "MultiChallenge Reliable Version Editing",
      "INTERACTIVE_PROBLEM_SOLVING": "MultiChallenge Self-Coherence",
      "SELF_COHERENCE": "MultiChallenge Self-Coherence",
      "Average": "MultiChallenge Average",
  }

  output_rows = []
  # Define headers exactly matching Sheet2.csv
  headers = [
      "LLM Model",
      "Methods",
      "",
      "HotPotQA Answer EM",
      "HotPotQA Answer F1",
      "HotPotQA Support EM",
      "HotPotQA Support F1",
      "HotPotQA Joint EM",
      "HotPotQA Joint F1",
      "MultiChallenge Instruction Retention",
      "MultiChallenge Inference Memory",
      "MultiChallenge Reliable Version Editing",
      "MultiChallenge Self-Coherence",
      "MultiChallenge Average",
      "CodeAct Accuracy",
      "SweBench-Lite % Resolved",
  ]

  # Sorting order for models, methods, conditions
  model_order = [
      "gemini-2.5-flash",
      "gemini-3.1-pro-preview",
      "gemini-3.5-flash",
      "grok-4.1-fast-non-reasoning",
  ]
  method_order = ["react"]
  condition_order = [
      "baseline",
      "w_pg_baseline",
      "static_onetime/100_samples",
      "static_incremental/100_samples",
      "scratch_onetime/100_samples",
      "scratch_incremental/100_samples",
  ]

  # Human-readable mapping for refinement_mode to condition name in sheet
  condition_names = {
      "baseline": "w/o PG",
      "w_pg_baseline": "w/ general PG",
      "static_onetime/100_samples": "w/ offline static PG",
      "static_incremental/100_samples": "w/ online static PG",
      "scratch_onetime/100_samples": "w/ offline scratch PG",
      "scratch_incremental/100_samples": "w/ online scratch PG",
  }

  last_model = None
  for m in model_order:
    m_df = final_df[final_df["model"] == m]
    if m_df.empty:
      continue

    last_method = None
    for method in method_order:
      meth_df = m_df[m_df["method"] == method]
      if meth_df.empty:
        continue

      for cond in condition_order:
        cond_df = meth_df[meth_df["refinement_mode"] == cond]
        if cond_df.empty:
          continue

        row_data = cond_df.iloc[0]

        row = {h: "" for h in headers}
        # Only print model/method name if it changes to match CSV style
        if m != last_model:
          row["LLM Model"] = m
          last_model = m
        if method != last_method:
          row["Methods"] = method.upper()
          last_method = method

        row[""] = condition_names.get(cond, cond)

        # Fill in MultiChallenge scores
        for orig, target in axis_mapping.items():
          val = row_data.get(orig)
          if pd.notnull(val):
            row[target] = f"{val:.4f}"

        output_rows.append(row)

  output_csv_path = os.path.join(
      script_dir, "results_0609/compiled_experiment_results_0609.csv"
  )
  if not os.path.exists(os.path.dirname(output_csv_path)):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
  with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    # Write first row with grouping titles like Sheet2.csv
    # In Sheet2.csv, row 1 is: LLM Model,Methods,,"HotPotQA (1000 samples, 7405 in total)",,,,,,MultiChallenge (100/166),,,,,"CodeAct (50 samples, 7139 in total)",SweBench-Lite (100/200)
    # We will write this metadata row manually
    writer.writerow({
        "LLM Model": "LLM Model",
        "Methods": "Methods",
        "": "",
        "HotPotQA Answer EM": "HotPotQA (1000 samples)",
        "MultiChallenge Instruction Retention": "MultiChallenge (166)",
        "CodeAct Accuracy": "CodeAct (50 samples)",
        "SweBench-Lite % Resolved": "SweBench-Lite (200)",
    })
    writer.writerows(output_rows)

  print(f"🎉 Big table CSV written successfully to: {output_csv_path}")

  # Print markdown version
  print("\n### Procedural Graph Evaluation Big Table (0609 Run)")
  # Format DataFrame for printing
  try:
    print(pd.DataFrame(output_rows).to_markdown(index=False))
  except Exception:
    df_print = pd.DataFrame(output_rows)
    print(df_print.to_string(index=False))


if __name__ == "__main__":
  app.run(main)
