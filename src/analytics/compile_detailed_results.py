#!/usr/bin/env python3
import glob
import json
import os
import re
from collections import Counter
import pandas as pd

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    import string
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0.0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def compute_em(prediction, ground_truth):
  return 1.0 if (normalize_answer(prediction) == normalize_answer(ground_truth)) else 0.0

def analyze_hotpotqa(results_dir):
  print("Processing HotpotQA...")
  run_names = ["baseline_react_test", "mode1_test", "mode2_test", "mode3_test", "mode4_test", "mode5_test"]
  summary_data = []

  for run_name in run_names:
    pattern = os.path.join(
        results_dir,
        f"hotpotqa/gemini-3.5-flash/react/pg_construction_comparison/{run_name}/**/trajectory.json"
    )
    files = glob.glob(pattern, recursive=True)
    if not files:
      print(f"  Warning: No files found for {run_name}")
      continue

    total_samples = 0
    em_sum = 0.0
    f1_sum = 0.0
    success_sum = 0.0

    for fpath in files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          data = json.load(f)
        
        traj = data.get("trajectory", [])
        if not traj:
          continue

        total_samples += 1
        # Success from metrics
        success = data.get("metrics", {}).get("success", 0.0)
        success_sum += success

        # Parse pred and gt from last observation
        last_obs = traj[-1]
        match = re.search(r"Answer submitted:\s*(.*?)\.\s*Ground truth:\s*(.*?)\.\s*Score:", last_obs)
        if match:
          pred = match.group(1).strip()
          gt = match.group(2).strip()
          em = compute_em(pred, gt)
          f1 = compute_f1(pred, gt)
          em_sum += em
          f1_sum += f1
        else:
          # If match fails, fallback to score as EM and F1
          em_sum += success
          f1_sum += success
      except Exception as e:
        print(f"Error reading {fpath}: {e}")

    if total_samples > 0:
      avg_em = em_sum / total_samples
      avg_f1 = f1_sum / total_samples
      avg_success = success_sum / total_samples
      print(f"  {run_name}: Samples={total_samples}, Success={avg_success:.2%}, EM={avg_em:.2%}, F1={avg_f1:.2%}")
      summary_data.append({
          "Run Name": run_name,
          "Samples": total_samples,
          "Success Rate": avg_success,
          "Ans EM": avg_em,
          "Ans F1": avg_f1
      })
    else:
      print(f"  {run_name}: No valid samples processed")

  df = pd.DataFrame(summary_data)
  return df

def analyze_multichallenge(results_dir, test_parquet_path):
  print("Processing MultiChallenge...")
  df_test = pd.read_parquet(test_parquet_path)
  
  run_names = ["baseline_react_test", "mode1_test", "mode2_test", "mode3_test", "mode4_test", "mode5_test"]
  summary_data = []

  # We will group by axis
  axes = sorted(df_test["axis"].unique().tolist())
  print(f"  Detected axes: {axes}")

  for run_name in run_names:
    pattern = os.path.join(
        results_dir,
        f"multichallenge/gemini-3.5-flash/react/pg_construction_comparison/{run_name}/**/trajectory.json"
    )
    files = glob.glob(pattern, recursive=True)
    if not files:
      print(f"  Warning: No files found for {run_name}")
      continue

    results_by_sample = {}
    for fpath in files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          data = json.load(f)
        sample_id = data.get("sample_id")
        if not sample_id:
          continue
        
        # sample_id is like "sample_001"
        idx = int(sample_id.split("_")[1]) - 1
        success = data.get("metrics", {}).get("success", 0.0)
        results_by_sample[idx] = success
      except Exception as e:
        print(f"Error reading {fpath}: {e}")

    # Calculate overall and per-axis success
    mode_results = {"Run Name": run_name}
    
    # Overall
    overall_success = []
    for idx, success in results_by_sample.items():
      overall_success.append(success)
    mode_results["Overall"] = sum(overall_success) / len(overall_success) if overall_success else 0.0

    # Per axis
    for axis in axes:
      axis_indices = df_test[df_test["axis"] == axis].index.tolist()
      axis_success = []
      for idx in axis_indices:
        if idx in results_by_sample:
          axis_success.append(results_by_sample[idx])
      mode_results[axis] = sum(axis_success) / len(axis_success) if axis_success else 0.0

    summary_data.append(mode_results)
    print(f"  {run_name}: Overall={mode_results['Overall']:.2%}")
    for axis in axes:
      print(f"    {axis}={mode_results[axis]:.2%}")

  df = pd.DataFrame(summary_data)
  return df


def analyze_memory_runs(runs_dict):
  summary_data = []
  for run_name, path in runs_dict.items():
    files = glob.glob(os.path.join(path, "**/trajectory.json"), recursive=True)
    if not files:
      print(f"  Warning: No files found for {run_name} at {path}")
      continue

    total_samples = 0
    em_sum = 0.0
    f1_sum = 0.0
    success_sum = 0.0
    steps_sum = 0.0
    tokens_sum = 0.0

    for fpath in files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          data = json.load(f)
        
        traj = data.get("trajectory", [])
        if not traj:
          continue

        total_samples += 1
        metrics = data.get("metrics", {})
        success = metrics.get("success", 0.0)
        success_sum += success
        steps_sum += metrics.get("steps", 0.0)
        tokens_sum += metrics.get("token_cost", 0.0)

        last_obs = traj[-1]
        match = re.search(r"Answer submitted:\s*(.*?)\.\s*Ground truth:\s*(.*?)\.\s*Score:", last_obs)
        if match:
          pred = match.group(1).strip()
          gt = match.group(2).strip()
          em = compute_em(pred, gt)
          f1 = compute_f1(pred, gt)
          em_sum += em
          f1_sum += f1
        else:
          em_sum += success
          f1_sum += success
      except Exception as e:
        pass

    if total_samples > 0:
      summary_data.append({
          "Method": run_name,
          "Samples": total_samples,
          "Success Rate": success_sum / total_samples,
          "Ans EM": em_sum / total_samples,
          "Ans F1": f1_sum / total_samples,
          "Avg Steps": steps_sum / total_samples,
          "Avg Tokens": tokens_sum / total_samples,
      })
  return pd.DataFrame(summary_data)


def analyze_multichallenge_memory_runs(runs_dict, test_parquet_path):
  df_test = pd.read_parquet(test_parquet_path)
  axes = sorted(df_test["axis"].unique().tolist())
  summary_data = []

  for run_name, path in runs_dict.items():
    files = glob.glob(os.path.join(path, "**/trajectory.json"), recursive=True)
    if not files:
      print(f"  Warning: No files found for {run_name} at {path}")
      continue

    results_by_sample = {}
    steps_by_sample = {}
    tokens_by_sample = {}

    for fpath in files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          data = json.load(f)
        sample_id = data.get("sample_id")
        if not sample_id:
          continue
        
        idx = int(sample_id.split("_")[1]) - 1
        metrics = data.get("metrics", {})
        results_by_sample[idx] = metrics.get("success", 0.0)
        steps_by_sample[idx] = metrics.get("steps", 0.0)
        tokens_by_sample[idx] = metrics.get("token_cost", 0.0)
      except Exception as e:
        pass

    mode_results = {
        "Method": run_name,
        "Samples": len(results_by_sample),
        "Overall": sum(results_by_sample.values()) / len(results_by_sample) if results_by_sample else 0.0,
        "Avg Steps": sum(steps_by_sample.values()) / len(steps_by_sample) if steps_by_sample else 0.0,
        "Avg Tokens": sum(tokens_by_sample.values()) / len(tokens_by_sample) if tokens_by_sample else 0.0,
    }

    for axis in axes:
      axis_indices = df_test[df_test["axis"] == axis].index.tolist()
      axis_success = [results_by_sample[idx] for idx in axis_indices if idx in results_by_sample]
      mode_results[axis] = sum(axis_success) / len(axis_success) if axis_success else 0.0

    summary_data.append(mode_results)
  return pd.DataFrame(summary_data)


def analyze_cfo_runs(runs_dict):
  summary_data = []
  for run_name, path in runs_dict.items():
    files = glob.glob(os.path.join(path, "**/trajectory.json"), recursive=True)
    if not files:
      print(f"  Warning: No files found for {run_name} at {path}")
      continue

    total_samples = 0
    success_sum = 0.0
    steps_sum = 0.0
    tokens_sum = 0.0
    latency_sum = 0.0
    survived_sum = 0.0
    score_m_sum = 0.0
    crisis1_sum = 0.0
    crisis2_sum = 0.0
    crisis3_sum = 0.0
    tools_per_mo_sum = 0.0
    actions_sum = 0.0
    raised_m_sum = 0.0

    for fpath in files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          data = json.load(f)
        
        metrics = data.get("metrics", {})
        if not metrics:
          continue

        total_samples += 1
        success_sum += metrics.get("success", 0.0)
        steps_sum += metrics.get("steps", 0.0)
        tokens_sum += metrics.get("token_cost", 0.0)
        latency_sum += metrics.get("latency", 0.0)
        survived_sum += metrics.get("cfo_survived", 0.0)
        score_m_sum += metrics.get("cfo_score_m", 0.0)
        crisis1_sum += metrics.get("cfo_crisis1", 0.0)
        crisis2_sum += metrics.get("cfo_crisis2", 0.0)
        crisis3_sum += metrics.get("cfo_crisis3", 0.0)
        tools_per_mo_sum += metrics.get("cfo_tools_per_mo", 0.0)
        actions_sum += metrics.get("cfo_actions", 0.0)
        raised_m_sum += metrics.get("cfo_raised_m", 0.0)
      except Exception as e:
        print(f"Error reading {fpath}: {e}")

    if total_samples > 0:
      summary_data.append({
          "Experiment ID": run_name,
          "Samples": f"{total_samples}/{total_samples}",
          "Success Rate": f"{success_sum / total_samples:.1%}",
          "Avg Steps": f"{steps_sum / total_samples:.2f}",
          "Avg Token Cost": f"{tokens_sum / total_samples:.1f}",
          "Avg Latency (s)": f"{latency_sum / total_samples:.2f}",
          "Full Surv. %": f"{survived_sum / total_samples:.1%}",
          "Avg. Score ($M)": f"${score_m_sum / total_samples:.2f}M",
          "1st Crisis %": f"{crisis1_sum / total_samples:.1%}",
          "2nd Crisis %": f"{crisis2_sum / total_samples:.1%}",
          "3rd Crisis %": f"{crisis3_sum / total_samples:.1%}",
          "Tools/Mo": f"{tools_per_mo_sum / total_samples:.2f}",
          "Actions": f"{actions_sum / total_samples:.1f}",
          "Raised ($M)": f"${raised_m_sum / total_samples:.2f}M",
      })
  return pd.DataFrame(summary_data)


def update_cfo_dashboard(workspace_dir):
  print("\n=== Updating CFO Dashboard ===")
  results_dir = os.path.join(
      workspace_dir,
      "results/cfo_generative_pg_memory_comparison",
  )
  dashboard_path = os.path.join(results_dir, "dashboard.csv")

  if not os.path.exists(dashboard_path):
    print(f"Error: {dashboard_path} not found.")
    return

  new_runs = {
      "exp_cfo_anthropic-claude-sonnet-4-6_react_baseline_comp_cfo_claude": (
          os.path.join(
              results_dir,
              "cfo/anthropic-claude-sonnet-4-6/react/comp_cfo_claude/baseline",
          )
      ),
      "exp_cfo_anthropic-claude-sonnet-4-6_react_memory_summarization_unknown_samples_comp_cfo_claude": (
          os.path.join(
              results_dir,
              "cfo/anthropic-claude-sonnet-4-6/react/comp_cfo_claude/memory_summarization",
          )
      ),
  }

  df_new = analyze_cfo_runs(new_runs)
  if df_new.empty:
    print("No new CFO results analyzed.")
    return

  print("New CFO Results:")
  print(df_new.to_string(index=False))

  df_existing = pd.read_csv(dashboard_path)

  # Update or append
  df_existing.set_index("Experiment ID", inplace=True)
  df_new.set_index("Experiment ID", inplace=True)

  for idx, row in df_new.iterrows():
    df_existing.loc[idx] = row

  df_existing.reset_index(inplace=True)

  df_existing.to_csv(dashboard_path, index=False)
  print(f"Updated {dashboard_path} successfully.")


def main():
  workspace_dir = os.environ.get(
      "PG_WORKSPACE_DIR",
      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
  )

  # HotpotQA
  hotpotqa_results_dir = os.path.join(
      workspace_dir,
      "results/hotpotqa_pg_construction_comparison_1000",
  )
  df_hotpotqa = analyze_hotpotqa(hotpotqa_results_dir)
  print("\nHotpotQA Summary Table:")
  print(df_hotpotqa.to_string(index=False))

  # MultiChallenge
  multichallenge_results_dir = os.path.join(
      workspace_dir,
      "results/multichallenge_pg_construction_comparison",
  )
  test_parquet_path = os.path.join(
      workspace_dir,
      "data/datasets/General/MultiChallenge/data/rich-self-evolution/test_fast_split.parquet",
  )
  df_multi = analyze_multichallenge(
      multichallenge_results_dir, test_parquet_path
  )
  print("\nMultiChallenge Summary Table:")
  print(df_multi.to_string(index=False))

  # HotpotQA Memory Comparison
  print("\n=== HotpotQA Memory Comparison Table (gemini-3.5-flash) ===")
  hotpotqa_memory_runs = {
      "Baseline (w/o memory)": os.path.join(
          workspace_dir,
          "results/hotpotqa_generative_pg_self_evolution_1000/hotpotqa/gemini-3.5-flash/react/generative_pg_self_evolution/baseline_val",
      ),
      "Memory Retrieval": os.path.join(
          workspace_dir,
          "results/hotpotqa_generative_pg_memory_comparison_1000/hotpotqa/gemini-3.5-flash/react/comp_1000/memory_retrieval",
      ),
      "Memory Summarization": os.path.join(
          workspace_dir,
          "results/hotpotqa_generative_pg_memory_comparison_1000/hotpotqa/gemini-3.5-flash/react/comp_1000/memory_summarization",
      ),
      "Procedural Graph (Ours)": os.path.join(
          workspace_dir,
          "results/hotpotqa_generative_pg_memory_comparison_1000/hotpotqa/gemini-3.5-flash/react/comp_1000/generative_pg_val",
      ),
  }
  df_hq_mem = analyze_memory_runs(hotpotqa_memory_runs)
  print(df_hq_mem.to_string(index=False))

  # MultiChallenge Memory Comparison
  print("\n=== MultiChallenge Memory Comparison Table (gemini-3.5-flash) ===")
  multichallenge_memory_runs = {
      "Baseline (w/o memory)": os.path.join(
          workspace_dir,
          "results/multichallenge_pg_construction_comparison/multichallenge/gemini-3.5-flash/react/pg_construction_comparison/baseline_react_test",
      ),
      "Memory Retrieval": os.path.join(
          workspace_dir,
          "results/multichallenge_rich_pg_memory_comparison/multichallenge/gemini-3.5-flash/react/20260615_233404/memory_retrieval",
      ),
      "Memory Summarization": os.path.join(
          workspace_dir,
          "results/multichallenge_rich_pg_memory_comparison/multichallenge/gemini-3.5-flash/react/20260615_233404/memory_summarization",
      ),
      "Procedural Graph (Ours)": os.path.join(
          workspace_dir,
          "results/multichallenge_rich_pg_memory_comparison/multichallenge/gemini-3.5-flash/react/20260615_233404/scratch_onetime",
      ),
  }
  df_mc_mem = analyze_multichallenge_memory_runs(
      multichallenge_memory_runs, test_parquet_path
  )
  print(df_mc_mem.to_string(index=False))

  # CFO Dashboard Update
  update_cfo_dashboard(workspace_dir)


if __name__ == "__main__":
  main()
