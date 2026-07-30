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

def calculate_val_f1_for_generation(results_dir, mode_prefix, gen):
  pattern = os.path.join(
      results_dir,
      f"hotpotqa/gemini-3.5-flash/react/pg_construction_comparison/{mode_prefix}_val_gen{gen}/**/trajectory.json"
  )
  if gen == 0:
    files = glob.glob(pattern, recursive=True)
    if not files:
      pattern_alt = os.path.join(
          results_dir,
          f"hotpotqa/gemini-3.5-flash/react/pg_construction_comparison/{mode_prefix}_val_gen0/**/trajectory.json"
      )
      files = glob.glob(pattern_alt, recursive=True)
  else:
    gen_str = f"{gen:02d}" if gen < 10 else str(gen)
    pattern = os.path.join(
        results_dir,
        f"hotpotqa/gemini-3.5-flash/react/pg_construction_comparison/{mode_prefix}_val_gen{gen_str}/**/trajectory.json"
    )
    files = glob.glob(pattern, recursive=True)
    if not files:
      pattern_alt = os.path.join(
          results_dir,
          f"hotpotqa/gemini-3.5-flash/react/pg_construction_comparison/{mode_prefix}_val_gen{gen}/**/trajectory.json"
      )
      files = glob.glob(pattern_alt, recursive=True)

  if not files:
    return None

  f1_sum = 0.0
  total_samples = 0

  for fpath in files:
    try:
      with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
      traj = data.get("trajectory", [])
      if not traj:
        continue
      
      success = data.get("metrics", {}).get("success", 0.0)

      last_obs = traj[-1]
      match = re.search(r"Answer submitted:\s*(.*?)\.\s*Ground truth:\s*(.*?)\.\s*Score:", last_obs)
      if match:
        pred = match.group(1).strip()
        gt = match.group(2).strip()
        f1 = compute_f1(pred, gt)
        f1_sum += f1
      else:
        f1_sum += success
      total_samples += 1
    except Exception as e:
      pass

  return f1_sum / total_samples if total_samples > 0 else None

def compile_token_and_parsing_stats(results_dir, dataset, run_name):
  pattern = os.path.join(
      results_dir,
      f"{dataset}/gemini-3.5-flash/react/pg_construction_comparison/{run_name}/**/trajectory.json"
  )
  files = glob.glob(pattern, recursive=True)
  if not files:
    return None

  total_samples = 0
  total_token_cost = 0.0
  total_steps = 0.0
  total_parsing_failures = 0

  for fpath in files:
    try:
      with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
      
      traj = data.get("trajectory", [])
      if not traj:
        continue

      total_samples += 1
      
      metrics = data.get("metrics", {})
      total_token_cost += metrics.get("token_cost", 0.0)
      total_steps += metrics.get("steps", 0.0)
      
      for obs in traj:
        obs_lower = obs.lower()
        if "failed to parse" in obs_lower or "error parsing action" in obs_lower or "invalid syntax" in obs_lower or "unterminated string literal" in obs_lower or "nonetype" in obs_lower:
          total_parsing_failures += 1

    except Exception as e:
      print(f"Error reading {fpath}: {e}")

  avg_token_cost = total_token_cost / total_samples if total_samples > 0 else 0.0
  avg_steps = total_steps / total_samples if total_samples > 0 else 0.0
  avg_parsing_failures = total_parsing_failures / total_samples if total_samples > 0 else 0.0

  return {
      "Run Name": run_name,
      "Samples": total_samples,
      "Avg Tokens/Sample": avg_token_cost,
      "Avg Steps/Sample": avg_steps,
      "Avg Parsing Failures/Sample": avg_parsing_failures,
      "Total Parsing Failures": total_parsing_failures
  }

def main():
  results_dir = os.environ.get("PG_RESULTS_DIR", "results")
  hotpotqa_results_dir = os.path.join(results_dir, "hotpotqa_pg_construction_comparison_1000")
  multichallenge_results_dir = os.path.join(results_dir, "multichallenge_pg_construction_comparison")

  print("=== Part 1: HotpotQA Validation F1 Scores by Generation ===")
  print("Gen | Mode 3 (Expert-Guided) F1 | Mode 5 (From-Scratch) F1")
  print("---------------------------------------------------------")
  for gen in range(11):
    f1_m3 = calculate_val_f1_for_generation(hotpotqa_results_dir, "mode3", gen)
    f1_m5 = calculate_val_f1_for_generation(hotpotqa_results_dir, "mode5", gen)
    
    val_m3_str = f"{f1_m3:.2%}" if f1_m3 is not None else "N/A"
    val_m5_str = f"{f1_m5:.2%}" if f1_m5 is not None else "N/A"
    print(f"{gen:<3} | {val_m3_str:<25} | {val_m5_str}")

  print("\n=== Part 2: Token and Parsing Stats Comparison ===")
  
  print("\n[HotpotQA Stats on Test Set (1,000 samples)]")
  hq_runs = ["baseline_react_test", "mode1_test", "mode2_test", "mode3_test", "mode4_test", "mode5_test"]
  hq_stats = []
  for r in hq_runs:
    stats = compile_token_and_parsing_stats(hotpotqa_results_dir, "hotpotqa", r)
    if stats:
      hq_stats.append(stats)
  print(pd.DataFrame(hq_stats).to_string(index=False))

  print("\n[MultiChallenge Stats on Test Set (56 samples)]")
  mc_runs = ["baseline_react_test", "mode1_test", "mode2_test", "mode3_test", "mode4_test", "mode5_test"]
  mc_stats = []
  for r in mc_runs:
    stats = compile_token_and_parsing_stats(multichallenge_results_dir, "multichallenge", r)
    if stats:
      mc_stats.append(stats)
  print(pd.DataFrame(mc_stats).to_string(index=False))

if __name__ == "__main__":
  main()
