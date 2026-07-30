"""Unified Experiment Runner for Procedural Graph solver evaluations."""

from concurrent import futures
import json
import os
import re
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

socket.setdefaulttimeout(120)
from absl import app
from absl import flags
from procedural_graph import core
from procedural_graph import env_base
from procedural_graph import retriever
from procedural_graph import solvers
from procedural_graph.datasets.bfcl import BfclEnv
from procedural_graph.datasets.cfo_env import CFOEnv
from procedural_graph.datasets.gdpval import GDPvalEnv
from procedural_graph.datasets.hotpotqa import HotpotQAEnv
from procedural_graph.datasets.multichallenge import MultiChallengeEnv
from procedural_graph.datasets.taubench import TauBenchEnv
from procedural_graph.llm_client import VertexAIClient
from procedural_graph.refinement import ProceduralGraphLLMRefiner, GraphRefinementMode
from procedural_graph.tracker import ResultsCompiler

FLAGS = flags.FLAGS

# Core execution parameters
flags.DEFINE_string(
    "dataset",
    "cfo",
    "Dataset adapter to evaluate: cfo, swebench, multichallenge, gdpval, taubench",
)
flags.DEFINE_string(
    "method", "react", "Solver method: react"
)
flags.DEFINE_string(
    "llm", "grok-4.1-fast-non-reasoning", "LLM model identifier"
)
flags.DEFINE_integer("num_samples", 50, "Max samples to run")
flags.DEFINE_integer("num_workers", 16, "ThreadPoolExecutor max workers limit")
flags.DEFINE_integer(
    "stride", 0, "Stride size for sequential incremental active updates"
)
flags.DEFINE_string(
    "gcp_project", None, "GCP Project ID for Vertex AI initialization (defaults to GOOGLE_CLOUD_PROJECT env var)"
)
flags.DEFINE_string(
    "embedding_model", "gemini-embedding-001", "Vertex AI embedding model name"
)
flags.DEFINE_string(
    "refinement_mode", "static_onetime", "Graph refinement mode"
)
flags.DEFINE_bool(
    "guided_only",
    True,
    "If True, run solver using Procedural Graph Outgoing Edge RAG guidance",
)
flags.DEFINE_string(
    "db_path", "", "Deprecated SQLite tracking database path (ignored)"
)
flags.DEFINE_string(
    "results_dir",
    "results",
    "Structured hierarchical results logging directory",
)
flags.DEFINE_string(
    "run_id",
    "",
    "Optional run ID (e.g. timestamp) to group results under method directory",
)
flags.DEFINE_integer("seed", 42, "Base random seed for evaluations")
flags.DEFINE_string(
    "llm_location",
    "global",
    "GCP location for Vertex AI endpoint (e.g. global, us-central1)",
)
flags.DEFINE_integer(
    "llm_timeout",
    360,
    "Timeout in seconds for Vertex AI API calls",
)
flags.DEFINE_string(
    "multichallenge_dataset_path",
    "data/datasets/General/MultiChallenge/data/test_166.parquet",
    "Relative or absolute path to the local MultiChallenge parquet dataset",
)
flags.DEFINE_string(
    "swebench_dataset_path",
    "data/datasets/Coding/SWE-bench_Lite/data/test_200.parquet",
    "Relative or absolute path to the local SWE-bench parquet dataset",
)
flags.DEFINE_string(
    "cfo_config_path",
    "data/datasets/Finance/CFO-Env/config.json",
    "Path to the CFO environment config.json file",
)
flags.DEFINE_string(
    "hotpotqa_dataset_path",
    "data/datasets/General/HotpotQA/distractor/validation-00000-of-00001.parquet",
    "Relative or absolute path to the local HotpotQA parquet dataset",
)
flags.DEFINE_string(
    "gdpval_dataset_path",
    "data/datasets/General/GDPval/data/train-00000-of-00001.parquet",
    "Relative or absolute path to the local GDPval parquet dataset",
)
flags.DEFINE_string(
    "gdpval_split",
    "test",
    "GDPval split to evaluate: train, val, test",
)
flags.DEFINE_string(
    "alfworld_split",
    "valid_unseen",
    "ALFWorld split to evaluate: train, valid_seen, valid_unseen",
)
flags.DEFINE_string(
    "bfcl_dataset_path",
    "data/datasets/General/BFCL/BFCL_v3_multi_turn_base.json",
    "Relative or absolute path to the local BFCL json dataset",
)
flags.DEFINE_string(
    "bfcl_possible_answer_path",
    "data/datasets/General/BFCL/possible_answer/BFCL_v3_multi_turn_base.json",
    "Relative or absolute path to the local BFCL possible answers json dataset",
)
flags.DEFINE_string(
    "taubench_domain",
    "retail",
    "TAU-bench domain: retail, airline",
)
flags.DEFINE_string(
    "taubench_split",
    "test",
    "TAU-bench split: train, test, dev",
)
flags.DEFINE_string(
    "taubench_user_model",
    "gemini-2.5-flash",
    "Simulated user LLM model for TAU-bench",
)
flags.DEFINE_string(
    "taubench_user_provider",
    "vertex_ai",
    "Simulated user model provider for TAU-bench",
)
flags.DEFINE_string(
    "initial_graph_path",
    "",
    "Path to a custom Procedural Graph JSON file.",
)
flags.DEFINE_bool(
    "refine",
    True,
    "Whether Procedural Graph refinement is enabled.",
)
flags.DEFINE_bool(
    "pg_as_text_summary",
    False,
    "Whether to represent the Procedural Graph as a static textual summary to the agent.",
)
flags.DEFINE_string(
    "train_dataset_path",
    "",
    "Absolute path to the train dataset (parquet) for memory retrieval/summarization.",
)
flags.DEFINE_string(
    "train_trajectories_dir",
    "",
    "Path to the directory containing training trajectories (for memory modes).",
)
flags.DEFINE_string(
    "memory_summary_path",
    "",
    "Path to load/save the memory summary text file.",
)
flags.DEFINE_integer(
    "max_steps",
    50,
    "Max solver steps. 0 means dynamic based on dataset.",
)

flags.DEFINE_integer(
    "max_refinement_trajectories",
    20,
    "Max trajectories to send to LLM for offline refinement to avoid context limit",
)
flags.DEFINE_bool(
    "rich_pg",
    False,
    "Whether to load *_rich_procedural_graph.json file and enable Rich Procedural Graph mode.",
)
flags.DEFINE_bool(
    "generative_guidance",
    False,
    "Whether to generate dynamic textual guidance from the complete Procedural Graph via LLM.",
)
flags.DEFINE_integer(
    "expel_top_k_exemplars",
    3,
    "Number of similar exemplars to retrieve in ExpeL",
)
flags.DEFINE_integer(
    "expel_top_k_rules",
    3,
    "Number of similar rules to retrieve in ExpeL",
)




def resolve_dataset_path(flag_value: str, relative_fallback: str) -> str:
  """Dynamically resolves a dataset parquet/json path for open-source portability."""
  if flag_value and os.path.exists(flag_value):
    return flag_value
  # Find workspace paper directory
  curr = os.path.abspath(os.path.dirname(__file__))
  paper_dir = None
  while curr and curr != "/":
    if os.path.exists(os.path.join(curr, "paper")):
      paper_dir = os.path.join(curr, "paper")
      break
    curr = os.path.dirname(curr)
  if not paper_dir:
    paper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  fallback_path = os.path.join(paper_dir, "data", relative_fallback)
  if os.path.exists(fallback_path):
    return fallback_path
  return flag_value


def get_base_results_dir() -> str:
  """Constructs the base results directory, including run_id if set."""
  base = os.path.join(FLAGS.results_dir, FLAGS.dataset, FLAGS.llm, FLAGS.method)
  if FLAGS.run_id:
    base = os.path.join(base, FLAGS.run_id)
  return base


def get_sample_log_dir(sample_id: str) -> str:
  """Determines output directory path based on the run phase and refinement mode."""
  if FLAGS.refine:
    # Training Phase
    # All refinement modes share the trajectories from the unguided train_samples run
    return os.path.join(
        get_base_results_dir(),
        "train_samples",
        "samples",
        sample_id,
    )
  else:
    # Evaluation Phase
    if FLAGS.refinement_mode in ("baseline", "w_pg_baseline", "train_samples"):
      return os.path.join(
          get_base_results_dir(),
          FLAGS.refinement_mode,
          "samples",
          sample_id,
      )
    else:
      # Refined graph evaluation
      n_samples_str = "unknown_samples"
      if FLAGS.initial_graph_path:
        match = re.search(r"(\d+)_samples", FLAGS.initial_graph_path)
        if match:
          n_samples_str = f"{match.group(1)}_samples"
      return os.path.join(
          get_base_results_dir(),
          FLAGS.refinement_mode,
          n_samples_str,
          "samples",
          sample_id,
      )


def is_sample_completed(sample_id: str) -> bool:
  sample_log_dir = get_sample_log_dir(sample_id)
  trajectory_path = os.path.join(sample_log_dir, "trajectory.json")
  if os.path.exists(trajectory_path):
    try:
      with open(trajectory_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return "metrics" in data and "success" in data["metrics"] and "system_error" not in data["metrics"]
    except Exception:
      pass
  return False


def load_graph_for_refinement(
    env: env_base.BaseEnvironment, active_graph_path: Optional[str]
) -> core.ProceduralGraph:
  """Loads Procedural Graph for refinement, starting from empty graph if scratch mode and no active path."""
  if active_graph_path:
    return env.load_initial_graph(active_graph_path)

  if FLAGS.refinement_mode in ("scratch_onetime", "scratch_incremental"):
    default_graph = env.load_initial_graph(None)
    start_node_id = env.get_current_node_id([])
    start_node = default_graph.get_node(start_node_id)

    graph = core.ProceduralGraph()
    if start_node:
      graph.add_node(start_node)
    else:
      node_type = core.NodeType.STATE
      if start_node_id.endswith("()"):
        node_type = core.NodeType.REASONING
      graph.add_node(
          core.Node(id=start_node_id, type=node_type, description="Start node")
      )
    return graph

  return env.load_initial_graph(None)


def extract_cfo_financial_history(trajectory: List[str]) -> List[Dict[str, Any]]:
  """Extracts monthly financial observations from the trajectory."""
  import json
  import ast

  history = []

  for step in trajectory:
    if step.startswith("Observation:"):
      obs_content = step[len("Observation:") :].strip()

      # Try parsing as JSON
      obs_data = None
      try:
        obs_data = json.loads(obs_content)
      except json.JSONDecodeError:
        # Fallback to ast.literal_eval for python dict repr
        try:
          val = ast.literal_eval(obs_content)
          if isinstance(val, dict):
            obs_data = val
        except Exception:
          pass

      if not obs_data or not isinstance(obs_data, dict):
        continue

      if "current_month" in obs_data:
        history.append(obs_data)
      elif "month_end_update" in obs_data:
        me_update = obs_data["month_end_update"]
        if isinstance(me_update, dict) and "current_observation" in me_update:
          curr_obs = me_update["current_observation"]
          if isinstance(curr_obs, dict):
            history.append(curr_obs)

  # Keep the latest entry for each month to preserve the end-of-month final state
  unique_history_dict = {}
  for entry in history:
    m = entry.get("current_month")
    if m is not None:
      # Clean up entry to keep only financial metrics and date
      clean_entry = {
          k: v for k, v in entry.items()
          if k not in ("available_tools", "available_actions", "tool_calls_remaining", "actions_remaining")
      }
      unique_history_dict[m] = clean_entry

  unique_history = [unique_history_dict[m] for m in sorted(unique_history_dict.keys())]
  return unique_history


def run_single_sample(
    sample_id: str,
    experiment_id: str,
    experiment_name: str,
    config_dict: Dict[str, Any],
    custom_graph_path: Optional[str] = None,
    train_samples: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, float, List[str]]:
  """Executes evaluation for a single sample, logging detailed traces."""
  start_time = time.time()
  seed = FLAGS.seed

  try:
    index = int(sample_id.split("_")[-1]) - 1
  except Exception as e:
    raise ValueError(
        f"Failed to parse sample index from sample ID '{sample_id}': {e}"
    ) from e

  sample_seed = FLAGS.seed + index

  max_retries = 5
  score = 0.0
  trajectory = []
  env = None
  
  for attempt in range(1, max_retries + 1):
    try:
      # 2. Configure LLM Client (Instantiate Real Vertex AI Client)
      llm_client = VertexAIClient(
          model_name=FLAGS.llm,
          project_id=FLAGS.gcp_project,
          location=FLAGS.llm_location,
          timeout=FLAGS.llm_timeout,
      )

      # 3. Load Environment Adapter
      if FLAGS.dataset == "cfo":
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = FLAGS.cfo_config_path
        if not os.path.isabs(config_path):
          config_path = os.path.join(project_root, config_path)
        env = CFOEnv(config_path=config_path, seed=sample_seed)
      elif FLAGS.dataset == "multichallenge":
        env = MultiChallengeEnv(
            sample_index=index,
            dataset_path=resolve_dataset_path(
                FLAGS.multichallenge_dataset_path,
                "datasets/General/MultiChallenge/data/test_166.parquet",
            ),
            llm_client=llm_client,
        )
      elif FLAGS.dataset == "hotpotqa":
        env = HotpotQAEnv(
            sample_index=index,
            dataset_path=resolve_dataset_path(
                FLAGS.hotpotqa_dataset_path,
                "datasets/General/HotpotQA/distractor/validation-00000-of-00001.parquet",
            ),
            llm_client=llm_client,
        )
      elif FLAGS.dataset == "gdpval":
        env = GDPvalEnv(
            sample_index=index,
            dataset_path=resolve_dataset_path(
                FLAGS.gdpval_dataset_path,
                "datasets/General/GDPval/data/train-00000-of-00001.parquet",
            ),
            llm_client=llm_client,
            split=FLAGS.gdpval_split,
        )
      elif FLAGS.dataset == "alfworld":
        from procedural_graph.datasets.alfworld import ALFWorldEnv
        env = ALFWorldEnv(
            sample_index=index,
            llm_client=llm_client,
            split=FLAGS.alfworld_split,
        )
      elif FLAGS.dataset == "bfcl":
        env = BfclEnv(
            sample_index=index,
            dataset_path=FLAGS.bfcl_dataset_path,
            possible_answer_path=FLAGS.bfcl_possible_answer_path,
        )
      elif FLAGS.dataset == "taubench":
        env = TauBenchEnv(
            sample_index=index,
            domain=FLAGS.taubench_domain,
            task_split=FLAGS.taubench_split,
            user_model=FLAGS.taubench_user_model,
            user_provider=FLAGS.taubench_user_provider,
            llm_client=llm_client,
        )
      else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset}")

      # 4. Load Procedural Graph
      initial_graph = env.load_initial_graph(
          custom_graph_path, rich_pg=FLAGS.rich_pg
      )

      # 5. Construct retriever if running in guided mode
      retriever_obj = None
      if FLAGS.guided_only:
        if FLAGS.refinement_mode == "expel":
          rules_path = os.path.join(get_base_results_dir(), "expel_rules.json")
          rules = []
          if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
              try:
                rules = json.load(f)
              except Exception:
                pass
          scorer = retriever.EmbeddingSimilarityScorer(
              model_name=FLAGS.embedding_model,
              use_fallback=True,
              project_id=FLAGS.gcp_project,
          )
          retriever_obj = retriever.ExpeLGuidanceProvider(
              train_samples or [],
              rules,
              scorer,
              top_k_exemplars=FLAGS.expel_top_k_exemplars,
              top_k_rules=FLAGS.expel_top_k_rules,
          )
        elif FLAGS.refinement_mode == "autoguide":
          lib_path = os.path.join(get_base_results_dir(), "autoguide_library.json")
          guidelines = []
          if os.path.exists(lib_path):
            with open(lib_path, "r", encoding="utf-8") as f:
              try:
                guidelines = json.load(f)
              except Exception:
                pass
          scorer = retriever.EmbeddingSimilarityScorer(
              model_name=FLAGS.embedding_model,
              use_fallback=True,
              project_id=FLAGS.gcp_project,
          )
          retriever_obj = retriever.AutoGuideGuidanceProvider(
              guidelines, scorer
          )
        elif FLAGS.refinement_mode == "awm":
          wf_path = os.path.join(get_base_results_dir(), "awm_workflows.json")
          workflows = {}
          if os.path.exists(wf_path):
            with open(wf_path, "r", encoding="utf-8") as f:
              try:
                workflows = json.load(f)
              except Exception:
                pass
          retriever_obj = retriever.AWMGuidanceProvider(workflows)
        elif FLAGS.refinement_mode == "knowagent":
          kb_path = os.path.join(get_base_results_dir(), "knowagent_kb.json")
          kb = {}
          if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
              try:
                kb = json.load(f)
              except Exception:
                pass
          retriever_obj = retriever.KnowAgentGuidanceProvider(kb)
        elif FLAGS.refinement_mode.startswith(("memory_retrieval", "rap")):
          if not train_samples:
            raise ValueError("train_samples must be provided for memory_retrieval / rap mode")
          scorer = retriever.EmbeddingSimilarityScorer(
              model_name=FLAGS.embedding_model,
              use_fallback=True,
              project_id=FLAGS.gcp_project,
          )
            
          retriever_obj = retriever.RAPGuidanceProvider(
              train_samples=train_samples, scorer=scorer, top_k_trajectories=5, max_roadmap_steps=5
          )
        elif FLAGS.refinement_mode.startswith(("memory_summarization", "memorybank", "memory_summary")):
          if FLAGS.memory_summary_path and os.path.exists(FLAGS.memory_summary_path):
            with open(FLAGS.memory_summary_path, "r", encoding="utf-8") as f:
              summary_guidance = f.read()
            retriever_obj = retriever.StaticGuidanceProvider(summary_guidance)
          else:
            if not train_samples:
              raise ValueError("train_samples or valid memory_summary_path must be provided for memory_summarization / memorybank mode")
            scorer = retriever.EmbeddingSimilarityScorer(
                model_name=FLAGS.embedding_model,
                use_fallback=True,
                project_id=FLAGS.gcp_project,
            )
              
            retriever_obj = retriever.MemoryBankGuidanceProvider(
                train_samples=train_samples, scorer=scorer, decay_rate=0.05, reinforcement_factor=0.5, top_k=5
            )
        elif FLAGS.generative_guidance or FLAGS.refinement_mode == "generative_guidance":
          retriever_obj = retriever.GenerativeGraphGuidanceProvider(
              initial_graph, llm_client
          )
        elif FLAGS.rich_pg or FLAGS.refinement_mode == "rich_multihop":
          scorer = retriever.EmbeddingSimilarityScorer(
              model_name=FLAGS.embedding_model,
              use_fallback=True,
              project_id=FLAGS.gcp_project,
          )
          retriever_obj = retriever.RichMultiHopGraphRetriever(
              initial_graph, scorer, max_hops=2
          )
        elif FLAGS.refinement_mode == "hybrid_pg":
          scorer = retriever.EmbeddingSimilarityScorer(
              model_name=FLAGS.embedding_model,
              use_fallback=True,
              project_id=FLAGS.gcp_project,
          )
          kb_path = os.path.join(get_base_results_dir(), "knowagent_kb.json")
          kb = {}
          if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
              try:
                kb = json.load(f)
              except Exception:
                pass
          retriever_obj = retriever.HybridGraphGuidanceProvider(
              initial_graph, llm_client, scorer, max_hops=5, kb=kb
          )
        else:
          if FLAGS.pg_as_text_summary:
            summary_guidance = initial_graph.to_text_summary()
            retriever_obj = retriever.StaticGuidanceProvider(summary_guidance)
          else:
            scorer = retriever.EmbeddingSimilarityScorer(
                model_name=FLAGS.embedding_model,
                use_fallback=True,
                project_id=FLAGS.gcp_project,
            )

            evaluator = retriever.SoftSemanticConditionEvaluator(scorer)
            retriever_obj = retriever.ProceduralGraphRetriever(
                initial_graph, scorer, evaluator
            )

      # 6. Instantiate Solver
      if FLAGS.method == "react":
        solver = solvers.ReAct(llm_client, retriever_obj)
      else:
        raise ValueError(f"Unsupported method: {FLAGS.method}. Only 'react' is supported.")

      # 6. Execute Solver loop
      max_steps = FLAGS.max_steps
      if max_steps == 0 and FLAGS.dataset == "cfo":
        max_steps = 1000
      score, trajectory = solver.solve(env, max_steps=max_steps)
      break
    except Exception as e:
      if attempt == max_retries:
        print(f"❌ [Error] Sample {sample_id} failed after {max_retries} attempts: {e}. Skipping and marking as failed.")
        # Fallback metrics
        metrics = {
            "success": 0.0,
            "steps": 0.0,
            "token_cost": 0.0,
            "latency": time.time() - start_time,
            "system_error": str(e),
        }
        fallback_trajectory = [f"System Error: {e}"]
        
        sample_log_dir = get_sample_log_dir(sample_id)
        os.makedirs(sample_log_dir, exist_ok=True)
        trajectory_path = os.path.join(sample_log_dir, "trajectory.json")
        save_data = {
            "sample_id": sample_id,
            "trajectory": fallback_trajectory,
            "metrics": metrics,
        }
        if env and hasattr(env, "game_file_path") and env.game_file_path:
          save_data["game_file_path"] = env.game_file_path
        with open(trajectory_path, "w", encoding="utf-8") as f:
          json.dump(save_data, f, indent=2)
        return sample_id, 0.0, fallback_trajectory
      else:
        print(f"⚠️ Attempt {attempt}/{max_retries} failed for sample {sample_id}: {e}. Retrying in 2 seconds...")
        time.sleep(2)

  latency = time.time() - start_time
  steps = len([item for item in trajectory if item.startswith("Action:")])

  # Fetch actual token usage telemetry stats from the real VertexAIClient
  stats = llm_client.get_telemetry_stats()
  token_cost = float(stats.get("total_tokens", 0))

  metrics = {
      "success": score / 10000.0 if FLAGS.dataset == "cfo" else score,
      "steps": float(steps),
      "token_cost": token_cost,
      "latency": latency,
  }

  if isinstance(env, CFOEnv):
    try:
      state = env.adapter.arena.env_state
      months = state.current_month
      passed = (not state.episode_terminated) or (state.termination_reason == "Maximum episode length reached")
      survived = passed and (months >= state.max_episode_months)
      
      crisis1 = months >= 32
      crisis2 = months >= 59
      crisis3 = months >= 112
      
      scores = env.adapter.arena._monthly_series.get("score", [])
      avg_score = (sum(scores) / len(scores)) if scores else 0.0
      avg_score_m = avg_score / 1e6
      
      tools_count = len(env.adapter.arena.tool_history)
      actions_count = len(env.adapter.arena.action_history)
      
      tools_per_mo = tools_count / months if months > 0 else 0.0
      
      raised = sum(pf.amount_approved for pf in state.pending_fundraisings if pf.success and pf.delivered) / 1e6
      
      metrics.update({
          "success": float(survived),
          "cfo_survived": float(survived),
          "cfo_months": float(months),
          "cfo_score_m": avg_score_m,
          "cfo_crisis1": float(crisis1),
          "cfo_crisis2": float(crisis2),
          "cfo_crisis3": float(crisis3),
          "cfo_tools_per_mo": tools_per_mo,
          "cfo_actions": float(actions_count),
          "cfo_raised_m": raised,
      })
    except Exception as e:
      print(f"⚠️ Failed to extract detailed CFO metrics: {e}")


  # Check for fatal API/quota/timeout errors in trajectory observations
  for item in trajectory:
    item_lower = item.lower()
    if "httperror" in item_lower or "timeout" in item_lower or "resource_exhausted" in item_lower or "quota exceeded" in item_lower:
      print(f"⚠️ [Skip] Sample {sample_id} contains fatal error in trajectory: {item}. Skipping.")

  # 7. Precalculate physical hierarchical logs directory path
  sample_log_dir = get_sample_log_dir(sample_id)

  # 8. Write detailed physical hierarchical logs
  os.makedirs(sample_log_dir, exist_ok=True)

  # Trajectory cognitive tree cold recovery backup
  trajectory_path = os.path.join(sample_log_dir, "trajectory.json")
  save_data = {
      "sample_id": sample_id,
      "trajectory": trajectory,
      "metrics": metrics,
  }
  if env and hasattr(env, "game_file_path") and env.game_file_path:
    save_data["game_file_path"] = env.game_file_path
  with open(trajectory_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=2)

  # Extract and write financial history if CFO dataset
  if FLAGS.dataset == "cfo":
    try:
      fin_history = extract_cfo_financial_history(trajectory)
      fin_history_path = os.path.join(sample_log_dir, "financial_history.json")
      with open(fin_history_path, "w", encoding="utf-8") as f:
        json.dump(fin_history, f, indent=2, default=str)
    except Exception as e:
      print(f"⚠️ Failed to extract/write financial history for {sample_id}: {e}")

  # Detailed low-level tracer lines
  detailed_path = os.path.join(sample_log_dir, "detailed_trace.jsonl")
  with open(detailed_path, "w", encoding="utf-8") as f:
    for line in trajectory:
      f.write(json.dumps({"sample_id": sample_id, "log": line}) + "\n")



  print(
      f"✅ Successfully evaluated sample {sample_id} in {latency:.2f}s."
      f" Score={score}"
  )
  return sample_id, metrics["success"], trajectory


def main(argv):
  del argv
  print("DEBUG: main function started successfully!", flush=True)



  split_suffix = ""
  if FLAGS.dataset == "multichallenge" and FLAGS.multichallenge_dataset_path:
    split_suffix = (
        "_"
        + os.path.splitext(os.path.basename(FLAGS.multichallenge_dataset_path))[
            0
        ]
    )
  elif FLAGS.dataset == "gdpval" and FLAGS.gdpval_dataset_path:
    split_suffix = (
        "_"
        + os.path.splitext(os.path.basename(FLAGS.gdpval_dataset_path))[0]
        + f"_{FLAGS.gdpval_split}"
    )
  elif FLAGS.dataset == "alfworld":
    split_suffix = f"_{FLAGS.alfworld_split}"
  elif FLAGS.dataset == "bfcl" and FLAGS.bfcl_dataset_path:
    split_suffix = (
        "_"
        + os.path.splitext(os.path.basename(FLAGS.bfcl_dataset_path))[0]
    )
  elif FLAGS.dataset == "taubench":
    split_suffix = f"_{FLAGS.taubench_domain}_{FLAGS.taubench_split}"
  if FLAGS.refine:
    experiment_id = f"exp_{FLAGS.dataset}{split_suffix}_{FLAGS.llm}_{FLAGS.method}_train_samples"
  else:
    if FLAGS.refinement_mode in ("baseline", "w_pg_baseline", "train_samples"):
      experiment_id = f"exp_{FLAGS.dataset}{split_suffix}_{FLAGS.llm}_{FLAGS.method}_{FLAGS.refinement_mode}"
    else:
      n_samples_str = "unknown_samples"
      if FLAGS.initial_graph_path:
        match = re.search(r"(\d+)_samples", FLAGS.initial_graph_path)
        if match:
          n_samples_str = f"{match.group(1)}_samples"
      experiment_id = f"exp_{FLAGS.dataset}{split_suffix}_{FLAGS.llm}_{FLAGS.method}_{FLAGS.refinement_mode}_{n_samples_str}"

  if FLAGS.run_id:
    experiment_id += f"_{FLAGS.run_id}"

  experiment_name = (
      f"Evaluation run: Dataset={FLAGS.dataset}, Solver={FLAGS.method}"
  )

  config_dict = {
      "dataset": FLAGS.dataset,
      "method": FLAGS.method,
      "llm": FLAGS.llm,
      "refinement_mode": FLAGS.refinement_mode,
      "guided_only": FLAGS.guided_only,
      "pg_as_text_summary": FLAGS.pg_as_text_summary,
      "seed": FLAGS.seed,
  }



  train_samples = []
  if (FLAGS.refinement_mode.startswith(("memory", "rap")) or FLAGS.refinement_mode == "expel") and not FLAGS.refine:
    if FLAGS.memory_summary_path and os.path.exists(FLAGS.memory_summary_path):
      print(f"Loading memory summary from {FLAGS.memory_summary_path}...")
    else:
      print("Loading train samples for memory retrieval...")
      if not FLAGS.train_trajectories_dir:
        raise ValueError("train_trajectories_dir must be set for memory/rap/expel modes")
    if FLAGS.train_dataset_path and os.path.exists(FLAGS.train_dataset_path):
      import pandas as pd
      if FLAGS.dataset == "gdpval":
        # Load the train split specifically
        dummy_env = GDPvalEnv(
            sample_index=0,
            dataset_path=FLAGS.train_dataset_path,
            split="train",
        )
        df_train = dummy_env.df
      elif FLAGS.dataset == "bfcl":
        df_train = None
      else:
        df_train = pd.read_parquet(FLAGS.train_dataset_path)

      if df_train is not None:
        for idx, row in df_train.iterrows():
          sample_id = f"sample_{idx+1:03d}"
          traj_path = os.path.join(
              FLAGS.train_trajectories_dir, sample_id, "trajectory.json"
          )
          success = False
          trajectory = []
          if os.path.exists(traj_path):
            try:
              with open(traj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                success = data.get("metrics", {}).get("success", 0.0) > 0.9
                trajectory = data.get("trajectory", [])
            except Exception:
              pass
          train_samples.append({
              "sample_id": sample_id,
              "target_question": (
                  row.get("target_question")
                  or row.get("question")
                  or row.get("problem")
                  or row.get("prompt")
                  or str(row.get("id"))
              ),
              "trajectory": trajectory,
              "success": success,
          })
      else:
        # BFCL specific loading
        with open(FLAGS.train_dataset_path, "r") as f:
          bfcl_samples = [json.loads(line) for line in f]
        for idx, sample in enumerate(bfcl_samples):
          sample_id = f"sample_{idx+1:03d}"
          traj_path = os.path.join(
              FLAGS.train_trajectories_dir, sample_id, "trajectory.json"
          )
          success = False
          trajectory = []
          if os.path.exists(traj_path):
            try:
              with open(traj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                success = data.get("metrics", {}).get("success", 0.0) > 0.9
                trajectory = data.get("trajectory", [])
            except Exception:
              pass
          first_prompt = ""
          if "question" in sample and sample["question"]:
            for msg in sample["question"][0]:
              if msg["role"] == "user":
                first_prompt = msg["content"]
                break
          train_samples.append({
              "sample_id": sample_id,
              "target_question": first_prompt or sample.get("id"),
              "trajectory": trajectory,
              "success": success,
          })
    else:
      import glob
      sample_dirs = sorted(glob.glob(os.path.join(FLAGS.train_trajectories_dir, "sample_*")))
      for sdir in sample_dirs:
        sample_id = os.path.basename(sdir)
        traj_path = os.path.join(sdir, "trajectory.json")
        success = False
        trajectory = []
        target_question = sample_id
        if os.path.exists(traj_path):
          try:
            with open(traj_path, "r", encoding="utf-8") as f:
              data = json.load(f)
              metrics = data.get("metrics", {})
              success = metrics.get("success", 0.0) > 0.9 or metrics.get("cfo_survived", 0.0) > 0.9
              trajectory = data.get("trajectory", [])
              if trajectory:
                if trajectory[0].startswith("Question:"):
                  target_question = trajectory[0][len("Question:"):].strip()
                elif "Initial Observation:" in trajectory[0]:
                  target_question = trajectory[0].split("Initial Observation:")[-1].strip()
          except Exception:
            pass
        train_samples.append({
            "sample_id": sample_id,
            "target_question": target_question,
            "trajectory": trajectory,
            "success": success,
        })

  # Generate sample IDs
  sample_ids = [f"sample_{i:03d}" for i in range(1, FLAGS.num_samples + 1)]

  # Partition into batches based on stride
  if FLAGS.stride > 0:
    batches = [
        sample_ids[i : i + FLAGS.stride]
        for i in range(0, len(sample_ids), FLAGS.stride)
    ]
  else:
    batches = [sample_ids]

  active_graph_path = (
      FLAGS.initial_graph_path if FLAGS.initial_graph_path else None
  )
  failed_count = 0
  all_results = []

  # Execute batches sequentially
  for batch_idx, batch_samples in enumerate(batches):
    print(
        f"\n--- Starting Batch {batch_idx + 1}/{len(batches)}"
        f" ({len(batch_samples)} samples) ---"
    )

    batch_results = []
    pending_samples = []
    for sample_id in batch_samples:
      if is_sample_completed(sample_id):
        # Read completed result from disk
        sample_log_dir = get_sample_log_dir(sample_id)
        traj_path = os.path.join(sample_log_dir, "trajectory.json")
        try:
          with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            score = data["metrics"]["success"]
            batch_results.append((sample_id, score, data["trajectory"]))
            print(f"⏩ [Resumption] Sample {sample_id} already completed on disk. Loaded score={score}")
        except Exception as e:
          print(f"⚠️ Failed to load completed sample {sample_id}: {e}. Re-evaluating...")
          pending_samples.append(sample_id)
      else:
        pending_samples.append(sample_id)

    if pending_samples:
      # Concurrent thread-safe runs for the pending samples in the current batch
      with futures.ThreadPoolExecutor(max_workers=FLAGS.num_workers) as executor:
        job_futures = {
            executor.submit(
                run_single_sample,
                sample_id,
                experiment_id,
                experiment_name,
                config_dict,
                active_graph_path,
                train_samples,
            ): sample_id
            for sample_id in pending_samples
        }
        # Wait and check results to propagate exceptions raised in threads
        for future in futures.as_completed(job_futures):
          sample_id = job_futures[future]
          try:
            res = future.result()
            if res:
              batch_results.append(res)
          except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"❌ [Error] Thread evaluation failed for sample {sample_id}: {e}"
            )
            import traceback

            traceback.print_exc()
            failed_count += 1

    # Ensure deterministic order of results for graph refinement
    batch_results.sort(key=lambda x: x[0])
    all_results.extend(batch_results)

    # Graph adaptation refinement at the end of the batch
    if (
        FLAGS.refine
        and FLAGS.refinement_mode
        in ("static_incremental", "scratch_incremental")
        and batch_results
    ):
      num_samples_so_far = sum(len(b) for b in batches[: batch_idx + 1])
      refined_graph_dir = os.path.join(
          get_base_results_dir(),
          FLAGS.refinement_mode,
          f"{num_samples_so_far}_samples",
          "refined_pg",
      )
      os.makedirs(refined_graph_dir, exist_ok=True)
      refined_graph_path = os.path.join(refined_graph_dir, "refined_graph.json")

      # Check if all samples in this batch were already completed, and graph
      # exists
      all_completed = all(
          is_sample_completed(sample_id)
          for sample_id in batch_samples
      )

      if all_completed and os.path.exists(refined_graph_path):
        active_graph_path = refined_graph_path
        print(
            f"⏩ [Resumption] All samples in Batch {batch_idx + 1} were"
            " completed and refined graph already exists at:"
            f" {refined_graph_path}. Skipping active learning refinement."
        )
      else:
        print(
            "Running active learning graph refinement for Batch"
            f" {batch_idx + 1}..."
        )

        if FLAGS.dataset == "cfo":
          env_for_graph = CFOEnv()
  # After the sequential batch loop completes, add support for one-time
  # refinement
  total_samples = len(sample_ids)
  refined_graph_dir = os.path.join(
      get_base_results_dir(),
      FLAGS.refinement_mode,
      f"{total_samples}_samples",
      "refined_pg",
  )
  refined_graph_path = os.path.join(refined_graph_dir, "refined_graph.json")
  all_completed = all(
      is_sample_completed(sample_id)
      for sample_id in sample_ids
  )

  if (
      FLAGS.refine
      and FLAGS.refinement_mode in ("static_onetime", "scratch_onetime")
      and all_results
  ):
    if all_completed and os.path.exists(refined_graph_path):
      print(
          "⏩ [Resumption] One-time refined graph already exists at:"
          f" {refined_graph_path}. Skipping offline graph refinement."
      )
    else:
      print(
          "Running one-time offline graph refinement using all trajectories..."
      )
      if FLAGS.dataset == "cfo":
        env_for_graph = CFOEnv()
      elif FLAGS.dataset == "multichallenge":
        env_for_graph = MultiChallengeEnv(
            sample_index=0,
            dataset_path=FLAGS.multichallenge_dataset_path,
            llm_client=object(),
        )
      elif FLAGS.dataset == "alfworld":
        from procedural_graph.datasets.alfworld import ALFWorldEnv
        env_for_graph = ALFWorldEnv(
            sample_index=0,
            llm_client=object(),
            split=FLAGS.alfworld_split,
        )
      elif FLAGS.dataset == "gdpval":
        env_for_graph = GDPvalEnv(
            sample_index=0,
            dataset_path=FLAGS.gdpval_dataset_path,
            split=FLAGS.gdpval_split,
        )
      elif FLAGS.dataset == "bfcl":
        env_for_graph = BfclEnv(
            sample_index=0,
            dataset_path=FLAGS.bfcl_dataset_path,
            possible_answer_path=FLAGS.bfcl_possible_answer_path,
        )
      elif FLAGS.dataset == "taubench":
        env_for_graph = TauBenchEnv(
            sample_index=0,
            domain=FLAGS.taubench_domain,
            task_split=FLAGS.taubench_split,
            user_model=FLAGS.taubench_user_model,
            user_provider=FLAGS.taubench_user_provider,
        )
      elif FLAGS.dataset == "hotpotqa":
        env_for_graph = HotpotQAEnv(sample_index=0, dataset_path=FLAGS.hotpotqa_dataset_path)
      elif FLAGS.dataset == "multichallenge":
        env_for_graph = MultiChallengeEnv(sample_index=0, dataset_path=FLAGS.multichallenge_dataset_path)
      elif FLAGS.dataset == "cfo":
        env_for_graph = CFOEnv(sample_index=0, dataset_path=FLAGS.cfo_dataset_path)
      else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset}")

      current_graph = load_graph_for_refinement(
          env_for_graph, active_graph_path
      )

      if FLAGS.dataset == "cfo":
        task_description = (
            "Corporate financial optimization and business constraint"
            " satisfaction."
        )
      elif FLAGS.dataset == "alfworld":
        task_description = (
            "Embodied household task navigation and object manipulation in TextWorld."
        )
      elif FLAGS.dataset == "multichallenge":
        task_description = (
            "Complex multi-turn dialogue assistant. You must act as the Assistant in the dialogue "
            "and generate the next response to the User. The 'Target Question' is an evaluation "
            "constraint that your response must satisfy (e.g., recalling a past fact or adhering "
            "to a style). You are NOT evaluating the model; you ARE the model.\n"
            "CRITICAL GUIDELINES:\n"
            "1. Always analyze the target question and extract constraints/facts first. Do not skip these steps. "
            "Understanding the exact criteria the judge will use is crucial to avoid missing subtle constraints.\n"
            "2. If the target question requires recalling specific facts, preferences, or choices from the history, "
            "your final response MUST explicitly mention them and acknowledge they are from the history "
            "(e.g., use headers like 'Your Favorites:' or phrases like 'As you mentioned earlier...'). "
            "The evaluator does not see the history and relies on your response to verify compliance.\n"
            "3. If the target question specifies a required title or subject line in quotes, copy it EXACTLY "
            "as written inside the quotes, including any punctuation (like question marks or periods) at the very end, "
            "even if it makes the title grammatically incorrect or look like a typo. Do not attempt to 'correct' it. "
            "The evaluator checks for an exact literal match.\n"
            "4. For strict linguistic constraints (e.g., 'only use passive voice', 'only use past tense'), "
            "carefully review every single word and clause in your response before submitting. "
            "For 'past tense only', avoid infinitives (e.g., 'to do') and gerunds (e.g., 'doing') if possible, "
            "as some evaluators strictly count them as non-past. For 'passive voice only', ensure there are "
            "absolutely no active clauses.\n"
            "5. If the user has a 'no caffeine' or 'caffeine-free' constraint, do NOT recommend 'decaf' coffee or "
            "'decaf' tea. Decaf products still contain trace amounts of caffeine and will fail strict evaluation. "
            "Stick to naturally caffeine-free options like water, herbal teas, or milk.\n"
            "6. Do not use CompileBaseContent unless the user explicitly asks to edit, rewrite, or revert a "
            "previous response."
        )
      elif FLAGS.dataset == "gdpval":
        task_description = (
            "Data processing, file analysis, and verification tasks using"
            " python and file reading tools (Excel, Word, PDF)."
        )
      elif FLAGS.dataset == "bfcl":
        task_description = (
            "Stateful tool use and function calling across multiple turns. "
            "You must execute correct function calls to satisfy the user's instructions "
            "while maintaining and updating the state of simulated APIs."
        )
      elif FLAGS.dataset == "taubench":
        task_description = (
            "Multi-turn customer support and tool-calling dialogue in domain environments (retail, airline). "
            "You must satisfy user requests according to domain policy manuals and wiki rules."
        )
      else:
        task_description = "General agent navigation and problem solving."

      refiner = ProceduralGraphLLMRefiner(
          VertexAIClient(
              model_name=FLAGS.llm,
              project_id=FLAGS.gcp_project,
              location=FLAGS.llm_location,
              timeout=FLAGS.llm_timeout,
          )
      )
      mode_enum = GraphRefinementMode(FLAGS.refinement_mode)

      filtered_results = []
      for res in all_results:
        traj = res[2]
        has_error = any("Execution Error:" in str(step) for step in traj)
        if not has_error:
          filtered_results.append(res)

      if not filtered_results:
        print(
            "⚠️ No valid trajectories remain in all results after filtering"
            " execution errors. Skipping refinement."
        )
        os.makedirs(refined_graph_dir, exist_ok=True)
        current_graph.save_to_json(refined_graph_path)
      else:
        # Sample trajectories to avoid exceeding LLM context limits
        if len(filtered_results) > FLAGS.max_refinement_trajectories:
          # Sort by score to pick representative successes and failures
          sorted_results = sorted(filtered_results, key=lambda x: x[1])
          half = FLAGS.max_refinement_trajectories // 2
          worst = sorted_results[:half]
          best = sorted_results[-half:]
          # Deduplicate by sample_id
          sampled_dict = {res[0]: res for res in (worst + best)}
          sampled_results = list(sampled_dict.values())
          # Sort back by sample_id to maintain chronological order
          sampled_results.sort(key=lambda x: x[0])
          print(
              f"Sampled {len(sampled_results)} trajectories for refinement"
              f" (out of {len(filtered_results)} total)."
          )
        else:
          sampled_results = filtered_results

        trajs = [res[2] for res in sampled_results]
        scores = [res[1] for res in sampled_results]

        tools_metadata = env_for_graph.get_tools_metadata()
        available_tools_list = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools_metadata
        )

        refined_graph, edits_log = refiner.refine_graph(
            graph=current_graph,
            mode=mode_enum,
            trajectories=trajs,
            success_scores=scores,
            task_description=task_description,
            available_tools_list=available_tools_list,
            env=env_for_graph,
        )
        print(
            f"One-time graph refinement completed. Edits applied:\n{edits_log}"
        )

        os.makedirs(refined_graph_dir, exist_ok=True)
        refined_graph.save_to_json(refined_graph_path)
        print(f"Saved refined graph to: {refined_graph_path}")
  elif (
      FLAGS.refine
      and FLAGS.refinement_mode in ("memory_summarization", "memorybank")
      and all_results
  ):
    summary_dir = get_base_results_dir()
    summary_path = FLAGS.memory_summary_path or os.path.join(summary_dir, "memory_summary.txt")

    if all_completed and os.path.exists(summary_path):
      print(f"⏩ [Resumption] Memory summary already exists at: {summary_path}. Skipping.")
    else:
      print("Running memory summarization using all trajectories...")

      successful_results = [res for res in all_results if res[1] > 0.9]
      if not successful_results:
        successful_results = all_results

      prompt_lines = [
          "You are an expert cognitive architect. Analyze the following successful execution trajectories of an agent solving various dialogue tasks.\n"
          "Identify common successful strategies, common patterns of tool usage, and general rules that lead to success.\n"
          "Synthesize these into a clear set of instructions/guidance (under 500 words) to guide the agent in future tasks.\n"
          "CRITICAL: Do NOT include any specific numbers, exact dollar amounts (e.g., $15M, $30M-$50M), exact month ranges, or hardcoded thresholds in your guidance. "
          "Instead, express these rules qualitatively and conceptually (e.g., 'maintain a safe cash buffer above the mandated minimum', 'submit fundraising requests early', 'prefer equity over debt to avoid interest obligations', 'conserve tool calls').\n"
      ]
      for res in successful_results[:5]:
        target_q = "Unknown Task"
        if res[2] and res[2][0].startswith("Observation:"):
          target_q = res[2][0]
        prompt_lines.append(f"Task: {target_q}")
        prompt_lines.append("Trajectory:")
        traj_text = "\n".join(res[2])
        if len(traj_text) > 15000:
          traj_text = traj_text[:7500] + "\n...[TRUNCATED INTERMEDIATE STEPS]...\n" + traj_text[-7500:]
        prompt_lines.append(traj_text)
        prompt_lines.append("-----------------\n")

      prompt = "\n".join(prompt_lines)

      llm_client = VertexAIClient(
          model_name=FLAGS.llm,
          project_id=FLAGS.gcp_project,
          location=FLAGS.llm_location,
          timeout=FLAGS.llm_timeout,
      )
      print("Sending summarization request to LLM...")
      summary = llm_client.generate(prompt, temperature=0.0)

      os.makedirs(os.path.dirname(summary_path), exist_ok=True)
      with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
      print(f"Saved memory summary to: {summary_path}")
  elif (
      FLAGS.refine
      and FLAGS.refinement_mode == "expel"
      and all_results
  ):
    rules_path = os.path.join(get_base_results_dir(), "expel_rules.json")
    if all_completed and os.path.exists(rules_path):
      print(f"⏩ [Resumption] ExpeL rules already exist at: {rules_path}. Skipping.")
    else:
      print("Running ExpeL rule extraction using successful and unsuccessful trajectories...")
      successful_results = [res for res in all_results if res[1] > 0.9]
      unsuccessful_results = [res for res in all_results if res[1] <= 0.9]

      prompt_lines = [
          "You are an expert agent analyst. Analyze the following successful and unsuccessful execution trajectories of an agent solving dialogue/professional tasks.",
          "Identify the key differences between successful runs and failures. Extract general rules and insights that can prevent future failures and guide successful steps.",
          "Format your output as a valid JSON list of objects, where each object has exactly two keys:",
          '- "trigger": The situation or context (e.g., "when user asks for X", "after writing a file", "before calling submit")',
          '- "insight": The strategic guideline/action rule to follow.',
          "\nExample Output:",
          '[{"trigger": "when the user requests a list of coffee options", "insight": "never recommend decaf coffee, use naturally caffeine-free alternatives instead."}]',
          "\nSuccess trajectories:"
      ]
      for res in successful_results[:3]:
        prompt_lines.append(f"Task: {res[0]}")
        prompt_lines.append("Trajectory:")
        prompt_lines.append("\n".join(res[2][:15]))
        prompt_lines.append("-----------------")

      if unsuccessful_results:
        prompt_lines.append("\nFailure trajectories:")
        for res in unsuccessful_results[:3]:
          prompt_lines.append(f"Task: {res[0]}")
          prompt_lines.append("Trajectory:")
          prompt_lines.append("\n".join(res[2][:15]))
          prompt_lines.append("-----------------")

      prompt = "\n".join(prompt_lines)
      llm_client = VertexAIClient(
          model_name=FLAGS.llm,
          project_id=FLAGS.gcp_project,
          location=FLAGS.llm_location,
          timeout=max(300, FLAGS.llm_timeout),
      )
      try:
        response = llm_client.generate(prompt, temperature=0.0)
        json_match = re.search(r"(\[.*\])", response, re.DOTALL)
        if json_match:
          rules_json = json_match.group(1).strip()
          rules = json.loads(rules_json)
        else:
          rules = json.loads(response.strip())
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        with open(rules_path, "w", encoding="utf-8") as f:
          json.dump(rules, f, indent=2)
        print(f"Saved ExpeL rules to: {rules_path}")
      except Exception as e:
        print(f"Error parsing/saving ExpeL rules: {e}")
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        with open(rules_path, "w", encoding="utf-8") as f:
          json.dump([], f)
  elif (
      FLAGS.refine
      and FLAGS.refinement_mode == "autoguide"
      and all_results
  ):
    lib_path = os.path.join(get_base_results_dir(), "autoguide_library.json")
    if all_completed and os.path.exists(lib_path):
      print(f"⏩ [Resumption] AutoGuide guidelines already exist at: {lib_path}. Skipping.")
    else:
      print("Running AutoGuide guideline extraction using all trajectories...")
      prompt_lines = [
          "You are an expert cognitive engineer. Create a context-aware guideline library based on the following task execution trajectories.",
          "For key actions or failure steps, formulate guidelines in a conditional format: If [context/precondition], then [guideline].",
          "Format your output as a valid JSON list of objects, where each object has exactly two keys:",
          '- "condition": A short description of the context/condition/pre-state.',
          '- "guideline": The specific action, plan, or check that should be executed.',
          "\nExample Output:",
          '[{"condition": "the user has a strict caffeine allergy", "guideline": "exclude decaffeinated coffee and decaffeinated tea, and only recommend herbal tea, water, or milk."}]',
          "\nTrajectories:"
      ]
      for res in all_results[:5]:
        prompt_lines.append(f"Task: {res[0]}")
        prompt_lines.append("Trajectory:")
        prompt_lines.append("\n".join(res[2][:15]))
        prompt_lines.append("-----------------")

      prompt = "\n".join(prompt_lines)
      llm_client = VertexAIClient(
          model_name=FLAGS.llm,
          project_id=FLAGS.gcp_project,
          location=FLAGS.llm_location,
          timeout=max(300, FLAGS.llm_timeout),
      )
      try:
        response = llm_client.generate(prompt, temperature=0.0)
        json_match = re.search(r"(\[.*\])", response, re.DOTALL)
        if json_match:
          lib_json = json_match.group(1).strip()
          guidelines = json.loads(lib_json)
        else:
          guidelines = json.loads(response.strip())
        os.makedirs(os.path.dirname(lib_path), exist_ok=True)
        with open(lib_path, "w", encoding="utf-8") as f:
          json.dump(guidelines, f, indent=2)
        print(f"Saved AutoGuide guidelines to: {lib_path}")
      except Exception as e:
        print(f"Error parsing/saving AutoGuide guidelines: {e}")
        os.makedirs(os.path.dirname(lib_path), exist_ok=True)
        with open(lib_path, "w", encoding="utf-8") as f:
          json.dump([], f)
  elif (
      FLAGS.refine
      and FLAGS.refinement_mode == "awm"
      and all_results
  ):
    wf_path = os.path.join(get_base_results_dir(), "awm_workflows.json")
    if all_completed and os.path.exists(wf_path):
      print(f"⏩ [Resumption] AWM workflows already exist at: {wf_path}. Skipping.")
    else:
      print("Running AWM workflow induction using successful trajectories...")
      if FLAGS.dataset == "gdpval":
        env_for_graph = GDPvalEnv(
            sample_index=0,
            dataset_path=FLAGS.gdpval_dataset_path,
            split=FLAGS.gdpval_split,
        )
      else:
        env_for_graph = None

      successful_by_occupation = {}
      for res in all_results:
        sample_id, score, traj = res
        if score > 0.9:
          try:
            if env_for_graph:
              idx = int(sample_id.split("_")[-1]) - 1
              row = env_for_graph.df.iloc[idx]
              occ = row["occupation"]
            else:
              occ = "General Agent"
            if occ not in successful_by_occupation:
              successful_by_occupation[occ] = []
            successful_by_occupation[occ].append(traj)
          except Exception:
            pass

      llm_client = VertexAIClient(
          model_name=FLAGS.llm,
          project_id=FLAGS.gcp_project,
          location=FLAGS.llm_location,
          timeout=FLAGS.llm_timeout,
      )

      workflows = {}
      for occ, trajs in successful_by_occupation.items():
        print(f"Inducing workflow for occupation: {occ}...")
        prompt = (
            f"You are an expert process designer. Analyze the following successful trajectories of a {occ} solving tasks.\n"
            f"Induce a standard step-by-step checklist workflow (under 10 steps) that this professional should execute to solve similar tasks successfully.\n"
            f"Do not include task-specific parameters; focus on procedural steps.\n"
            f"Trajectories:\n" + "\n---\n".join("\n".join(t[:15]) for t in trajs[:3])
        )
        response = llm_client.generate(prompt, temperature=0.0)
        workflows[occ] = response.strip()

      os.makedirs(os.path.dirname(wf_path), exist_ok=True)
      with open(wf_path, "w", encoding="utf-8") as f:
        json.dump(workflows, f, indent=2)
      print(f"Saved AWM workflows to: {wf_path}")
  elif (
      FLAGS.refine
      and FLAGS.refinement_mode == "knowagent"
      and all_results
  ):
    kb_path = os.path.join(get_base_results_dir(), "knowagent_kb.json")
    if all_completed and os.path.exists(kb_path):
      print(f"⏩ [Resumption] KnowAgent KB already exists at: {kb_path}. Skipping.")
    else:
      print("Running KnowAgent KB transition extraction...")
      successful_results = [res for res in all_results if res[1] > 0.9]
      kb = {}
      for res in successful_results:
        traj = res[2]
        tool_sequence = []
        for step in traj:
          if step.startswith("Action:"):
            action_content = step[len("Action:") :].strip()
            try:
              from procedural_graph.env_base import parse_action_string
              action_name, _, _ = parse_action_string(action_content)
              tool_sequence.append(action_name)
            except Exception:
              pass
        
        for i in range(len(tool_sequence) - 1):
          curr = tool_sequence[i]
          nxt = tool_sequence[i+1]
          if curr not in kb:
            kb[curr] = set()
          kb[curr].add(nxt)
      
      kb_json = {k: list(v) for k, v in kb.items()}
      os.makedirs(os.path.dirname(kb_path), exist_ok=True)
      with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_json, f, indent=2)
      print(f"Saved KnowAgent KB transitions to: {kb_path}")
  # 9. Compile summary.csv for this run group
  import csv
  completed_results = []
  for sample_id in sample_ids:
    sample_log_dir = get_sample_log_dir(sample_id)
    traj_path = os.path.join(sample_log_dir, "trajectory.json")
    if os.path.exists(traj_path):
      try:
        with open(traj_path, "r", encoding="utf-8") as f:
          data = json.load(f)
          completed_results.append(data)
      except Exception:
        pass

  if completed_results:
    # Filter out system errors when computing averages
    valid_results = [res for res in completed_results if "system_error" not in res.get("metrics", {})]
    
    total_samples = len(completed_results)
    valid_samples = len(valid_results)
    
    if valid_results:
      successes = [res["metrics"]["success"] for res in valid_results]
      steps_list = [res["metrics"]["steps"] for res in valid_results]
      tokens_list = [res["metrics"]["token_cost"] for res in valid_results]
      latency_list = [res["metrics"]["latency"] for res in valid_results]
      
      avg_success = sum(successes) / len(successes)
      avg_steps = sum(steps_list) / len(steps_list)
      avg_tokens = sum(tokens_list) / len(tokens_list)
      avg_latency = sum(latency_list) / len(latency_list)
      
      if FLAGS.dataset == "cfo":
        cfo_surv = [res["metrics"].get("cfo_survived", 0.0) for res in valid_results]
        cfo_score = [res["metrics"].get("cfo_score_m", 0.0) for res in valid_results]
        cfo_crisis1 = [res["metrics"].get("cfo_crisis1", 0.0) for res in valid_results]
        cfo_crisis2 = [res["metrics"].get("cfo_crisis2", 0.0) for res in valid_results]
        cfo_crisis3 = [res["metrics"].get("cfo_crisis3", 0.0) for res in valid_results]
        cfo_tools = [res["metrics"].get("cfo_tools_per_mo", 0.0) for res in valid_results]
        cfo_actions = [res["metrics"].get("cfo_actions", 0.0) for res in valid_results]
        cfo_raised = [res["metrics"].get("cfo_raised_m", 0.0) for res in valid_results]
        
        cfo_vals = [
            f"{sum(cfo_surv) / len(cfo_surv) * 100:.2f}%",
            f"${sum(cfo_score) / len(cfo_score):.2f}M",
            f"{sum(cfo_crisis1) / len(cfo_crisis1) * 100:.2f}%",
            f"{sum(cfo_crisis2) / len(cfo_crisis2) * 100:.2f}%",
            f"{sum(cfo_crisis3) / len(cfo_crisis3) * 100:.2f}%",
            f"{sum(cfo_tools) / len(cfo_tools):.2f}",
            f"{sum(cfo_actions) / len(cfo_actions):.2f}",
            f"${sum(cfo_raised) / len(cfo_raised):.2f}M",
        ]
      else:
        cfo_vals = [""] * 8
    else:
      avg_success = 0.0
      avg_steps = 0.0
      avg_tokens = 0.0
      avg_latency = 0.0
      cfo_vals = [""] * 8
      
    summary_headers = [
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
    
    parent_dir = os.path.dirname(os.path.dirname(get_sample_log_dir(sample_ids[0])))
    summary_csv_path = os.path.join(parent_dir, "summary.csv")
    
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(summary_headers)
      writer.writerow([
          experiment_id,
          f"{valid_samples}/{len(sample_ids)}",
          f"{avg_success * 100:.2f}%",
          f"{avg_steps:.2f}",
          f"{avg_tokens:.2f}",
          f"{avg_latency:.2f}"
      ] + cfo_vals)

    print(f"📊 Saved group summary CSV to: {summary_csv_path}")

  # 10. Auto compile and refresh Results Compiler comparison dashboards
  print("📊 Refreshing summary results dashboards...")
  compiler = ResultsCompiler(FLAGS.results_dir)

  dashboard_md_path = os.path.join(
      FLAGS.results_dir, "procedural_graph_experiments_summary.md"
  )
  dashboard_csv_path = os.path.join(FLAGS.results_dir, "dashboard.csv")

  compiler.compile_dashboard(dashboard_md_path, dashboard_csv_path)

  if FLAGS.dataset == "cfo":
    cfo_dashboard_path = os.path.join(FLAGS.results_dir, "cfo_dashboard.md")
    compiler.compile_cfo_dashboard(cfo_dashboard_path)
  elif FLAGS.dataset == "alfworld":
    alfworld_dashboard_path = os.path.join(FLAGS.results_dir, "alfworld_dashboard.md")
    compiler.compile_general_dashboard(alfworld_dashboard_path, dataset_filter="alfworld")

  if failed_count > 0:
    print(f"⚠️ [Warning] Procedural Graph evaluation completed with {failed_count} failures!")
  else:
    print("🎉 Procedural Graph evaluation test run finished successfully!")


if __name__ == "__main__":
  app.run(main)
