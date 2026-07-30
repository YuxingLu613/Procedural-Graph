"""HotpotQA multi-hop question answering evaluation environment adapter."""

import os
import string
import threading
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from .. import env_base


def normalize_answer(s: str) -> str:
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return (
        " "
        + " ".join([
            w
            for w in text.split()
            if w.lower() not in ("a", "an", "the", "in", "of", "at")
        ])
        + " "
    )

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()


class HotpotQAEnv(env_base.BaseEnvironment):
  """HotpotQA multi-hop QA evaluation environment adapter."""

  _df_cache = {}  # Class-level DataFrame cache to prevent duplicate Disk I/O
  _df_cache_lock = threading.Lock()  # Thread synchronization lock for caching

  def __init__(
      self,
      sample_index: int = 0,
      dataset_path: Optional[str] = None,
      llm_client: Optional[Any] = None,
  ):
    super().__init__()
    self.dataset_path = dataset_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/datasets/General/HotpotQA/distractor/validation-00000-of-00001.parquet"))
    with HotpotQAEnv._df_cache_lock:
      if self.dataset_path not in HotpotQAEnv._df_cache:
        if not os.path.exists(self.dataset_path):
          raise FileNotFoundError(
              "HotpotQA parquet dataset file not found at: "
              f"{self.dataset_path}"
          )
        HotpotQAEnv._df_cache[self.dataset_path] = pd.read_parquet(
            self.dataset_path
        )

    df = HotpotQAEnv._df_cache[self.dataset_path]
    if sample_index < 0 or sample_index >= len(df):
      raise IndexError(
          f"sample_index {sample_index} out of range (total rows: {len(df)})"
      )

    row = df.iloc[sample_index]
    self.question_id = row["id"]
    self.question = row["question"]
    self.ground_truth = row["answer"]
    self.question_type = row.get("type", "bridge")
    self.question_level = row.get("level", "medium")

    # Build passages index from context struct
    context = row.get("context")
    self.passages: Dict[str, str] = {}
    if isinstance(context, dict):
      titles = context.get("title", [])
      sentences_list = context.get("sentences", [])
      for t, s_arr in zip(titles, sentences_list):
        if isinstance(s_arr, (list, tuple)):
          self.passages[str(t)] = " ".join(s_arr)
        elif hasattr(s_arr, "tolist"):
          self.passages[str(t)] = " ".join(s_arr.tolist())
        else:
          self.passages[str(t)] = str(s_arr)

    self.terminated = False
    self.score = 0.0
    self.final_response = ""
    self.trajectory_history: List[str] = []

  def get_system_prompt(self) -> str:
    return (
        "You are an expert multi-hop QA Agent solving HotpotQA questions.\n"
        f"Question: {self.question}\n\n"
        "Available Tool Actions:\n"
        " - first_hop_retrieve(query='<title>'): Fetch the Wikipedia passage"
        " matching the first entity title.\n"
        " - scan_index(query='<keyword>'): Scan passages or context for bridge"
        " terms.\n"
        " - second_hop_retrieve(query='<title>'): Fetch the secondary Wikipedia"
        " passage using the bridge term.\n"
        " - generate_answer(answer='<final_answer>'): Submit your verified"
        " answer and finish."
    )

  def is_terminated(self) -> bool:
    return self.terminated

  def get_score(self) -> float:
    return self.score

  def reset(self) -> str:
    self.terminated = False
    self.score = 0.0
    self.final_response = ""
    self.trajectory_history = []
    return f"HotpotQA environment loaded. Question: {self.question}"

  def get_state(self) -> Any:
    return (
        self.terminated,
        self.score,
        self.final_response,
        list(self.trajectory_history),
    )

  def load_state(self, state: Any) -> None:
    self.terminated, self.score, self.final_response, traj = state
    self.trajectory_history = list(traj)

  # =========================================================================
  # Registered Tool Methods
  # =========================================================================

  @env_base.register_tool(
      "first_hop_retrieve", "Retrieve Wikipedia passage by title."
  )
  def first_hop_retrieve(self, query: str) -> str:
    query_clean = query.strip().lower()
    # 1. Exact match
    for title, content in self.passages.items():
      if query_clean == title.lower():
        return f"Title: {title}\nContent: {content}"
    # 2. Query is substring of title (e.g. query="Slettedahl", title="Keith Konrad Slettedahl")
    for title, content in self.passages.items():
      if query_clean in title.lower():
        return f"Title: {title}\nContent: {content}"
    # 3. Title is substring of query (e.g. query="Keith Konrad Slettedahl biography", title="Keith Konrad Slettedahl")
    for title, content in self.passages.items():
      if title.lower() in query_clean:
        return f"Title: {title}\nContent: {content}"
    # Fallback return first passage if fuzzy match fails
    if self.passages:
      first_t, first_c = next(iter(self.passages.items()))
      return f"Title: {first_t}\nContent: {first_c}"
    return "No passages available in context."

  @env_base.register_tool(
      "scan_index", "Scan context passages for keywords or bridge entities."
  )
  def scan_index(self, query: str) -> str:
    query_clean = query.strip().lower()
    matches = []
    for title, content in self.passages.items():
      if query_clean in content.lower() or query_clean in title.lower():
        matches.append(f"[{title}]: {content}")
    if matches:
      return "\n\n".join(matches[:3])
    return "No matching bridge terms found."

  @env_base.register_tool(
      "second_hop_retrieve",
      "Retrieve secondary Wikipedia passage using bridge entity.",
  )
  def second_hop_retrieve(self, query: str) -> str:
    return self.first_hop_retrieve(query)

  @env_base.register_tool(
      "generate_answer",
      "Submit final answer to the question.",
      is_state_changing=True,
  )
  def generate_answer(self, answer: str) -> str:
    self.final_response = answer
    self.terminated = True

    pred = normalize_answer(answer)
    gt = normalize_answer(self.ground_truth)

    if pred == gt or pred in gt or gt in pred:
      self.score = 1.0
    else:
      self.score = 0.0

    return (
        f"Answer submitted: {answer}. Ground truth: {self.ground_truth}."
        f" Score: {self.score}"
    )

  # =========================================================================
  # Procedural Graph Node Mapping Hook
  # =========================================================================

  def get_current_node_id(self, trajectory: List[str]) -> str:
    if not trajectory:
      return "START"

    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        try:
          action_name, _, _ = env_base.parse_action_string(action_content)
          name_lower = action_name.lower()
          if "first_hop" in name_lower:
            return "First_Hop_Retrieve"
          elif "scan" in name_lower:
            return "Scan_Index"
          elif "second_hop" in name_lower:
            return "Second_Hop_Retrieve"
          elif (
              "answer" in name_lower
              or "finish" in name_lower
              or "submit" in name_lower
          ):
            return "Generate_Answer"
        except Exception:
          pass
    return "START"

  def get_retrieval_query_hook(
      self, trajectory: List[str], current_obs: str
  ) -> str:
    return f"Question: {self.question}"
