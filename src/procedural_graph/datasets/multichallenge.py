"""MultiChallenge Environment Adapter subclassing BaseEnvironment."""

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from .. import env_base
from ..llm_client import VertexAIClient


class MultiChallengeEnv(env_base.BaseEnvironment):
  """MultiChallenge conversational evaluation environment adapter."""

  _df_cache = {}  # Class-level DataFrame cache to prevent duplicate Disk I/O
  _df_cache_lock = threading.Lock()  # Thread synchronization lock for caching

  def __init__(
      self,
      sample_index: int,
      dataset_path: Optional[str] = None,
      llm_client: Optional[Any] = None,
  ):
    super().__init__()
    self.questions_path = dataset_path or (
        "MultiChallenge/data/test_166.parquet"
    )
    with MultiChallengeEnv._df_cache_lock:
      if self.questions_path not in MultiChallengeEnv._df_cache:
        if not os.path.exists(self.questions_path):
          raise FileNotFoundError(
              "MultiChallenge parquet dataset file not found at: "
              f"{self.questions_path}"
          )
        MultiChallengeEnv._df_cache[self.questions_path] = pd.read_parquet(
            self.questions_path
        )

    df = MultiChallengeEnv._df_cache[self.questions_path]
    if sample_index < 0 or sample_index >= len(df):
      raise IndexError(
          f"sample_index {sample_index} out of range (total rows: {len(df)})"
      )

    row = df.iloc[sample_index]
    self.sample_index = sample_index
    self.question_id = row["question_id"]
    self.axis = row["axis"]
    self.target_question = row["target_question"]
    self.pass_criteria = row["pass_criteria"]

    # Dynamically convert parallel list structure from Parquet struct to
    # list of dicts
    conversation_struct = row.get("conversation")
    if isinstance(conversation_struct, dict):
      roles = conversation_struct.get("role")
      roles = roles if roles is not None else []
      contents = conversation_struct.get("content")
      contents = contents if contents is not None else []
    elif conversation_struct is not None:
      roles = getattr(conversation_struct, "role", None)
      roles = roles if roles is not None else []
      contents = getattr(conversation_struct, "content", None)
      contents = contents if contents is not None else []
    else:
      roles, contents = [], []

    if len(roles) != len(contents):
      raise ValueError(
          "Mismatched roles/contents parallel arrays inside conversation struct"
          f" at index {sample_index}"
      )

    self.conversation = [
        {"role": r, "content": c} for r, c in zip(roles, contents)
    ]
    self.llm_client = llm_client or VertexAIClient(
        model_name="grok-4.1-fast-non-reasoning"
    )
    self.judge_client = VertexAIClient(
        model_name="gemini-2.5-pro",
        location="global",
        timeout=180,
    )

    self.terminated = False
    self.score = 0.0
    self.final_response = ""
    self.trajectory_history: List[str] = []
    self._allow_verify = False

    # Pre-render history string
    history_parts = []
    for turn in self.conversation:
      role = "User" if turn.get("role") == "user" else "Assistant"
      history_parts.append(f"{role}: {turn.get('content', '')}")
    self.history_str = "\n\n".join(history_parts)

  def get_system_prompt(self) -> str:
    return (
        "You are an AI Agent solving complex multi-turn conversation challenges.\n"
        "You must act as the Assistant in the dialogue and generate the next response to the User.\n"
        "The 'Target Question' is an evaluation constraint that your response must satisfy (e.g., recalling a past fact or adhering to a style).\n"
        "You are NOT evaluating the model; you ARE the model. Your final response submitted via Finish() must be the actual dialogue response to the user that satisfies this constraint.\n\n"
        "Available Tool Actions:\n"
        " - ParseHistory(): Returns the conversation turns history as a formatted string.\n"
        " - ExtractConstraints(): Parses and extracts explicit instruction requirements from the conversation.\n"
        " - ExtractUserFacts(): Extracts user attributes, history, and preferences from the conversation.\n"
        " - AnalyzeTargetQuestion(): Returns the target evaluation criteria.\n"
        " - CompileBaseContent(): Returns baseline content/document for editing tasks.\n"
        " - Finish(final_response=\"<final_response>\"):\n"
        "   Terminal state-changing tool to submit and finish. CRITICAL: Use escaped newlines (\\n) for line breaks. Do not use raw unescaped newlines."
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
    return (
        "Dialogue environment loaded. Target Question to address: "
        f"{self.target_question}"
    )

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
  # Expose MultiChallenge tools via register_tool decorators
  # =========================================================================

  @env_base.register_tool(
      "ParseHistory", "Returns the conversation turns history."
  )
  def ParseHistory(self) -> str:
    return self.history_str

  @env_base.register_tool(
      "ExtractConstraints",
      "Parses and extracts explicit instruction requirements.",
  )
  def ExtractConstraints(self) -> str:
    client = self.llm_client
    prompt = (
        "Read the following conversation history between a User and an "
        "Assistant:\n"
        f"--- History ---\n{self.history_str}\n---------------\n\n"
        "Identify and list all explicit instructions, constraints, or "
        "guidelines specified by the User that the Assistant is expected to "
        "follow throughout the entire conversation.\n"
        "If no explicit constraints are found, respond with 'None'."
    )
    return client.generate(prompt)

  @env_base.register_tool(
      "ExtractUserFacts", "Extracts user attributes, history, and preferences."
  )
  def ExtractUserFacts(self) -> str:
    client = self.llm_client
    prompt = (
        "Read the following conversation history between a User and an "
        "Assistant:\n"
        f"--- History ---\n{self.history_str}\n---------------\n\n"
        "Identify and list all key facts, preferences, relationship details, "
        "dates, locations, or restrictions mentioned by the User.\n"
        "If no facts are found, respond with 'None'."
    )
    return client.generate(prompt)

  @env_base.register_tool(
      "AnalyzeTargetQuestion", "Returns the target evaluation criteria."
  )
  def AnalyzeTargetQuestion(self) -> str:
    return f"Target Question Evaluation Criteria: {self.target_question}"

  @env_base.register_tool(
      "CompileBaseContent", "Returns baseline content for editing tasks."
  )
  def CompileBaseContent(self) -> str:
    assistant_responses = [
        turn["content"]
        for turn in self.conversation
        if turn["role"] == "assistant"
    ]
    if assistant_responses:
      return assistant_responses[-1]
    return "No baseline assistant response found in history."

  @env_base.register_tool(
      "VerifyConstraints", "Triggers online LLM-as-a-judge verification."
  )
  def VerifyConstraints(self, draft_response: str = "") -> str:
    if not getattr(self, "_allow_verify", False):
      return (
          "Error: VerifyConstraints cannot be called directly during the reasoning loop. "
          "VerifyConstraints is only executed automatically at the end of the task. "
          "Please continue reasoning or submit your final response using Finish(final_response=\"...\")."
      )
    client = self.judge_client
    prompt = (
        "You are tasked with evaluating a model response to see if it meets a"
        " specific criteria.\nThe criteria will always be YES/NO"
        " evaluation.\n\nThe model response is as"
        f" follows:\n<MODEL_RESPONSE>\n{draft_response}\n</MODEL_RESPONSE>\n\nThe"
        " criteria that the model response must meet is as follows. Be VERY"
        f" STRICT!:\n<CRITERIA>\n{self.target_question}\n</CRITERIA>\n\nPrint"
        ' your reasoning followed by your verdict, either "YES" or "NO".'
    )
    judge_response = client.generate(prompt)

    clean_response = judge_response.strip().upper()
    import re

    words = re.findall(r"\b(YES|NO)\b", clean_response)
    verdict = "NO"
    if words:
      verdict = words[-1]
    else:
      if "YES" in clean_response:
        verdict = "YES"
      else:
        verdict = "NO"

    if verdict == self.pass_criteria.strip().upper():
      self.score = 1.0
    else:
      self.score = 0.0

    print(
        f"\n[DEBUG Judge] Sample: {self.sample_index} ({self.question_id}) | "
        f"Verdict: {verdict} (Expected: {self.pass_criteria}) | "
        f"Judge Response:\n{judge_response}\n",
        flush=True,
    )

    return (
        f"Judge Explanation and Verdict:\n{judge_response}\n\nParsed Verdict: "
        f"{verdict} (Expected: {self.pass_criteria}) | Assigned Score: "
        f"{self.score}"
    )

  @env_base.register_tool(
      "Finish",
      "Terminal state-changing tool that submits final response. CRITICAL: Use escaped newlines (\\n) for line breaks. Do not use raw unescaped newlines.",
      is_state_changing=True,
  )
  def Finish(self, final_response: str = "") -> str:
    self.terminated = True
    self.final_response = final_response
    if self.score == 0.0:
      self._allow_verify = True
      try:
        self.VerifyConstraints(final_response)
      finally:
        self._allow_verify = False
    return f"Task terminated. Submitted final response: {final_response}"

  # =========================================================================
  # Override Hooks
  # =========================================================================

  def get_current_node_id(self, trajectory: List[str]) -> str:
    if not trajectory:
      return "START"

    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        # Try regex first to handle syntax/argument parsing errors gracefully
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", action_content)
        if match:
          action_name = match.group(1)
          if action_name in (
              "Finish",
              "VerifyConstraints",
              "ParseHistory",
              "ExtractConstraints",
              "ExtractUserFacts",
              "AnalyzeTargetQuestion",
              "CompileBaseContent",
          ):
            return action_name
        try:
          action_name, _, _ = env_base.parse_action_string(action_content)
          if action_name in (
              "Finish",
              "VerifyConstraints",
              "ParseHistory",
              "ExtractConstraints",
              "ExtractUserFacts",
              "AnalyzeTargetQuestion",
              "CompileBaseContent",
          ):
            return action_name
          else:
            return action_name
        except Exception:
          pass
    return "START"

  def get_retrieval_query_hook(
      self, trajectory: List[str], current_obs: str
  ) -> str:
    return f"AXIS: {self.axis} | CRITERIA: {self.target_question}"

  def parse_proposal_hook(self, llm_proposal: str) -> str:
    return llm_proposal.strip()


