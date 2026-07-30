"""TAU-bench evaluation environment adapter subclassing BaseEnvironment."""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure tau-bench repository path is on python path
_curr = os.path.abspath(os.path.dirname(__file__))
while _curr != "/" and not os.path.exists(os.path.join(_curr, "dataset/Agent/tau-bench")):
  _curr = os.path.dirname(_curr)
_tau_bench_dir = os.path.join(_curr, "dataset/Agent/tau-bench")
if os.path.exists(_tau_bench_dir) and _tau_bench_dir not in sys.path:
  sys.path.insert(0, _tau_bench_dir)

# Propagate GCP project environment variables for sub-libraries if set
_gcp_project = (
    os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("VERTEX_PROJECT_ID")
    or os.environ.get("GCLOUD_PROJECT")
)
if _gcp_project:
  os.environ.setdefault("VERTEX_PROJECT", _gcp_project)
  os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _gcp_project)
  os.environ.setdefault("VERTEXAI_PROJECT", _gcp_project)
os.environ.setdefault("VERTEX_LOCATION", "us-central1")
os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")

from .. import env_base

_tau_import_error = None
try:
  from tau_bench.envs import get_env as get_tau_env
  from tau_bench.types import Action, RESPOND_ACTION_NAME
except ImportError as e:
  get_tau_env = None
  Action = None
  RESPOND_ACTION_NAME = "respond"
  _tau_import_error = e


class TauBenchEnv(env_base.BaseEnvironment):
  """TAU-bench evaluation environment adapter."""

  def __init__(
      self,
      sample_index: int = 0,
      domain: str = "retail",
      task_split: str = "test",
      user_model: str = "gemini-2.5-flash",
      user_provider: str = "vertex_ai",
      llm_client: Optional[Any] = None,
  ):
    super().__init__()
    if get_tau_env is None:
      raise ImportError(
          f"Failed to import tau_bench from {_tau_bench_dir}. Error: {_tau_import_error}"
      ) from _tau_import_error
    self.domain = domain
    self.task_split = task_split
    self.sample_index = sample_index
    self.user_model = user_model
    self.user_provider = user_provider

    # Initialize tau-bench environment
    self.tau_env = get_tau_env(
        env_name=self.domain,
        user_strategy="llm",
        user_model=self.user_model,
        user_provider=self.user_provider,
        task_split=self.task_split,
        task_index=self.sample_index,
    )

    # Initial user observation from env reset
    res = self.tau_env.reset(task_index=self.sample_index)
    self.initial_user_obs = res.observation
    self.task = res.info.task
    self.instruction = getattr(self.task, "instruction", str(self.task))

    self.terminated = False
    self.score = 0.0
    self.final_response = ""
    self.trajectory_history: List[str] = []
    self.last_observation = self.initial_user_obs

    # Tools mapping and schema
    self.tool_schemas = self.tau_env.tools_info
    self.tools_map = self.tau_env.tools_map

  def get_system_prompt(self) -> str:
    """Returns system prompt detailing wiki policy rules and tool interfaces."""
    tools_desc = []
    for info in self.tool_schemas:
      fn = info.get("function", {})
      name = fn.get("name", "")
      desc = fn.get("description", "")
      params = fn.get("parameters", {})
      tools_desc.append(
          f"- {name}: {desc}\n  Parameters: {json.dumps(params)}"
      )

    tools_str = "\n".join(tools_desc)
    prompt = f"""You are a helpful customer support agent operating in the {self.domain.upper()} domain.
Your goal is to assist the customer according to the domain policy manual and rules provided below.

=== DOMAIN POLICY MANUAL & WIKI ===
{self.tau_env.wiki}

=== AVAILABLE TOOLS ===
- respond(content: str): Send a message to the customer.
{tools_str}

=== INSTRUCTIONS & FORMAT ===
- At each step, analyze the conversation history and execute either a domain tool call or send a message to the customer using respond(content="...").
- Tool calls MUST use standard function call format, e.g.:
  `get_order_details(order_id="12345")` or `respond(content="Hello! How can I help you today?")`
- When you have completed all requested tasks and answered all questions, send a final confirmation to the user using `respond(content="...")`.

=== INITIAL CUSTOMER MESSAGE ===
Customer: {self.initial_user_obs}
"""
    return prompt

  def is_terminated(self) -> bool:
    return self.terminated

  def get_score(self) -> float:
    return self.score

  def reset(self) -> str:
    """Resets the environment and returns the initial observation."""
    res = self.tau_env.reset(task_index=self.sample_index)
    self.initial_user_obs = res.observation
    self.terminated = False
    self.score = 0.0
    self.trajectory_history = []
    self.last_observation = self.initial_user_obs
    return f"Customer: {self.initial_user_obs}"

  def get_state(self) -> Any:
    """Clones and returns current state checkpoint."""
    return {
        "sample_index": self.sample_index,
        "terminated": self.terminated,
        "score": self.score,
        "trajectory_history": list(self.trajectory_history),
        "last_observation": self.last_observation,
        "actions": list(self.tau_env.actions),
    }

  def load_state(self, state: Any) -> None:
    """Restores environment state from checkpoint."""
    self.sample_index = state["sample_index"]
    self.terminated = state["terminated"]
    self.score = state["score"]
    self.trajectory_history = list(state["trajectory_history"])
    self.last_observation = state["last_observation"]

  def execute_action(self, action_str: str) -> Tuple[str, bool]:
    """Parses and executes an action call string in the tau-bench environment."""
    if self.terminated:
      return "Environment is already terminated.", True

    self.trajectory_history.append(f"Action: {action_str}")
    try:
      tool_name, args, kwargs = env_base.parse_action_string(action_str)
    except Exception as e:
      err_msg = f"Failed to parse action '{action_str}': {e}"
      self.trajectory_history.append(f"Observation: {err_msg}")
      return err_msg, self.terminated

    # Handle action dispatch
    if tool_name == "respond":
      msg_content = kwargs.get("content", args[0] if args else "")
      tau_action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": msg_content})
    elif tool_name in self.tools_map:
      tau_action = Action(name=tool_name, kwargs=kwargs)
    else:
      err_msg = f"Unknown tool '{tool_name}'. Available tools: respond, {list(self.tools_map.keys())}"
      self.trajectory_history.append(f"Observation: {err_msg}")
      return err_msg, self.terminated

    # Step tau-bench environment
    try:
      response = self.tau_env.step(tau_action)
      self.score = float(response.reward)
      self.terminated = response.done
      obs = response.observation

      if tool_name == "respond":
        obs_text = f"Customer: {obs}"
      else:
        obs_text = f"Tool Result ({tool_name}): {obs}"

      self.last_observation = obs_text
      self.trajectory_history.append(f"Observation: {obs_text}")

      if self.terminated:
        self.final_response = obs_text

      return obs_text, self.terminated
    except Exception as e:
      err_msg = f"Execution error in tool '{tool_name}': {e}"
      self.trajectory_history.append(f"Observation: {err_msg}")
      return err_msg, self.terminated

  # =========================================================================
  # Procedural Graph Node & Retrieval Hooks
  # =========================================================================

  def get_retrieval_query_hook(self, trajectory: List[str], last_obs: str) -> str:
    """Dynamic multi-turn retrieval query hook capturing recent observations and current goal."""
    recent_obs = last_obs or self.last_observation or self.initial_user_obs
    return f"Domain: {self.domain} | Task Goal: {self.instruction} | Latest User Observation: {recent_obs}"

  def get_current_node_id(self, trajectory: List[str]) -> str:
    """Deduces the current Node ID or high-level state milestone from execution trajectory history."""
    if not trajectory:
      return "START"

    last_tool = "START"
    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", action_content)
        if match:
          last_tool = match.group(1)
          break
        try:
          action_name, _, _ = env_base.parse_action_string(action_content)
          last_tool = action_name
          break
        except Exception:
          pass

    # Map tool actions to high-level procedural states if matching graph node exists
    if last_tool in ["find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"]:
      return "AUTHENTICATE_USER"
    elif last_tool in ["get_order_details", "get_product_details"]:
      return "CHECK_POLICY_RULES"
    elif last_tool in ["cancel_order", "modify_order_address", "modify_order_items", "return_order"]:
      return "EXECUTE_TOOL"
    elif last_tool == "respond":
      return "COMMUNICATE_CUSTOMER"

    return last_tool if last_tool != "START" else "START"
