"""Generalized Agent Solvers utilizing Procedural Graph and RAG guidance."""

import abc
import dataclasses
import re
from typing import Any, List, Optional, Protocol, Tuple
from . import prompts
from . import retriever
from .env_base import BaseEnvironment


class LLMClient(Protocol):
  """Decoupled protocol for LLM text generation and reasoning."""

  def generate(
      self,
      prompt: str,
      temperature: float = 0.0,
      system_instruction: Optional[str] = None,
  ) -> str:
    """Generates text completion from LLM."""
    ...


def parse_llm_response(response: str) -> Tuple[str, str]:
  """Extracts Thought and Action blocks robustly from LLM response."""
  response = response.strip()

  # 1. If "Action:" is present in the response
  action_prefix_match = re.search(r"\bAction:", response, re.IGNORECASE)
  if action_prefix_match:
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\bAction:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    action_match = re.search(
        r"\bAction:\s*(.*)", response, re.DOTALL | re.IGNORECASE
    )

    if thought_match:
      thought = thought_match.group(1).strip()
    else:
      # If Thought: prefix is missing, treat everything before Action: as thought
      before_action = response.split(action_prefix_match.group(0))[0]
      thought = before_action.strip()
      thought = re.sub(
          r"^Thought:\s*", "", thought, flags=re.IGNORECASE
      ).strip()

    action = action_match.group(1).strip() if action_match else ""

  # 2. If "Action:" is NOT present in the response
  else:
    lines = response.split("\n")
    last_line = lines[-1].strip()

    # Check if the last line matches a standard function call pattern
    if re.match(r"^\w+\(.*\)$", last_line):
      action = last_line
      thought = "\n".join(lines[:-1]).strip()
      thought = re.sub(
          r"^Thought:\s*", "", thought, flags=re.IGNORECASE
      ).strip()
    else:
      # No action found: treat the entire response as the thought reasoning
      thought = re.sub(
          r"^Thought:\s*", "", response, flags=re.IGNORECASE
      ).strip()
      action = ""

  # Unwrap triple backticks from the action if LLM formatted it inside code blocks
  markdown_match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```", action, re.DOTALL)
  if markdown_match:
    action = markdown_match.group(1).strip()

  # Clean trailing formatting over-generation (e.g. Observation: or Thought:)
  # from action
  if action:
    action = re.split(
        r"\b(Observation|Thought):", action, flags=re.IGNORECASE
    )[0].strip()

  return thought, action


class BaseSolver(abc.ABC):
  """Abstract base class representing a generic Procedural Graph solver."""

  def __init__(
      self,
      llm: LLMClient,
      retriever_obj: Optional[retriever.ProceduralGraphRetriever] = None,
  ):
    self.llm = llm
    self.retriever = retriever_obj

  @abc.abstractmethod
  def solve(
      self, env: BaseEnvironment, max_steps: int = 15
  ) -> Tuple[float, List[str]]:
    """Executes reasoning exploration to solve task and return trajectory."""
    pass

  def _get_guidance(self, env: BaseEnvironment, trajectory: List[str]) -> str:
    """Helper to retrieve and build procedural graph guidance."""
    if not self.retriever:
      return ""
    if hasattr(self.retriever, "get_guidance"):
      return self.retriever.get_guidance(env, trajectory)
    curr_node = env.get_current_node_id(trajectory)
    last_obs = ""
    for item in reversed(trajectory):
      if item.startswith("Observation:"):
        last_obs = item[len("Observation:") :].strip()
        break
    query = env.get_retrieval_query_hook(trajectory, last_obs)
    top_nodes = self.retriever.retrieve_top_k_nodes(
        curr_node, trajectory, query
    )
    return retriever.GuideBuilder.build_guidance(top_nodes)


class ReAct(BaseSolver):
  """ReAct solver with Notepad memory and dynamic history resetting."""

  def solve(
      self, env: BaseEnvironment, max_steps: int = 15
  ) -> Tuple[float, List[str]]:
    obs = env.reset()
    trajectory = [f"Observation: {obs}"]
    notepad = "Recent Notepad Notes: None"
    local_history: List[str] = []

    for _ in range(max_steps):
      if env.is_terminated():
        break

      # Dynamically reset local history/trajectory if step exceeds 6 turns
      # while capturing memory summaries inside notepad notes.
      if len(local_history) > 12:
        # Ask LLM to consolidate notepad notes with recent history safely in XML
        summary_prompt = prompts.NOTEPAD_CONSOLIDATION_PROMPT.format(
            existing_notepad=notepad,
            recent_history="\n".join(local_history),
        )
        summary = self.llm.generate(summary_prompt, temperature=0.0)
        notepad = f"Recent Notepad Notes: {summary.strip()}"
        local_history = []

      guidance = self._get_guidance(env, trajectory)

      raw_prompt = prompts.REACT_LOOP_INSTRUCTION.format(
          system_prompt=env.get_system_prompt(),
          procedural_graph_guidance="{procedural_graph_guidance}",
          notepad_history=notepad,
          trajectory="\n".join(trajectory[-9:]),
      )
      prompt = env.format_guidance_prompt_hook(guidance, raw_prompt)

      response = self.llm.generate(prompt, temperature=0.0)
      thought, action = parse_llm_response(response)

      if not action:
        action = response.strip()

      trajectory.append(f"Thought: {thought}")
      trajectory.append(f"Action: {action}")
      local_history.append(f"Thought: {thought}")
      local_history.append(f"Action: {action}")

      obs, _ = env.execute_action(action)
      if obs.startswith("Error parsing action:") or obs.startswith("Error: Action '"):
        trajectory[-1] = "Action: [Invalid Action Call Format - Failed to Parse]"
        local_history[-1] = "Action: [Invalid Action Call Format - Failed to Parse]"
      trajectory.append(f"Observation: {obs}")
      local_history.append(f"Observation: {obs}")

    return env.get_score(), trajectory


