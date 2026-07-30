import abc
import ast
import dataclasses
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclasses.dataclass
class ToolMetadata:
  """Metadata for tools registered in the environment."""

  name: str
  description: str
  is_state_changing: bool = False


def register_tool(name: str, description: str, is_state_changing: bool = False):
  """Decorator to register a method of BaseEnvironment subclass as a tool."""

  def decorator(func: Callable[..., Any]):
    if not hasattr(func, "_tool_metadata"):
      func._tool_metadata = []
    func._tool_metadata.append((name, description, is_state_changing))
    return func

  return decorator


def parse_action_string(
    action_str: str,
) -> Tuple[str, List[Any], Dict[str, Any]]:
  """Parses a function call string like 'tool_name(arg1, arg2, kw1="val1")'."""
  action_str = action_str.strip()
  if not action_str:
    raise ValueError("Action string is empty.")

  try:
    # Use AST to parse safely
    tree = ast.parse(action_str, mode="eval")
    if not isinstance(tree.body, ast.Call):
      raise ValueError("Action must be a single function call.")

    # Extract function name
    if isinstance(tree.body.func, ast.Name):
      func_name = tree.body.func.id
    else:
      raise ValueError("Function name must be a simple identifier.")

    # Extract positional arguments
    args = []
    for arg_node in tree.body.args:
      args.append(ast.literal_eval(arg_node))

    # Extract keyword arguments
    kwargs = {}
    for keyword in tree.body.keywords:
      if keyword.arg is None:
        raise ValueError("Position-only keywords are not allowed.")
      kwargs[keyword.arg] = ast.literal_eval(keyword.value)

    return func_name, args, kwargs
  except Exception as e:
    raise ValueError(
        f"Failed to parse action call string '{action_str}': {e}"
    ) from e


class BaseEnvironment(abc.ABC):
  """Abstract base class for task environments with dynamic tool interfaces."""

  def __init__(self):
    self._tools: Dict[str, Tuple[Callable[..., Any], ToolMetadata]] = {}
    # Discover and bind registered tools
    for attr_name in dir(self):
      attr = getattr(self, attr_name)
      if callable(attr) and hasattr(attr, "_tool_metadata"):
        for name, desc, is_state_changing in attr._tool_metadata:
          metadata = ToolMetadata(
              name=name, description=desc, is_state_changing=is_state_changing
          )
          # The attribute attr is already bound to self on lookup
          self._tools[name] = (attr, metadata)

  def get_tools_metadata(self) -> List[ToolMetadata]:
    """Returns a list of metadata for all registered tools."""
    return [meta for _, meta in self._tools.values()]

  @abc.abstractmethod
  def get_system_prompt(self) -> str:
    """Returns the system prompt describing instructions/rules for the task."""
    pass

  @abc.abstractmethod
  def is_terminated(self) -> bool:
    """Returns True if the task is terminated (success or failure)."""
    pass

  @abc.abstractmethod
  def get_score(self) -> float:
    """Returns the current evaluation score of the environment."""
    pass

  @abc.abstractmethod
  def reset(self) -> str:
    """Resets the environment and returns the initial observation."""
    pass

  @abc.abstractmethod
  def get_state(self) -> Any:
    """Clones and returns the current environment state checkpoint."""
    pass

  @abc.abstractmethod
  def load_state(self, state: Any) -> None:
    """Restores the environment state from a checkpoint."""
    pass

  def execute_action(self, action_str: str) -> Tuple[str, bool]:
    """Parses and dispatches action proposal string to the registered tool."""
    try:
      sanitized_action = self.parse_proposal_hook(action_str)
      print(f"🏃‍♂️ [BaseEnvironment] Executing: {sanitized_action}", flush=True)
      action_name, args, kwargs = parse_action_string(sanitized_action)
    except Exception as e:
      print(f"❌ [BaseEnvironment] Failed parsing action '{action_str}': {e}", flush=True)
      return f"Error parsing action: {e}", self.is_terminated()

    if action_name not in self._tools:
      print(f"❌ [BaseEnvironment] Action '{action_name}' is not registered.", flush=True)
      return (
          f"Error: Action '{action_name}' is not registered.",
          self.is_terminated(),
      )

    func, _ = self._tools[action_name]
    try:
      sig = inspect.signature(func)
      bound = sig.bind(*args, **kwargs)
      result = func(*bound.args, **bound.kwargs)
      res_str = str(result)
      max_obs_len = 100000
      if len(res_str) > max_obs_len:
        print(f"⚠️ [Truncation] Observation too long ({len(res_str)} chars). Truncating to {max_obs_len} chars.", flush=True)
        res_str = res_str[:max_obs_len // 2] + "\n...[TRUNCATED TO PROTECT CONTEXT WINDOW]...\n" + res_str[-max_obs_len // 2:]
      print(f"👉 [BaseEnvironment] Result: {res_str[:300]}...", flush=True)
      return res_str, self.is_terminated()
    except TypeError as e:
      return (
          f"TypeError executing '{action_name}': {e}",
          self.is_terminated(),
      )
    except Exception as e:
      return (
          f"Error executing '{action_name}': {e}",
          self.is_terminated(),
      )

  def load_initial_graph(
      self, custom_path: Optional[str] = None, rich_pg: bool = False
  ) -> Any:
    """Loads the Procedural Graph for this environment (custom or default JSON)."""
    from .core import ProceduralGraph

    if custom_path:
      return ProceduralGraph.load_from_json(custom_path)

    import os

    # Inspect the module file path of the concrete subclass
    module_file = inspect.getfile(self.__class__)
    module_dir = os.path.dirname(os.path.abspath(module_file))

    # Convention: cfo_env -> cfo, swebench -> swebench
    class_name = self.__class__.__name__.lower()
    if class_name.endswith("environment"):
      prefix = class_name[:-11]
    elif class_name.endswith("env"):
      prefix = class_name[:-3]
    else:
      prefix = class_name

    pg_filename = f"{prefix}_procedural_graph.json"
    legacy_filename = (
        f"{prefix}_rich_procedural_graph.json"
        if rich_pg
        else f"{prefix}_procedural_graph.json"
    )
    candidates = [
        os.path.abspath(os.path.join(module_dir, "../../../data/graphs", pg_filename)),
        os.path.abspath(os.path.join(module_dir, "../../data/graphs", pg_filename)),
        os.path.abspath(os.path.join(module_dir, "../data/graphs", pg_filename)),
        os.path.join(module_dir, pg_filename),
        os.path.join(module_dir, "datasets", pg_filename),
        # Fallbacks for legacy filenames during transition
        os.path.abspath(os.path.join(module_dir, "../../../data/graphs", legacy_filename)),
        os.path.abspath(os.path.join(module_dir, "../../data/graphs", legacy_filename)),
    ]
    json_path = None
    for cand in candidates:
      if os.path.exists(cand):
        json_path = cand
        break

    if not json_path:
      raise FileNotFoundError(
          "Could not load initial procedural graph asset. JSON file not found in any candidates:"
          f" {candidates}"
      )

    return ProceduralGraph.load_from_json(json_path)

  # =========================================================================
  #  Dataset-specific Hooks with default implementations
  # =========================================================================

  def get_current_node_id(self, trajectory: List[str]) -> str:
    """[Hook 1/4] Deduces the current Node ID from execution trajectory history."""
    if not trajectory:
      return "START"

    for item in reversed(trajectory):
      # Try to locate the last action in trajectory to deduce the current state
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        # Try regex first to handle syntax/argument parsing errors gracefully
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", action_content)
        if match:
          return match.group(1)
        try:
          action_name, _, _ = parse_action_string(action_content)
          return action_name
        except Exception:
          pass
    return "START"

  def get_retrieval_query_hook(
      self, trajectory: List[str], current_obs: str
  ) -> str:
    """[Hook 2/4] Prepares the query string for similarity scorer."""
    return current_obs

  def parse_proposal_hook(self, llm_proposal: str) -> str:
    """[Hook 3/4] Sanitizes LLM action string proposals."""
    return llm_proposal.strip()

  def format_guidance_prompt_hook(
      self, rules: str, prompt_template: str
  ) -> str:
    """[Hook 4/4] Injects RAG procedural graph guidance into prompt templates."""
    prompt_template = prompt_template.replace("{procedural_graph_guidance}", rules)
    return prompt_template.replace("{procedural_graph_guidance}", rules)

  def compress_trajectory_hook(self, trajectory: List[str]) -> str:
    """[Hook 5/5] Compresses long raw trajectory lists into a compact, semantically dense representation for LLM refinement context."""
    # Default fallback: raw join of steps
    return "\n".join(trajectory)
