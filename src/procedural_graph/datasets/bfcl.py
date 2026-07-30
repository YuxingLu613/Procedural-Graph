"""BFCL Dataset Environment Adapter subclassing BaseEnvironment."""

import json
import os
import copy
import re
import functools
from typing import Any, Dict, List, Optional, Tuple
from .. import env_base
from . import bfcl_apis

def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """Prepends the instance name to the function name in the call string."""
    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"
    return re.sub(pattern, replace_function, function_call_string)

class ToolRedirector:
    """Redirects tool calls dynamically to the current instance in env."""
    def __init__(self, env: "BfclEnv", class_name: str, attr_name: str):
        self.env = env
        self.class_name = class_name
        self.attr_name = attr_name
        
    def __call__(self, *args, **kwargs):
        instance = self.env.instances[self.class_name]
        method = getattr(instance, self.attr_name)
        try:
            res = method(*args, **kwargs)
            self.env._current_turn_results.append(res)
            return res
        except Exception as e:
            self.env._current_turn_results.append(f"Error during execution: {e}")
            raise e

class BfclEnv(env_base.BaseEnvironment):
  """BFCL evaluation environment adapter."""

  def __init__(
      self,
      sample_index: int,
      dataset_path: Optional[str] = None,
      possible_answer_path: Optional[str] = None,
  ):
    super().__init__()
    # Use workspace paths
    self.dataset_path = dataset_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/datasets/General/BFCL/BFCL_v3_multi_turn_base.json"))

    if not os.path.exists(self.dataset_path):
        raise FileNotFoundError(f"BFCL dataset not found at {self.dataset_path}")
    if not os.path.exists(self.possible_answer_path):
        raise FileNotFoundError(f"BFCL possible answers not found at {self.possible_answer_path}")

    with open(self.dataset_path, "r") as f:
        self.examples = [json.loads(line) for line in f]
    with open(self.possible_answer_path, "r") as f:
        self.answers = [json.loads(line) for line in f]

    if sample_index < 0 or sample_index >= len(self.examples):
        raise IndexError(f"sample_index {sample_index} out of range (total: {len(self.examples)})")

    self.example = self.examples[sample_index]
    self.answer = self.answers[sample_index]
    assert self.example["id"] == self.answer["id"], f"ID mismatch: {self.example['id']} vs {self.answer['id']}"

    self.sample_index = sample_index
    self.task_id = self.example["id"]
    self.questions = self.example["question"]
    self.initial_config = self.example["initial_config"]
    self.involved_classes = self.example["involved_classes"]

    self.current_turn = 0
    self.total_turns = len(self.questions)
    self.terminated = False
    self.score = 0.0

    self.instances: Dict[str, Any] = {}
    self.gt_instances: Dict[str, Any] = {}
    self.model_responses_by_turn: List[List[Any]] = []
    self.gt_responses_by_turn: List[List[Any]] = []
    self._current_turn_results: List[Any] = []

    self._setup_instances()
    self._register_dynamic_tools()

  def _setup_instances(self):
    self.instances = {}
    self.gt_instances = {}
    for class_name in self.involved_classes:
      if not hasattr(bfcl_apis, class_name):
          raise AttributeError(f"Class {class_name} not found in bfcl_apis")
      cls = getattr(bfcl_apis, class_name)
      
      # Model instance
      instance = cls()
      if hasattr(instance, "_load_scenario"):
        class_initial_config = self.initial_config.get(class_name, {})
        long_context = "long_context" in self.task_id or "composite" in self.task_id
        instance._load_scenario(copy.deepcopy(class_initial_config), long_context=long_context)
      self.instances[class_name] = instance
      
      # GT instance
      gt_instance = cls()
      if hasattr(gt_instance, "_load_scenario"):
        class_initial_config = self.initial_config.get(class_name, {})
        long_context = "long_context" in self.task_id or "composite" in self.task_id
        gt_instance._load_scenario(copy.deepcopy(class_initial_config), long_context=long_context)
      self.gt_instances[class_name] = gt_instance

  def _register_dynamic_tools(self):
    for class_name, instance in self.instances.items():
      for attr_name in dir(instance):
        if attr_name.startswith("_"):
          continue
        bound_method = getattr(instance, attr_name)
        if callable(bound_method):
          doc = bound_method.__doc__ or f"Method {attr_name} of class {class_name}"
          metadata = env_base.ToolMetadata(
              name=attr_name,
              description=doc,
              is_state_changing=True
          )
          redirector = ToolRedirector(self, class_name, attr_name)
          functools.update_wrapper(redirector, bound_method)
          self._tools[attr_name] = (redirector, metadata)

  def get_system_prompt(self) -> str:
    tools_desc = []
    for name, (_, metadata) in self._tools.items():
      if name == "Finish":
        continue
      tools_desc.append(f" - {name}: {metadata.description.strip()}")
    tools_str = "\n".join(tools_desc)
    
    return (
        "You are an AI Agent solving complex multi-turn tool-use challenges.\n"
        "You must execute the necessary tool calls to satisfy the user's request.\n"
        "At each turn, you will receive an instruction from the user.\n"
        "You can call multiple tools if needed.\n"
        "When you have completed all actions for the current turn, you MUST call Finish() to submit and get the next instruction.\n\n"
        "Available Tool Actions:\n"
        f"{tools_str}\n"
        " - Finish(final_response=\"<final_response>\"):\n"
        "   Call this to complete the current turn. CRITICAL: Use escaped newlines (\\n) for line breaks. Do not use raw unescaped newlines."
    )

  def is_terminated(self) -> bool:
    return self.terminated

  def get_score(self) -> float:
    return self.score

  def reset(self) -> str:
    self.current_turn = 0
    self.terminated = False
    self.score = 0.0
    self.model_responses_by_turn = []
    self.gt_responses_by_turn = []
    self._current_turn_results = []
    self._setup_instances()
    return self.get_current_turn_prompt()

  def get_current_turn_prompt(self) -> str:
    messages = self.questions[self.current_turn]
    for msg in messages:
      if msg["role"] == "user":
        return msg["content"]
    return ""

  def get_state(self) -> Any:
    cloned_instances = copy.deepcopy(self.instances)
    cloned_gt_instances = copy.deepcopy(self.gt_instances)
    return (
        self.current_turn,
        self.terminated,
        self.score,
        copy.deepcopy(self.model_responses_by_turn),
        copy.deepcopy(self.gt_responses_by_turn),
        copy.deepcopy(self._current_turn_results),
        cloned_instances,
        cloned_gt_instances
    )

  def load_state(self, state: Any) -> None:
    (
        self.current_turn,
        self.terminated,
        self.score,
        model_resp,
        gt_resp,
        curr_results,
        cloned_instances,
        cloned_gt_instances
    ) = state
    
    self.model_responses_by_turn = copy.deepcopy(model_resp)
    self.gt_responses_by_turn = copy.deepcopy(gt_resp)
    self._current_turn_results = copy.deepcopy(curr_results)
    self.instances = cloned_instances
    self.gt_instances = cloned_gt_instances

  @env_base.register_tool(
      "Finish",
      "Terminal state-changing tool that submits final response for the current turn.",
      is_state_changing=True,
  )
  def Finish(self, final_response: str = "") -> str:
    self.model_responses_by_turn.append(self._current_turn_results)
    
    # Execute GT calls
    gt_calls = self.answer["ground_truth"][self.current_turn]
    gt_results = self._execute_gt_calls(gt_calls)
    self.gt_responses_by_turn.append(gt_results)
    
    # Run checks for this turn
    state_check = self._check_states()
    if not state_check["valid"]:
      self.score = 0.0
      self.terminated = True
      print(f"❌ [BfclEnv] State mismatch at turn {self.current_turn}: {state_check['error_message']}", flush=True)
      return f"Task failed at turn {self.current_turn} due to state mismatch. Error: {state_check['error_message']}"
      
    flat_model_res = [item for sublist in self.model_responses_by_turn for item in sublist]
    flat_gt_res = [item for sublist in self.gt_responses_by_turn for item in sublist]
    
    response_check = self._check_responses(flat_model_res, flat_gt_res)
    if not response_check["valid"]:
      self.score = 0.0
      self.terminated = True
      print(f"❌ [BfclEnv] Response mismatch at turn {self.current_turn}: {response_check['error_message']}", flush=True)
      return f"Task failed at turn {self.current_turn} due to response mismatch. Error: {response_check['error_message']}"
      
    # Progress to next turn
    self.current_turn += 1
    if self.current_turn >= self.total_turns:
      self.terminated = True
      self.score = 1.0
      print(f"✅ [BfclEnv] Task completed successfully (all turns passed)", flush=True)
      return f"Task completed successfully. Final response: {final_response}"
    else:
      self._current_turn_results = []
      next_prompt = self.get_current_turn_prompt()
      print(f"⏭️ [BfclEnv] Turn {self.current_turn-1} passed. Starting Turn {self.current_turn}", flush=True)
      return f"Turn completed. Next user instruction: {next_prompt}"

  def _execute_gt_calls(self, gt_calls: List[str]) -> List[Any]:
    namespace = {}
    class_method_name_mapping = {}
    for class_name, instance in self.gt_instances.items():
      var_name = f"gt_{class_name.lower()}"
      namespace[var_name] = instance
      for attr_name in dir(instance):
        if attr_name.startswith("_"):
          continue
        attr = getattr(instance, attr_name)
        if callable(attr):
          class_method_name_mapping[attr_name] = var_name
          
    results = []
    for call_str in gt_calls:
      processed_call = _process_method_calls(call_str, class_method_name_mapping)
      try:
        # Use a safe eval with empty globals and our namespace as locals
        res = eval(processed_call, {"__builtins__": {}}, namespace)
        results.append(res)
      except Exception as e:
        results.append(f"Error during GT execution: {e}")
        print(f"Error executing GT call '{processed_call}': {e}", flush=True)
    return results

  def _check_states(self) -> dict:
    for class_name, gt_instance in self.gt_instances.items():
      model_instance = self.instances[class_name]
      valid, differences = self._compare_instances(model_instance, gt_instance)
      if not valid:
        return {
            "valid": False,
            "error_message": f"Model instance for {class_name} does not match the state with ground truth instance. Diff: {differences}",
            "differences": differences
        }
    return {"valid": True}
    
  def _compare_instances(self, model_obect, ground_truth_object):
    assert type(model_obect) == type(ground_truth_object), "Objects are not of the same type."
    differences = {}
    valid = True
    for attr_name in vars(ground_truth_object):
      if attr_name.startswith("_"):
        continue
      model_attr = getattr(model_obect, attr_name)
      ground_truth_attr = getattr(ground_truth_object, attr_name)
      if model_attr != ground_truth_attr:
        valid = False
        differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}
    return valid, differences

  def _check_responses(self, model_res: list, gt_res: list) -> dict:
    is_subsequence, missing_items = self._is_subsequence_unordered(gt_res, model_res)
    if not is_subsequence:
      return {
          "valid": False,
          "error_message": f"Model response execution results so far does not contain all the ground truth response execution results. Missing: {missing_items}",
          "missing_items": missing_items
      }
    return {"valid": True}

  def _is_subsequence_unordered(self, list1, list2) -> tuple[bool, list]:
    list2_copy = list2[:]
    missing_elements = []
    for item in list1:
      try:
        list2_copy.remove(item)
      except ValueError:
        missing_elements.append(item)
    is_subsequence = len(missing_elements) == 0
    return is_subsequence, missing_elements

  # Override hooks if needed
  def get_current_node_id(self, trajectory: List[str]) -> str:
    # Deduced from trajectory
    if not trajectory:
      return "START"
    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", action_content)
        if match:
          return match.group(1)
    return "START"
