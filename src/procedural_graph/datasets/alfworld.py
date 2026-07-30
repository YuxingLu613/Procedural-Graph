"""ALFWorld TextWorld evaluation environment adapter subclassing BaseEnvironment."""

import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .. import env_base


class ALFWorldEnv(env_base.BaseEnvironment):
  """ALFWorld TextWorld evaluation environment adapter."""

  _env_cache = None
  _env_cache_lock = threading.Lock()

  def __init__(
      self,
      sample_index: int = 0,
      dataset_path: Optional[str] = None,
      llm_client: Optional[Any] = None,
      split: str = "valid_unseen",
  ):
    super().__init__()
    self.split = split

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    _alf_candidates = [
        os.path.join(project_root, "dataset/Agent/ALFWorld"),
        os.path.abspath(os.path.join(project_root, "../data/datasets/Agent/ALFWorld")),
        os.path.abspath(os.path.join(project_root, "../../dataset/Agent/ALFWorld")),
    ]
    alfworld_data_dir = _alf_candidates[0]
    for _dir in _alf_candidates:
      if os.path.exists(_dir):
        alfworld_data_dir = _dir
        _env_lib = os.path.join(_dir, "alfworld_env")
        if os.path.exists(_env_lib) and _env_lib not in sys.path:
          sys.path.insert(0, _env_lib)
        break
    os.environ["ALFWORLD_DATA"] = alfworld_data_dir

    # Create config dict dynamically
    config = {
        "dataset": {
            "data_path": os.path.join(alfworld_data_dir, "json_2.1.1/train"),
            "eval_id_data_path": os.path.join(
                alfworld_data_dir, "json_2.1.1/valid_seen"
            ),
            "eval_ood_data_path": os.path.join(
                alfworld_data_dir, "json_2.1.1/valid_unseen"
            ),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": os.path.join(alfworld_data_dir, "logic/alfred.pddl"),
            "grammar": os.path.join(alfworld_data_dir, "logic/alfred.twl2"),
        },
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "controller": {
            "type": "oracle",
            "debug": False,
            "load_receps": True,
        },
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "visdom": False,
            "task": "alfred",
            "training_method": "dqn",  # Prevent AlfredExpert wrapper from attaching and calling inventory.pop() during eval
            "observation_pool_capacity": 3,
            "hide_init_receptacles": False,
        },
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": 50,
            }
        },
        "rl": {
            "training": {
                "max_nb_steps_per_episode": 50,
            }
        },
    }

    import alfworld
    import alfworld.agents.environment as environment

    # Monkey-patch textworld PDDL parser to be thread-safe
    try:
      import textworld.envs.pddl.logic as pddl_logic
      if not hasattr(pddl_logic, "_locked_parse_and_convert"):
        orig_pddl_logic_parse = pddl_logic._parse_and_convert
        pddl_logic_lock = threading.RLock()
        def locked_pddl_logic_parse(*args, **kwargs):
          with pddl_logic_lock:
            return orig_pddl_logic_parse(*args, **kwargs)
        pddl_logic._parse_and_convert = locked_pddl_logic_parse
        pddl_logic._locked_parse_and_convert = True
        print("Successfully monkey-patched textworld PDDL parser with lock.")
    except Exception as e:
      print(f"Warning: Failed to monkey-patch textworld parser: {e}")

    try:
      import textworld.envs.pddl.textgen as pddl_textgen
      if not hasattr(pddl_textgen, "_locked_parse_and_convert"):
        orig_pddl_textgen_parse = pddl_textgen._parse_and_convert
        pddl_textgen_lock = threading.RLock()
        def locked_pddl_textgen_parse(*args, **kwargs):
          with pddl_textgen_lock:
            return orig_pddl_textgen_parse(*args, **kwargs)
        pddl_textgen._parse_and_convert = locked_pddl_textgen_parse
        pddl_textgen._locked_parse_and_convert = True
        print("Successfully monkey-patched textworld PDDL textgen parser with lock.")
    except Exception as e:
      print(f"Warning: Failed to monkey-patch textworld textgen parser: {e}")

    try:
      import textworld.logic as tw_logic
      if not hasattr(tw_logic, "_locked_parse_and_convert"):
        orig_tw_logic_parse = tw_logic._parse_and_convert
        tw_logic_lock = threading.RLock()
        def locked_tw_logic_parse(*args, **kwargs):
          with tw_logic_lock:
            return orig_tw_logic_parse(*args, **kwargs)
        tw_logic._parse_and_convert = locked_tw_logic_parse
        tw_logic._locked_parse_and_convert = True
        print("Successfully monkey-patched textworld logic parser with lock.")
    except Exception as e:
      print(f"Warning: Failed to monkey-patch textworld logic parser: {e}")

    # Monkey-patch fast_downward.pddl2sas to be thread-safe
    try:
      import fast_downward
      if not hasattr(fast_downward, "_locked_pddl2sas"):
        orig_pddl2sas = fast_downward.pddl2sas
        pddl2sas_lock = threading.RLock()
        def locked_pddl2sas(*args, **kwargs):
          with pddl2sas_lock:
            return orig_pddl2sas(*args, **kwargs)
        fast_downward.pddl2sas = locked_pddl2sas
        fast_downward._locked_pddl2sas = True
        print("Successfully monkey-patched fast_downward.pddl2sas with lock.")
    except Exception as e:
      print(f"Warning: Failed to monkey-patch fast_downward: {e}")

    # Map split to internal alfworld training/eval split name
    if self.split == "train":
      train_eval_mapped = "train"
    elif self.split in ("valid_seen", "valid_unseen"):
      train_eval_mapped = "eval_out_of_distribution"
    else:
      raise ValueError(f"Unknown split: {self.split}")

    # Initialize environment generator with thread-isolated instantiation
    with ALFWorldEnv._env_cache_lock:
      if (
          ALFWorldEnv._env_cache is None
          or ALFWorldEnv._env_cache[0] != self.split
      ):
        env_class = environment.get_environment(config["env"]["type"])
        base_env = env_class(config, train_eval=train_eval_mapped)
        ALFWorldEnv._env_cache = (self.split, base_env)

      self.base_env = ALFWorldEnv._env_cache[1]
      self.game_env = self.base_env.init_env(batch_size=1)

      # Skip to the specified game sample index
      total_games = len(self.base_env.game_files)
      if sample_index < 0 or sample_index >= total_games:
        raise IndexError(
            f"sample_index {sample_index} out of range (total games: {total_games})"
        )
      self.game_env.skip(sample_index)

    self.sample_index = sample_index
    self.current_obs = ""
    self.admissible_actions = []
    self.terminated = False
    self.score = 0.0
    self.trajectory_history = []
    self.game_file_path = ""

  def close(self) -> None:
    if hasattr(self, "game_env") and self.game_env:
      try:
        self.game_env.close()
      except Exception:
        pass

  def get_system_prompt(self) -> str:
    return (
        "You are an expert agent solving an embodied household task.\n"
        f"Initial State Observation:\n{self.current_obs}\n\n"
        "Available Tool Actions:\n"
        " - take_action(action='<action_string>'): Execute a text action (e.g. 'go to drawer 1', 'take spatula 1 from drawer 1').\n"
        " - look(): Re-observe the room environment state.\n"
        " - check_valid_actions(): List all currently valid/admissible text actions."
    )

  def is_terminated(self) -> bool:
    return self.terminated

  def get_score(self) -> float:
    return self.score

  def reset(self) -> str:
    self.terminated = False
    self.score = 0.0
    self.trajectory_history = []

    obs, info = self.game_env.reset()
    self.current_obs = self.clean_obs(obs[0])
    self.target_question = self.current_obs
    self.admissible_actions = info["admissible_commands"][0]
    self.game_file_path = info["extra.gamefile"][0]

    return f"Environment loaded. Initial Observation:\n{self.current_obs}"

  def get_state(self) -> Any:
    return (
        self.terminated,
        self.score,
        self.current_obs,
        list(self.admissible_actions),
        list(self.trajectory_history),
    )

  def load_state(self, state: Any) -> None:
    self.terminated, self.score, self.current_obs, actions, traj = state
    self.admissible_actions = list(actions)
    self.trajectory_history = list(traj)

  @env_base.register_tool("look", "Observe the environment again.")
  def look(self) -> str:
    try:
      obs, _, done, info = self.game_env.step(["look"])
      self.current_obs = self.clean_obs(obs[0])
      self.admissible_actions = info["admissible_commands"][0]
      self.terminated = done[0]
    except Exception as e:
      return f"Observation: Nothing happens (environment notice: {str(e)})."
    return self.current_obs

  @env_base.register_tool(
      "check_valid_actions", "Get the list of currently admissible actions."
  )
  def check_valid_actions(self) -> str:
    valid_str = ", ".join(self.admissible_actions)
    return f"Admissible actions: {valid_str}"

  @env_base.register_tool(
      "take_action",
      "Execute a valid text action in the room.",
      is_state_changing=True,
  )
  def take_action(self, action: str) -> str:
    action_clean = action.strip()
    try:
      obs, rewards, done, info = self.game_env.step([action_clean])
      self.current_obs = self.clean_obs(obs[0])
      self.admissible_actions = info["admissible_commands"][0]
      self.terminated = done[0]
      self.score = float(rewards[0])
    except Exception as e:
      return f"Observation: Invalid action '{action_clean}' (environment notice: {str(e)})."
    return self.current_obs

  def clean_obs(self, obs: str) -> str:
    cleaned = re.sub(r'[-_=\\\/\$\|>]', "", obs).strip()
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    # Strip textworld banner
    if "TextWorld" in cleaned:
      cleaned = "\n".join(cleaned.split("\n")[2:])
    return cleaned.strip()

  # =========================================================================
  # Procedural Graph Node Mapping Hook
  # =========================================================================

  def parse_proposal_hook(self, llm_proposal: str) -> str:
    prop = llm_proposal.strip()
    # Strip trailing Observation/Thought keywords if LLM over-generated
    prop = re.split(r"\b(Observation|Thought):", prop, flags=re.IGNORECASE)[0].strip()
    if not re.match(r"^\w+\(.*\)$", prop):
      # Wrap raw textworld command in take_action tool call
      clean_prop = prop.replace('"', '\\"')
      return f'take_action(action="{clean_prop}")'
    return prop

  def get_current_node_id(self, trajectory: List[str]) -> str:
    if not trajectory:
      return "START"

    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        act_str = ""
        try:
          action_name, args, kwargs = env_base.parse_action_string(
              action_content
          )
          name_lower = action_name.lower()
          if "look" in name_lower:
            return "Look"
          elif "check" in name_lower:
            return "Check_Valid_Actions"
          elif "take_action" in name_lower:
            if args:
              act_str = str(args[0])
            elif "action" in kwargs:
              act_str = str(kwargs["action"])
        except Exception:
          # Fallback for raw commands or parse failures
          match = re.match(r"^take_action\(\s*(?:action=)?['\"](.*?)['\"]\s*\)", action_content, re.IGNORECASE)
          if match:
            act_str = match.group(1)
          else:
            act_str = action_content

        if act_str:
          act_str = act_str.strip().lower()
          if act_str.startswith("go to") or act_str.startswith("go "):
            return "Go_To_Location"
          elif act_str.startswith("take") or act_str.startswith("pick up"):
            return "Take_Object"
          elif act_str.startswith("put") or act_str.startswith("place"):
            return "Put_Object"
          elif act_str.startswith("clean") or act_str.startswith("wash"):
            return "Clean_Object"
          elif act_str.startswith("heat") or act_str.startswith("warm"):
            return "Heat_Object"
          elif act_str.startswith("cool") or act_str.startswith("freeze") or act_str.startswith("chill"):
            return "Cool_Object"
          elif any(act_str.startswith(k) for k in ("use", "toggle", "turn on", "turn off", "open", "close", "examine", "slice")):
            return "Use_Object"
          return "Take_Action"
    return "START"

  def get_retrieval_query_hook(
      self, trajectory: List[str], current_obs: str
  ) -> str:
    return f"Task: {self.target_question}. Observation: {current_obs}"
