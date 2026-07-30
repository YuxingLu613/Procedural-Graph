"""CFO Financial Simulation Environment Adapter subclassing BaseEnvironment."""

from typing import Any, Dict, List, Optional, Tuple
from .. import env_base

import copy
import dataclasses
import json
import os
import sys

_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_cfo_candidates = [
    os.path.join(_DIR, "CFO-Env-F1B9"),
    os.path.abspath(os.path.join(_DIR, "../data/datasets/Finance/CFO-Env/CFO-Env-F1B9")),
    os.path.abspath(os.path.join(_DIR, "../../CFO-Env-F1B9")),
]
for cfo_env_dir in _cfo_candidates:
  if os.path.exists(cfo_env_dir) and cfo_env_dir not in sys.path:
    sys.path.insert(0, cfo_env_dir)
    break

# pytype: disable=import-error
from arena import create_arena, EnterpriseArena
# pytype: enable=import-error


@dataclasses.dataclass
class Note:
  id: str
  content: str
  created_month: int
  tags: List[str] = dataclasses.field(default_factory=list)


class AgentMemory:
  """Simple note-taking memory for the agent."""

  def __init__(self):
    self.notes: Dict[str, Note] = {}
    self._counter = 0

  def save_note(self, content: str, tags: Optional[List[str]] = None, current_month: int = 0) -> Dict[str, Any]:
    self._counter += 1
    nid = f"note_{self._counter}"
    self.notes[nid] = Note(id=nid, content=content, created_month=current_month, tags=tags or [])
    return {"success": True, "note_id": nid, "message": f"Note saved: {nid}"}

  def recall_notes(self, query: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 10) -> Dict[str, Any]:
    matches = []
    for note in self.notes.values():
      if tags and not any(t in note.tags for t in tags):
        continue
      if query and query.lower() not in note.content.lower():
        continue
      matches.append({
          "id": note.id,
          "content": note.content,
          "created_month": note.created_month,
          "tags": note.tags,
      })
    matches.sort(key=lambda x: x["created_month"], reverse=True)
    return {"success": True, "count": len(matches[:limit]), "notes": matches[:limit]}

  def clear(self):
    self.notes.clear()
    self._counter = 0


class CFOEnvAdapter:
  """Wraps EnterpriseArena into the standard Procedural Graph Environment interface."""

  def __init__(self, config_path: str, expected_response: str = ""):
    self.config_path = os.path.abspath(config_path)
    self.expected_response = expected_response
    self.arena = create_arena(self.config_path)
    self.llm = None
    self.memory = AgentMemory()
    self.reset()

  def set_llm_engine(self, llm_engine: Any):
    """No-op: CFO-Env executes actual python code and does not simulate tools."""
    self.llm = llm_engine

  def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
    """Resets the arena to Month 0."""
    self.memory.clear()
    return self.arena.reset(seed=seed)

  # ---- Info-only Tools ------------------------------------------------

  def check_cash_in_bank(self, **kwargs) -> str:
    """Tool: Checks cash balances, debt, and customer gross totals."""
    try:
      res, success = self.arena.use_tool("check_cash_in_bank")
      if not success:
        return f"Error: {res.message}"
      data = res.data
      if kwargs:
        data["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(data, indent=2, default=str)
    except Exception as e:
      return f"Error executing check_cash_in_bank: {e}"

  def access_financial_docs(self, action: Optional[str] = None, filename: str = "", **kwargs) -> str:
    """Tool: List or retrieve financial documents from the document registry."""
    try:
      if not action:
        return "Error: 'action' is a required argument for access_financial_docs() (must be 'list' or 'retrieve')."
      res, success = self.arena.use_tool("access_financial_docs", action=action, filename=filename)
      if not success:
        return f"Error: {res.message}"
      data = res.data
      if kwargs:
        data["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(data, indent=2, default=str)
    except Exception as e:
      return f"Error executing access_financial_docs: {e}"

  def cash_flow_forecast_calculation(self, months: int = 3, growth_rate_override: Optional[float] = None, **kwargs) -> str:
    """Tool: Generates multi-month cash flow forecasts (heuristic projections)."""
    try:
      kwargs_dict = {"months": months}
      if growth_rate_override is not None:
        kwargs_dict["growth_rate_override"] = growth_rate_override
      res, success = self.arena.use_tool("cash_flow_forecast_calculation", **kwargs_dict)
      if not success:
        return f"Error: {res.message}"
      data = res.data
      if kwargs:
        data["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(data, indent=2, default=str)
    except Exception as e:
      return f"Error executing cash_flow_forecast_calculation: {e}"

  def check_market_data(self, data_type: Optional[str] = None, start_month: int = 0, end_month: Optional[int] = None, **kwargs) -> str:
    """Tool: Query historical macroeconomic and credit indicators."""
    try:
      if not data_type:
        return (
            "Error: 'data_type' is a required argument for check_market_data(). "
            "Supported types: ['GDP', 'CPI', 'UNRATE', 'SOFR', 'FEDFUNDS', 'Baa_Yield', "
            "'Tsy2Y', 'Tsy5Y', 'Tsy10Y', 'Tsy30Y', 'VIX', 'PE_Ratio', 'PS_Ratio', "
            "'Gross_Margin', 'EBITDA_Margin', 'Monthly_User_Growth', 'Annual_User_Growth']"
        )
      kwargs_dict = {"data_type": data_type, "start_month": start_month}
      if end_month is not None:
        kwargs_dict["end_month"] = end_month
      res, success = self.arena.use_tool("check_market_data", **kwargs_dict)
      if not success:
        return f"Error: {res.message}"
      data = res.data
      if kwargs:
        data["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(data, indent=2, default=str)
    except Exception as e:
      return f"Error executing check_market_data: {e}"

  # ---- Memory Tools (Notpad persistent notepad) ------------------------

  def save_note(self, content: Optional[str] = None, tags: str = "", **kwargs) -> str:
    """Memory: Saves a persistent note to carry context across months."""
    try:
      if not content:
        return "Error: 'content' is a required argument for save_note()."
      tag_list = [t.strip() for t in tags.split(",")] if tags else []
      res = self.memory.save_note(content=content, tags=tag_list, current_month=self.arena.env_state.current_month)
      if kwargs:
        res["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(res, indent=2)
    except Exception as e:
      return f"Error executing save_note: {e}"

  def recall_notes(self, query: str = "", tags: str = "", limit: int = 10, **kwargs) -> str:
    """Memory: Recalls persistent notes by keyword or tags."""
    try:
      tag_list = [t.strip() for t in tags.split(",")] if tags else None
      res = self.memory.recall_notes(query=query or None, tags=tag_list, limit=limit)
      if kwargs:
        res["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(res, indent=2)
    except Exception as e:
      return f"Error executing recall_notes: {e}"

  # ---- State-changing Actions (automatically rolls the month) ---------

  def fund_raising_request(self, type: Optional[str] = None, amount: Optional[float] = None, **kwargs) -> str:
    """Action: Submits debt or equity request and advances the calendar."""
    try:
      if not type or amount is None:
        return "Error: both 'type' (equity/debt) and 'amount' are required arguments for fund_raising_request()."
      res, success = self.arena.take_action("fund_raising_request", type=type, amount=amount)
      
      # Record agent decision log in arena (Part 2: trajectory)
      self.arena.log_agent_event("action_execution", f"Executed fundraising of type={type} for amount={amount}")
      
      # Roll the month automatically!
      step_res = self.arena.end_step()
      
      out = {
          "action_result": {
              "success": success,
              "message": res.message,
              "data": res.data
          },
          "month_end_update": {
              "new_month": self.arena.env_state.current_month,
              "terminated": step_res.terminated,
              "reason": step_res.info.get("termination_reason"),
              "reward": step_res.reward,
              "current_observation": step_res.observation
          }
      }
      if kwargs:
        out["action_result"]["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(out, indent=2, default=str)
    except Exception as e:
      return f"Error executing fund_raising_request: {e}"

  def book_closing(self, **kwargs) -> str:
    """Action: Generates Year-to-Date financials and advances the calendar."""
    try:
      res, success = self.arena.take_action("book_closing")
      
      self.arena.log_agent_event("action_execution", "Executed book closing")
      
      # Roll the month automatically!
      step_res = self.arena.end_step()
      
      out = {
          "action_result": {
              "success": success,
              "message": res.message,
              "data": res.data
          },
          "month_end_update": {
              "new_month": self.arena.env_state.current_month,
              "terminated": step_res.terminated,
              "reason": step_res.info.get("termination_reason"),
              "reward": step_res.reward,
              "current_observation": step_res.observation
          }
      }
      if kwargs:
        out["action_result"]["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(out, indent=2, default=str)
    except Exception as e:
      return f"Error executing book_closing: {e}"

  def pass_action(self, **kwargs) -> str:
    """Action: Passes this month without capital moves and advances the calendar."""
    try:
      res, success = self.arena.take_action("pass")
      
      self.arena.log_agent_event("action_execution", "Executed pass action")
      
      # Roll the month automatically!
      step_res = self.arena.end_step()
      
      out = {
          "action_result": {
              "success": success,
              "message": res.message,
              "data": res.data
          },
          "month_end_update": {
              "new_month": self.arena.env_state.current_month,
              "terminated": step_res.terminated,
              "reason": step_res.info.get("termination_reason"),
              "reward": step_res.reward,
              "current_observation": step_res.observation
          }
      }
      if kwargs:
        out["action_result"]["warning"] = f"Unexpected keyword argument(s) were ignored: {list(kwargs.keys())}"
      return json.dumps(out, indent=2, default=str)
    except Exception as e:
      return f"Error executing pass_action: {e}"

  def get_state(self) -> Any:
    """Returns a deepcopied checkpoint tuple of the arena and memory state."""
    return (copy.deepcopy(self.arena), copy.deepcopy(self.memory))

  def load_state(self, checkpoint: Any):
    """Restores the arena and memory from a checkpoint tuple."""
    self.arena, self.memory = checkpoint


class CFOEnv(env_base.BaseEnvironment):
  """CFO liquidity management environment adapter."""

  def __init__(
      self, config_path: str = "config.json", seed: Optional[int] = None
  ):
    super().__init__()
    self.adapter = CFOEnvAdapter(config_path=config_path)
    self.seed = seed

  def get_system_prompt(self) -> str:
    return (
        "You are a CFO Agent managing enterprise cash balance and capital"
        " raising. Avoid bankruptcy (ending cash < 0) and maximize enterprise"
        " valuation.\n\n"
        "Available Tools:\n"
        " - check_cash_in_bank(): Checks physical cash balance in treasury.\n"
        " - access_financial_docs(action: str, filename: str = ''): Accesses financial documents.\n"
        "     * action: 'list' to list available documents, or 'retrieve' to view a document.\n"
        "     * filename: required if action is 'retrieve'. Use the exact filename from the list.\n"
        " - cash_flow_forecast_calculation(months: int = 3, growth_rate_override: float = None): Projects cash flow.\n"
        " - check_market_data(data_type: str, start_month: int = 0, end_month: int = None): Queries macro indicators.\n"
        "     * data_type: 'GDP', 'CPI', 'UNRATE', 'SOFR', 'FEDFUNDS', 'Baa_Yield', 'Tsy2Y', 'Tsy5Y', 'Tsy10Y', 'Tsy30Y', 'VIX', 'PE_Ratio', 'PS_Ratio', 'Gross_Margin', 'EBITDA_Margin', 'Monthly_User_Growth', 'Annual_User_Growth'.\n"
        " - save_note(content: str, tags: str = ''): Saves a persistent note.\n"
        " - recall_notes(query: str = '', tags: str = '', limit: int = 10): Recalls notes.\n"
        " - fund_raising_request(type: str, amount: float): Submits fundraising request ('equity' or 'debt').\n"
        " - book_closing(): Generates YTD financials and advances to next month.\n"
        " - pass_action(): Passes current month without changes and advances to next month."
    )

  def is_terminated(self) -> bool:
    # Check if the underlying arena is terminated
    if hasattr(self.adapter, "arena"):
      return self.adapter.arena.is_terminated()
    return getattr(self.adapter, "terminated", False)

  def get_score(self) -> float:
    if hasattr(self.adapter, "arena"):
      return self.adapter.arena.get_score()
    return getattr(self.adapter, "score", 10000.0)

  def reset(self, seed: Optional[int] = None) -> str:
    s = seed if seed is not None else self.seed
    obs = self.adapter.reset(seed=s)
    if isinstance(obs, dict) and "available_actions" in obs:
      obs["available_actions"] = [
          "pass_action" if a == "pass" else a for a in obs["available_actions"]
      ]
    return str(obs)

  def execute_action(self, action_str: str) -> Tuple[str, bool]:
    result_str, terminated = super().execute_action(action_str)

    # Intercept tool budget exhaustion and automatically advance the month
    if "Tool budget exhausted" in result_str:
      result_str = self.pass_action()
      terminated = self.is_terminated()

    try:
      import json

      data = json.loads(result_str)
      if isinstance(data, dict):
        if (
            "month_end_update" in data
            and "current_observation" in data["month_end_update"]
        ):
          obs = data["month_end_update"]["current_observation"]
          if "available_actions" in obs:
            obs["available_actions"] = [
                "pass_action" if a == "pass" else a
                for a in obs["available_actions"]
            ]
        result_str = json.dumps(data, indent=2, default=str)
    except Exception:
      pass
    return result_str, terminated

  def get_state(self) -> Any:
    return self.adapter.get_state()

  def load_state(self, state: Any) -> None:
    self.adapter.load_state(state)

  # =========================================================================
  # Expose CFO tools via register_tool decorators
  # =========================================================================

  @env_base.register_tool(
      "check_cash_in_bank", "Checks cash balance, debt, gross totals."
  )
  def check_cash_in_bank(self, **kwargs) -> str:
    return self.adapter.check_cash_in_bank(**kwargs)

  @env_base.register_tool(
      "access_financial_docs", "List or retrieve documents."
  )
  def access_financial_docs(
      self, action: str = "", filename: str = "", **kwargs
  ) -> str:
    return self.adapter.access_financial_docs(
        action=action, filename=filename, **kwargs
    )

  @env_base.register_tool(
      "cash_flow_forecast_calculation", "Cash flow forecast projection."
  )
  def cash_flow_forecast_calculation(
      self,
      months: int = 3,
      growth_rate_override: Optional[float] = None,
      **kwargs,
  ) -> str:
    return self.adapter.cash_flow_forecast_calculation(
        months=months, growth_rate_override=growth_rate_override, **kwargs
    )

  @env_base.register_tool(
      "check_market_data", "Query historical macroeconomic indicators."
  )
  def check_market_data(
      self,
      data_type: str = "",
      start_month: int = 0,
      end_month: Optional[int] = None,
      **kwargs,
  ) -> str:
    return self.adapter.check_market_data(
        data_type=data_type,
        start_month=start_month,
        end_month=end_month,
        **kwargs,
    )

  @env_base.register_tool(
      "save_note", "Saves a persistent note to carry context."
  )
  def save_note(self, content: str = "", tags: str = "", **kwargs) -> str:
    return self.adapter.save_note(content=content, tags=tags, **kwargs)

  @env_base.register_tool(
      "recall_notes", "Recalls persistent notes by keyword/tags."
  )
  def recall_notes(
      self, query: str = "", tags: str = "", limit: int = 10, **kwargs
  ) -> str:
    return self.adapter.recall_notes(
        query=query, tags=tags, limit=limit, **kwargs
    )

  @env_base.register_tool(
      "fund_raising_request",
      "Submits debt or equity request.",
      is_state_changing=True,
  )
  def fund_raising_request(
      self, type: str = "", amount: float = 0.0, **kwargs
  ) -> str:
    return self.adapter.fund_raising_request(type=type, amount=amount, **kwargs)

  @env_base.register_tool(
      "book_closing",
      "Generates Year-to-Date financials.",
      is_state_changing=True,
  )
  def book_closing(self, **kwargs) -> str:
    return self.adapter.book_closing(**kwargs)

  @env_base.register_tool(
      "pass_action",
      "Passes current month without capital moves.",
      is_state_changing=True,
  )
  def pass_action(self, **kwargs) -> str:
    return self.adapter.pass_action(**kwargs)

  # =========================================================================
  # Override hooks
  # =========================================================================

  def get_current_node_id(self, trajectory: List[str]) -> str:
    """Deduces node ID mapping CFO milestones to Procedural Graph nodes."""
    if not trajectory:
      return "START"

    last_relevant_action = None
    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        try:
          action_name, _, _ = env_base.parse_action_string(action_content)
          if action_name in (
              "check_cash_in_bank",
              "check_market_data",
              "cash_flow_forecast_calculation",
              "save_note",
              "pass_action",
              "book_closing",
              "fund_raising_request",
          ):
            last_relevant_action = action_name
            break
        except Exception:
          pass

    if not last_relevant_action:
      return "START"

    if last_relevant_action == "check_cash_in_bank":
      return "check_cash_in_bank"
    elif last_relevant_action == "check_market_data":
      return "check_market_data"
    elif last_relevant_action == "cash_flow_forecast_calculation":
      return "cash_flow_forecast_calculation"
    elif last_relevant_action == "save_note":
      return "Decide_Capital"
    elif last_relevant_action in (
        "pass_action",
        "book_closing",
        "fund_raising_request",
    ):
      return "Month_Start"

    return "START"

  def compress_trajectory_hook(self, trajectory: List[str]) -> str:
    """Compresses long CFO balance sheet logs into a compact markdown monthly decision log."""
    import ast
    
    # 1. Parse monthly states
    monthly_data = {}
    actions_by_month = {}
    current_m = 0
    
    # Keep track of active actions taken in each month
    for step in trajectory:
      # Track action under current month
      if step.startswith("Action:"):
        action_str = step[len("Action:"):].strip()
        if current_m not in actions_by_month:
          actions_by_month[current_m] = []
        actions_by_month[current_m].append(action_str)
        
      elif step.startswith("Observation:"):
        obs_content = step[len("Observation:"):].strip()
        obs_data = None
        try:
          obs_data = json.loads(obs_content)
        except Exception:
          try:
            val = ast.literal_eval(obs_content)
            if isinstance(val, dict):
              obs_data = val
          except Exception:
            pass
            
        if obs_data and isinstance(obs_data, dict):
          # Extract month and financial metrics
          m = obs_data.get("current_month")
          if m is None and "month_end_update" in obs_data:
            m = obs_data["month_end_update"].get("new_month")
            me_obs = obs_data["month_end_update"].get("current_observation")
            if isinstance(me_obs, dict):
              obs_data = me_obs
              
          if m is not None:
            current_m = m
            # Keep the last observation for this month
            monthly_data[current_m] = {
                "cash": obs_data.get("cash_balance", 0.0),
                "debt": obs_data.get("total_debt", 0.0),
                "users": obs_data.get("active_users", 0),
                "lending_rate": obs_data.get("lending_rate"),
                "terminated": obs_data.get("episode_terminated", False),
                "reason": obs_data.get("termination_reason"),
            }
            
    # 2. Format into a highly compact timeline table
    compressed_lines = [
        "--- Compact CFO Monthly Decision Timeline ---",
        "| Month | Cash Balance | Total Debt | Active Users | Lending Rate | Actions Taken / Outcomes |",
        "|---|---|---|---|---|---|",
    ]
    
    for m in sorted(monthly_data.keys()):
      state = monthly_data[m]
      cash_m = f"${state['cash']/1e6:.2f}M"
      debt_m = f"${state['debt']/1e6:.2f}M" if state['debt'] > 0 else "$0.00"
      users = f"{state['users']}"
      rate = f"{state['lending_rate']*100:.2f}%" if state['lending_rate'] is not None else "N/A"
      
      actions = actions_by_month.get(m, [])
      actions_str = ", ".join(actions) if actions else "None"
      
      if state["terminated"]:
        actions_str += f" | ❌ TERMINATED: {state['reason']}"
        
      compressed_lines.append(
          f"| Month {m} | {cash_m} | {debt_m} | {users} | {rate} | {actions_str} |"
      )
      
    return "\n".join(compressed_lines)
