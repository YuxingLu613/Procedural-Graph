"""Procedural Graph memory refinement and LLM-based offline distillation engines."""

import enum
import json
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple
from .core import ProceduralGraph, Edge, GraphValidator, Node, NodeType, RelationType

logger = logging.getLogger(__name__)


class GraphRefinementMode(enum.Enum):
  STATIC_ONETIME = "static_onetime"
  STATIC_INCREMENTAL = "static_incremental"
  SCRATCH_ONETIME = "scratch_onetime"
  SCRATCH_INCREMENTAL = "scratch_incremental"


class RefinementLLMClient(Protocol):
  """Decoupled LLM client for refinement text completion."""

  @property
  def max_token_len(self) -> int:
    """Returns the maximum input token limit for the model."""
    ...

  def generate(self, prompt: str, temperature: float = 0.0) -> str:
    """Generates text completion from LLM."""
    ...


class ProceduralGraphRefiner:
  """Heuristic statistical failure-rate pruner that prunes edges based on online feedback."""

  def __init__(
      self,
      graph: ProceduralGraph,
      failure_threshold: float = 0.7,
      min_attempts: int = 3,
  ):
    self.graph = graph
    self.failure_threshold = failure_threshold
    self.min_attempts = min_attempts
    # Map edge signature "source->target" to (attempts, failures)
    self.edge_stats: Dict[str, Tuple[int, int]] = {}

  def record_edge_step(self, source: str, target: str, success: bool) -> None:
    sig = f"{source}->{target}"
    attempts, failures = self.edge_stats.get(sig, (0, 0))
    attempts += 1
    if not success:
      failures += 1
    self.edge_stats[sig] = (attempts, failures)

  def prune_failed_edges(self) -> List[str]:
    """Prunes edges that exceed failure rates.

    Returns list of pruned edge signatures.
    """
    pruned_signatures = []
    edges_to_keep = []

    for edge in self.graph.edges:
      sig = f"{edge.source}->{edge.target}"
      attempts, failures = self.edge_stats.get(sig, (0, 0))

      if attempts >= self.min_attempts:
        fail_rate = failures / attempts
        if fail_rate >= self.failure_threshold:
          pruned_signatures.append(f"{sig} (Fail Rate: {fail_rate * 100:.1f}%)")
          continue  # Skip adding to keep list, pruning it

      edges_to_keep.append(edge)

    # Rebuild graph topology safely
    self.graph.edges = edges_to_keep
    self.graph._rebuild_adjacency()
    return pruned_signatures


class ProceduralGraphLLMRefiner:
  """Offline LLM-based graph distiller supporting the 4 refinement modes with transaction rollbacks."""

  def __init__(self, llm: RefinementLLMClient):
    self.llm = llm

  def refine_graph(
      self,
      graph: ProceduralGraph,
      mode: GraphRefinementMode,
      trajectories: List[List[str]],
      success_scores: List[float],
      task_description: str,
      available_tools_list: str = "",
      env: Optional[Any] = None,
      allow_cycles: bool = False,
      rejected_history: Optional[List[Tuple[Dict[str, Any], float]]] = None,
  ) -> Tuple[ProceduralGraph, List[str]]:
    """Main refinement orchestrator with regression safeguarding and topological rollback.

    Args:
      graph: The current Procedural Graph to refine.
      mode: The refinement mode.
      trajectories: Trajectories of recent runs.
      success_scores: Validation success scores of the runs.
      task_description: High-level task instructions.
      available_tools_list: Bulleted list of valid actions.
      env: The environment wrapper instance.
      allow_cycles: Whether to allow cycles in the graph.
      rejected_history: Optional history of rejected candidates.

    Returns:
      Tuple of (refined_graph, list_of_applied_edits_or_errors).
    """
    edits_log = []
    # 1. Perform transaction backup
    backup_dict = graph.to_dict()

    # 2. Determine prompt context constraints and truncate trajectories if necessary
    max_token_len = getattr(self.llm, "max_token_len", 100000)
    # Estimate max characters (using conservative 3 characters per token for code/logs)
    max_prompt_chars = max_token_len * 3

    # Estimate base prompt overhead size without attempts_block
    # Base instructions + task context + JSON graph length + safety margin
    graph_json_len = len(json.dumps(backup_dict, indent=2, ensure_ascii=False))
    base_prompt_len = 2000 + len(task_description) + graph_json_len

    allowed_attempts_chars = max_prompt_chars - base_prompt_len - 100
    allowed_attempts_chars = max(allowed_attempts_chars, 1000)  # Min 1k chars

    num_trajs = len(trajectories)
    max_traj_chars = 1000  # Default fallback minimum
    if num_trajs > 0:
      max_traj_chars = max(1000, (allowed_attempts_chars // num_trajs) - 100)

    # Helper function to truncate from the middle
    def truncate_trajectory(t_str: str, limit: int) -> str:
      if len(t_str) <= limit:
        return t_str
      if limit <= 100:
        return t_str[:limit]
      half = (limit - 50) // 2
      return (
          t_str[:half]
          + "\n... [TRUNCATED DUE TO LLM CONTEXT LIMITS] ...\n"
          + t_str[-half:]
      )

    # Assemble failures vs successes
    attempts_summary = []
    for idx, (traj, score) in enumerate(zip(trajectories, success_scores)):
      if env and hasattr(env, "compress_trajectory_hook"):
        compressed_traj_str = env.compress_trajectory_hook(traj)
      else:
        traj_str = "\n".join(traj)
        compressed_traj_str = truncate_trajectory(traj_str, max_traj_chars)
      attempts_summary.append(
          f"### Attempt {idx+1} (Score:"
          f" {score:.2f})\nTrajectory:\n{compressed_traj_str}\n"
      )

    attempts_block = "\n".join(attempts_summary)

    rejected_block = ""
    if rejected_history:
      rejected_summaries = []
      for idx, (rej_graph_dict, score) in enumerate(rejected_history):
        rejected_summaries.append(
            f"### Rejected Candidate {idx+1} (Val Success Rate: {score:.1%})\n"
            f"Graph Representation:\n"
            f"{json.dumps(rej_graph_dict, indent=2, ensure_ascii=False)}\n"
        )
      rejected_block = (
          "\nHere are the candidate Procedural Graphs you previously proposed"
          " that were REJECTED because they did not improve the validation"
          " success rate over the current best graph. DO NOT propose these"
          " exact graphs or edits that lead to these graphs again. Propose"
          " different modifications:\n"
          + "\n".join(rejected_summaries)
      )

    prompt = f"""You are an expert cognitive architect optimizing an Procedural Graph for an intelligent agent.
The Procedural Graph represents a structured cognitive procedure guidance.

Task context: {task_description}
Refinement Mode: {mode.value}

Available Tool Actions (The agent can only execute these actions):
{available_tools_list}

Here are the execution trajectories of recent attempts by the agent:
{attempts_block}

Here is the current Procedural Graph representation:
{json.dumps(backup_dict, indent=2, ensure_ascii=False)}
{rejected_block}

Your job is to refine the Procedural Graph. Follow these guidelines based on the mode:
- **static_onetime / static_incremental**: Prune edges/nodes that lead to loops, deadlocks, or failures. Add missing nodes and edges that could fix the failures and improve performance for future tasks.
- **scratch_onetime / scratch_incremental**:
  * If starting from scratch (only a START node exists), synthesize a brand new, complete Procedural Graph using the Available Tool Actions list, Status, and successful patterns in the trajectories.
  * Otherwise, prune edges/nodes that lead to loops, deadlocks, or failures, and add missing nodes and edges based on the given procedural graph.

Rules for nodes and edges:
1. **Node Types**: You can propose both `STATE` nodes (representing high-level procedural milestones, e.g. 'AUTHENTICATE_USER', 'CHECK_RETURN_POLICY', 'EXECUTE_TOOL') and `ACTION` nodes matching tool names in the "Available Tool Actions" list.
2. **Transition Conditions**: If an edge has a `condition`, provide a natural language semantic precondition under which this transition should fire (e.g., "When dialogue history has been parsed but target constraints are unknown"). Use `null` if the transition is unconditional.
3. **Execution Guidance**: For every edge added in `add_edges`, you MUST provide a `guidance` string detailing exactly what policy rule or action to take next and the strategic rationale behind it (e.g., "Verify user identity first using email or name+zip before querying order details").
4. **Pitfalls**: Provide a `pitfalls` string warning about premature actions, policy violations (e.g., executing refund before checking delivery date), or common formatting pitfalls to avoid during this step.
5. **Generality & Leak Prevention**: The updated Procedural Graph must guide the agent effectively without overfitting to specific details of a single trajectory. Use high-level conceptual descriptions. CRITICAL: Do NOT include any specific numbers, exact dollar amounts (e.g., $15M, $30M-$50M), exact month ranges, or hardcoded thresholds in your node descriptions, edge conditions, guidance, or pitfalls. Instead, express these rules qualitatively and conceptually.
6. **Node ID Compatibility**: If refining an existing graph (static modes), preserve existing node IDs where applicable.
7. **Directed Acyclic Graph (DAG) Constraint**: The Procedural Graph MUST remain a Directed Acyclic Graph (DAG) and MUST NOT contain any cyclic dependencies or loops. The environment loop handles multi-turn dialogue automatically. Any cyclic proposal will have its back-edge severed.

Please propose the exact set of edits to perform. You must output your edits as a single valid JSON block containing four arrays: "add_nodes", "delete_nodes", "add_edges", and "delete_edges".

Output Format must be exactly:
```json
{{
  "add_nodes": [
    {{"id": "node_id", "type": "ACTION", "description": "..."}}
  ],
  "delete_nodes": ["node_id"],
  "add_edges": [
    {{"source": "node_1", "target": "node_2", "relation": "LEADS_TO", "condition": "...", "guidance": "...", "pitfalls": "..."}}
  ],
  "delete_edges": [
    {{"source": "node_a", "target": "node_b"}}
  ]
}}
```

Make sure to output ONLY the raw JSON block.
"""
    response = self.llm.generate(prompt, temperature=0.0)

    import re
    # Try to find content inside ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
      json_block = match.group(1).strip()
    else:
      # Fallback: find the outermost curly braces { ... }
      match = re.search(r"(\{.*\})", response, re.DOTALL)
      if match:
        json_block = match.group(1).strip()
      else:
        json_block = response.strip()

    try:
      edits = json.loads(json_block)
    except Exception as e:
      err = f"LLM proposed invalid JSON: {e}. Output was:\n{response}"
      logger.warning(err)
      return graph, [err]

    # Edits will be applied to the sandbox graph, and any actual cyclic back-edges will be dynamically healed by the generic cycle breaker below.
    pass

    # Apply changes in a transaction-safe sandbox
    sandbox_graph = ProceduralGraph.from_dict(backup_dict)

    try:
      # 1. Delete edges first (to prevent hanging references before deleting nodes)
      for edge_data in edits.get("delete_edges", []):
        sandbox_graph.delete_edge(edge_data["source"], edge_data["target"])
        edits_log.append(
            f"Deleted Edge: {edge_data['source']} -> {edge_data['target']}"
        )

      # 2. Delete nodes
      for node_id in edits.get("delete_nodes", []):
        sandbox_graph.delete_node(node_id)
        edits_log.append(f"Deleted Node: {node_id}")

      # 3. Add nodes
      for node_data in edits.get("add_nodes", []):
        node = Node(
            id=node_data["id"],
            type=NodeType(node_data["type"]),
            description=node_data["description"],
            metadata=node_data.get("metadata", {}),
        )
        if sandbox_graph.get_node(node.id):
          edits_log.append(f"Skipped adding existing Node: {node.id}")
          continue
        sandbox_graph.add_node(node)
        edits_log.append(f"Added Node: {node.id}")

      # 4. Add edges
      for edge_data in edits.get("add_edges", []):
        edge = Edge(
            source=edge_data["source"],
            target=edge_data["target"],
            relation=RelationType(edge_data["relation"]),
            condition=edge_data.get("condition"),
            guidance=edge_data.get("guidance") or edge_data.get("purpose"),
            pitfalls=edge_data.get("pitfalls"),
            purpose=edge_data.get("purpose"),
            metadata=edge_data.get("metadata", {}),
        )
        # Check if identical edge already exists
        exists = False
        for existing_edge in sandbox_graph.edges:
          if (
              existing_edge.source == edge.source
              and existing_edge.target == edge.target
              and existing_edge.relation == edge.relation
          ):
            exists = True
            break
        if exists:
          edits_log.append(
              f"Skipped adding existing Edge: {edge.source} -> {edge.target}"
          )
          continue
        sandbox_graph.add_edge(edge)
        edits_log.append(f"Added Edge: {edge.source} -> {edge.target}")

      # Actively heal any cyclic dependencies proposed by the LLM to preserve DAG structure
      if not allow_cycles:
        while True:
          has_cycle, cycle_path = GraphValidator._has_cycle(sandbox_graph)
          if not has_cycle:
            break
          source = cycle_path[-2]
          target = cycle_path[-1]
          sandbox_graph.delete_edge(source, target)
          edits_log.append(
              f"Safeguard (Cycle Breaker): Severed cyclic back-edge to preserve DAG: {source} -> {target}"
          )

      # Validate Topological Validity (cycle & deadlock check)
      is_valid, errors = GraphValidator.validate(sandbox_graph, allow_cycles=allow_cycles)
      if not is_valid:
        err_msg = (
            f"Topological verification failed after edits: {', '.join(errors)}."
            " Rolling back graph changes."
        )
        logger.warning(err_msg)
        return graph, [err_msg]

      # Validation passed, commit refined graph
      return sandbox_graph, edits_log

    except Exception as e:
      err_msg = (
          f"Exception occurred during graph refinement execution: {e}. Rolling"
          " back changes."
      )
      logger.warning(err_msg)
      return graph, [err_msg]
