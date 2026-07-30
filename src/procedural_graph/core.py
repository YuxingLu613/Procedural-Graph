"""Core Procedural Graph definition and verification engine."""

import collections
from dataclasses import dataclass, field
import enum
import json
from typing import Any, Dict, List, Optional, Set, Tuple


class RelationType(enum.Enum):
  LEADS_TO = "LEADS_TO"
  TRIGGERS = "TRIGGERS"
  DEPENDS_ON = "DEPENDS_ON"
  CONVERGES_TO = "CONVERGES_TO"
  PROVIDES_INPUT_FOR = "PROVIDES_INPUT_FOR"


class NodeType(enum.Enum):
  ACTION = "ACTION"
  REASONING = "REASONING"
  STATE = "STATE"


@dataclass(frozen=True)
class Node:
  """Strongly type-checked representation of an Procedural Graph Node."""

  id: str
  type: NodeType
  description: str
  metadata: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "id": self.id,
        "type": self.type.value,
        "description": self.description,
        "metadata": self.metadata,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Node":
    return cls(
        id=data["id"],
        type=NodeType(data["type"]),
        description=data.get("description", data.get("label", "")),
        metadata=data.get("metadata", {}),
    )


@dataclass(frozen=True)
class Edge:
  """Strongly type-checked representation of an Procedural Graph Edge."""

  source: str
  target: str
  relation: RelationType
  condition: Optional[str] = None
  guidance: Optional[str] = None
  pitfalls: Optional[str] = None
  purpose: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "source": self.source,
        "target": self.target,
        "relation": self.relation.value,
        "condition": self.condition,
        "guidance": self.guidance or self.purpose,
        "pitfalls": self.pitfalls,
        "metadata": self.metadata,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Edge":
    return cls(
        source=data["source"],
        target=data["target"],
        relation=RelationType(data["relation"]),
        condition=data.get("condition"),
        guidance=data.get("guidance") or data.get("purpose"),
        pitfalls=data.get("pitfalls"),
        purpose=data.get("purpose"),
        metadata=data.get("metadata", {}),
    )


class ProceduralGraph:
  """Procedural Graph manager handling Node/Edge CRUD, adjacency index, and JSON."""

  def __init__(self):
    self.nodes: Dict[str, Node] = {}
    self.edges: List[Edge] = []
    # Adjacency index for outgoing edges: source_id -> list of Edge objects
    self._adjacency: Dict[str, List[Edge]] = collections.defaultdict(list)

  def add_node(self, node: Node) -> None:
    if node.id in self.nodes:
      raise ValueError(f"Node '{node.id}' already exists in Procedural Graph.")
    self.nodes[node.id] = node

  def get_node(self, node_id: str) -> Optional[Node]:
    return self.nodes.get(node_id)

  def update_node(self, node_id: str, **kwargs: Any) -> None:
    if node_id not in self.nodes:
      raise KeyError(f"Node '{node_id}' not found.")
    if "id" in kwargs:
      raise ValueError("Updating node ID is not permitted.")
    node = self.nodes[node_id]
    # Since Node is frozen, we need to recreate it when updating fields
    updated_fields = {
        "id": node.id,
        "type": node.type,
        "description": node.description,
        "metadata": node.metadata,
    }
    for k, v in kwargs.items():
      if k in updated_fields:
        updated_fields[k] = v
      else:
        raise AttributeError(f"Node has no attribute '{k}'")
    self.nodes[node_id] = Node(
        id=updated_fields["id"],
        type=updated_fields["type"],
        description=updated_fields["description"],
        metadata=updated_fields["metadata"],
    )

  def delete_node(self, node_id: str) -> None:
    if node_id in self.nodes:
      del self.nodes[node_id]
    self.edges = [
        e for e in self.edges if e.source != node_id and e.target != node_id
    ]
    self._rebuild_adjacency()

  def add_edge(self, edge: Edge) -> None:
    if edge.source not in self.nodes:
      raise ValueError(f"Source node '{edge.source}' does not exist.")
    if edge.target not in self.nodes:
      raise ValueError(f"Target node '{edge.target}' does not exist.")
    self.edges.append(edge)
    self._adjacency[edge.source].append(edge)

  def get_outgoing_edges(self, node_id: str) -> List[Edge]:
    return self._adjacency.get(node_id, [])

  def delete_edge(
      self, source: str, target: str, relation: Optional[RelationType] = None
  ) -> None:
    self.edges = [
        e
        for e in self.edges
        if not (
            e.source == source
            and e.target == target
            and (relation is None or e.relation == relation)
        )
    ]
    self._rebuild_adjacency()

  def _rebuild_adjacency(self) -> None:
    self._adjacency.clear()
    for edge in self.edges:
      self._adjacency[edge.source].append(edge)

  def to_dict(self) -> Dict[str, Any]:
    return {
        "nodes": [n.to_dict() for n in self.nodes.values()],
        "edges": [e.to_dict() for e in self.edges],
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "ProceduralGraph":
    graph = cls()
    for node_data in data.get("nodes", []):
      graph.add_node(Node.from_dict(node_data))
    for edge_data in data.get("edges", []):
      graph.add_edge(Edge.from_dict(edge_data))
    return graph

  def to_text_summary(self) -> str:
    """Generates a structured textual summary of the Procedural Graph."""
    lines = ["Procedural Graph Nodes:"]
    for node in self.nodes.values():
      lines.append(f"- [{node.id}] (Type: {node.type.value}): {node.description}")

    lines.append("\nProcedural Graph Transitions:")
    for edge in self.edges:
      cond_str = (
          f" (Condition: {edge.condition})"
          if edge.condition and edge.condition.strip()
          else ""
      )
      lines.append(f"- Transition: [{edge.source}] -> [{edge.target}]{cond_str}")
      guidance = edge.guidance or edge.purpose or edge.metadata.get("strategic_rationale")
      if guidance:
        lines.append(f"  * Guidance: {guidance}")
      pitfalls = edge.pitfalls or edge.metadata.get("pitfalls")
      if pitfalls:
        lines.append(f"  * Pitfalls: {pitfalls}")
    return "\n".join(lines)

  def save_to_json(self, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

  @classmethod
  def load_from_json(cls, filepath: str) -> "ProceduralGraph":
    with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)
    return cls.from_dict(data)


class GraphValidator:
  """Validation engine enforcing topological cycle detection and deadlock terminal checks."""

  @staticmethod
  def validate(
      graph: ProceduralGraph, allow_cycles: bool = False
  ) -> Tuple[bool, List[str]]:
    """Runs all validation checks. Returns (is_valid, list of errors)."""
    errors = []

    # 1. Hanging Edge validation
    for edge in graph.edges:
      if edge.source not in graph.nodes:
        errors.append(
            f"Hanging edge: source '{edge.source}' does not exist in nodes."
        )
      if edge.target not in graph.nodes:
        errors.append(
            f"Hanging edge: target '{edge.target}' does not exist in nodes."
        )

    if errors:
      return False, errors

    # 2. Cycle Detection (Topological DFS Check)
    if not allow_cycles:
      has_cycle, cycle_path = GraphValidator._has_cycle(graph)
      if has_cycle:
        errors.append(f"Cyclic dependency detected: {' -> '.join(cycle_path)}")

    # 3. Deadlock / Terminal Reachability Check via Backward BFS
    unreachable_nodes = GraphValidator._check_terminal_reachability(graph)
    if unreachable_nodes:
      errors.append(
          "Deadlock / Terminal Unreachability detected for nodes:"
          f" {sorted(list(unreachable_nodes))}"
      )

    return len(errors) == 0, errors

  @staticmethod
  def _has_cycle(graph: ProceduralGraph) -> Tuple[bool, List[str]]:
    visited: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}

    for start_node in graph.nodes:
      if visited[start_node] != 0:
        continue

      stack: List[Tuple[str, int]] = [(start_node, 0)]  # (node, edge_index)
      visited[start_node] = 1
      path: List[str] = [start_node]

      while stack:
        u, edge_idx = stack[-1]
        outgoing = graph.get_outgoing_edges(u)

        if edge_idx < len(outgoing):
          stack[-1] = (u, edge_idx + 1)
          v = outgoing[edge_idx].target

          if visited[v] == 1:
            # Cycle detected, return cycle path
            try:
              cycle_start = path.index(v)
              return True, path[cycle_start:] + [v]
            except ValueError:
              return True, path + [v]
          elif visited[v] == 0:
            visited[v] = 1
            path.append(v)
            stack.append((v, 0))
        else:
          stack.pop()
          path.pop()
          visited[u] = 2

    return False, []

  @staticmethod
  def _check_terminal_reachability(graph: ProceduralGraph) -> Set[str]:
    """Returns a set of node IDs that cannot reach any terminal node (out_degree = 0)."""
    terminal_nodes = {
        node_id
        for node_id in graph.nodes
        if not graph.get_outgoing_edges(node_id)
    }

    if not terminal_nodes:
      if graph.nodes:
        return set(graph.nodes.keys())
      return set()

    backward_adj = collections.defaultdict(list)
    for edge in graph.edges:
      backward_adj[edge.target].append(edge.source)

    queue = collections.deque(terminal_nodes)
    visited = set(terminal_nodes)

    while queue:
      u = queue.popleft()
      for parent in backward_adj[u]:
        if parent not in visited:
          visited.add(parent)
          queue.append(parent)

    all_nodes = set(graph.nodes.keys())
    return all_nodes - visited
