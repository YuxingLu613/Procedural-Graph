"""Retrieval-Augmented Generation (RAG) engine for dynamic Procedural Graph guidance."""

import dataclasses
import logging
import random
import re
import time
import json
import urllib.request
import urllib.error
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple
from .core import ProceduralGraph, Edge

logger = logging.getLogger(__name__)


def _get_task_description(env: Any) -> str:
  """Safely retrieves the task description, question, or prompt from the environment."""
  return (
      getattr(env, "target_question", None)
      or getattr(env, "question", None)
      or getattr(env, "prompt", None)
      or getattr(env, "current_obs", "")
  )


class SimilarityScorer(Protocol):
  """Protocol defining the similarity scoring interface."""

  def score(self, query: str, candidates: List[str]) -> List[float]:
    """Returns list of similarity scores mapping the query to candidates."""
    ...


class ConditionEvaluator(Protocol):
  """Protocol defining historical trajectory logic condition check."""

  def evaluate(self, condition: Optional[str], trajectory: List[str]) -> bool:
    """Returns True if the condition is satisfied by the history."""
    ...


def call_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
  """Helper running the function with exponential backoff and random jitter."""
  delay = initial_delay
  for attempt in range(max_retries):
    try:
      return func(*args, **kwargs)
    except Exception as e:
      if attempt == max_retries - 1:
        raise e
      sleep_time = delay + random.uniform(0, 0.1 * delay)
      time.sleep(sleep_time)
      delay *= backoff_factor


class EmbeddingSimilarityScorer:
  """Vertex AI Text Embedding Similarity Scorer with Jaccard fallback and caching."""

  def __init__(
      self,
      model_name: Any = "text-embedding-005",
      use_fallback: bool = True,
      project_id: Optional[str] = None,
  ):
    self.model_name = model_name
    self.use_fallback = use_fallback
    self.project_id = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("VERTEX_PROJECT_ID")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if not self.project_id and not self.use_fallback:
      raise ValueError(
          "GCP project ID is required for Vertex AI embedding. Please set the "
          "GOOGLE_CLOUD_PROJECT environment variable or pass project_id explicitly."
      )
    self.credentials = None
    self.auth_request = None
    self._initialized = False
    self._embedding_cache: Dict[str, List[float]] = {}

  def _lazy_init(self) -> None:
    if self._initialized:
      return
    if not isinstance(self.model_name, str):
      self._initialized = True
      return
    try:
      import google.auth
      from google.auth.transport import requests as auth_requests

      self.credentials, _ = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
          quota_project_id=self.project_id,
      )
      self.auth_request = auth_requests.Request()
      self.credentials.refresh(self.auth_request)
    except Exception as e:
      if not self.use_fallback:
        raise e
      logger.warning(
          "Could not initialize Google Auth Credentials for REST Embedding: %s."
          " Fallback Jaccard will be used.",
          e,
      )
      self.credentials = None
    self._initialized = True

  def _get_embeddings_rest(self, texts: List[str]) -> List[List[float]]:
    if not self.credentials:
      raise RuntimeError("Credentials not initialized")
    
    if self.credentials.expired:
      self.credentials.refresh(self.auth_request)

    location = "us-central1"
    is_gemini = "gemini" in self.model_name.lower()
    method = "embedContent" if is_gemini else "predict"
    
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/"
        f"locations/{location}/publishers/google/models/{self.model_name}:{method}"
    )

    if is_gemini:
      embeddings = []
      for text in texts:
        payload = {
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        emb = self._call_rest_api(url, payload, is_gemini)
        embeddings.append(emb)
      return embeddings
    else:
      payload = {
          "instances": [
              {"content": t} for t in texts
          ]
      }
      return self._call_rest_api(url, payload, is_gemini)

  def _call_rest_api(self, url: str, payload: Dict[str, Any], is_gemini: bool) -> Any:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {self.credentials.token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": self.project_id,
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
      result = json.loads(response.read().decode("utf-8"))
      if is_gemini:
        embedding = result.get("embedding", {})
        values = embedding.get("values", [])
        return values
      else:
        predictions = result.get("predictions", [])
        if not predictions:
          raise RuntimeError(f"No predictions in REST response: {result}")
        return [p.get("embeddings", {}).get("values", []) for p in predictions]

  def score(self, query: str, candidates: List[str]) -> List[float]:
    self._lazy_init()
    if not self.credentials or not candidates:
      if not self.credentials and not self.use_fallback:
        raise RuntimeError(
            "Vertex AI credentials failed to initialize and fallback similarity is"
            " disabled."
        )
      return [self._fallback_score(query, cand) for cand in candidates]

    try:
      import numpy as np

      def get_embeddings() -> Tuple[List[float], List[List[float]]]:
        # Query embedding caching check
        if query not in self._embedding_cache:
          self._embedding_cache[query] = self._get_embeddings_rest([query])[0]
        q_emb = self._embedding_cache[query]

        # Identify candidates requiring embedding requests
        to_embed = [c for c in candidates if c not in self._embedding_cache]
        if to_embed:
          c_embs_fetched = self._get_embeddings_rest(to_embed)
          for cand, emb in zip(to_embed, c_embs_fetched):
            self._embedding_cache[cand] = emb

        c_embs_all = [self._embedding_cache[c] for c in candidates]
        return q_emb, c_embs_all

      q_vector, c_vectors = call_with_backoff(get_embeddings)

      # Compute cosine similarities manually using numpy
      scores = []
      q_arr = np.array(q_vector)
      q_norm = np.linalg.norm(q_arr)
      for c_vector in c_vectors:
        c_arr = np.array(c_vector)
        c_norm = np.linalg.norm(c_arr)
        if q_norm > 0 and c_norm > 0:
          cosine_sim = float(np.dot(q_arr, c_arr) / (q_norm * c_norm))
        else:
          cosine_sim = 0.0
        scores.append(cosine_sim)
      return scores

    except Exception as e:
      if not self.use_fallback:
        raise e
      logger.warning(
          "Vertex AI embedding similarity score failed: %s. Falling back to"
          " string hybrid overlap score.",
          e,
      )
      return [self._fallback_score(query, cand) for cand in candidates]

  def _fallback_score(self, query: str, candidate: str) -> float:
    """Jaccard overlap based string similarity fallback with underscore normalization."""

    def get_words(text: str) -> Set[str]:
      # Normalize underscores to whitespace to treat words inside underscores correctly
      normalized_text = text.replace("_", " ").lower()
      return set(re.findall(r"\w+", normalized_text))

    q_words = get_words(query)
    c_words = get_words(candidate)
    if not q_words and not c_words:
      return 0.0
    intersection = q_words & c_words
    union = q_words | c_words
    return len(intersection) / len(union) if union else 0.0


class SoftSemanticConditionEvaluator:
  """Evaluates soft semantic conditions using similarity scorer or keyword overlap."""

  def __init__(self, scorer: SimilarityScorer, threshold: float = 0.25):
    self.scorer = scorer
    self.threshold = threshold

  def evaluate(self, condition: Optional[str], trajectory: List[str]) -> bool:
    if not condition or not condition.strip():
      return True

    cond_str = condition.strip()
    if cond_str in ("TRUE", "null", "None"):
      return True
    if cond_str == "FALSE":
      return False

    if not trajectory:
      return True

    # Extract latest observations or thoughts from trajectory
    recent_items = [
        item
        for item in trajectory[-10:]
        if item.startswith("Observation:") or item.startswith("Thought:")
    ]
    if not recent_items:
      recent_items = trajectory[-5:]

    context_str = "\n".join(recent_items)

    try:
      scores = self.scorer.score(cond_str, [context_str])
      if scores and scores[0] >= self.threshold:
        return True
    except Exception:
      pass

    # Fallback to keyword/substring check
    norm_cond = cond_str.lower()
    norm_ctx = context_str.lower()

    words = [w for w in re.findall(r"\w+", norm_cond) if len(w) > 3]
    if not words:
      return True

    return any(w in norm_ctx for w in words)


class SimpleConditionEvaluator:
  """Evaluates logic metadata conditions against action trajectory strings with parentheses support."""

  def evaluate(self, condition: Optional[str], trajectory: List[str]) -> bool:
    if not condition:
      return True

    expr = condition.strip()
    if not expr:
      return True

    # Normalize spaces and convert logical operators to uppercase for case-insensitivity
    expr = re.sub(r"\band\b", "AND", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bor\b", "OR", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bnot\b", "NOT", expr, flags=re.IGNORECASE)

    # 1. Handle Parentheses Grouping recursively
    while "(" in expr:
      end_idx = expr.find(")")
      if end_idx == -1:
        raise ValueError(
            f"Unbalanced parentheses in condition expression: {condition}"
        )
      start_idx = expr.rfind("(", 0, end_idx)
      if start_idx == -1:
        raise ValueError(
            f"Unbalanced parentheses in condition expression: {condition}"
        )

      inner_expr = expr[start_idx + 1 : end_idx]
      inner_result = self.evaluate(inner_expr, trajectory)
      # Replace the parenthesis block with literal boolean strings
      expr = (
          expr[:start_idx]
          + ("TRUE" if inner_result else "FALSE")
          + expr[end_idx + 1 :]
      )

    expr_stripped = expr.strip()
    if expr_stripped == "TRUE":
      return True
    if expr_stripped == "FALSE":
      return False

    # 2. Resolve basic flat logical expressions
    if expr_stripped.startswith("NOT "):
      return not self.evaluate(expr_stripped[4:], trajectory)

    # Handle OR (lowest precedence, evaluated last)
    if " OR " in expr_stripped:
      parts = expr_stripped.split(" OR ")
      return any(self.evaluate(p, trajectory) for p in parts)

    # Handle AND (higher precedence, evaluated first)
    if " AND " in expr_stripped:
      parts = expr_stripped.split(" AND ")
      return all(self.evaluate(p, trajectory) for p in parts)

    # Precise Variable boundary check
    return self._check_variable(expr_stripped, trajectory)

  def _check_variable(self, var: str, trajectory: List[str]) -> bool:
    v = var.strip()
    if v == "TRUE":
      return True
    if v == "FALSE":
      return False
    # Use regex word boundary checks to prevent partial matches (like "A" matching "Action:")
    pattern = re.compile(r"\b" + re.escape(v) + r"\b")
    return any(pattern.search(step) is not None for step in trajectory)


class ProceduralGraphRetriever:
  """Performs outbound edge-constrained local RAG search on Procedural Graphs."""

  def __init__(
      self,
      graph: ProceduralGraph,
      scorer: SimilarityScorer,
      evaluator: ConditionEvaluator,
  ):
    self.graph = graph
    self.scorer = scorer
    self.evaluator = evaluator

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    """Returns dynamic Procedural Graph guidance based on current state."""
    curr_node = env.get_current_node_id(trajectory)
    last_obs = ""
    for item in reversed(trajectory):
      if item.startswith("Observation:"):
        last_obs = item[len("Observation:") :].strip()
        break
    query = env.get_retrieval_query_hook(trajectory, last_obs)
    top_nodes = self.retrieve_top_k_nodes(curr_node, trajectory, query)
    return GuideBuilder.build_guidance(top_nodes)


  def retrieve_top_k_nodes(
      self,
      current_node_id: str,
      trajectory: List[str],
      query: str,
      top_k: int = 5,
  ) -> List[Tuple[Edge, float]]:
    """Finds Top-K valid outgoing transitions constrained by topology and conditions."""
    if current_node_id not in self.graph.nodes:
      logger.warning(
          "ProceduralGraphRetriever called with missing active node ID: '%s'",
          current_node_id,
      )
      return []

    direct_edges = list(self.graph.get_outgoing_edges(current_node_id))
    all_outgoing = list(direct_edges)

    seen = {(e.source, e.target) for e in direct_edges}
    for e1 in direct_edges:
      for e2 in self.graph.get_outgoing_edges(e1.target):
        if (e2.source, e2.target) not in seen:
          seen.add((e2.source, e2.target))
          all_outgoing.append(e2)

    # Also retrieve incoming transitions to prevent losing path constraints when stuck on a tool node
    incoming_edges = [e for e in self.graph.edges if e.target == current_node_id]
    for e in incoming_edges:
      if (e.source, e.target) not in seen:
        seen.add((e.source, e.target))
        all_outgoing.append(e)

    # 1. Evaluate conditions on local edges
    valid_local_edges = []
    for edge in all_outgoing:
      if self.evaluator.evaluate(edge.condition, trajectory):
        valid_local_edges.append(edge)

    # 2. Get all valid edges globally (to allow semantic jump if agent goes off-track)
    all_valid_global_edges = []
    for edge in self.graph.edges:
      if self.evaluator.evaluate(edge.condition, trajectory):
        all_valid_global_edges.append(edge)

    if not all_valid_global_edges:
      return []

    # 3. Score global edges to find semantically relevant ones
    global_descriptions = []
    for edge in all_valid_global_edges:
      target_node = self.graph.get_node(edge.target)
      desc = target_node.description if target_node else ""
      if edge.purpose:
        desc += f" Purpose: {edge.purpose}"
      rationale = edge.metadata.get("strategic_rationale")
      if rationale:
        desc += f" Rationale: {rationale}"
      global_descriptions.append(desc)

    global_scores = self.scorer.score(query, global_descriptions)
    scored_global = list(zip(all_valid_global_edges, global_scores))
    scored_global.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 2 global edges
    top_global_edges = [e for e, s in scored_global[:2]]

    # 4. Merge local and top global edges (preserving order and removing duplicates)
    merged_edges = list(valid_local_edges)
    seen_merged = {(e.source, e.target) for e in merged_edges}
    for e in top_global_edges:
      if (e.source, e.target) not in seen_merged:
        seen_merged.add((e.source, e.target))
        merged_edges.append(e)

    if not merged_edges:
      return []

    # 5. Score the merged candidate set
    candidates_descriptions = []
    for edge in merged_edges:
      target_node = self.graph.get_node(edge.target)
      desc = target_node.description if target_node else ""
      if edge.purpose:
        desc += f" Purpose: {edge.purpose}"
      rationale = edge.metadata.get("strategic_rationale")
      if rationale:
        desc += f" Rationale: {rationale}"
      candidates_descriptions.append(desc)

    scores = self.scorer.score(query, candidates_descriptions)

    scored_edges = list(zip(merged_edges, scores))
    scored_edges.sort(key=lambda x: x[1], reverse=True)
    return scored_edges[:top_k]


class GuideBuilder:
  """Formats the retrieved action transitions into highly readable prescriptions."""

  @staticmethod
  def build_guidelines(scored_edges: List[Tuple[Edge, float]]) -> str:
    """Formats retrieved action transitions with rich strategic guidance and pitfall avoidance."""
    if not scored_edges:
      return ""

    lines = ["=== RECOMMENDED PROCEDURAL GRAPH GUIDANCE ==="]
    for edge, _ in scored_edges:
      cond_str = (
          f" [Precondition: {edge.condition}]"
          if edge.condition and edge.condition.strip()
          else ""
      )
      lines.append(f"• Stage Transition: [{edge.source}] ➔ [{edge.target}]{cond_str}")
      guidance = edge.guidance or edge.purpose or edge.metadata.get("strategic_rationale")
      if guidance:
        lines.append(f"  - Strategic Policy Guidance: {guidance}")
      pitfalls = edge.pitfalls or edge.metadata.get("pitfalls")
      if pitfalls:
        lines.append(f"  - Critical Pitfalls & Policy Warnings: {pitfalls}")

    return "\n".join(lines)

  @staticmethod
  def build_guidance(scored_edges: List[Tuple[Edge, float]]) -> str:
    return GuideBuilder.build_guidelines(scored_edges)


@dataclasses.dataclass
class MemoryBankEntry:
  """Represents a discrete, time-aware structured memory entry in MemoryBank."""
  id: str
  content: str
  category: str  # e.g., 'task_insight', 'successful_rule', 'error_pattern'
  timestamp: float
  access_count: int = 1
  initial_strength: float = 1.0


class MemoryBankGuidanceProvider:
  """Faithful implementation of MemoryBank (arXiv:2305.10250) with Ebbinghaus forgetting decay,

  structured memory induction, dual-threshold filtering, and dynamic access reinforcement.
  """

  def __init__(
      self,
      train_samples: List[Dict[str, Any]],
      scorer: SimilarityScorer,
      decay_rate: float = 0.05,
      reinforcement_factor: float = 0.5,
      sim_threshold: float = 0.18,
      strength_threshold: float = 0.25,
      top_k: int = 5,
  ):
    self.scorer = scorer
    self.decay_rate = decay_rate
    self.reinforcement_factor = reinforcement_factor
    self.sim_threshold = sim_threshold
    self.strength_threshold = strength_threshold
    self.top_k = top_k
    self.memory_bank: List[MemoryBankEntry] = self._induce_memory_bank(train_samples)

  def _induce_memory_bank(self, samples: List[Dict[str, Any]]) -> List[MemoryBankEntry]:
    """Dynamically parses historical trajectories into structured memory entries with timestamps."""
    entries: List[MemoryBankEntry] = []
    base_time = time.time() - (len(samples) * 3600.0)

    for idx, sample in enumerate(samples):
      sample_id = sample.get("sample_id", f"sample_{idx}")
      question = sample.get("target_question", "").strip()
      trajectory = sample.get("trajectory", [])
      success = sample.get("success", True)
      entry_time = base_time + (idx * 3600.0)

      if not question or not trajectory:
        continue

      if success:
        actions = [s[len("Action:"):].strip() for s in trajectory if s.startswith("Action:")]
        milestone_str = " -> ".join(actions[:6])
        if len(actions) > 6:
          milestone_str += " -> ..."
        content = f"Task: '{question}'. Successful strategy milestones: [{milestone_str}]."
        entries.append(
            MemoryBankEntry(
                id=f"{sample_id}_rule",
                content=content,
                category="successful_rule",
                timestamp=entry_time,
                initial_strength=1.0,
            )
        )
      else:
        for step in reversed(trajectory):
          if step.startswith("Observation:") and any(
              err_kw in step.lower() for err_kw in ("error", "fail", "invalid", "exception")
          ):
            err_obs = step[len("Observation:"):].strip()
            content = f"Task: '{question}'. Pitfall observed: {err_obs}. Avoid repeating failed action sequence."
            entries.append(
                MemoryBankEntry(
                    id=f"{sample_id}_error",
                    content=content,
                    category="error_pattern",
                    timestamp=entry_time,
                    initial_strength=0.8,
                )
            )
            break

    logger.info("MemoryBank initialized with %d discrete memory entries.", len(entries))
    return entries

  def _compute_active_strength(self, entry: MemoryBankEntry, current_time: float) -> float:
    """Computes active memory strength S(e, t) using the Ebbinghaus forgetting curve."""
    elapsed_hours = max(0.0, (current_time - entry.timestamp) / 3600.0)
    import math
    decay_denominator = 1.0 + self.reinforcement_factor * max(0, entry.access_count - 1)
    decay_exponent = -(self.decay_rate * elapsed_hours) / decay_denominator
    return float(entry.initial_strength * math.exp(decay_exponent))

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    """Retrieves relevant memory items exceeding active strength and similarity thresholds."""
    if not self.memory_bank:
      return ""

    current_time = time.time()
    recent_items = [
        item for item in trajectory[-6:]
        if item.startswith("Observation:") or item.startswith("Thought:")
    ]
    query = "\n".join(recent_items) if recent_items else _get_task_description(env)
    if not query:
      return ""

    active_candidates: List[Tuple[MemoryBankEntry, float]] = []
    for entry in self.memory_bank:
      strength = self._compute_active_strength(entry, current_time)
      if strength >= self.strength_threshold:
        active_candidates.append((entry, strength))

    if not active_candidates:
      return ""

    candidate_contents = [c[0].content for c in active_candidates]
    try:
      sim_scores = self.scorer.score(query, candidate_contents)
    except Exception:
      return ""

    scored_items: List[Tuple[MemoryBankEntry, float, float]] = []
    for (entry, strength), sim in zip(active_candidates, sim_scores):
      if sim >= self.sim_threshold:
        combined_score = sim * strength
        scored_items.append((entry, strength, combined_score))

    if not scored_items:
      return ""

    scored_items.sort(key=lambda x: x[2], reverse=True)
    top_items = scored_items[:self.top_k]

    for entry, _, _ in top_items:
      entry.access_count += 1
      entry.timestamp = current_time

    lines = ["\n[MemoryBank Active Recall (Time-Aware Long-Term Memory)]"]
    rules = [item for item in top_items if item[0].category == "successful_rule"]
    errors = [item for item in top_items if item[0].category == "error_pattern"]

    if rules:
      lines.append("Successful Strategic Rules & Milestones:")
      for entry, strength, score in rules:
        lines.append(f"- (Strength: {strength:.2f}, Access: {entry.access_count}x) {entry.content}")
    if errors:
      lines.append("\nCritical Error Patterns to Avoid:")
      for entry, strength, score in errors:
        lines.append(f"- (Strength: {strength:.2f}, Access: {entry.access_count}x) {entry.content}")

    return "\n".join(lines)


@dataclasses.dataclass
class StateActionTransition:
  """Represents a discrete state-action transition (s, a -> s') extracted from a trajectory."""
  state_summary: str
  action: str
  next_state_summary: str
  success: bool
  trajectory_id: str
  step_idx: int


class PlanningTreeNode:
  """Represents a node in the dynamically constructed RAP planning tree / subgraph."""
  def __init__(self, node_id: str, state_summary: str):
    self.node_id: str = node_id
    self.state_summary: str = state_summary
    self.outgoing_transitions: List[StateActionTransition] = []
    self.visit_count: int = 0


class RAPGuidanceProvider:
  """Faithful implementation of RAP (Retrieval-Augmented Planning, arXiv:2402.03610) that parses

  retrieved trajectories into state-action transition subgraphs and formats structured planning trees.
  """

  def __init__(
      self,
      train_samples: List[Dict[str, Any]],
      scorer: SimilarityScorer,
      top_k_trajectories: int = 5,
      max_roadmap_steps: int = 5,
  ):
    self.train_samples = train_samples
    self.scorer = scorer
    self.top_k_trajectories = top_k_trajectories
    self.max_roadmap_steps = max_roadmap_steps
    self._subgraph_cache: Dict[str, Dict[str, PlanningTreeNode]] = {}

  def _extract_transitions(self, sample_id: str, trajectory: List[str], success: bool) -> List[StateActionTransition]:
    """Parses raw trajectory strings into structured state-action transition tuples."""
    transitions: List[StateActionTransition] = []
    curr_state = "INITIAL_STATE"

    for idx, step in enumerate(trajectory):
      if step.startswith("Observation:") or step.startswith("Question:") or step.startswith("Thought:"):
        clean_step = step.split(":", 1)[-1].strip()
        curr_state = (clean_step[:117] + "...") if len(clean_step) > 120 else clean_step
      elif step.startswith("Action:"):
        action_str = step[len("Action:"):].strip()
        next_state = "TERMINAL_STATE"
        if idx + 1 < len(trajectory) and trajectory[idx + 1].startswith("Observation:"):
          next_obs = trajectory[idx + 1][len("Observation:"):].strip()
          next_state = (next_obs[:117] + "...") if len(next_obs) > 120 else next_obs
        
        transitions.append(
            StateActionTransition(
                state_summary=curr_state,
                action=action_str,
                next_state_summary=next_state,
                success=success,
                trajectory_id=sample_id,
                step_idx=len(transitions),
            )
        )
        curr_state = next_state

    return transitions

  def _build_planning_subgraph(self, target_question: str) -> Dict[str, PlanningTreeNode]:
    """Retrieves Top-K trajectories and builds a dynamic planning tree / transition subgraph."""
    if target_question in self._subgraph_cache:
      return self._subgraph_cache[target_question]

    candidates = [s for s in self.train_samples if s.get("trajectory")]
    if not candidates:
      return {}

    candidate_questions = [c.get("target_question", "") for c in candidates]
    try:
      scores = self.scorer.score(target_question, candidate_questions)
    except Exception:
      return {}

    scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_candidates = [sc[0] for sc in scored_candidates[:self.top_k_trajectories]]

    subgraph: Dict[str, PlanningTreeNode] = {}
    for cand in top_candidates:
      sample_id = str(cand.get("sample_id", "unknown"))
      success = bool(cand.get("success", True))
      transitions = self._extract_transitions(sample_id, cand["trajectory"], success)

      for t in transitions:
        node_key = f"step_{t.step_idx}"
        if node_key not in subgraph:
          subgraph[node_key] = PlanningTreeNode(node_id=node_key, state_summary=t.state_summary)
        
        subgraph[node_key].visit_count += 1
        subgraph[node_key].outgoing_transitions.append(t)

    self._subgraph_cache[target_question] = subgraph
    return subgraph

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    """Matches active execution step against retrieved planning tree and formats checklist prior."""
    target_question = _get_task_description(env)
    if not target_question:
      return ""

    subgraph = self._build_planning_subgraph(target_question)
    if not subgraph:
      return ""

    current_action_count = len([s for s in trajectory if s.startswith("Action:")])
    active_node_key = f"step_{current_action_count}"

    lines = ["\n[RAP (Retrieval-Augmented Planning) Structured Decision Tree Prior]"]
    
    active_node = subgraph.get(active_node_key)
    if active_node and active_node.outgoing_transitions:
      lines.append(f"Current Execution Stage (Step {current_action_count}):")
      successful_branches = [t for t in active_node.outgoing_transitions if t.success]
      failed_branches = [t for t in active_node.outgoing_transitions if not t.success]

      if successful_branches:
        lines.append("Recommended Branch Transitions (from successful past trajectories):")
        seen_actions = set()
        for t in successful_branches:
          if t.action not in seen_actions:
            seen_actions.add(t.action)
            lines.append(f"  * Branch [{t.trajectory_id}]: Take Action -> {t.action} | Expected State: {t.next_state_summary}")
      
      if failed_branches:
        lines.append("Pitfall Branches to Avoid (led to failure in similar runs):")
        seen_actions = set()
        for t in failed_branches:
          if t.action not in seen_actions:
            seen_actions.add(t.action)
            lines.append(f"  * Avoid Action -> {t.action} | Observed Bad State: {t.next_state_summary}")

    future_steps = []
    for step_idx in range(current_action_count + 1, current_action_count + 1 + self.max_roadmap_steps):
      future_key = f"step_{step_idx}"
      if future_key in subgraph and subgraph[future_key].outgoing_transitions:
        succ_trans = [t for t in subgraph[future_key].outgoing_transitions if t.success]
        if succ_trans:
          future_steps.append((step_idx, succ_trans[0].action))

    if future_steps:
      lines.append("\nDownstream Execution Roadmap (Planning Tree Checklist):")
      for step_idx, act in future_steps:
        lines.append(f"  [ ] Step {step_idx} Target Transition -> {act}")

    return "\n".join(lines)


class RichMultiHopGraphRetriever:
  """Performs node-centric embedding match and generates a 2-hop neighborhood textual summary."""

  def __init__(
      self,
      graph: ProceduralGraph,
      scorer: SimilarityScorer,
      max_hops: int = 2,
      fallback_to_embedding: bool = True,
  ):
    self.graph = graph
    self.scorer = scorer
    self.max_hops = max_hops
    self.fallback_to_embedding = fallback_to_embedding

  def _filter_guidance_text(self, text: str, query: str, top_k: int = 5) -> str:
    if not text or not self.scorer or not query:
      return text

    # Split into sentences (preserving the ending dot)
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    if len(sentences) <= top_k:
      return text

    try:
      scores = self.scorer.score(query, sentences)
      scored_sentences = list(zip(sentences, scores))
      scored_sentences.sort(key=lambda x: x[1], reverse=True)
      top_sentences = [s for s, score in scored_sentences[:top_k]]
      return " ".join(top_sentences)
    except Exception as e:
      logger.warning("Failed to filter guidance sentences: %s", e)
      return text

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    """Matches active cognitive node via embedding and returns a 2-hop neighborhood textual summary."""
    if not self.graph.nodes:
      return ""

    # 1. Prepare query string from recent observations / thoughts
    last_obs = ""
    for item in reversed(trajectory):
      if item.startswith("Observation:"):
        last_obs = item[len("Observation:") :].strip()
        break

    if hasattr(env, "get_retrieval_query_hook"):
      query = env.get_retrieval_query_hook(trajectory, last_obs)
    else:
      query = last_obs

    if not query:
      if trajectory:
        query = trajectory[-1]
      else:
        query = "START"

    # 2. Identify active node: Prioritize env.get_current_node_id if available and valid
    matched_node_id = None
    if hasattr(env, "get_current_node_id"):
      try:
        curr_id = env.get_current_node_id(trajectory)
        if curr_id in self.graph.nodes:
          matched_node_id = curr_id
          logger.info(
              "RichMultiHopGraphRetriever matched node via get_current_node_id: %s",
              matched_node_id,
          )
      except Exception as e:
        logger.warning("Failed to get current node ID from env: %s", e)

    # If no exact node ID matched, return empty string so caller can fallback
    if matched_node_id is None:
      logger.info(
          "RichMultiHopGraphRetriever: No tool match found in Procedural Graph."
      )
      return ""

    matched_node = self.graph.get_node(matched_node_id)
    if not matched_node:
      return ""

    # 3. Form 2-Hop Neighborhood Adjacency Subgraph Summary
    lines = [
        "=== Procedural Graph Guidance ===",
        "The following are recommended transitions based on the active node. Use them as reference guidance, but you may choose other actions if they are more appropriate for the user request.",
        f"Active Cognitive Node: [{matched_node.id}] (Type: {matched_node.type.value})"
    ]
    if matched_node.description:
      lines.append(f"Description: {matched_node.description}")

    # Hop 1 Outgoing
    hop1_edges = list(self.graph.get_outgoing_edges(matched_node.id))
    if hop1_edges:
      lines.append("\nImmediate Transition Options (Hop 1):")
      for edge in hop1_edges:
        cond_str = (
            f" (Condition: {edge.condition})"
            if edge.condition and edge.condition.strip()
            else ""
        )
        lines.append(
            f"- Transition: [{edge.source}] -> [{edge.target}]{cond_str}"
        )
        guidance = (
            edge.guidance
            or edge.purpose
            or edge.metadata.get("strategic_rationale")
        )
        if guidance:
          filtered_guidance = self._filter_guidance_text(guidance, query)
          lines.append(f"  * Guidance: {filtered_guidance}")
        pitfalls = edge.pitfalls or edge.metadata.get("pitfalls")
        if pitfalls:
          filtered_pitfalls = self._filter_guidance_text(pitfalls, query)
          lines.append(f"  * Pitfalls to Avoid: {filtered_pitfalls}")

    # Hop 2 Outgoing
    hop2_edges = []
    seen = {(e.source, e.target) for e in hop1_edges}
    for e1 in hop1_edges:
      for e2 in self.graph.get_outgoing_edges(e1.target):
        if (e2.source, e2.target) not in seen:
          seen.add((e2.source, e2.target))
          hop2_edges.append(e2)

    if hop2_edges:
      lines.append("\nSubsequent Horizon (Hop 2):")
      for edge in hop2_edges:
        cond_str = (
            f" (Condition: {edge.condition})"
            if edge.condition and edge.condition.strip()
            else ""
        )
        lines.append(
            f"- Transition: [{edge.source}] -> [{edge.target}]{cond_str}"
        )
        guidance = (
            edge.guidance
            or edge.purpose
            or edge.metadata.get("strategic_rationale")
        )
        if guidance:
          filtered_guidance = self._filter_guidance_text(guidance, query)
          lines.append(f"  * Guidance: {filtered_guidance}")
        pitfalls = edge.pitfalls or edge.metadata.get("pitfalls")
        if pitfalls:
          filtered_pitfalls = self._filter_guidance_text(pitfalls, query)
          lines.append(f"  * Pitfalls to Avoid: {filtered_pitfalls}")

    # 4. Global Semantic Fallback (true hybrid search)
    global_edges = []
    if self.scorer and query:
      candidates_desc = []
      all_edges = self.graph.edges
      for edge in all_edges:
        target_node = self.graph.get_node(edge.target)
        desc = target_node.description if target_node else ""
        if edge.purpose:
          desc += f" Purpose: {edge.purpose}"
        rationale = edge.metadata.get("strategic_rationale")
        if rationale:
          desc += f" Rationale: {rationale}"
        candidates_desc.append(desc)
      
      try:
        global_scores = self.scorer.score(query, candidates_desc)
        scored_global = list(zip(all_edges, global_scores))
        scored_global.sort(key=lambda x: x[1], reverse=True)
        # Take top 2 global edges that are NOT already in the hop1/hop2 sets
        seen = {(e.source, e.target) for e in hop1_edges + hop2_edges}
        for e, s in scored_global:
          if (e.source, e.target) not in seen:
            global_edges.append(e)
            if len(global_edges) >= 2:
              break
      except Exception as e:
        logger.warning("Global semantic edge retrieval failed: %s", e)

    if global_edges:
      lines.append("\nSemantically Relevant Guidelines (Global Search):")
      for edge in global_edges:
        cond_str = (
            f" (Condition: {edge.condition})"
            if edge.condition and edge.condition.strip()
            else ""
        )
        lines.append(
            f"- Transition: [{edge.source}] -> [{edge.target}]{cond_str}"
        )
        guidance = (
            edge.guidance
            or edge.purpose
            or edge.metadata.get("strategic_rationale")
        )
        if guidance:
          filtered_guidance = self._filter_guidance_text(guidance, query)
          lines.append(f"  * Guidance: {filtered_guidance}")
        pitfalls = edge.pitfalls or edge.metadata.get("pitfalls")
        if pitfalls:
          filtered_pitfalls = self._filter_guidance_text(pitfalls, query)
          lines.append(f"  * Pitfalls to Avoid: {filtered_pitfalls}")

    return "\n".join(lines)


class GenerativeGraphGuidanceProvider:
  """Generates situational textual guidance by prompting an LLM with the full Procedural Graph and current trajectory."""

  def __init__(
      self,
      graph: ProceduralGraph,
      llm_client: Any,
  ):
    self.graph = graph
    self.llm_client = llm_client

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    """Prompts LLM with full Procedural Graph and trajectory to generate situational guidance."""
    if not self.graph.nodes:
      return ""

    graph_summary = self.graph.to_text_summary()

    last_obs = ""
    for item in reversed(trajectory):
      if item.startswith("Observation:"):
        last_obs = item[len("Observation:") :].strip()
        break

    if hasattr(env, "get_retrieval_query_hook"):
      query = env.get_retrieval_query_hook(trajectory, last_obs)
    else:
      query = last_obs

    if not query:
      query = trajectory[-1] if trajectory else "START"

    recent_context = (
        "\n".join(trajectory[-6:])
        if trajectory
        else "No prior steps (Initial state)."
    )

    is_cfo = "cfo" in env.__class__.__name__.lower()
    constraint_prefix = ""
    constraint_suffix = ""
    if is_cfo:
      constraint_prefix = (
          "[CRITICAL EXECUTION CONSTRAINT]\n"
          "The agent you are guiding operates strictly in a ReAct loop and has NO ability to output text, code, "
          "or deliverables directly to the user. ALL deliverables (queries, scripts, markdown files, letters, etc.) "
          "MUST be programmatically written to files in the workspace using 'write_file' or 'run_python' before "
          "calling 'submit()'. You MUST NEVER suggest that the agent output text directly or provide answers in the chat. "
          "Any recommendation to output text directly will cause the agent to crash.\n\n"
      )
      constraint_suffix = (
          "\n\n[REITERATED CRITICAL CONSTRAINT]\n"
          "Remember: Do NOT tell the agent to output the query or content directly. Tell it explicitly to write "
          "it to a file (e.g. 'instructions.md') using 'write_file' and then submit."
      )
    else:
      constraint_prefix = (
          "[EXECUTION CONSTRAINT]\n"
          "The agent you are guiding operates strictly in a ReAct loop and calls tool functions to interact with the environment. "
          "First identify the target domain of the user request (e.g., TradingBot, TravelAPI, TicketAPI, TwitterAPI, MessageAPI, VehicleControlAPI, MathAPI, CmdClient). "
          "Always suggest the next logical tool function to call from that target domain based on the Procedural Graph transitions. "
          "Do NOT recommend tools belonging to an unrelated domain (e.g. do NOT recommend Twitter or bash tools when user asks for stock trading, tickets, or flight bookings).\n\n"
      )

    prompt = (
        f"{constraint_prefix}"
        "You are an expert cognitive architect and execution guide for an AI"
        f" agent solving the task: {_get_task_description(env)}\n\n"
        "Here is the complete Procedural Graph governing the task"
        f" structure and strategic guidance:\n{graph_summary}\n\nHere is the"
        f" current active query / observation:\n{query}\n\nHere is the agent's"
        f" recent execution trajectory:\n{recent_context}\n\nAnalyze the"
        " complete Procedural Graph in the context of the agent's current progress."
        " Generate clear, actionable guidance (in 2-4 sentences) advising the"
        " agent on exactly what step or strategy to pursue next, and what"
        " pitfalls to avoid."
        f"{constraint_suffix}"
    )

    try:
      guidance_text = self.llm_client.generate(prompt, temperature=0.0)
      return (
          "Recommended Execution Guidance (Generated from Action"
          f" Graph):\n{guidance_text.strip()}"
      )
    except Exception as e:
      logger.warning("Generative guidance generation failed: %s", e)
      return ""


class ExpeLGuidanceProvider:
  """Retrieves similar exemplars and triggers of rules for ExpeL."""

  def __init__(
      self,
      train_samples: List[Dict[str, Any]],
      rules: List[Dict[str, Any]],
      scorer: SimilarityScorer,
      top_k_exemplars: int = 3,
      top_k_rules: int = 3,
  ):
    self.train_samples = train_samples
    self.rules = rules
    self.scorer = scorer
    self.top_k_exemplars = top_k_exemplars
    self.top_k_rules = top_k_rules

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    target_question = _get_task_description(env)
    if not target_question:
      return ""

    guidance_lines = []

    # 1. Retrieve similar exemplars
    successful_samples = [s for s in self.train_samples if s.get("success", True) and s.get("trajectory")]
    if successful_samples:
      candidate_questions = [s["target_question"] for s in successful_samples]
      scores = self.scorer.score(target_question, candidate_questions)
      scored = sorted(zip(successful_samples, scores), key=lambda x: x[1], reverse=True)
      top_samples = [s[0] for s in scored[:self.top_k_exemplars]]

      guidance_lines.append("[ExpeL Retrieved Experiences]")
      for i, cand in enumerate(top_samples):
        guidance_lines.append(f"--- Example {i+1} ---")
        guidance_lines.append(f"Task: {cand['target_question']}")
        guidance_lines.append("Trajectory:")
        traj_steps = cand["trajectory"]
        if len(traj_steps) > 30:
          truncated_steps = traj_steps[:15] + ["...[TRUNCATED INTERMEDIATE STEPS]..."] + traj_steps[-15:]
        else:
          truncated_steps = traj_steps
        for step in truncated_steps:
          guidance_lines.append(step)
        guidance_lines.append("-----------------\n")

    # 2. Retrieve relevant rules/insights
    if self.rules:
      recent_items = [
          item for item in trajectory[-10:]
          if item.startswith("Observation:") or item.startswith("Thought:")
      ]
      query_context = "\n".join(recent_items) if recent_items else target_question
      
      triggers = [r.get("trigger", "") for r in self.rules]
      rule_scores = self.scorer.score(query_context, triggers)
      scored_rules = sorted(zip(self.rules, rule_scores), key=lambda x: x[1], reverse=True)
      top_rules = [r[0] for r in scored_rules[:self.top_k_rules] if r[1] > 0.1]

      if top_rules:
        guidance_lines.append("[ExpeL Abstracted Rules]")
        for r in top_rules:
          guidance_lines.append(f"- Under context: {r.get('trigger')}\n  Guideline: {r.get('insight')}")

    return "\n".join(guidance_lines)


class AutoGuideGuidanceProvider:
  """Retrieves context-aware guidelines for AutoGuide."""

  def __init__(
      self,
      guidelines: List[Dict[str, Any]],
      scorer: SimilarityScorer,
      top_k: int = 5,
  ):
    self.guidelines = guidelines
    self.scorer = scorer
    self.top_k = top_k

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    if not self.guidelines:
      return ""

    recent_items = [
        item for item in trajectory[-10:]
        if item.startswith("Observation:") or item.startswith("Thought:")
    ]
    query_context = "\n".join(recent_items) if recent_items else _get_task_description(env)

    conditions = [g.get("condition", "") for g in self.guidelines]
    scores = self.scorer.score(query_context, conditions)
    scored = sorted(zip(self.guidelines, scores), key=lambda x: x[1], reverse=True)
    top_guidelines = [g[0] for g in scored[:self.top_k] if g[1] > 0.15]

    if not top_guidelines:
      return ""

    guidance_lines = ["\n[AutoGuide Selected Guidelines]"]
    for g in top_guidelines:
      guidance_lines.append(f"- If {g.get('condition')}\n  Then: {g.get('guideline')}")
    return "\n".join(guidance_lines)


class AWMGuidanceProvider:
  """Retrieves step-by-step induced workflows for Agent Workflow Memory."""

  def __init__(self, workflows: Dict[str, Any]):
    self.workflows = workflows

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    occupation = getattr(env, "occupation", "").strip()
    if not occupation or occupation not in self.workflows:
      if not self.workflows:
        return ""
      occupation = next(iter(self.workflows.keys()))

    workflow = self.workflows.get(occupation, "")
    if not workflow:
      return ""

    return f"\n[Agent Workflow Memory (AWM) Checklist Plan]\nOccupation: {occupation}\nWorkflow steps to execute:\n{workflow}"


class KnowAgentGuidanceProvider:
  """Retrieves next-action planning suggestions from Action Knowledge Base (KB)."""

  def __init__(self, kb: Dict[str, Any]):
    self.kb = kb

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    last_tool = "reset"
    for item in reversed(trajectory):
      if item.startswith("Action:"):
        action_content = item[len("Action:") :].strip()
        try:
          from procedural_graph.env_base import parse_action_string
          action_name, _, _ = parse_action_string(action_content)
          last_tool = action_name
          break
        except Exception:
          pass

    next_options = self.kb.get(last_tool, [])
    if not next_options:
      return ""

    options_str = ", ".join(next_options)
    return (
        f"\n[KnowAgent Action Knowledge Base Guidance]\n"
        f"Based on successful historical runs, after calling '{last_tool}', "
        f"the recommended next action(s)/tool(s) are: {options_str}."
    )


class StaticGuidanceProvider:
  """Simple provider that returns a static guidance string."""

  def __init__(self, guidance: str):
    self.guidance = guidance

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    return self.guidance
class HybridGraphGuidanceProvider:
  """Generates situational textual guidance by prompting an LLM with a local subgraph and current trajectory."""

  def __init__(
      self,
      graph: ProceduralGraph,
      llm_client: Any,
      scorer: SimilarityScorer,
      max_hops: int = 1,
      kb: Optional[Dict[str, Any]] = None,
  ):
    self.graph = graph
    self.llm_client = llm_client
    self.scorer = scorer
    self.max_hops = max_hops
    self.kb = kb or {}
    # Reuse RichMultiHopGraphRetriever to extract the subgraph summary
    # Disable embedding fallback so we know if the tool match failed
    self.subgraph_extractor = RichMultiHopGraphRetriever(
        graph, scorer, max_hops=max_hops, fallback_to_embedding=False
    )

  def get_guidance(self, env: Any, trajectory: List[str]) -> str:
    """Prompts LLM with local subgraph and trajectory to generate situational guidance."""
    if not self.graph.nodes:
      return ""

    # 1. Extract local subgraph text
    subgraph_summary = self.subgraph_extractor.get_guidance(env, trajectory)
    
    if not subgraph_summary:
      # Fallback to full graph if we couldn't match the tool
      subgraph_summary = self.graph.to_text_summary()
      graph_source = "Full Procedural Graph (Fallback due to unmatched state)"
      graph_context_desc = "the complete Procedural Graph governing the task structure and strategic guidance"
    else:
      graph_source = "Local Procedural Graph Partition"
      graph_context_desc = "the relevant local partition of the Procedural Graph governing the current state and immediate transition options"

    # 2. Prepare query and context
    last_obs = ""
    for item in reversed(trajectory):
      if item.startswith("Observation:"):
        last_obs = item[len("Observation:") :].strip()
        break

    if hasattr(env, "get_retrieval_query_hook"):
      query = env.get_retrieval_query_hook(trajectory, last_obs)
    else:
      query = last_obs

    if not query:
      query = trajectory[-1] if trajectory else "START"

    recent_context = (
        "\n".join(trajectory[-6:])
        if trajectory
        else "No prior steps (Initial state)."
    )

    # 3. Construct prompt with local subgraph or full graph fallback
    is_cfo = "cfo" in env.__class__.__name__.lower()
    constraint_prefix = ""
    constraint_suffix = ""
    if is_cfo:
      constraint_prefix = (
          "[CRITICAL EXECUTION CONSTRAINT]\n"
          "The agent you are guiding operates strictly in a ReAct loop and has NO ability to output text, code, "
          "or deliverables directly to the user. ALL deliverables (queries, scripts, markdown files, letters, etc.) "
          "MUST be programmatically written to files in the workspace using 'write_file' or 'run_python' before "
          "calling 'submit()'. You MUST NEVER suggest that the agent output text directly or provide answers in the chat. "
          "Any recommendation to output text directly will cause the agent to crash.\n\n"
      )
      constraint_suffix = (
          "\n\n[REITERATED CRITICAL CONSTRAINT]\n"
          "Remember: Do NOT tell the agent to output or content directly. Tell it explicitly to write "
          "it to a file (e.g. 'instructions.md') using 'write_file' and then submit."
      )
    else:
      constraint_prefix = (
          "[EXECUTION CONSTRAINT]\n"
          "The agent you are guiding operates strictly in a ReAct loop and calls tool functions to interact with the environment. "
          "First identify the target domain of the user request (e.g., TradingBot, TravelAPI, TicketAPI, TwitterAPI, MessageAPI, VehicleControlAPI, MathAPI, CmdClient). "
          "Always suggest the next logical tool function to call from that target domain based on the Procedural Graph transitions. "
          "Do NOT recommend tools belonging to an unrelated domain (e.g. do NOT recommend Twitter or bash tools when user asks for stock trading, tickets, or flight bookings).\n\n"
      )

    prompt = (
        f"{constraint_prefix}"
        f"Task: {_get_task_description(env)}\n"
        f"Current Active Query / Observation: {query}\n"
        f"Recent Execution Trajectory:\n{recent_context}\n\n"
        f"Procedural Graph Subgraph Policy & Transitions:\n{subgraph_summary}\n\n"
        "Generate extremely concise, direct bullet points advising the agent:\n"
        "1. NEXT TOOL FUNCTION TO CALL and exact parameter names. Perform ALL query lookups (get_product_details) BEFORE calling respond.\n"
        "2. CRITICAL PRE-CONDITIONS & PITFALLS (e.g. 10-digit variant item IDs, payment_method_id, order status).\n"
        "Keep the total guidance under 80 words. Direct bullet points only. No conversational filler."
        f"{constraint_suffix}"
    )

    try:
      guidance_text = self.llm_client.generate(prompt, temperature=0.0)
      return (
          "Recommended Execution Guidance (Generated from Local Action"
          f" Graph Partition):\n{guidance_text.strip()}"
      )
    except Exception as e:
      logger.warning("Hybrid guidance generation failed: %s", e)
      return ""

