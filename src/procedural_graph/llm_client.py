"""Real Vertex AI LLM Client utilizing google auth credentials and generateContent API."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request


class VertexAIClient:
  """Real Vertex AI Client conforming to solvers.LLMClient protocol."""

  def __init__(
      self,
      model_name: str = "grok-4.1-fast-non-reasoning",
      project_id: Optional[str] = None,
      location: Optional[str] = None,
      timeout: int = 180,
  ):
    self.model_name = model_name
    self.project_id = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("VERTEX_PROJECT_ID")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if not self.project_id:
      raise ValueError(
          "GCP project ID is required for Vertex AI. Please set the "
          "GOOGLE_CLOUD_PROJECT environment variable or pass project_id explicitly."
      )
    self.timeout = timeout

    if location is not None:
      self.location = location
    else:
      self.location = "us-central1"

    # Set environment variables for sub-libraries if needed
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.project_id)
    os.environ.setdefault("GCLOUD_PROJECT", self.project_id)

    import google.auth  # pytype: disable=import-error
    from google.auth.transport import requests as auth_requests  # pytype: disable=import-error

    try:
      self.credentials, _ = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
          quota_project_id=self.project_id,
      )
      self.auth_request = auth_requests.Request()
      self.credentials.refresh(self.auth_request)
    except Exception as e:
      raise RuntimeError(
          f"Failed to initialize Google Auth Credentials for Vertex AI: {e}. "
          "Please check your GCP environment authentication settings."
      ) from e

    self.prompt_tokens = 0
    self.completion_tokens = 0
    self.api_calls = 0
    self.total_latency = 0.0

  @property
  def max_token_len(self) -> int:
    name = self.model_name.lower()
    if "gemini-3.1-pro-preview" in name or "gemini-1.5-pro" in name:
      return 2000000
    elif (
        "gemini-2.5-flash" in name
        or "gemini-3.5-flash" in name
        or "gemini-1.5-flash" in name
    ):
      return 1000000
    elif "claude" in name:
      return 200000
    elif "grok" in name:
      return 131072
    return 100000  # Conservative default

  def generate(
      self,
      prompt: str,
      temperature: float = 0.0,
      system_instruction: Optional[str] = None,
  ) -> str:
    """Generates text completions from the Vertex model (conforming to LLMClient Protocol)."""
    print(f"🤖 [VertexAIClient] Calling {self.model_name} (location={self.location}, prompt_len={len(prompt)})...", flush=True)
    publisher = "google"
    model_id = self.model_name
    
    if "grok" in self.model_name.lower():
      publisher = "xai"
    elif "claude" in self.model_name.lower():
      publisher = "anthropic"
      if model_id.startswith("anthropic-"):
        model_id = model_id[len("anthropic-"):]

    method = "rawPredict" if publisher == "anthropic" else "generateContent"
    if self.location == "global":
      url = (
          f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/"
          f"locations/global/publishers/{publisher}/models/{model_id}:{method}"
      )
    else:
      url = (
          f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/"
          f"locations/{self.location}/publishers/{publisher}/models/{model_id}:{method}"
      )

    if publisher == "anthropic":
      # Anthropic Messages API payload
      payload: Dict[str, Any] = {
          "anthropic_version": "vertex-2023-10-16",
          "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 4096,  # Claude requires max_tokens to be set explicitly
          "temperature": temperature,
      }
      if system_instruction:
        payload["system"] = system_instruction
    else:
      # Google generateContent API payload
      payload: Dict[str, Any] = {
          "contents": [{"role": "user", "parts": [{"text": prompt}]}],
          "generationConfig": {
              "temperature": temperature,
              "seed": 42,
          },
      }
      if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {self.credentials.token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": self.project_id,
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    max_retries = 10
    for attempt in range(1, max_retries + 1):
      start_time = time.time()
      try:
        # Refresh token if expired
        if self.credentials.expired:
          self.credentials.refresh(self.auth_request)
          headers["Authorization"] = f"Bearer {self.credentials.token}"
          req = urllib.request.Request(
              url, data=data, headers=headers, method="POST"
          )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
          result = json.loads(response.read().decode("utf-8"))

          if publisher == "anthropic":
            usage = result.get("usage", {})
            p_tokens = usage.get("input_tokens", 0)
            c_tokens = usage.get("output_tokens", 0)
            text = ""
            for content_part in result.get("content", []):
              if content_part.get("type") == "text":
                text += content_part.get("text", "")
          else:
            usage = result.get("usageMetadata", {})
            p_tokens = usage.get("promptTokenCount", 0)
            c_tokens = usage.get("candidatesTokenCount", 0)
            candidates = result.get("candidates", [])
            if not candidates:
              print(
                  f"⚠️ [VertexAIClient] Warning: Empty candidates in response: {result}",
                  flush=True,
              )
              text = ""
            else:
              parts = candidates[0].get("content", {}).get("parts", [])
              text = "".join(part.get("text", "") for part in parts if "text" in part)
          self.prompt_tokens += p_tokens
          self.completion_tokens += c_tokens
          self.api_calls += 1
          latency = time.time() - start_time
          self.total_latency += latency

          print(f"✅ [VertexAIClient] {self.model_name} responded: tokens_in={p_tokens}, tokens_out={c_tokens}, latency={latency:.2f}s", flush=True)
          return text
      except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        # Retry on common server/throttling errors
        retry_codes = [401, 403, 429, 500, 502, 503, 504]
        if attempt < max_retries and e.code in retry_codes:
          sleep_time = (2.5 ** attempt) + random.uniform(1, 4)
          print(f"⚠️ Vertex API HTTPError {e.code} for model {self.model_name} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s...")
          time.sleep(sleep_time)
          continue
        raise RuntimeError(
            f"Vertex AI API HTTPError {e.code} for model "
            f"{self.model_name}: {err_body}"
        ) from e
      except (TimeoutError, urllib.error.URLError) as e:
        is_timeout = isinstance(e, TimeoutError) or (isinstance(e, urllib.error.URLError) and "timed out" in str(e).lower())
        if is_timeout and attempt >= max_retries:
          print(f"⚠️ Vertex API Timeout for model {self.model_name} after {attempt} attempts: {e}", flush=True)
          raise RuntimeError(
              f"Vertex AI API Timeout error for model {self.model_name}: {e}"
          ) from e
        sleep_time = min(30, (2.0 ** attempt) + random.uniform(1, 3))
        print(f"⚠️ Vertex API Network/Timeout error for model {self.model_name}: {e} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s...", flush=True)
        time.sleep(sleep_time)
        continue
      except Exception as e:
        if attempt < max_retries:
          sleep_time = min(30, (2.0 ** attempt) + random.uniform(1, 3))
          print(f"⚠️ Vertex API Network error for model {self.model_name}: {e} (Attempt {attempt}/{max_retries}). Retrying in {sleep_time:.2f}s...", flush=True)
          time.sleep(sleep_time)
          continue
        raise RuntimeError(
            f"Vertex AI API Network error for model {self.model_name}: {e}"
        ) from e

    raise RuntimeError(
        f"Vertex AI API generateContent failed after {max_retries} retries for"
        f" model {self.model_name}."
    )

  def get_telemetry_stats(self) -> Dict[str, Any]:
    return {
        "prompt_tokens": self.prompt_tokens,
        "completion_tokens": self.completion_tokens,
        "total_tokens": self.prompt_tokens + self.completion_tokens,
        "api_calls": self.api_calls,
        "total_latency": self.total_latency,
    }
