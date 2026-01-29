from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shinka.llm.models.result import QueryResult


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def load_mock_statements(path: Optional[str]) -> List[str]:
    """Load a list of Lean statements for MockLLM.

    Supported formats:
    - `.json`: JSON list of strings.
    - otherwise: non-empty lines as single-line statements.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
            return []
                                                                        
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []


_DECL_NAME_RE = re.compile(
    r"^\s*(?:noncomputable\s+)?"
    r"(theorem|lemma|def|example|axiom|abbrev|opaque|instance)\s+"
    r"([A-Za-z_][A-Za-z0-9_']*)\b",
    re.MULTILINE,
)


def _rewrite_decl_name(stmt: str, new_name: str) -> str:
    m = _DECL_NAME_RE.search(stmt)
    if not m:
        return stmt
    name_span = m.span(2)
    return stmt[: name_span[0]] + new_name + stmt[name_span[1] :]


def _ensure_by_sorry(stmt: str) -> str:
    s = stmt.strip()
    if not s:
        return s
                                             
    if re.search(r"\bsorry\s*$", s):
        return s
                                                                                  
    if ":=" in s:
        head = s.split(":=", 1)[0].rstrip()
        return head + " := by sorry"
    return s + " := by sorry"


@dataclass
class LLMAvailability:
    ok: bool
    reason: str = ""


def probe_openai_base_url(base_url: Optional[str], timeout_s: float = 0.8) -> LLMAvailability:
    """Best-effort reachability probe for an OpenAI-compatible server.

    - Never raises.
    - Treats "local"/"mock" URLs as unavailable.
    - If base_url is empty, we assume the OpenAI SDK default endpoint is usable
      *only when* an API key is provided (so `--llm_mode=auto` can work with
      the official OpenAI endpoint without hard-coding a base_url).
    """
    url = (base_url or "").strip()
    if not url:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if api_key and api_key.upper() != "EMPTY":
            return LLMAvailability(ok=True, reason="openai_sdk_default")
        return LLMAvailability(ok=False, reason="empty_base_url_no_api_key")
    if url.lower() in {"local", "mock"} or url.lower().startswith("mock://"):
        return LLMAvailability(ok=False, reason=f"non_http_base_url:{url}")
    if not (url.startswith("http://") or url.startswith("https://")):
        return LLMAvailability(ok=False, reason=f"non_http_base_url:{url}")

                                                   
    try:
        import requests

        models_url = url.rstrip("/") + "/models"
        resp = requests.get(models_url, timeout=timeout_s)
        if 200 <= resp.status_code < 300:
            return LLMAvailability(ok=True, reason="ok")
        return LLMAvailability(ok=False, reason=f"http_status:{resp.status_code}")
    except Exception as e:
        return LLMAvailability(ok=False, reason=f"exception:{type(e).__name__}:{e}")


class MockLLMClient:
    """Offline LLM for smoke tests (no network).

    Contract: implements the subset of `shinka.llm.LLMClient` used by this repo:
    - `get_kwargs() -> dict`
    - `query(...) -> QueryResult`
    - `total_calls` counter
    """

    def __init__(
        self,
        statements: Optional[List[str]] = None,
        model_name: str = "mock-llm",
        seed: int = 0,
    ):
        self.model_name = model_name
        self.total_calls = 0
        self._counter = 0

                                       
        pool = list(statements or [])
        if not pool:
            pool = [
                "theorem my_mock : True := by sorry",
                "theorem my_mock (n : Nat) : n = n := by sorry",
            ]
        self._statements = pool

                                                                                 
        self._emit_broken_once = _truthy(os.environ.get("AUTOFORMAL_MOCK_EMIT_BROKEN_ONCE"))
        self._broken_emitted = False

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": 0.0,
            "max_tokens": 0,
        }

    def _next_statement(self, kind: str) -> str:
        self._counter += 1

                                                                                   
        if kind == "patch" and self._emit_broken_once and not self._broken_emitted:
            self._broken_emitted = True
            return f"theorem my_mock_broken_{self._counter:04d} : True := by"

        base = self._statements[(self._counter - 1) % max(len(self._statements), 1)]
        name = f"my_mock_{self._counter:04d}"
        stmt = _rewrite_decl_name(base, name)
        stmt = _ensure_by_sorry(stmt)
        return stmt.strip()

    def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> QueryResult:
        self.total_calls += 1

        kind = "patch"
        if "Compile Error" in (msg or "") or "FIX a Lean4 theorem" in (system_msg or ""):
            kind = "repair"

        stmt = self._next_statement(kind=kind)
        content = (
            f"<NAME>mock_{self.total_calls:04d}</NAME>\n"
            f"<DESCRIPTION>MockLLM offline response ({kind})</DESCRIPTION>\n"
            f"```lean\n{stmt}\n```"
        )
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[],
            model_name=self.model_name,
            kwargs=llm_kwargs or {},
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
        )


class ReplayLLMClient:
    """Offline replay client reading pre-recorded QueryResult payloads from JSONL."""

    def __init__(self, jsonl_path: str, model_name: str = "replay-llm"):
        self.model_name = model_name
        self.total_calls = 0
        self._records: List[Dict[str, Any]] = []
        self._idx = 0

        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(f"Replay file not found: {jsonl_path}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and "content" in rec:
                self._records.append(rec)

        if not self._records:
            raise ValueError(f"Replay file has no usable records: {jsonl_path}")

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": 0.0,
            "max_tokens": 0,
        }

    def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> QueryResult:
        self.total_calls += 1

        rec = (
            self._records[self._idx]
            if self._idx < len(self._records)
            else self._records[-1]
        )
        self._idx += 1

        content = str(rec.get("content", ""))
        cost = float(rec.get("cost", 0.0) or 0.0)
        model_name = str(rec.get("model_name", self.model_name) or self.model_name)

        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[],
            model_name=model_name,
            kwargs=llm_kwargs or {},
            input_tokens=int(rec.get("input_tokens", 0) or 0),
            output_tokens=int(rec.get("output_tokens", 0) or 0),
            cost=cost,
        )
