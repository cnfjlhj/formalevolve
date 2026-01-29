                      
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

_DEFAULT_INFRA_ERROR_PATTERNS: List[str] = [
    r"^TimeoutError\b",
    r"^ReadTimeout\b",
    r"^ConnectTimeout\b",
    r"^RemoteProtocolError\b",
    r"^HTTPStatusError: Server error '50[0234]\b",
                                                                          
                                                                                
                                           
    r"maximum context length",
                                                                                
                                                                               
                                 
    r"^HTTPStatusError: Client error '400\b",
    r"^JSONDecodeError\b",
]


def _normalize_base_url(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    s = s.rstrip("/")
    if s.endswith("/v1"):
        return s
    if s.endswith("/v1/"):
        return s[:-1]
    return s + "/v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


async def _openai_chat(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    n: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> List[str]:
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "n": int(n),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    resp: Optional[httpx.Response] = None
    try:
        resp = await asyncio.wait_for(
            client.post(url, headers=headers, json=payload),
            timeout=float(timeout_s),
        )

                                                                          
                                                                              
                                                                       
        if int(resp.status_code) >= 400:
            msg = ""
            body = ""
            try:
                body = resp.text
            except Exception:
                body = ""
            try:
                obj = resp.json()
                if isinstance(obj, dict):
                    err = obj.get("error") or {}
                    if isinstance(err, dict):
                        msg = str(err.get("message") or "").strip()
            except Exception:
                msg = ""

            detail = (msg or body or "").strip()
            if len(detail) > 500:
                detail = detail[:500] + "...(truncated)"
            raise httpx.HTTPStatusError(
                f"Client error '{resp.status_code}' for url '{url}': {detail}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()
    finally:
        if resp is not None:
            try:
                await resp.aclose()
            except Exception:
                pass
    out: List[str] = []
    try:
        for ch in data.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") or {}
            if isinstance(msg, dict):
                out.append(str(msg.get("content") or ""))
    except Exception:
        return []
    return out


def _force_by_sorry(stmt: str) -> str:
    s = str(stmt or "").strip()
    if not s:
        return ""
                                                                                    
                                                                        
    if ":=" not in s:
        return s
    head = s.split(":=", 1)[0].rstrip()
    return head + " := by sorry"


_G_SUFFIX_PAT = re.compile(r"_g(?P<g>\d+)$")


def _parse_g_idx(name: str) -> Optional[int]:
    m = _G_SUFFIX_PAT.search(str(name or "").strip())
    if not m:
        return None
    try:
        return int(m.group("g"))
    except Exception:
        return None


def _record_infra_retry_count(record: Dict[str, Any]) -> int:
    try:
        return int(record.get("infra_retry") or 0)
    except Exception:
        return 0


def _is_retryable_infra_failure(
    record: Dict[str, Any],
    *,
    infra_error_patterns: List[re.Pattern[str]],
) -> bool:
    err = str(record.get("error") or "").strip()
    if not err:
        return False
    for pat in infra_error_patterns:
        if pat.search(err):
            return True
    return False


def _effective_infra_retry_count(
    record: Dict[str, Any],
    *,
    infra_error_patterns: List[re.Pattern[str]],
) -> int:
    """Return an integer counter used to cap infra retries.

    Historical records may not have an `infra_retry` field. In that case, we still
    want to count an existing infra failure as having consumed 1 retry budget.
    """
    base = _record_infra_retry_count(record)
    if base > 0:
        return base
    if _is_retryable_infra_failure(record, infra_error_patterns=infra_error_patterns):
        return 1
    return 0


def _looks_like_context_length_error(err: str) -> bool:
    s = str(err or "").lower()
    return ("maximum context length" in s) or ("context length" in s) or ("requested" in s and "tokens" in s)


def _compute_safe_max_tokens_from_error(err: str, *, requested_max_tokens: int) -> Optional[int]:
    """Best-effort derive a smaller max_tokens from a vLLM context-length error string."""
    text = str(err or "")
    lower = text.lower()
    if "maximum context length" not in lower:
        return None

    model_max: Optional[int] = None
    msg_tokens: Optional[int] = None

    m = re.search(r"maximum context length is\\s+(\\d+)\\s+tokens", lower)
    if m:
        try:
            model_max = int(m.group(1))
        except Exception:
            model_max = None

    m = re.search(r"\\((\\d+)\\s+in the messages,\\s+(\\d+)\\s+in the completion\\)", lower)
    if m:
        try:
            msg_tokens = int(m.group(1))
        except Exception:
            msg_tokens = None

    buffer_tokens = 256
    if model_max is not None and msg_tokens is not None:
        safe = int(model_max) - int(msg_tokens) - int(buffer_tokens)
        safe = max(256, safe)
        if safe >= int(requested_max_tokens):
            return None
        return safe

                                                                       
    safe = int(requested_max_tokens) - 512
    safe = max(256, safe)
    if safe >= int(requested_max_tokens):
        return None
    return safe


def _load_existing_by_origin(output_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    path = output_dir / "to_inference_codes.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}

    by_origin: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        origin = str(r.get("origin_problem_id") or "").strip()
        if not origin:
            continue
        name = str(r.get("problem_id") or r.get("name") or "").strip()
        g_idx = _parse_g_idx(name)
        if g_idx is None:
            continue
        by_origin.setdefault(origin, {})[int(g_idx)] = r
    return by_origin


def _compute_resume_targets(
    *,
    loaded: List[Item],
    existing_by_origin: Dict[str, Dict[int, Dict[str, Any]]],
    resume_retry_infra: bool,
    infra_error_patterns: List[re.Pattern[str]],
    max_infra_retries: int,
    resume_retry_max_tokens_mismatch: bool,
    expected_max_tokens: int,
) -> Dict[str, List[int]]:
    def _record_has_successful_output(rec: Dict[str, Any]) -> bool:
        if not isinstance(rec, dict):
            return False
        err = str(rec.get("error") or "").strip()
        if err:
            return False
        out = str(rec.get("model_output") or "").strip()
        if out:
            return True
                                                                                  
        full_code = str(rec.get("full_code") or "").strip()
        return bool(full_code and full_code != "None")

    targets: Dict[str, List[int]] = {}
    for it in loaded:
        exp_n = int(it.n)
        if exp_n <= 0:
            continue
        by_g = existing_by_origin.get(it.origin_problem_id, {})
        todo: List[int] = []
        for g in range(exp_n):
            rec = by_g.get(int(g))
            if rec is None:
                todo.append(int(g))
                continue
            if bool(resume_retry_max_tokens_mismatch):
                try:
                    prev_mt = int(rec.get("max_tokens") or 0)
                except Exception:
                    prev_mt = 0
                if int(prev_mt) != int(expected_max_tokens) and not _record_has_successful_output(rec):
                    todo.append(int(g))
                    continue
            if not bool(resume_retry_infra):
                continue
            if not _is_retryable_infra_failure(rec, infra_error_patterns=infra_error_patterns):
                continue
            if int(max_infra_retries) > 0 and _effective_infra_retry_count(
                rec, infra_error_patterns=infra_error_patterns
            ) >= int(max_infra_retries):
                continue
            todo.append(int(g))
        if todo:
            targets[it.origin_problem_id] = todo
    return targets


def _build_prompt(
    *,
    handler: str,
    lean4_code: str,
) -> List[Dict[str, str]]:
    if handler == "dpskcot":
                                                                                      
        formal_statement = lean4_code.split(":= by")[0] + ":= by sorry"
        prompt = (
            "Complete the following Lean 4 code:\n\n"
            f"```lean4\n{formal_statement}```\n\n"
            "Before producing the Lean 4 code to formally prove the given theorem, "
            "provide a detailed proof plan outlining the main proof steps and strategies.\n"
            "The plan should highlight key ideas, intermediate lemmas, and proof structures "
            "that will guide the construction of the final formal proof."
        )
        return [{"role": "user", "content": prompt}]
    if handler == "kiminacot":
                                                                                    
        formal_statement = lean4_code.split(":= by")[0] + ":= by"
                                                    
        lines = str(formal_statement or "").splitlines()
        cleaned = "\n".join(
            [
                line
                for line in lines
                if not (
                    line.startswith("import")
                    or line.startswith("set_option")
                    or line.startswith("open")
                    or line.strip() == ""
                )
            ]
        )
        user = (
            "Think about and solve the following problem step by step in Lean 4."
            f"\n# Problem:{cleaned}"
            f"\n# Formal statement:\n```lean4\n{formal_statement}\n```\n"
        )
        return [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": user},
        ]
    if handler == "dpsknoncot":
                                                                                         
        formal_statement = lean4_code.split(":= by")[0] + ":= by sorry"
        prompt = (
            "Complete the following Lean 4 code.\n"
            "Return ONLY a complete Lean 4 file inside a ```lean4 code fence (no explanations).\n\n"
            f"```lean4\n{formal_statement}\n```\n"
        )
        return [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4. Output only Lean 4 code."},
            {"role": "user", "content": prompt},
        ]
    raise ValueError(f"Unknown handler: {handler!r}")


def _extract_code_from_fence(text: str) -> str:
                                                                                  
    if not text:
        return ""
    for pat in (r"```lean4\n(.*?)\n```", r"```lean4\n(.*?)```", r"```lean\n(.*?)```"):
        matches = re.findall(pat, text, re.DOTALL)
        if matches:
            return str(matches[-1]).strip()
                                                                          
    for fence in ("```lean4\n", "```lean\n"):
        idx = text.find(fence)
        if idx >= 0:
            return text[idx + len(fence) :].strip()
                                                                                   
    m = re.search(
        r"(?m)^\s*(?:import|set_option|open|abbrev|opaque|axiom|instance|"
        r"(?:noncomputable\s+)?(?:theorem|lemma|def|example))\b",
        text,
    )
    if m:
        return text[m.start() :].strip()
    return ""


def _replace_statement_in_proof(statement: str, proof: str) -> str:
    """Merge model-generated proof into the given statement.

    This mirrors Goedel's `replace_statement_in_proof` behavior but avoids importing
    the full Goedel codebase (and its extra deps).
    """
    st = str(statement or "")
    pr = str(proof or "")
    if not st.strip() or not pr.strip():
        return ""

                                                                               
                                       
    marker_re = re.compile(r":=\s*by\b", flags=re.IGNORECASE)
    m0 = marker_re.search(st)
    if not m0:
        return ""
    head = st[: m0.end()].rstrip()

    body = pr.strip()
                                                                                     
    m1 = marker_re.search(body)
    if m1:
        body = body[m1.end() :]
    else:
                                                         
        m_by = re.match(r"^\s*by\b", body)
        if m_by:
            body = body[m_by.end() :]

    body_lines = body.splitlines()
                                                                               
                                                                            
                                                                    
    prefix: Optional[str] = None
    for ln in body_lines:
        if not ln.strip():
            continue
        m = re.match(r"^(\s+)", ln)
        prefix = m.group(1) if m else ""
        break
    if prefix:
        body_lines = [ln[len(prefix) :] if ln.startswith(prefix) else ln for ln in body_lines]

    merged = head + "\n" + "\n".join(body_lines).lstrip("\n")
    return merged.strip() + "\n"


def _load_resume_records(
    *,
    output_dir: Path,
    n: int,
    expected_n_by_origin: Optional[Dict[str, int]] = None,
    resume_retry_infra: bool = False,
    infra_error_patterns: Optional[List[re.Pattern[str]]] = None,
    max_infra_retries: int = 0,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    by_origin = _load_existing_by_origin(output_dir)

    completed: Dict[str, List[Dict[str, Any]]] = {}
    completed_origins: List[str] = []
    for origin, by_g in by_origin.items():
        exp_n = int(n)
        if expected_n_by_origin is not None:
            exp_n = int(expected_n_by_origin.get(origin, exp_n))
        if exp_n <= 0:
            continue
        if not all(i in by_g for i in range(int(exp_n))):
            continue

                                                                                                   
                                                                                                  
                                                                    
        if bool(resume_retry_infra):
            pats = infra_error_patterns or []
                                                                             
            max_prev_retry = max((_record_infra_retry_count(by_g[i]) for i in range(int(exp_n))), default=0)
            if int(max_infra_retries) > 0 and int(max_prev_retry) >= int(max_infra_retries):
                completed[origin] = [by_g[i] for i in range(int(exp_n))]
                completed_origins.append(origin)
                continue
            if any(_is_retryable_infra_failure(by_g[i], infra_error_patterns=pats) for i in range(int(exp_n))):
                continue

        completed[origin] = [by_g[i] for i in range(int(exp_n))]
        completed_origins.append(origin)
    return completed, completed_origins


@dataclass(frozen=True)
class Item:
    origin_problem_id: str
    lean4_code: str
    n: int
    input_item: Dict[str, Any]


def _load_items(items: Iterable[Dict[str, Any]], *, default_n: int) -> List[Item]:
    out: List[Item] = []
    for obj in items:
        origin_id = obj.get("origin_problem_id", obj.get("problem_id", obj.get("name")))
        origin_id = str(origin_id or "").strip()
        if not origin_id:
            continue
        lean4_code = str(obj.get("lean4_code") or "").strip()
        if not lean4_code:
            continue
        lean4_code = _force_by_sorry(lean4_code)
        n_raw = obj.get("n", default_n)
        try:
            n_i = int(n_raw)
        except Exception:
            n_i = int(default_n)
        if n_i <= 0:
                                                            
            continue
        out.append(Item(origin_problem_id=origin_id, lean4_code=lean4_code, n=int(n_i), input_item=dict(obj)))
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


async def _run_all(
    *,
    loaded: List[Item],
    output_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    inference_handler: str,
    default_n: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    output_mode: str,
    workers: int,
    max_retries: int,
    retry_backoff_s: float,
    save_every: int,
    existing_by_origin: Dict[str, Dict[int, Dict[str, Any]]],
    resume_targets: Dict[str, List[int]],
    infra_error_patterns: List[re.Pattern[str]],
    max_request_n: int,
) -> None:
    total = len(loaded)
    sem = asyncio.Semaphore(max(1, int(workers)))
    t_start = time.time()

    records_by_origin: Dict[str, Dict[int, Dict[str, Any]]] = dict(existing_by_origin)

    limits = httpx.Limits(
        max_connections=max(8, int(workers) * 2),
        max_keepalive_connections=max(8, int(workers) * 2),
    )
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        clamp_warned = False

        async def _process_one(it: Item) -> Tuple[str, Dict[int, Dict[str, Any]]]:
            async with sem:
                messages = _build_prompt(handler=str(inference_handler), lean4_code=it.lean4_code)
                n_i = int(it.n) if int(it.n) > 0 else int(default_n)

                targets = list(resume_targets.get(it.origin_problem_id) or [])
                if not targets:
                    targets = [g for g in range(int(n_i)) if g not in (records_by_origin.get(it.origin_problem_id) or {})]
                targets = [int(g) for g in targets if 0 <= int(g) < int(n_i)]
                if not targets:
                    return it.origin_problem_id, records_by_origin.get(it.origin_problem_id, {})

                                      
                                                                                                    
                                                                                                 
                                                                                               
                                                                                                  
                                               
                chunk_n = int(max_request_n)
                if chunk_n <= 0:
                    chunk_n = len(targets)

                merged: Dict[int, Dict[str, Any]] = dict(records_by_origin.get(it.origin_problem_id, {}))

                for off in range(0, len(targets), max(1, chunk_n)):
                    sub_targets = targets[off : off + max(1, chunk_n)]
                    request_n = len(sub_targets)
                    raws: List[str] = []
                    err = ""
                    t0 = time.time()
                                                                                                       
                                                                                                     
                                                         
                    effective_max_tokens = min(int(max_tokens), 8000)
                    nonlocal clamp_warned
                    if not clamp_warned and int(effective_max_tokens) != int(max_tokens):
                        _warn(f"clamp: requested max_tokens={max_tokens} -> {effective_max_tokens} (safety cap)")
                        clamp_warned = True

                    if request_n > 1:
                        _warn(f"request: origin={it.origin_problem_id} request_n={request_n} offset={off} total_targets={len(targets)}")

                    for attempt in range(int(max_retries) + 1):
                        try:
                                                   
                                                                                                   
                                                                                                    
                            ctx_reductions = 0
                            while True:
                                try:
                                    raws = await _openai_chat(
                                        client=client,
                                        base_url=base_url,
                                        api_key=str(api_key),
                                        model=str(model),
                                        messages=messages,
                                        n=int(request_n),
                                        temperature=float(temperature),
                                        max_tokens=int(effective_max_tokens),
                                        timeout_s=float(timeout_s),
                                    )
                                    break
                                except Exception as e:
                                    msg = f"{type(e).__name__}: {e}"
                                    if (
                                        ctx_reductions < 3
                                        and _looks_like_context_length_error(msg)
                                        and int(effective_max_tokens) > 256
                                    ):
                                        new_mt = _compute_safe_max_tokens_from_error(
                                            msg, requested_max_tokens=int(effective_max_tokens)
                                        )
                                        if new_mt is not None and int(new_mt) < int(effective_max_tokens):
                                            _warn(
                                                f"context_guard: origin={it.origin_problem_id} "
                                                f"request_n={request_n} max_tokens {effective_max_tokens} -> {new_mt}"
                                            )
                                            effective_max_tokens = int(new_mt)
                                            ctx_reductions += 1
                                            continue
                                    raise
                            err = ""
                            break
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"
                            raws = []
                            if attempt < int(max_retries):
                                await asyncio.sleep(float(retry_backoff_s) * (2**attempt))

                    if not raws:
                        raws = [""] * int(request_n)
                    if len(raws) < int(request_n):
                        raws = list(raws) + [""] * (int(request_n) - len(raws))

                    latency = float(time.time() - t0)
                    is_infra_err = bool(err) and any(p.search(str(err)) for p in infra_error_patterns)

                    for j, raw in enumerate(raws[: int(request_n)]):
                        g_idx = int(sub_targets[int(j)])
                        extracted = _extract_code_from_fence(raw)
                        full_code = "None"
                        if extracted:
                            if str(output_mode) == "full":
                                full_code = extracted.strip() + "\n"
                            else:
                                replaced = _replace_statement_in_proof(it.lean4_code, extracted)
                                if replaced and not str(replaced).startswith("**Error**"):
                                    full_code = str(replaced)

                        prev = merged.get(int(g_idx)) or {}
                        prev_retry = _effective_infra_retry_count(prev, infra_error_patterns=infra_error_patterns)
                        infra_retry = prev_retry + 1 if is_infra_err else prev_retry
                        merged[int(g_idx)] = {
                            "problem_id": f"{it.origin_problem_id}_g{g_idx}",
                            "origin_problem_id": it.origin_problem_id,
                            "lean4_code": it.lean4_code,
                            "model_input": messages[-1]["content"] if messages else "",
                            "messages_history_list": messages,
                            "inference_handler": str(inference_handler),
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                            "max_tokens_effective": int(effective_max_tokens),
                            "latency_s": latency,
                            "error": err,
                            "model_output": raw,
                            "full_code": full_code,
                            "infra_retry": int(infra_retry),
                            "input_item": it.input_item,
                        }
                return it.origin_problem_id, merged

        pending: List[Item] = []
        for it in loaded:
            n_i = int(it.n) if int(it.n) > 0 else int(default_n)
            by_g = records_by_origin.get(it.origin_problem_id, {})
            targets = list(resume_targets.get(it.origin_problem_id) or [])
            if not targets:
                targets = [g for g in range(int(n_i)) if g not in by_g]
            if targets:
                pending.append(it)
        if not pending:
            _warn("resume: nothing to do (all items already completed)")
            return

        tasks = [asyncio.create_task(_process_one(it)) for it in pending]
        done = sum(1 for it in loaded if it.origin_problem_id in records_by_origin and len(records_by_origin[it.origin_problem_id]) >= max(1, int(it.n)))
        for fut in asyncio.as_completed(tasks):
            origin, recs = await fut
            records_by_origin[origin] = recs
            done += 1
            _warn(f"progress: {done}/{total} items elapsed_s={time.time()-t_start:.1f}")

            if int(save_every) > 0 and (done % int(save_every) == 0 or done == total):
                ordered: List[Dict[str, Any]] = []
                for it in loaded:
                    by_g = records_by_origin.get(it.origin_problem_id) or {}
                    n_i = int(it.n) if int(it.n) > 0 else int(default_n)
                    for g in range(int(n_i)):
                        r = by_g.get(int(g))
                        if r:
                            ordered.append(r)
                txt = json.dumps(ordered, indent=2, ensure_ascii=False)
                _atomic_write(output_dir / "to_inference_codes.json", txt)
                _atomic_write(output_dir / "full_records.json", txt)
                _warn(f"checkpoint: wrote {len(ordered)} records -> {output_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Goedel-Prover-V2 inference via an OpenAI-compatible remote server (vLLM)."
    )
    ap.add_argument("--input_jsonl", required=True, help="Input problems JSONL (same format as Goedel inference).")
    ap.add_argument("--output_dir", required=True, help="Output directory (writes to_inference_codes.json).")
    ap.add_argument("--openai_base_url", required=True, help="OpenAI-compatible base URL (host:port[/v1]).")
    ap.add_argument("--model", required=True, help="Model name/id (must appear in /v1/models).")
    ap.add_argument(
        "--inference_handler",
        default="dpskcot",
        choices=["dpskcot", "dpsknoncot", "kiminacot"],
        help="Prompting style (default: dpskcot).",
    )
    ap.add_argument(
        "--output_mode",
        default="merge",
        choices=["merge", "full"],
        help=(
            "How to construct `full_code` for compilation: "
            "- merge (default): merge the model proof body into the input statement (robust for theorem-proving). "
            "- full: use the model's extracted Lean code block as-is (needed for CombiBench Fine-Eval to fill answers)."
        ),
    )
    ap.add_argument("--n", type=int, default=1, help="Number of attempts per input item (default: 1).")
    ap.add_argument("--temperature", type=float, default=1.0)
                                                                                                            
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--timeout_s", type=float, default=600.0)
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "").strip())
    ap.add_argument("--max_problems", type=int, default=0, help="If >0, only run the first N input items.")
    ap.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Write intermediate to_inference_codes.json every N items (default: 1; safer for long runs).",
    )
    ap.add_argument("--workers", type=int, default=1, help="Max concurrent remote requests over problems (default: 1).")
    ap.add_argument("--resume", action="store_true", help="Resume if output_dir already has to_inference_codes.json")
    ap.add_argument("--max_retries", type=int, default=1, help="Retries per problem on failure/timeout (default: 1).")
    ap.add_argument(
        "--retry_backoff_s",
        type=float,
        default=2.0,
        help="Base backoff seconds between retries (default: 2.0; exponential).",
    )
    ap.add_argument(
        "--resume_retry_infra",
        action="store_true",
        help=(
            "When resuming, treat infra failures (e.g., TimeoutError / 5xx) as NOT completed and rerun them. "
            "This fills empty attempt slots without increasing the intended attempt budget."
        ),
    )
    ap.add_argument(
        "--infra_error_regex",
        action="append",
        default=[],
        help=(
            "Regex pattern (repeatable) for classifying an `error` string as retryable infra failure. "
            "Defaults to a conservative built-in list if omitted."
        ),
    )
    ap.add_argument(
        "--max_infra_retries",
        type=int,
        default=5,
        help="Max infra re-runs per problem when --resume_retry_infra is set (default: 5).",
    )
    ap.add_argument(
        "--resume_retry_max_tokens_mismatch",
        action="store_true",
        help="When resuming, rerun slots whose stored record.max_tokens differs from current --max_tokens.",
    )
    ap.add_argument(
        "--max_request_n",
        type=int,
        default=8,
        help="Cap OpenAI `n` per request by chunking missing slots (default: 8; 0 disables). Helps avoid vLLM KV cache overload.",
    )
    args = ap.parse_args()

    base_url = _normalize_base_url(str(args.openai_base_url))
    if not base_url:
        raise SystemExit("Invalid --openai_base_url")

    input_path = Path(str(args.input_jsonl)).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list(_read_jsonl(input_path))
    if int(args.max_problems) > 0:
        items = items[: int(args.max_problems)]
    loaded = _load_items(items, default_n=int(args.n))
    if not loaded:
        _warn("No items to run (empty input?)")
        return 2

    patterns_raw: List[str] = list(args.infra_error_regex or [])
    if not patterns_raw:
        patterns_raw = list(_DEFAULT_INFRA_ERROR_PATTERNS)
    infra_pats: List[re.Pattern[str]] = [re.compile(p) for p in patterns_raw]

    existing_by_origin: Dict[str, Dict[int, Dict[str, Any]]] = {}
    resume_targets: Dict[str, List[int]] = {}
    if bool(args.resume):
        existing_by_origin = _load_existing_by_origin(output_dir)
        resume_targets = _compute_resume_targets(
            loaded=loaded,
            existing_by_origin=existing_by_origin,
            resume_retry_infra=bool(args.resume_retry_infra),
            infra_error_patterns=infra_pats,
            max_infra_retries=int(args.max_infra_retries),
            resume_retry_max_tokens_mismatch=bool(args.resume_retry_max_tokens_mismatch),
            expected_max_tokens=int(args.max_tokens),
        )
        total_slots = sum(len(v) for v in resume_targets.values())
        _warn(
            f"resume: loaded {len(existing_by_origin)} origins; "
            f"will run {len(resume_targets)} origins ({total_slots} slots)"
        )

    asyncio.run(
        _run_all(
            loaded=loaded,
            output_dir=output_dir,
            base_url=base_url,
            api_key=str(args.api_key),
            model=str(args.model),
            inference_handler=str(args.inference_handler),
            default_n=int(args.n),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_s=float(args.timeout_s),
            output_mode=str(args.output_mode),
            workers=int(args.workers),
            max_retries=int(args.max_retries),
            retry_backoff_s=float(args.retry_backoff_s),
            save_every=int(args.save_every),
            existing_by_origin=existing_by_origin,
            resume_targets=resume_targets,
            infra_error_patterns=infra_pats,
            max_request_n=int(args.max_request_n),
        )
    )

    print(f"Wrote: {output_dir / 'to_inference_codes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
