                      
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_code_fence(text: str, lang: str = "lean") -> str:
    if not text:
        return ""
    fence = f"```{lang}"
    if fence in text:
        _, tail = text.split(fence, 1)
        if "```" in tail:
            body, _ = tail.split("```", 1)
            return body.strip()
                               
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return text.strip()


_THEOREM_START_RE = re.compile(r"(?m)^\s*(?:noncomputable\s+)?theorem\b")


def _theorem_block(code: str) -> str:
    m = _THEOREM_START_RE.search(code or "")
    if not m:
        return ""
    return (code[m.start() :] or "").strip()


def _theorem_has_sorry(code: str) -> bool:
    blk = _theorem_block(code)
    if not blk:
        return True                                    
    return "sorry" in blk


def _read_program_rows(run_dir: Path) -> List[Tuple[str, float, Dict[str, Any]]]:
    db_path = run_dir / "evolution_db.sqlite"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT code, combined_score, public_metrics FROM programs")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []

    out: List[Tuple[str, float, Dict[str, Any]]] = []
    for code, score, pm_raw in rows:
        if pm_raw is None:
            continue
        try:
            pm = json.loads(pm_raw) if isinstance(pm_raw, str) else dict(pm_raw)
        except Exception:
            continue
        try:
            s = float(score) if score is not None else 0.0
        except Exception:
            s = 0.0
        out.append((str(code or ""), s, pm if isinstance(pm, dict) else {}))
    return out


def _is_ok(pm: Dict[str, Any], key: str) -> bool:
    try:
        return int(pm.get(key, 0) or 0) == 1
    except Exception:
        return False


@dataclass(frozen=True)
class Candidate:
    code: str
    combined_score: float
    compile_ok: int
    semantic_ok: int


def select_candidates(
    run_dir: Path,
    *,
    k: int,
    prefer_semantic_ok: bool,
) -> List[Candidate]:
    rows = _read_program_rows(run_dir)
    cands: List[Candidate] = []
    for code, score, pm in rows:
        c_ok = 1 if _is_ok(pm, "compile_ok") else 0
        s_ok = 1 if _is_ok(pm, "semantic_ok") else 0
        if c_ok != 1:
            continue
        cands.append(
            Candidate(
                code=str(code),
                combined_score=float(score),
                compile_ok=int(c_ok),
                semantic_ok=int(s_ok),
            )
        )

    if prefer_semantic_ok:
        sem = [c for c in cands if c.semantic_ok == 1]
        if sem:
            cands = sem

    cands.sort(key=lambda c: (c.combined_score, c.semantic_ok, c.compile_ok), reverse=True)
    return cands[: max(0, int(k))]


def openai_chat(
    *,
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    api_key: str = "",
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        timeout=float(timeout_s),
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"] or "")
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run a downstream prover on top-k candidates from an autoformalization run.\n"
            "Success criterion (default): Lean compiles AND theorem block contains no `sorry`.\n"
            "\n"
            "NOTE: CombiBench headers may contain `:= sorry` for solution abbreviations; this script\n"
            "allows `sorry` in the header but forbids it inside the theorem declaration."
        )
    )
    ap.add_argument("--run_root", required=True, help="Run root directory (contains manifest.json).")
    ap.add_argument(
        "--baseline",
        default="",
        help="Baseline/method name inside run_root (e.g., ours/strong_semantic). If empty, auto-detect when unique.",
    )
    ap.add_argument("--k", type=int, default=5, help="Candidates to try per problem (default 5).")
    ap.add_argument(
        "--prefer_semantic_ok",
        action="store_true",
        default=True,
        help="Prefer semantic_ok=1 candidates; fallback to compile_ok if none (default: on).",
    )
    ap.add_argument("--prover_url", type=str, default=os.environ.get("PROVER_URL", "").strip())
    ap.add_argument("--prover_model", type=str, default=os.environ.get("PROVER_MODEL", "").strip())
    ap.add_argument("--prover_api_key", type=str, default=os.environ.get("PROVER_API_KEY", "").strip())
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--timeout_s", type=float, default=180.0)
    ap.add_argument("--compile_timeout", type=int, default=600)
    ap.add_argument("--lean_server_url", type=str, default=os.environ.get("LEAN_SERVER_URL", "").strip())
    ap.add_argument("--write", type=str, default="", help="Write JSON report to this path.")
    args = ap.parse_args()

    run_root = Path(str(args.run_root)).expanduser().resolve()
    manifest = _read_json(run_root / "manifest.json") or {}
    problems = manifest.get("problems") or []
    if not isinstance(problems, list) or not problems:
        raise SystemExit(f"Invalid manifest problems: {run_root / 'manifest.json'}")

                          
    baselines = manifest.get("baselines")
    if isinstance(baselines, list) and baselines:
        baselines = [str(b) for b in baselines]
    else:
        runs_dir = run_root / "runs"
        baselines = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.is_dir() else []
    baseline = str(args.baseline or "").strip()
    if not baseline:
        if len(baselines) != 1:
            raise SystemExit(f"Need --baseline (run_root has {baselines})")
        baseline = baselines[0]
    if baseline not in baselines:
        raise SystemExit(f"Unknown baseline {baseline!r} (run_root has {baselines})")

    prover_url = str(args.prover_url).strip()
    prover_model = str(args.prover_model).strip()
    if not prover_url or not prover_model:
        raise SystemExit("Need --prover_url/--prover_model (or PROVER_URL/PROVER_MODEL env)")

                                            
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from evaluate import check_lean_compile, normalize_lean_code                
    except Exception as e:
        raise SystemExit(f"Failed to import evaluate.py helpers: {type(e).__name__}: {e}")

    report: Dict[str, Any] = {
        "run_root": str(run_root),
        "baseline": baseline,
        "k": int(args.k),
        "prefer_semantic_ok": bool(args.prefer_semantic_ok),
        "prover_url": prover_url,
        "prover_model": prover_model,
        "compile_timeout": int(args.compile_timeout),
        "problems": [],
        "summary": {},
    }

    solved = 0
    attempted = 0
    for p in problems:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("problem_id") or "").strip()
        pname = str(p.get("problem_name") or "").strip()
        runs = p.get("runs") or {}
        if not pid or not isinstance(runs, dict):
            continue
        run_dir_raw = runs.get(baseline)
        if not run_dir_raw:
            _warn(f"Missing run dir for problem_id={pid} baseline={baseline}")
            continue
        run_dir = Path(str(run_dir_raw)).expanduser().resolve()
        if not run_dir.exists():
            _warn(f"Run dir not found for problem_id={pid} baseline={baseline}: {run_dir}")
            continue

        cands = select_candidates(
            run_dir,
            k=int(args.k),
            prefer_semantic_ok=bool(args.prefer_semantic_ok),
        )
        attempted += 1

        prob_row: Dict[str, Any] = {
            "problem_id": pid,
            "problem_name": pname,
            "run_dir": str(run_dir),
            "num_candidates": len(cands),
            "success": False,
            "attempts": [],
        }

        for idx, cand in enumerate(cands):
            code0 = normalize_lean_code(cand.code or "")
            if not code0:
                continue

            sys_msg = (
                "You are an expert Lean 4 theorem prover.\n"
                "Given a Lean 4 file that contains exactly one theorem with `:= by sorry`, "
                "replace the proof with a complete proof that compiles.\n"
                "Do NOT change the statement (type/binders) of the theorem.\n"
                "Do NOT introduce any new `sorry` inside the theorem.\n"
                "Return a complete Lean 4 file inside a ```lean code fence.\n"
            )
            user_msg = (
                "Prove the theorem in the following Lean 4 file.\n\n"
                "```lean\n"
                f"{code0}\n"
                "```\n"
            )

            t0 = time.time()
            ok = False
            err = ""
            out_code = ""
            try:
                raw = openai_chat(
                    url=prover_url,
                    model=prover_model,
                    api_key=str(args.prover_api_key),
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    timeout_s=float(args.timeout_s),
                )
                out_code = normalize_lean_code(_extract_code_fence(raw, lang="lean"))
                if not out_code:
                    raise RuntimeError("empty prover output")
                if _theorem_has_sorry(out_code):
                    raise RuntimeError("theorem block still contains `sorry`")
                lean_ok, lean_err = check_lean_compile(
                    out_code,
                    timeout=int(args.compile_timeout),
                    lean_server_url=str(args.lean_server_url) if str(args.lean_server_url) else None,
                )
                if not lean_ok:
                    raise RuntimeError(f"Lean compile failed: {lean_err}")
                ok = True
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"

            prob_row["attempts"].append(
                {
                    "rank": idx,
                    "combined_score": float(cand.combined_score),
                    "compile_ok": int(cand.compile_ok),
                    "semantic_ok": int(cand.semantic_ok),
                    "elapsed_s": round(time.time() - t0, 3),
                    "ok": bool(ok),
                    "error": err,
                }
            )
            if ok:
                prob_row["success"] = True
                solved += 1
                break

        report["problems"].append(prob_row)

    report["summary"] = {
        "attempted_problems": attempted,
        "solved_problems": solved,
        "success_rate": (float(solved) / float(attempted)) if attempted > 0 else 0.0,
    }

    print(
        f"[ProverEval] solved {solved}/{attempted} "
        f"(rate={report['summary']['success_rate']:.3f}) "
        f"baseline={baseline} run_root={run_root}"
    )

    if str(args.write).strip():
        out_path = Path(str(args.write)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Write] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
