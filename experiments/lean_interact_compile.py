#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def _msg_to_dict(m: Any) -> Dict[str, Any]:
    # lean_interact.interface.Message is a pydantic model with model_dump().
    try:
        raw = m.model_dump()  # type: ignore[attr-defined]
    except Exception:
        raw = {}
    start_pos = raw.get("start_pos") or {}
    end_pos = raw.get("end_pos") or {}
    return {
        "severity": raw.get("severity"),
        "pos": {"line": start_pos.get("line"), "column": start_pos.get("column")},
        "endPos": {"line": end_pos.get("line"), "column": end_pos.get("column")},
        "data": raw.get("data"),
    }


def _partition_messages(messages: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    for m in messages:
        d = _msg_to_dict(m)
        sev = str(d.get("severity") or "").lower()
        if sev == "error":
            errors.append(d)
        elif sev == "warning":
            warnings.append(d)
        else:
            infos.append(d)
    return errors, warnings, infos


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    # This repo standardizes proving compilation on an HTTP Lean server to avoid
    # local `lean_interact` OOMs and environment drift.
    if os.environ.get("AUTOFORMAL_ALLOW_LEAN_INTERACT", "").strip().lower() not in {"1", "true", "yes"}:
        _warn(
            "lean_interact_compile.py is disabled by default.\n"
            "Use experiments/lean_server_compile.py (HTTP Lean server) instead, or set "
            "AUTOFORMAL_ALLOW_LEAN_INTERACT=1 to override."
        )
        return 2

    ap = argparse.ArgumentParser(
        description=(
            "Compile Goedel-Prover-V2 inference outputs with lean_interact (Mathlib).\n"
            "Input: Goedel to_inference_codes.json (list of dicts with problem_id/full_code).\n"
            "Output: code_compilation_repl.json compatible with Goedel summarize scripts.\n"
        )
    )
    ap.add_argument("--input_path", required=True, help="Path to to_inference_codes.json")
    ap.add_argument("--output_path", required=True, help="Path to write code_compilation_repl.json")
    ap.add_argument("--timeout_s", type=int, default=600, help="Per-proof timeout seconds (default: 600)")
    ap.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Write intermediate code_compilation_repl.json every N new records (default: 0=only at end).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output_path by skipping already-compiled names.",
    )
    args = ap.parse_args()

    input_path = Path(str(args.input_path)).expanduser().resolve()
    if not input_path.exists():
        _warn(f"input not found: {input_path}")
        return 2
    output_path = Path(str(args.output_path)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure we can import `autoformalization.*` from the FormalEvol repo root.
    # experiments/xxx.py -> autoformalization_v1 -> examples -> FormalEvol
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    try:
        from autoformalization.lean_env import get_lean_server  # type: ignore
        from lean_interact import Command  # type: ignore
        from lean_interact.interface import CommandResponse, LeanError  # type: ignore
    except Exception as e:
        _warn(f"import failed: {type(e).__name__}: {e}")
        return 2

    try:
        server = get_lean_server()
    except Exception as e:
        _warn(f"lean server init failed: {type(e).__name__}: {e}")
        return 2

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        _warn(f"expected a JSON list: {input_path}")
        return 2

    results_by_name: Dict[str, Dict[str, Any]] = {}
    if bool(args.resume) and output_path.exists():
        try:
            prev = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(prev, list):
                for r in prev:
                    if not isinstance(r, dict):
                        continue
                    nm = str(r.get("name") or r.get("problem_id") or "").strip()
                    if nm:
                        results_by_name[nm] = r
                _warn(f"resume: loaded {len(results_by_name)} records from {output_path}")
            else:
                _warn(f"resume: output_path exists but is not a JSON list: {output_path}")
        except Exception as e:
            _warn(f"resume: failed to load {output_path}: {type(e).__name__}: {e}")

    num_pass = 0
    num_complete = 0
    t0 = time.time()
    _warn(
        f"start: inputs={len(raw)} resume={bool(args.resume)} save_every={int(args.save_every)} "
        f"timeout_s={int(args.timeout_s)}"
    )

    def _recompute_counts() -> Tuple[int, int]:
        p = 0
        c = 0
        for r in results_by_name.values():
            comp = r.get("compilation_result") or {}
            if isinstance(comp, dict) and comp.get("pass"):
                p += 1
            if isinstance(comp, dict) and comp.get("complete"):
                c += 1
        return p, c

    num_pass, num_complete = _recompute_counts()

    def _checkpoint() -> None:
        ordered: List[Dict[str, Any]] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            nm = str(r.get("name") or r.get("problem_id") or "").strip()
            if not nm:
                continue
            rec = results_by_name.get(nm)
            if rec is not None:
                ordered.append(rec)
        _atomic_write_json(output_path, ordered)

    processed_new = 0
    skipped_existing = 0
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue

        name = str(r.get("name") or r.get("problem_id") or "").strip()
        if not name:
            continue
        if name in results_by_name:
            skipped_existing += 1
            continue

        full_code = r.get("full_code")
        if full_code is None:
            full_code = ""
        code_str = str(full_code)
        if code_str.strip() in ("", "None"):
            parsed_result = {
                "sorries": [],
                "tactics": [],
                "errors": [
                    {
                        "severity": "error",
                        "pos": {"line": None, "column": None},
                        "endPos": {"line": None, "column": None},
                        "data": "Empty/None code",
                    }
                ],
                "warnings": [],
                "infos": [],
                "ast": {},
                "system_errors": None,
                "pass": False,
                "complete": False,
            }
            results_by_name[name] = {"name": name, "problem_id": name, "code": code_str, "compilation_result": parsed_result}
            processed_new += 1
            continue

        try:
            res = server.run(Command(cmd=code_str), timeout=int(args.timeout_s))
        except Exception as e:
            parsed_result = {
                "pass": False,
                "complete": False,
                "system_errors": f"RUN ERROR: {type(e).__name__}: {e}",
            }
            results_by_name[name] = {"name": name, "problem_id": name, "code": code_str, "compilation_result": parsed_result}
            processed_new += 1
            continue

        if isinstance(res, LeanError):
            parsed_result = {
                "sorries": [],
                "tactics": [],
                "errors": [
                    {
                        "severity": "error",
                        "pos": {"line": None, "column": None},
                        "endPos": {"line": None, "column": None},
                        "data": str(res),
                    }
                ],
                "warnings": [],
                "infos": [],
                "ast": {},
                "system_errors": str(res),
                "pass": False,
                "complete": False,
            }
            results_by_name[name] = {"name": name, "problem_id": name, "code": code_str, "compilation_result": parsed_result}
            processed_new += 1
            continue

        if not isinstance(res, CommandResponse):
            parsed_result = {
                "pass": False,
                "complete": False,
                "system_errors": f"Unknown response type: {type(res).__name__}",
            }
            results_by_name[name] = {"name": name, "problem_id": name, "code": code_str, "compilation_result": parsed_result}
            processed_new += 1
            continue

        errors, warnings, infos = _partition_messages(list(res.messages or []))
        sorries = [w for w in warnings if "declaration uses 'sorry'" in str(w.get("data") or "")]

        pass_ok = bool(res.lean_code_is_valid()) and not errors
        complete_ok = bool(
            pass_ok
            and not sorries
            and not any("declaration uses 'sorry'" in str(w.get("data") or "") or "failed" in str(w.get("data") or "") for w in warnings)
        )
        if pass_ok:
            num_pass += 1
        if complete_ok:
            num_complete += 1

        parsed_result = {
            "sorries": sorries,
            "tactics": [],
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "ast": {},
            "system_errors": None,
            "pass": pass_ok,
            "complete": complete_ok,
        }
        results_by_name[name] = {"name": name, "problem_id": name, "code": code_str, "compilation_result": parsed_result}
        processed_new += 1

        if (i + 1) % 10 == 0:
            _warn(
                f"progress: {i+1}/{len(raw)} processed_new={processed_new} skipped={skipped_existing} "
                f"pass={num_pass} complete={num_complete} elapsed_s={time.time()-t0:.1f}"
            )

        save_every = int(args.save_every)
        if save_every > 0 and processed_new > 0 and (processed_new % save_every == 0):
            _checkpoint()
            _warn(f"checkpoint: wrote {len(results_by_name)} total records -> {output_path}")

    _checkpoint()
    num_pass, num_complete = _recompute_counts()
    _warn(f"wrote: {output_path}")
    _warn(
        f"summary: total={len(results_by_name)} processed_new={processed_new} skipped={skipped_existing} "
        f"pass={num_pass} complete={num_complete} elapsed_s={time.time()-t0:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
