                      
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ProblemItem:
    index: int
    problem_name: str
    full_name: str
    informal: str
    header: str
    ground_truth: str


def _read_jsonl_first_n(path: Path, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_benchmark(dataset_path: Path, n: int) -> List[ProblemItem]:
    rows = _read_jsonl_first_n(dataset_path, n)
    problems: List[ProblemItem] = []
    for i, obj in enumerate(rows):
        problem_name = str(obj.get("problem_name") or f"problem_{i:04d}").strip()
        full_name = str(obj.get("full_name") or problem_name).strip()
        informal = str(obj.get("informal_stmt") or "").strip()
        header = str(obj.get("header") or "import Mathlib").strip()
        ground_truth = str(obj.get("formal_stmt") or "").strip()
        problems.append(
            ProblemItem(
                index=i,
                problem_name=problem_name,
                full_name=full_name,
                informal=informal,
                header=header,
                ground_truth=ground_truth,
            )
        )
    return problems


def _dataset_path(cwd: Path, dataset: str, dataset_path_override: str) -> Path:
    if dataset_path_override:
        p = Path(dataset_path_override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--dataset_path not found: {p}")
        return p

    bench = cwd / "benchmark"
    if dataset == "proofnet_test":
        p = bench / "proofnet_lean4.15.0_test.jsonl"
    elif dataset == "proofnet_full":
        p = bench / "proofnet_lean4.15.0.jsonl"
    elif dataset == "combibench":
        p = bench / "combibench_lean4.15.0.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not p.exists():
        raise FileNotFoundError(f"Bundled dataset not found: {p}")
    return p


_ABS_PATH_RE = re.compile(r"(?P<p>/(?:home|Users)/[^\\s:'\"`]+)")


def _redact_paths(s: str) -> str:
    if not s:
        return ""
    return _ABS_PATH_RE.sub("[REDACTED_PATH]", s)


def _safe_problem_dirname(problem: ProblemItem) -> str:
    raw = f"{problem.index:04d}_{problem.problem_name}"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw)


def _iter_chunks(seq: List[ProblemItem], size: int) -> Iterable[List[ProblemItem]]:
    if size <= 0:
        raise ValueError(f"chunk size must be >0, got {size}")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _write_problem_config(out_dir: Path, problem: ProblemItem, *, compile_timeout: int, lean_server_url: str) -> Path:
    cfg_path = out_dir / "problem_config.json"
    cfg = {
        "informal": problem.informal,
        "header": problem.header,
        "ground_truth": problem.ground_truth,
        "use_beq": False,
        "use_semantic": False,
        "use_cycle_consistency": False,
        "compile_timeout": int(compile_timeout),
        "lean_server_url": str(lean_server_url),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg_path


def _write_gt_file(out_dir: Path, problem: ProblemItem) -> Path:
    p = out_dir / "gt.lean"
    code = f"{problem.header.strip()}\n\n{problem.ground_truth.strip()}\n"
    p.write_text(code, encoding="utf-8")
    return p


def _run_evaluate(evaluate_py: Path, *, program_path: Path, results_dir: Path, env: Dict[str, str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            os.fspath(evaluate_py),
            "--program_path",
            os.fspath(program_path),
            "--results_dir",
            os.fspath(results_dir),
        ],
        cwd=str(evaluate_py.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        (results_dir / "evaluate_stdout.log").write_text(_redact_paths(proc.stdout or ""), encoding="utf-8")
        (results_dir / "evaluate_stderr.log").write_text(_redact_paths(proc.stderr or ""), encoding="utf-8")
        raise RuntimeError(f"evaluate.py failed (returncode={proc.returncode}) for {program_path}")


def _load_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit compile_ok for dataset-provided ground-truth statements.")
    ap.add_argument("--dataset", type=str, default="proofnet_test", choices=["proofnet_test", "proofnet_full", "combibench"])
    ap.add_argument("--dataset_path", type=str, default="", help="Optional override JSONL path (otherwise uses bundled benchmark/*)")
    ap.add_argument("--num_problems", type=int, default=50, help="How many problems (from the start of JSONL)")
    ap.add_argument("--out_root", type=str, default="", help="Output root. Default: /tmp/formalevolve_gt_compile_audit_*")
    ap.add_argument("--compile_timeout", type=int, default=60)
    ap.add_argument("--lean_server_url", type=str, default="local")
    ap.add_argument("--batch_size", type=int, default=10, help="Write outputs in batches (progress-friendly)")
    args = ap.parse_args(argv)

    cwd = Path(__file__).resolve().parents[1]
    evaluate_py = cwd / "evaluate.py"
    if not evaluate_py.exists():
        raise FileNotFoundError(f"evaluate.py not found at: {evaluate_py}")

    dataset_path = _dataset_path(cwd, str(args.dataset), str(args.dataset_path))
    problems = _load_benchmark(dataset_path, int(args.num_problems))
    if not problems:
        raise RuntimeError("No problems loaded (empty dataset slice?)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else Path(f"/tmp/formalevolve_gt_compile_audit_{args.dataset}_n{len(problems)}_{ts}")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["AUTOFORMAL_DISABLE_SEMANTIC"] = "1"
    env["CYCLE_API_BASE_URL"] = ""

    ok = 0
    total = 0
    rows: List[Dict[str, Any]] = []

    for chunk in _iter_chunks(problems, int(args.batch_size)):
        for p in chunk:
            total += 1
            prob_dir = out_root / _safe_problem_dirname(p)
            prob_dir.mkdir(parents=True, exist_ok=True)
            _write_problem_config(
                prob_dir,
                p,
                compile_timeout=int(args.compile_timeout),
                lean_server_url=str(args.lean_server_url),
            )
            gt_path = _write_gt_file(prob_dir, p)
            results_dir = prob_dir / "results"
            try:
                _run_evaluate(evaluate_py, program_path=gt_path, results_dir=results_dir, env=env)
            except Exception as e:
                metrics = _load_metrics(results_dir / "metrics.json")
                compile_ok = int(metrics.get("compile_ok", 0) or 0)
                rows.append(
                    {
                        "index": int(p.index),
                        "problem_name": p.problem_name,
                        "full_name": p.full_name,
                        "compile_ok": compile_ok,
                        "compile_error_type": str(metrics.get("compile_error_type", "") or "evaluator_failed"),
                        "compile_error_msg": _redact_paths(str(metrics.get("compile_error_msg", "") or str(e))),
                    }
                )
                continue

            metrics = _load_metrics(results_dir / "metrics.json")
            compile_ok = int(metrics.get("compile_ok", 0) or 0)
            if compile_ok == 1:
                ok += 1
            rows.append(
                {
                    "index": int(p.index),
                    "problem_name": p.problem_name,
                    "full_name": p.full_name,
                    "compile_ok": compile_ok,
                    "compile_error_type": str(metrics.get("compile_error_type", "") or ""),
                    "compile_error_msg": _redact_paths(str(metrics.get("compile_error_msg", "") or "")),
                }
            )

        (out_root / "compile_audit.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        (out_root / "summary.json").write_text(
            json.dumps(
                {
                    "created_at": ts,
                    "dataset": str(args.dataset),
                    "num_problems": len(problems),
                    "compile_ok_count": int(ok),
                    "compile_ok_ratio": (float(ok) / float(total)) if total else 0.0,
                    "compile_timeout": int(args.compile_timeout),
                    "lean_server_url": str(args.lean_server_url),
                    "note": "Local artifact; do not commit to an anonymous repo.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[progress] {total}/{len(problems)} compile_ok={ok}")

    print(f"[DONE] out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

