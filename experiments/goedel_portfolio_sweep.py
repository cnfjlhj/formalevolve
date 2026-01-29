                      
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.dont_write_bytecode = True


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr)


def _parse_int_list(spec: str) -> List[int]:
    raw = str(spec or "").strip()
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"Invalid int in list: {part!r}")
        out.append(int(p))
    return out


def _parse_field_list(spec: str) -> List[str]:
    raw = str(spec or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p not in ("complete", "pass"):
            raise ValueError(f"Invalid field: {p!r} (expected 'complete' or 'pass')")
        out.append(p)
    return out


def _run(cmd: List[str], *, cwd: Optional[Path] = None, dry_run: bool) -> int:
    s = " ".join(cmd)
    if dry_run:
        print(s)
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return int(proc.returncode)


@dataclass(frozen=True)
class SweepJob:
    k: int
    n_per_statement: int
    out_dir: Path
    dataset_jsonl: Path
    index_json: Path
    goedel_out_dir: Path


def _build_jobs(
    *,
    out_dir: Path,
    k_list: List[int],
    budget: int,
) -> List[SweepJob]:
    jobs: List[SweepJob] = []
    for k in k_list:
        if k <= 0:
            raise ValueError(f"Invalid k (must be >0): {k}")
        if budget <= 0:
            raise ValueError(f"Invalid budget (must be >0): {budget}")
        if budget % k != 0:
            raise ValueError(f"budget={budget} must be divisible by k={k} (to keep total attempts fixed)")
        n_per_statement = budget // k
        tag = f"k{k:02d}_n{n_per_statement:02d}_b{budget:02d}"
        job_dir = out_dir / tag
        jobs.append(
            SweepJob(
                k=int(k),
                n_per_statement=int(n_per_statement),
                out_dir=job_dir,
                dataset_jsonl=job_dir / "dataset.jsonl",
                index_json=job_dir / "selection_index.json",
                goedel_out_dir=job_dir / "goedel_out",
            )
        )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep Goedel proving under a fixed total proof-attempt budget per problem (portfolio evaluation).\n"
            "\n"
            "For each K in --k_list, we export top-K unique statements per problem, then run Goedel\n"
            "with --n = budget/K attempts per statement so that total attempts per problem is fixed.\n"
            "\n"
            "This implements your preferred selection strategy B:\n"
            "- Prefer semantic_ok=1 statements; fill remaining slots with compile_ok=1.\n"
        )
    )
    ap.add_argument("--run_root", required=True, help="Autoformal run root (contains manifest.json + runs/).")
    ap.add_argument(
        "--dataset_mode",
        default="selected",
        choices=["selected", "ground_truth"],
        help="Dataset mode: selected (from evolution_db.sqlite) or ground_truth (from inputs/*/problem_config.json).",
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="Baseline name under run_root/runs/. If empty and only one exists, auto-pick it.",
    )
    ap.add_argument(
        "--solution_mode",
        default=os.environ.get("GOEDEL_SOLUTION_MODE", "without_solution"),
        choices=["without_solution", "with_solution"],
        help="Ground-truth export mode: keep *_solution := sorry (without_solution) or fill GT solution (with_solution).",
    )
    ap.add_argument("--k_list", default="1,2,4,8,16", help="Comma-separated K values (default: 1,2,4,8,16).")
    ap.add_argument(
        "--budget",
        type=int,
        default=16,
        help="Total proof attempts per problem (default: 16). Requires budget %% K == 0.",
    )
    ap.add_argument(
        "--max_problems",
        type=int,
        default=0,
        help="If >0, only export the first N problems in manifest order (useful for smoke tests).",
    )
    ap.add_argument(
        "--problems_file",
        type=str,
        default="",
        help="Optional file with one problem_id per line (only export those).",
    )
    ap.add_argument(
        "--hf_dataset",
        type=str,
        default="",
        help=(
            "Optional HuggingFace dataset id to override ground-truth formal statements "
            "(e.g. AI-MO/CombiBench). Only applies to --dataset_mode ground_truth."
        ),
    )
    ap.add_argument(
        "--hf_split",
        type=str,
        default="",
        help=(
            "HF split name when using --hf_dataset (e.g. test or test_with_solution). "
            "Only applies to --dataset_mode ground_truth."
        ),
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (will create subdirs per K).",
    )

    ap.add_argument(
        "--goedel_root",
        default="proofmodel/Goedel-Prover-V2",
        help="Path to Goedel-Prover-V2 repo root (default: proofmodel/Goedel-Prover-V2).",
    )
    ap.add_argument(
        "--inference_backend",
        default=os.environ.get("GOEDEL_INFERENCE_BACKEND", "local"),
        choices=["local", "remote"],
        help="Inference backend: local (Goedel inference.py) or remote (OpenAI-compatible server).",
    )
    ap.add_argument(
        "--model_path",
        default=os.environ.get("GOEDEL_MODEL_PATH", "Goedel-LM/Goedel-Prover-V2-8B"),
        help="Goedel model path/name (env: GOEDEL_MODEL_PATH).",
    )
    ap.add_argument(
        "--openai_base_url",
        default=os.environ.get("GOEDEL_OPENAI_BASE_URL", ""),
        help="Remote OpenAI-compatible base URL for --inference_backend remote (env: GOEDEL_OPENAI_BASE_URL).",
    )
    ap.add_argument(
        "--remote_model",
        default=os.environ.get("GOEDEL_REMOTE_MODEL", ""),
        help="Remote model name/id for --inference_backend remote (env: GOEDEL_REMOTE_MODEL).",
    )
    ap.add_argument(
        "--remote_max_tokens",
        type=int,
        default=int(os.environ.get("GOEDEL_REMOTE_MAX_TOKENS", "8192")),
        help="Remote max_tokens (default: 8192; env: GOEDEL_REMOTE_MAX_TOKENS).",
    )
    ap.add_argument(
        "--remote_timeout_s",
        type=float,
        default=float(os.environ.get("GOEDEL_REMOTE_TIMEOUT_S", "600")),
        help="Remote request timeout seconds (default: 600; env: GOEDEL_REMOTE_TIMEOUT_S).",
    )
    ap.add_argument(
        "--remote_workers",
        type=int,
        default=int(os.environ.get("GOEDEL_REMOTE_WORKERS", "1")),
        help="Max concurrent remote requests over problems (default: 1; env: GOEDEL_REMOTE_WORKERS).",
    )
    ap.add_argument(
        "--remote_resume",
        action="store_true",
        help="Resume remote inference if output already exists (passes --resume to goedel_remote_inference.py).",
    )
    ap.add_argument(
        "--remote_max_retries",
        type=int,
        default=int(os.environ.get("GOEDEL_REMOTE_MAX_RETRIES", "1")),
        help="Retries per problem on remote failure/timeout (default: 1; env: GOEDEL_REMOTE_MAX_RETRIES).",
    )
    ap.add_argument(
        "--remote_retry_backoff_s",
        type=float,
        default=float(os.environ.get("GOEDEL_REMOTE_RETRY_BACKOFF_S", "2.0")),
        help="Base backoff seconds between remote retries (default: 2.0; env: GOEDEL_REMOTE_RETRY_BACKOFF_S).",
    )
    ap.add_argument(
        "--remote_output_mode",
        default=os.environ.get("GOEDEL_REMOTE_OUTPUT_MODE", "merge"),
        choices=["merge", "full"],
        help=(
            "Output mode passed to experiments/goedel_remote_inference.py (default: merge; "
            "env: GOEDEL_REMOTE_OUTPUT_MODE). Use `full` for CombiBench Fine-Eval."
        ),
    )
    ap.add_argument(
        "--remote_api_key",
        default=os.environ.get("OPENAI_API_KEY", "").strip(),
        help="Remote API key (default: env OPENAI_API_KEY).",
    )
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("GOEDEL_GPU", "4")))
    ap.add_argument("--node", type=int, default=int(os.environ.get("GOEDEL_NODE", "1")))
    ap.add_argument(
        "--inference_handler",
        default=os.environ.get("GOEDEL_HANDLER", "dpskcot"),
        choices=["dpskcot", "dpsknoncot", "kiminacot"],
    )
    ap.add_argument("--temp", type=float, default=float(os.environ.get("GOEDEL_TEMP", "1.0")))
    ap.add_argument("--max_model_len", type=int, default=int(os.environ.get("GOEDEL_MAX_MODEL_LEN", "131072")))
    ap.add_argument("--cpu", type=int, default=int(os.environ.get("GOEDEL_CPU", "32")))
    ap.add_argument(
        "--compile_backend",
        default=os.environ.get("GOEDEL_COMPILE_BACKEND", "lean_server_http"),
        choices=["lean_server_http", "lean_interact", "goedel_repl"],
        help=(
            "Compilation backend for proving. This repo enforces HTTP Lean server only.\n"
            "- lean_server_http (REQUIRED): compile via LEAN_SERVER_URL (e.g. http://127.0.0.1:8002/api/check)\n"
            "- lean_interact / goedel_repl: disabled to avoid OOM & inconsistent environments."
        ),
    )
    ap.add_argument(
        "--lean_server_url",
        default=os.environ.get("LEAN_SERVER_URL", "http://127.0.0.1:8002/api/check"),
        help="HTTP Lean server URL (default: env LEAN_SERVER_URL or http://127.0.0.1:8002/api/check).",
    )
    ap.add_argument(
        "--compile_timeout_s",
        type=int,
        default=int(os.environ.get("GOEDEL_COMPILE_TIMEOUT_S", "600")),
        help="Per-proof compile timeout seconds for HTTP Lean server (default: 600; env: GOEDEL_COMPILE_TIMEOUT_S).",
    )
    ap.add_argument(
        "--compile_workers",
        type=int,
        default=int(os.environ.get("GOEDEL_COMPILE_WORKERS", "8")),
        help="Parallel workers over problems for compilation (default: 8; env: GOEDEL_COMPILE_WORKERS).",
    )
    ap.add_argument(
        "--compile_early_stop_field",
        default=os.environ.get("GOEDEL_COMPILE_EARLY_STOP_FIELD", "complete"),
        choices=["complete", "pass"],
        help="Per-problem early-stop criterion during compilation (default: complete).",
    )
    ap.add_argument(
        "--compile_fine_eval",
        action="store_true",
        help="Enable CombiBench-style Fine-Eval checks in experiments/lean_server_compile.py.",
    )
    ap.add_argument(
        "--compile_fine_eval_forbid",
        type=str,
        default="axiom,local_instance",
        help="Comma-separated forbidden keywords for --compile_fine_eval (default: axiom,local_instance).",
    )
    ap.add_argument(
        "--compile_fine_eval_answer_hf_dataset",
        type=str,
        default="",
        help=(
            "Optional HF dataset id to enforce CombiBench-style answer checks for fill-in-the-blank tasks "
            "(e.g. AI-MO/CombiBench). Requires --compile_fine_eval."
        ),
    )
    ap.add_argument(
        "--compile_fine_eval_answer_hf_split",
        type=str,
        default="",
        help=(
            "HF split name to load answers from (e.g. test). Requires --compile_fine_eval and "
            "--compile_fine_eval_answer_hf_dataset."
        ),
    )
    ap.add_argument(
        "--compile_fine_eval_answer_hf_index_column",
        type=str,
        default="theorem_name",
        help="HF index column for answer map (default: theorem_name).",
    )
    ap.add_argument(
        "--compile_fine_eval_answer_hf_answer_column",
        type=str,
        default="answer",
        help="HF answer column (default: answer).",
    )
    ap.add_argument(
        "--compile_resume",
        action="store_true",
        help="Pass --resume to experiments/lean_server_compile.py (skip already-compiled attempts).",
    )
    ap.add_argument(
        "--pipeline_compile",
        action="store_true",
        help=(
            "Overlap compilation with inference (remote backend only) by periodically resuming compilation "
            "while remote inference is still running."
        ),
    )
    ap.add_argument(
        "--pipeline_compile_poll_s",
        type=float,
        default=float(os.environ.get("GOEDEL_PIPELINE_COMPILE_POLL_S", "15")),
        help="Seconds between compilation resume passes while inference is running (default: 15).",
    )
    ap.add_argument(
        "--field",
        default="complete",
        choices=["complete", "pass"],
        help="Success field in compilation_result to summarize (default: complete).",
    )
    ap.add_argument(
        "--fields",
        default="",
        help="Optional comma-separated list of fields to summarize (e.g., pass,complete). Overrides --field.",
    )
    ap.add_argument(
        "--require_header_no_sorry",
        action="store_true",
        help="Skip problems whose header contains 'sorry' when exporting datasets (for meaningful `complete`).",
    )
    ap.add_argument(
        "--sanitize_header_sorry",
        action="store_true",
        help="Rewrite `abbrev x : T := sorry` in headers into `opaque x : T` (to make `complete` meaningful).",
    )
    ap.add_argument(
        "--skip_goedel",
        action="store_true",
        help="Only export datasets/indexes and print the Goedel commands; do not run inference/compile/summarize.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing anything.",
    )
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_root = Path(str(args.run_root)).expanduser().resolve()
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = _parse_int_list(str(args.k_list))
    if not k_list:
        raise SystemExit("Empty --k_list")
    fields = _parse_field_list(str(args.fields)) if str(args.fields or "").strip() else [str(args.field)]

                                                     
                                                                                                                 
    if (
        str(args.dataset_mode) == "ground_truth"
        and str(args.solution_mode) == "without_solution"
        and str(args.inference_backend) == "remote"
        and str(args.remote_output_mode) == "merge"
    ):
        _warn("--solution_mode=without_solution implies --remote_output_mode=full; overriding merge -> full")
        args.remote_output_mode = "full"

    jobs = _build_jobs(out_dir=out_dir, k_list=k_list, budget=int(args.budget))
    build_script = workdir / "experiments" / "build_goedel_prover_dataset.py"
    sum_script = workdir / "experiments" / "summarize_goedel_prover_results.py"
    if not build_script.exists():
        _warn(f"missing: {build_script}")
        return 2
    if not sum_script.exists():
        _warn(f"missing: {sum_script}")
        return 2

    goedel_root = Path(str(args.goedel_root)).expanduser()
    if not goedel_root.is_absolute():
        goedel_root = (workdir / goedel_root).resolve()
    infer_py = goedel_root / "src" / "inference.py"
    compile_py = goedel_root / "src" / "compile.py"
    if str(args.inference_backend) == "local" and not infer_py.exists():
        _warn(f"missing: {infer_py} (required for --inference_backend local)")
        return 2
    if not compile_py.exists():
        if str(args.compile_backend) == "goedel_repl":
            _warn(f"missing: {compile_py}")
            return 2

    remote_infer_py = workdir / "experiments" / "goedel_remote_inference.py"
    if str(args.inference_backend) == "remote":
        if not remote_infer_py.exists():
            _warn(f"missing: {remote_infer_py} (required for --inference_backend remote)")
            return 2
        if not str(args.openai_base_url or "").strip():
            _warn("--openai_base_url is required for --inference_backend remote")
            return 2
        if not str(args.remote_model or "").strip():
            _warn("--remote_model is required for --inference_backend remote")
            return 2

    baseline = str(args.baseline or "").strip()
    if str(args.dataset_mode) == "ground_truth":
        baseline = baseline or "ground_truth"

    if str(args.compile_backend) != "lean_server_http":
        _warn(
            f"--compile_backend={args.compile_backend} is disabled. "
            "Use --compile_backend lean_server_http (HTTP Lean server only)."
        )
        return 2

    http_compile_py = workdir / "experiments" / "lean_server_compile.py"
    if not http_compile_py.exists():
        _warn(f"missing: {http_compile_py}")
        return 2

    for job in jobs:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        job.goedel_out_dir.mkdir(parents=True, exist_ok=True)

                            
        export_cmd = [
            sys.executable,
            str(build_script),
            "--run_root",
            str(run_root),
            "--mode",
            str(args.dataset_mode),
            "--baseline",
            str(baseline),
            "--solution_mode",
            str(args.solution_mode),
            "--k",
            str(int(job.k)),
            "--prefer_semantic",
            "--out",
            str(job.dataset_jsonl),
            "--index_out",
            str(job.index_json),
        ]
        if str(args.hf_dataset or "").strip() and str(args.hf_split or "").strip():
            export_cmd.extend(["--hf_dataset", str(args.hf_dataset), "--hf_split", str(args.hf_split)])
        if int(args.max_problems) > 0:
            export_cmd.extend(["--max_problems", str(int(args.max_problems))])
        if str(args.problems_file or "").strip():
            export_cmd.extend(["--problems_file", str(args.problems_file)])
        if bool(args.require_header_no_sorry):
            export_cmd.append("--require_header_no_sorry")
        if bool(args.sanitize_header_sorry):
            export_cmd.append("--sanitize_header_sorry")
        _info(f"[Export] k={job.k} -> {job.dataset_jsonl}")
        rc = _run(export_cmd, cwd=workdir, dry_run=bool(args.dry_run))
        if rc != 0:
            _warn(f"export failed (k={job.k}): returncode={rc}")
            return rc

                                            
        if str(args.inference_backend) == "remote":
            infer_cmd = [
                sys.executable,
                str(remote_infer_py),
                "--input_jsonl",
                str(job.dataset_jsonl),
                "--output_dir",
                str(job.goedel_out_dir),
                "--openai_base_url",
                str(args.openai_base_url),
                "--model",
                str(args.remote_model),
                "--inference_handler",
                str(args.inference_handler),
                "--n",
                str(int(job.n_per_statement)),
                "--temperature",
                str(float(args.temp)),
                "--max_tokens",
                str(int(args.remote_max_tokens)),
                "--timeout_s",
                str(float(args.remote_timeout_s)),
                "--output_mode",
                str(args.remote_output_mode),
                "--workers",
                str(int(args.remote_workers)),
                "--max_retries",
                str(int(args.remote_max_retries)),
                "--retry_backoff_s",
                str(float(args.remote_retry_backoff_s)),
            ]
            if bool(args.remote_resume):
                infer_cmd.append("--resume")
            if str(args.remote_api_key or "").strip():
                infer_cmd.extend(["--api_key", str(args.remote_api_key)])
        else:
            infer_cmd = [
                sys.executable,
                str(infer_py),
                "--input_path",
                str(job.dataset_jsonl),
                "--model_path",
                str(args.model_path),
                "--output_dir",
                str(job.goedel_out_dir),
                "--n",
                str(int(job.n_per_statement)),
                "--gpu",
                str(int(args.gpu)),
                "--node",
                str(int(args.node)),
                "--inference_handler",
                str(args.inference_handler),
                "--temp",
                str(float(args.temp)),
                "--max_model_len",
                str(int(args.max_model_len)),
                "--split",
                "none",
                "--correction_round",
                "0",
        ]
        compile_out = job.goedel_out_dir / "code_compilation_repl.json"
        compile_cmd = [
            sys.executable,
            str(http_compile_py),
            "--input_path",
            str(job.goedel_out_dir / "to_inference_codes.json"),
            "--output_path",
            str(compile_out),
            "--lean_server_url",
            str(args.lean_server_url),
            "--timeout_s",
            str(int(args.compile_timeout_s)),
            "--workers",
            str(int(args.compile_workers)),
            "--early_stop_field",
            str(args.compile_early_stop_field),
            "--save_every",
            "10",
        ]
        if bool(args.compile_resume) or bool(args.pipeline_compile):
            compile_cmd.append("--resume")
        if bool(args.compile_fine_eval):
            compile_cmd.append("--fine_eval")
            if str(args.compile_fine_eval_forbid or "").strip():
                compile_cmd.extend(["--fine_eval_forbid", str(args.compile_fine_eval_forbid)])
            if str(args.compile_fine_eval_answer_hf_dataset or "").strip() and str(
                args.compile_fine_eval_answer_hf_split or ""
            ).strip():
                compile_cmd.extend(
                    [
                        "--fine_eval_answer_hf_dataset",
                        str(args.compile_fine_eval_answer_hf_dataset),
                        "--fine_eval_answer_hf_split",
                        str(args.compile_fine_eval_answer_hf_split),
                        "--fine_eval_answer_index_column",
                        str(args.compile_fine_eval_answer_hf_index_column),
                        "--fine_eval_answer_column",
                        str(args.compile_fine_eval_answer_hf_answer_column),
                    ]
                )
        compile_cwd = workdir
        summarize_cmds: List[Tuple[str, List[str], Path]] = []
        for f in fields:
            summary_out = job.goedel_out_dir / f"portfolio_summary_{f}"
            summarize_cmds.append(
                (
                    f,
                    [
                        sys.executable,
                        str(sum_script),
                        "--compilation_json",
                        str(compile_out),
                        "--field",
                        str(f),
                        "--index_json",
                        str(job.index_json),
                        "--out_dir",
                        str(summary_out),
                    ],
                    summary_out,
                )
            )

        if bool(args.skip_goedel):
            print("\n# --- Goedel pipeline ---")
            print("# Inference:")
            print(" ".join(infer_cmd))
            print("# Compile:")
            print(" ".join(compile_cmd))
            for f, cmd, _out in summarize_cmds:
                print(f"# Summarize ({f}):")
                print(" ".join(cmd))
            continue

        _info(f"[Goedel] k={job.k} n={job.n_per_statement} -> {job.goedel_out_dir}")
        if bool(args.pipeline_compile) and str(args.inference_backend) == "remote":
            if bool(args.dry_run):
                print(" ".join(infer_cmd))
                print(" ".join(compile_cmd))
            else:
                infer_proc = subprocess.Popen(infer_cmd, cwd=str(workdir))
                input_path = job.goedel_out_dir / "to_inference_codes.json"
                poll_s = max(1.0, float(args.pipeline_compile_poll_s))

                while True:
                    if input_path.exists():
                        rc2 = subprocess.run(compile_cmd, cwd=str(compile_cwd)).returncode
                        if rc2 != 0:
                            _warn(f"pipeline compile: returncode={rc2} (will retry)")
                    if infer_proc.poll() is not None:
                        break
                    time.sleep(poll_s)

                infer_rc = int(infer_proc.wait())
                if infer_rc != 0:
                    _warn(f"goedel inference failed (k={job.k}): returncode={infer_rc}")
                    return infer_rc

                                                                     
                if input_path.exists():
                    rc3 = subprocess.run(compile_cmd, cwd=str(compile_cwd)).returncode
                    if rc3 != 0:
                        _warn(f"goedel compile failed (k={job.k}): returncode={rc3}")
                        return rc3
        else:
            rc = _run(
                infer_cmd,
                cwd=(goedel_root if str(args.inference_backend) == "local" else workdir),
                dry_run=bool(args.dry_run),
            )
            if rc != 0:
                _warn(f"goedel inference failed (k={job.k}): returncode={rc}")
                return rc
            rc = _run(compile_cmd, cwd=compile_cwd, dry_run=bool(args.dry_run))
            if rc != 0:
                _warn(f"goedel compile failed (k={job.k}): returncode={rc}")
                return rc
        for f, cmd, outp in summarize_cmds:
            rc = _run(cmd, cwd=workdir, dry_run=bool(args.dry_run))
            if rc != 0:
                _warn(f"summarize failed (k={job.k} field={f}): returncode={rc}")
                return rc
            _info(f"[OK] wrote: {outp / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
