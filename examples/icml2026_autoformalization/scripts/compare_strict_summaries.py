                      
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(hit: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(hit) / float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two strict_summary.json files (trend-level).")
    ap.add_argument("--a", type=str, required=True, help="Path to strict_summary.json (run A)")
    ap.add_argument("--b", type=str, required=True, help="Path to strict_summary.json (run B)")
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    args = ap.parse_args()

    a_path = Path(args.a).expanduser().resolve()
    b_path = Path(args.b).expanduser().resolve()
    if not a_path.exists():
        raise FileNotFoundError(a_path)
    if not b_path.exists():
        raise FileNotFoundError(b_path)

    a = _load_json(a_path)
    b = _load_json(b_path)

    a_n = int(a.get("num_problems") or 0)
    b_n = int(b.get("num_problems") or 0)
    if a_n != b_n:
                                              
        print(f"[warn] num_problems differ: {args.label_a}={a_n} vs {args.label_b}={b_n}")

    for label, obj, n in [(args.label_a, a, a_n), (args.label_b, b, b_n)]:
        ch = int(obj.get("compile_hit_db") or 0)
        sh = int(obj.get("semantic_hit_db") or 0)
        print(f"{label}: run_dir={obj.get('run_dir')}")
        print(f"{label}: compile_hit_db = {ch}/{n} = {_rate(ch, n):.3f}")
        print(f"{label}: semantic_hit_db = {sh}/{n} = {_rate(sh, n):.3f}")

                   
    a_ch = int(a.get("compile_hit_db") or 0)
    a_sh = int(a.get("semantic_hit_db") or 0)
    b_ch = int(b.get("compile_hit_db") or 0)
    b_sh = int(b.get("semantic_hit_db") or 0)
    n = max(a_n, b_n)
    print("----")
    print(f"Δ compile_hit_db (B-A): {b_ch - a_ch} ({_rate(b_ch, n) - _rate(a_ch, n):+.3f} rate)")
    print(f"Δ semantic_hit_db (B-A): {b_sh - a_sh} ({_rate(b_sh, n) - _rate(a_sh, n):+.3f} rate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

