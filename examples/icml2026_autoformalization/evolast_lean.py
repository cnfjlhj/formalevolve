"""
EvolAST-style equivalence-preserving rewrites for Lean4 theorem statements.

Goal: provide a *low-cost* syntactic mutation operator inspired by EvolProver (Tian et al., 2025):
- Parse (a subset of) Lean expressions into a small AST
- Recursively traverse nodes; with probability p apply one applicable rewrite rule
- Emit a rewritten Lean theorem statement (keeping `:= by sorry` body unchanged)

This module is intentionally conservative:
- Only rewrites within binder types and the goal type (theorem's `: ... :=`)
- Avoids touching imports/preamble and proof bodies
- Limits supported operators to common arithmetic/logic/relation symbols
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Public configuration surface
# ----------------------------

RULE_KEYS = (
    "hyp_reorder",
    "commutativity",
    "associativity",
    "distributivity",
    "de_morgan",
    "sym_swap",
    "dual_relation",
)


def parse_rule_weights(raw: Any) -> Dict[str, float]:
    """Parse rule weights from JSON or `k=v,k=v` form; defaults to uniform."""
    if raw is None:
        return {k: 1.0 for k in RULE_KEYS}
    if isinstance(raw, dict):
        out: Dict[str, float] = {k: 0.0 for k in RULE_KEYS}
        for k, v in raw.items():
            ks = str(k).strip()
            if ks in out:
                try:
                    out[ks] = float(v)
                except Exception:
                    pass
        if sum(out.values()) <= 0:
            return {k: 1.0 for k in RULE_KEYS}
        return out

    s = str(raw).strip()
    if not s:
        return {k: 1.0 for k in RULE_KEYS}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return parse_rule_weights(obj)
    except Exception:
        pass

    out = {k: 0.0 for k in RULE_KEYS}
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in out:
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    if sum(out.values()) <= 0:
        return {k: 1.0 for k in RULE_KEYS}
    return out


def similarity_ratio(a: str, b: str) -> float:
    """Whitespace-robust similarity ratio; normalizes decl names to ignore naming noise."""
    aa = _canonicalize_for_similarity(a)
    bb = _canonicalize_for_similarity(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return float(SequenceMatcher(a=aa, b=bb).ratio())


# -----------------
# Lean code helpers
# -----------------

_DECL_RE = re.compile(r"(?m)^\s*(?:noncomputable\s+)?theorem\b")


def split_preamble_and_decl(code: str) -> Tuple[str, str]:
    """Split a Lean file into preamble and first theorem declaration (best-effort)."""
    s = str(code or "").strip()
    if not s:
        return "", ""
    m = _DECL_RE.search(s)
    if not m:
        return "", s
    pre = s[: m.start()].rstrip()
    decl = s[m.start() :].strip()
    return pre, decl


def _canonicalize_for_similarity(code: str) -> str:
    s = str(code or "")
    pre, decl = split_preamble_and_decl(s)
    decl = re.sub(r"\s+", " ", decl).strip()
    decl = re.sub(
        r"^(noncomputable\s+)?theorem\s+[^\s:(]+",
        r"\1theorem __NAME__",
        decl,
    )
    pre = "\n".join(ln.strip() for ln in (pre or "").splitlines() if ln.strip())
    return (pre + "\n" + decl).strip() if pre else decl


def _theorem_parts(decl: str) -> Optional[Tuple[str, str, str]]:
    """Return (head_up_to_colon, type_part, tail_from_coloneq) for a theorem decl."""
    s = str(decl or "").strip()
    if not s:
        return None
    # Find the `:` that starts the type at depth 0.
    colon_idx = _find_at_toplevel(s, ":")
    if colon_idx is None:
        return None
    # Find the `:=` that starts the body at depth 0.
    #
    # IMPORTANT: Lean propositions may contain `let x := ...` inside the goal type,
    # so there can be multiple top-level `:=` tokens between `:` and the theorem's
    # actual `:= by ...` body delimiter. Picking the first would corrupt the decl:
    #   `theorem t : let x := 1; P x := by ...`
    # We locate the *theorem-body* `:=` by skipping `:=` that belong to `let` binders
    # at the same delimiter depth (tracked with `;`).
    coloneq_idx = _find_theorem_body_coloneq(s, start=colon_idx + 1)
    if coloneq_idx is None or coloneq_idx <= colon_idx:
        return None
    head = s[:colon_idx].rstrip()
    typ = s[colon_idx + 1 : coloneq_idx].strip()
    tail = s[coloneq_idx:].lstrip()
    return head, typ, tail


def _is_ident_char(ch: str) -> bool:
    # Lean identifiers allow letters/digits/underscore and primes.
    return ch.isalnum() or ch in {"_", "'"}


def _is_keyword_at(s: str, i: int, kw: str) -> bool:
    if not s.startswith(kw, i):
        return False
    before_ok = i == 0 or (not _is_ident_char(s[i - 1]))
    after_i = i + len(kw)
    after_ok = after_i >= len(s) or (not _is_ident_char(s[after_i]))
    return bool(before_ok and after_ok)


def _find_theorem_body_coloneq(s: str, *, start: int) -> Optional[int]:
    """Find the theorem-body `:=` delimiter, robust to `let x := ...; ...` in the goal type."""
    s = str(s or "")
    if not s:
        return None

    depths = {"(": 0, "[": 0, "{": 0}
    closes = {")": "(", "]": "[", "}": "{"}

    i = max(0, int(start))
    in_string = False
    block_comment_depth = 0

    # Track whether we are in the *binder* part of a `let` expression at depth 0:
    # `let x := <binding>; <body>`
    in_let_binding = False

    while i < len(s):
        if block_comment_depth > 0:
            if s.startswith("/-", i):
                block_comment_depth += 1
                i += 2
                continue
            if s.startswith("-/", i):
                block_comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == "\"":
                in_string = False
                i += 1
                continue
            i += 1
            continue

        # Comments / strings
        if s.startswith("/-", i):
            block_comment_depth = 1
            i += 2
            continue
        if s.startswith("--", i):
            nl = s.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
            continue
        if s[i] == "\"":
            in_string = True
            i += 1
            continue

        # Delimiter nesting
        ch = s[i]
        if ch in depths:
            depths[ch] += 1
            i += 1
            continue
        if ch in closes:
            opener = closes[ch]
            if depths[opener] > 0:
                depths[opener] -= 1
            i += 1
            continue

        if not all(v == 0 for v in depths.values()):
            i += 1
            continue

        # Depth 0: handle `let` binders (only the binder assignment uses `:=`).
        if _is_keyword_at(s, i, "let"):
            in_let_binding = True
            i += 3
            continue

        if in_let_binding:
            if ch == ";":
                in_let_binding = False
                i += 1
                continue
            # The binder assignment itself.
            if s.startswith(":=", i):
                i += 2
                continue
        else:
            if s.startswith(":=", i):
                return i

        i += 1

    return None


def _find_all_at_toplevel(s: str, needle: str) -> List[int]:
    """Find all occurrences of `needle` at delimiter depth 0.

    Notes:
    - Tracks only (), [], {} nesting (good enough for our best-effort split).
    - Skips line (`-- ...`) and block (`/- ... -/`, nested) comments.
    - Skips string literals (`"..."`) to avoid false positives in messages.
    """
    s = str(s or "")
    if not s or not needle:
        return []

    depths = {"(": 0, "[": 0, "{": 0}
    closes = {")": "(", "]": "[", "}": "{"}

    out: List[int] = []
    i = 0
    in_string = False
    block_comment_depth = 0

    while i < len(s):
        if block_comment_depth > 0:
            if s.startswith("/-", i):
                block_comment_depth += 1
                i += 2
                continue
            if s.startswith("-/", i):
                block_comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == "\"":
                in_string = False
                i += 1
                continue
            i += 1
            continue

        # Comments / strings
        if s.startswith("/-", i):
            block_comment_depth = 1
            i += 2
            continue
        if s.startswith("--", i):
            nl = s.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
            continue
        if s[i] == "\"":
            in_string = True
            i += 1
            continue

        # Delimiter nesting
        ch = s[i]
        if ch in depths:
            depths[ch] += 1
            i += 1
            continue
        if ch in closes:
            opener = closes[ch]
            if depths[opener] > 0:
                depths[opener] -= 1
            i += 1
            continue

        if all(v == 0 for v in depths.values()) and s.startswith(needle, i):
            out.append(i)
            i += len(needle)
            continue

        i += 1

    return out


def _find_at_toplevel(s: str, needle: str) -> Optional[int]:
    for idx in _find_all_at_toplevel(s, needle):
        return idx
    return None


def _extract_binders(head: str) -> Tuple[str, List[str]]:
    """Split `theorem name ...` head into (prefix, binder_groups)."""
    s = str(head or "").rstrip()
    # Find first binder open at depth 0.
    idx = None
    for j, ch in enumerate(s):
        if ch in "([{":
            idx = j
            break
    if idx is None:
        return s, []
    prefix = s[:idx].rstrip()
    rest = s[idx:].lstrip()
    groups: List[str] = []
    i = 0
    while i < len(rest):
        while i < len(rest) and rest[i].isspace():
            i += 1
        if i >= len(rest):
            break
        if rest[i] not in "([{":
            # Unusual binder surface form; stop and keep remaining as prefix tail.
            prefix = (prefix + " " + rest[i:]).strip()
            break
        open_ch = rest[i]
        close_ch = { "(": ")", "[": "]", "{": "}" }[open_ch]
        depth = 0
        start = i
        while i < len(rest):
            if rest[i] == open_ch:
                depth += 1
            elif rest[i] == close_ch:
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        groups.append(rest[start:i].strip())
    return prefix, groups


# ----------------------------
# Expression parsing / printing
# ----------------------------

_MULTI_OPS = ("↔", "≤", "≥", "≠", ":=", "->", "→")
_SINGLE_OP_CHARS = set("()[]{}:+*=/<>^¬∧∨")


def _tokenize(expr: str) -> List[str]:
    s = str(expr or "")
    tokens: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        matched = False
        for op in _MULTI_OPS:
            if s.startswith(op, i):
                tokens.append(op)
                i += len(op)
                matched = True
                break
        if matched:
            continue
        if ch in _SINGLE_OP_CHARS:
            tokens.append(ch)
            i += 1
            continue
        # Atom: read until whitespace or operator-ish char.
        j = i
        while j < len(s) and (not s[j].isspace()) and (s[j] not in _SINGLE_OP_CHARS):
            # Stop before any multi-op.
            if any(s.startswith(op, j) for op in _MULTI_OPS):
                break
            j += 1
        atom = s[i:j]
        tokens.append(atom)
        i = j
    return tokens


_PREC: Dict[str, int] = {
    "↔": 10,
    "=": 20,
    "≠": 20,
    "<": 20,
    ">": 20,
    "≤": 20,
    "≥": 20,
    "∨": 30,
    "∧": 40,
    "+": 50,
    "*": 60,
    "^": 70,
    "->": 5,
    "→": 5,
}


@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Atom(Expr):
    text: str


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    arg: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


class _Parser:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> str:
        return self.toks[self.i] if self.i < len(self.toks) else ""

    def _eat(self, tok: str) -> bool:
        if self._peek() == tok:
            self.i += 1
            return True
        return False

    def parse(self) -> Expr:
        return self._parse_expr(0)

    def _parse_expr(self, min_prec: int) -> Expr:
        node = self._parse_prefix()
        while True:
            op = self._peek()
            if op not in _PREC:
                break
            prec = _PREC[op]
            if prec < min_prec:
                break
            self.i += 1
            # Right-assoc for arrow and power; left-assoc otherwise.
            next_min = prec + (0 if op in {"->", "→", "^"} else 1)
            rhs = self._parse_expr(next_min)
            node = Binary(op=op, left=node, right=rhs)
        return node

    def _parse_prefix(self) -> Expr:
        if self._eat("¬"):
            return Unary("¬", self._parse_expr(80))
        if self._eat("("):
            inner = self._parse_expr(0)
            self._eat(")")
            return inner
        tok = self._peek()
        if tok == "":
            return Atom("")
        self.i += 1
        return Atom(tok)


def _parse_expr(expr: str) -> Expr:
    toks = _tokenize(expr)
    if not toks:
        return Atom("")
    try:
        parser = _Parser(toks)
        node = parser.parse()
        # If we fail to consume all tokens, the grammar is outside our supported subset
        # (e.g., `∀/∃` binders, commas, `:` binder syntax). In that case, fall back to a
        # raw Atom to avoid truncating/changing the expression.
        if parser.i != len(toks):
            return Atom(str(expr).strip())
        return node
    except Exception:
        return Atom(str(expr).strip())


def _prec_of(e: Expr) -> int:
    if isinstance(e, Binary):
        return int(_PREC.get(e.op, 100))
    if isinstance(e, Unary):
        return 90
    return 100


def _to_str(e: Expr, parent_prec: int = 0) -> str:
    if isinstance(e, Atom):
        return e.text
    if isinstance(e, Unary):
        s = f"{e.op}{_to_str(e.arg, 90)}"
        return s if 90 >= parent_prec else f"({s})"
    if isinstance(e, Binary):
        p = int(_PREC.get(e.op, 100))
        left = _to_str(e.left, p + 1)
        right = _to_str(e.right, p + (0 if e.op in {"->", "→", "^"} else 1))
        s = f"{left} {e.op} {right}"
        return s if p >= parent_prec else f"({s})"
    return ""


# ----------------------------
# Rewrite rules (7-rule surface)
# ----------------------------

_COMM_OPS = {"+", "*", "∧", "∨"}
_ASSOC_OPS = {"+", "*", "∧", "∨"}
_SYM_REL_OPS = {"=", "↔", "≠"}
_DUAL_REL = {">": "<", "<": ">", "≥": "≤", "≤": "≥"}


def apply_evolast_to_lean_code(
    code: str,
    *,
    p: float = 0.35,
    rule_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    max_rewrites: int = 32,
    mode: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Rewrite a Lean file (preamble + theorem) and return (new_code, info).

    Safety:
    - Default mode is `safe` (semantics-preserving): we only add redundant parentheses
      around the theorem goal type.
    - `aggressive` enables the original rule-based mutation for the supported
      expression subset, with a strict fallback to raw text when parsing is partial.
    """
    mode_eff = str(
        (mode or os.environ.get("AUTOFORMAL_EVOLAST_MODE") or "safe")
    ).strip().lower()
    if mode_eff not in {"aggressive", "unsafe"}:
        return apply_evolast_safe_to_lean_code(code)

    rng = random.Random(seed)
    weights = parse_rule_weights(rule_weights)

    pre, decl = split_preamble_and_decl(code)
    parts = _theorem_parts(decl)
    if parts is None:
        return code, {"ok": False, "reason": "no_theorem_parts"}
    head, typ, tail = parts

    prefix, binders = _extract_binders(head)
    binder_info, binders2 = _rewrite_binders(binders, p=p, weights=weights, rng=rng, max_rewrites=max_rewrites)
    type_info, typ2 = _rewrite_expr_text(typ, p=p, weights=weights, rng=rng, max_rewrites=max_rewrites)

    head2 = (prefix + (" " + " ".join(binders2) if binders2 else "")).strip()
    decl2 = f"{head2} : {typ2} {tail}".strip()
    out = (pre + "\n\n" + decl2).strip() if pre else decl2

    info = {
        "ok": True,
        "p": float(p),
        "max_rewrites": int(max_rewrites),
        "mode": mode_eff,
        "rule_weights": dict(weights),
        "binder": binder_info,
        "type": type_info,
    }
    return out, info


def apply_evolast_safe_to_lean_code(code: str) -> Tuple[str, Dict[str, Any]]:
    """Semantics-preserving EvolAST fallback.

    We only add redundant parentheses around the theorem goal type:
        `theorem ... : T := ...` -> `theorem ... : (T) := ...`
    This guarantees the proposition is unchanged.
    """
    pre, decl = split_preamble_and_decl(str(code or ""))
    parts = _theorem_parts(decl)
    if parts is None:
        return code, {"ok": False, "mode": "safe", "reason": "no_theorem_parts"}
    head, typ, tail = parts
    typ_s = str(typ or "").strip()
    if not typ_s:
        return code, {"ok": False, "mode": "safe", "reason": "empty_type"}
    typ2 = f"({typ_s})"
    decl2 = f"{head} : {typ2} {tail}".strip()
    out = (pre + "\n\n" + decl2).strip() if pre else decl2
    return out, {"ok": True, "mode": "safe", "transformation": "wrap_goal_type_parens"}


def goal_has_reflexive_relation(code: str) -> bool:
    """Heuristic guard: reject mutations that make the goal contain obvious `t = t` / `t ↔ t` / `t ≠ t`."""
    _pre, decl = split_preamble_and_decl(code)
    parts = _theorem_parts(decl)
    if parts is None:
        return False
    _head, typ, _tail = parts
    ast = _parse_expr(typ)
    return _has_reflexive_relation(ast)


def _rewrite_binders(
    binders: List[str],
    *,
    p: float,
    weights: Dict[str, float],
    rng: random.Random,
    max_rewrites: int,
) -> Tuple[Dict[str, Any], List[str]]:
    counts = {k: 0 for k in RULE_KEYS}

    out = list(binders)
    # Rule 1: hypothesis reordering (conservative adjacent swaps).
    if out and rng.random() < float(p) and weights.get("hyp_reorder", 0.0) > 0.0:
        idxs = [i for i, b in enumerate(out) if _is_hypothesis_binder(b)]
        swaps = 0
        for _ in range(min(8, len(idxs) * 2 + 1)):
            if swaps >= max_rewrites:
                break
            if len(idxs) < 2:
                break
            j = rng.randrange(0, len(idxs) - 1)
            i1 = idxs[j]
            i2 = idxs[j + 1]
            if _safe_swap_binders(out[i1], out[i2]):
                out[i1], out[i2] = out[i2], out[i1]
                swaps += 1
        if swaps > 0:
            counts["hyp_reorder"] += 1

    # Apply expression rewrites inside each binder's type.
    rewritten = []
    for b in out:
        b2 = _rewrite_binder_type(b, p=p, weights=weights, rng=rng, counts=counts, max_rewrites=max_rewrites)
        rewritten.append(b2)

    return {"counts": counts, "num_binders": len(out)}, rewritten


def _is_hypothesis_binder(b: str) -> bool:
    s = str(b or "").strip()
    if ":" not in s:
        return False
    # Extract the binder name (best-effort).
    m = re.match(r"^[({\[]\s*([A-Za-z_][A-Za-z0-9_']*)\b", s)
    name = (m.group(1) if m else "").strip()
    if name.lower().startswith(("h", "hyp", "assm")):
        return True
    # Heuristic: logical/relation symbols suggest Prop-like binder.
    return bool(re.search(r"(=|≠|<|>|≤|≥|∧|∨|↔|¬|→|->|∣)", s))


def _binder_name_and_type(b: str) -> Tuple[str, str]:
    s = str(b or "").strip()
    # Split at first ':' at top level inside this binder group.
    # Binder groups are already balanced, so a simple split is OK. Be careful to
    # strip ONLY the outermost binder delimiters; do NOT strip nested parentheses
    # in the type, otherwise we may corrupt e.g. `Finset (EuclideanSpace ...))`.
    if not s or ":" not in s:
        return "", ""

    open_ch = s[0] if s[0] in "([{" else ""
    close_ch = {"(": ")", "[": "]", "{": "}"}.get(open_ch, "")
    inner = s
    if open_ch and inner.startswith(open_ch):
        inner = inner[1:]
    if close_ch and inner.endswith(close_ch):
        inner = inner[:-1]
    inner = inner.strip()

    if ":" not in inner:
        return "", ""
    left, right = inner.split(":", 1)

    # left like "hn " or "x y "
    left = left.strip()
    name = left.split()[0] if left.split() else ""
    typ = right.strip()
    return name, typ


def _safe_swap_binders(b1: str, b2: str) -> bool:
    n1, t1 = _binder_name_and_type(b1)
    n2, t2 = _binder_name_and_type(b2)
    if not n1 or not n2:
        return False
    # Prevent swapping if either type mentions the other binder name.
    if re.search(rf"\\b{re.escape(n1)}\\b", t2):
        return False
    if re.search(rf"\\b{re.escape(n2)}\\b", t1):
        return False
    return True


def _rewrite_binder_type(
    b: str,
    *,
    p: float,
    weights: Dict[str, float],
    rng: random.Random,
    counts: Dict[str, int],
    max_rewrites: int,
) -> str:
    s = str(b or "").strip()
    if ":" not in s:
        return b

    # IMPORTANT: strip ONLY ONE outer delimiter pair. Using `lstrip`/`rstrip`
    # is unsafe because it removes *all* matching characters, which can corrupt
    # nested parentheses in types (a real bug observed in large runs).
    open_ch = s[0] if (s and s[0] in "([{") else "("
    close_ch = {"(": ")", "[": "]", "{": "}"}.get(open_ch, ")")
    inner = s
    if inner.startswith(open_ch):
        inner = inner[1:]
    if inner.endswith(close_ch):
        inner = inner[:-1]
    inner = inner.strip()

    if ":" not in inner:
        return b
    left, right = inner.split(":", 1)
    typ = right.strip()

    info, typ2 = _rewrite_expr_text(typ, p=p, weights=weights, rng=rng, max_rewrites=max_rewrites)
    # Accumulate counts (excluding hyp_reorder which is already handled).
    for k, v in (info.get("counts") or {}).items():
        if k in counts and k != "hyp_reorder":
            counts[k] += int(v)
    return f"{open_ch}{left.strip()} : {typ2}{close_ch}"


def _rewrite_expr_text(
    expr: str,
    *,
    p: float,
    weights: Dict[str, float],
    rng: random.Random,
    max_rewrites: int,
) -> Tuple[Dict[str, Any], str]:
    counts = {k: 0 for k in RULE_KEYS}
    ast = _parse_expr(expr)
    remaining = int(max(0, max_rewrites))

    def go(node: Expr) -> Expr:
        nonlocal remaining
        if remaining <= 0:
            return node

        # Recurse first (top-down keeps changes coarse).
        if isinstance(node, Unary):
            node = Unary(node.op, go(node.arg))
        elif isinstance(node, Binary):
            node = Binary(node.op, go(node.left), go(node.right))

        if remaining <= 0:
            return node

        applicable = _applicable_rules(node)
        if not applicable:
            return node
        if rng.random() >= float(p):
            return node

        picked = _weighted_pick(applicable, weights, rng)
        if picked is None:
            return node

        new_node = _apply_rule(node, picked, rng)
        if new_node is node:
            return node
        counts[picked] += 1
        remaining -= 1
        return new_node

    ast2 = go(ast)
    return {"counts": counts, "rewrites_used": int(max_rewrites - remaining)}, _to_str(ast2, 0).strip()


def _has_reflexive_relation(node: Expr) -> bool:
    if isinstance(node, Binary) and node.op in {"=", "↔", "≠"}:
        if node.left == node.right:
            return True
    if isinstance(node, Unary):
        return _has_reflexive_relation(node.arg)
    if isinstance(node, Binary):
        return _has_reflexive_relation(node.left) or _has_reflexive_relation(node.right)
    return False


def _weighted_pick(applicable: Iterable[str], weights: Dict[str, float], rng: random.Random) -> Optional[str]:
    items = []
    total = 0.0
    for k in applicable:
        w = float(weights.get(k, 0.0) or 0.0)
        if w <= 0:
            continue
        items.append((k, w))
        total += w
    if total <= 0 or not items:
        return None
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if r <= acc:
            return k
    return items[-1][0]


def _applicable_rules(node: Expr) -> List[str]:
    out: List[str] = []
    if isinstance(node, Binary):
        if node.op in _COMM_OPS:
            out.append("commutativity")
        if node.op in _ASSOC_OPS and (isinstance(node.left, Binary) and node.left.op == node.op or isinstance(node.right, Binary) and node.right.op == node.op):
            out.append("associativity")
        if node.op in {"*", "∧"} and (
            isinstance(node.left, Binary)
            and node.left.op in {"+", "∨"}
            or isinstance(node.right, Binary)
            and node.right.op in {"+", "∨"}
        ):
            out.append("distributivity")
        # Factoring form: (a*b + a*c) -> a*(b+c); ((A∧B) ∨ (A∧C)) -> A∧(B∨C)
        if node.op in {"+", "∨"}:
            out.append("distributivity")
        if node.op in _SYM_REL_OPS:
            out.append("sym_swap")
        if node.op in _DUAL_REL:
            out.append("dual_relation")
    if isinstance(node, Unary) and node.op == "¬" and isinstance(node.arg, Binary) and node.arg.op in {"∧", "∨"}:
        out.append("de_morgan")
    # Reverse De Morgan: (¬A ∨ ¬B) -> ¬(A ∧ B); (¬A ∧ ¬B) -> ¬(A ∨ B)
    if isinstance(node, Binary) and node.op in {"∨", "∧"}:
        if isinstance(node.left, Unary) and node.left.op == "¬" and isinstance(node.right, Unary) and node.right.op == "¬":
            out.append("de_morgan")
    return out


def _apply_rule(node: Expr, rule: str, rng: random.Random) -> Expr:
    if rule == "commutativity" and isinstance(node, Binary) and node.op in _COMM_OPS:
        return Binary(node.op, node.right, node.left)
    if rule == "sym_swap" and isinstance(node, Binary) and node.op in _SYM_REL_OPS:
        return Binary(node.op, node.right, node.left)
    if rule == "dual_relation" and isinstance(node, Binary) and node.op in _DUAL_REL:
        return Binary(_DUAL_REL[node.op], node.right, node.left)
    if rule == "associativity" and isinstance(node, Binary) and node.op in _ASSOC_OPS:
        # ((a op b) op c) <-> (a op (b op c))
        if isinstance(node.left, Binary) and node.left.op == node.op:
            # (a op b) op c -> a op (b op c)
            a = node.left.left
            b = node.left.right
            c = node.right
            return Binary(node.op, a, Binary(node.op, b, c))
        if isinstance(node.right, Binary) and node.right.op == node.op:
            # a op (b op c) -> (a op b) op c
            a = node.left
            b = node.right.left
            c = node.right.right
            return Binary(node.op, Binary(node.op, a, b), c)
        return node
    if rule == "de_morgan" and isinstance(node, Unary) and node.op == "¬" and isinstance(node.arg, Binary):
        a = node.arg.left
        b = node.arg.right
        if node.arg.op == "∧":
            return Binary("∨", Unary("¬", a), Unary("¬", b))
        if node.arg.op == "∨":
            return Binary("∧", Unary("¬", a), Unary("¬", b))
        return node
    if (
        rule == "de_morgan"
        and isinstance(node, Binary)
        and node.op in {"∨", "∧"}
        and isinstance(node.left, Unary)
        and node.left.op == "¬"
        and isinstance(node.right, Unary)
        and node.right.op == "¬"
    ):
        a = node.left.arg
        b = node.right.arg
        if node.op == "∨":
            return Unary("¬", Binary("∧", a, b))
        return Unary("¬", Binary("∨", a, b))
    if rule == "distributivity" and isinstance(node, Binary):
        # Arithmetic: a*(b+c) -> a*b + a*c ; (a+b)*c -> a*c + b*c
        # Logic: A ∧ (B ∨ C) -> (A ∧ B) ∨ (A ∧ C) ; (A ∨ B) ∧ C -> (A ∧ C) ∨ (B ∧ C)
        # Factoring (reverse distributivity) for both domains.
        if node.op == "*" and isinstance(node.right, Binary) and node.right.op == "+":
            a = node.left
            b = node.right.left
            c = node.right.right
            return Binary("+", Binary("*", a, b), Binary("*", a, c))
        if node.op == "*" and isinstance(node.left, Binary) and node.left.op == "+":
            a = node.left.left
            b = node.left.right
            c = node.right
            return Binary("+", Binary("*", a, c), Binary("*", b, c))
        if node.op == "∧" and isinstance(node.right, Binary) and node.right.op == "∨":
            a = node.left
            b = node.right.left
            c = node.right.right
            return Binary("∨", Binary("∧", a, b), Binary("∧", a, c))
        if node.op == "∧" and isinstance(node.left, Binary) and node.left.op == "∨":
            a = node.left.left
            b = node.left.right
            c = node.right
            return Binary("∨", Binary("∧", a, c), Binary("∧", b, c))
        if node.op == "+" and isinstance(node.left, Binary) and node.left.op == "*" and isinstance(node.right, Binary) and node.right.op == "*":
            l1, l2 = node.left.left, node.left.right
            r1, r2 = node.right.left, node.right.right
            # Try factor on left side: (a*b + a*c) -> a*(b+c)
            if l1 == r1:
                return Binary("*", l1, Binary("+", l2, r2))
            if l1 == r2:
                return Binary("*", l1, Binary("+", l2, r1))
            if l2 == r1:
                return Binary("*", l2, Binary("+", l1, r2))
            if l2 == r2:
                return Binary("*", l2, Binary("+", l1, r1))
            return node
        if node.op == "∨" and isinstance(node.left, Binary) and node.left.op == "∧" and isinstance(node.right, Binary) and node.right.op == "∧":
            l1, l2 = node.left.left, node.left.right
            r1, r2 = node.right.left, node.right.right
            # (A∧B) ∨ (A∧C) -> A ∧ (B ∨ C)
            if l1 == r1:
                return Binary("∧", l1, Binary("∨", l2, r2))
            if l1 == r2:
                return Binary("∧", l1, Binary("∨", l2, r1))
            if l2 == r1:
                return Binary("∧", l2, Binary("∨", l1, r2))
            if l2 == r2:
                return Binary("∧", l2, Binary("∨", l1, r1))
            return node
        return node
    return node
