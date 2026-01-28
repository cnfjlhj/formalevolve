from pathlib import Path
from typing import Optional, Union
from .apply_diff import (
    write_git_diff,
    _mutable_ranges,
    _ensure_evolve_marker_newlines,
    EVOLVE_START,
    EVOLVE_END,
)
from shinka.llm import extract_between
import logging
import re

logger = logging.getLogger(__name__)


def _extract_last_fenced_block_for_lean(raw: str) -> Optional[str]:
    """Prefer the *last* Lean-ish fenced code block that looks like an answer.

    Why:
    - Models may echo the prompt (which often contains fenced code blocks) before the final answer.
    - A naive first-match extraction can pick up the echoed prompt or an earlier draft.
    - This led to observed pathologies where `rewrite.txt` (saved from the model's final block)
      differed from the actually applied `main.lean` (extracted from an earlier block).
    """
    if not raw:
        return None

    # Collect candidates; later patterns are more permissive.
    patterns = [
        r"```(?:lean4?|lean)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    blocks: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, raw, re.DOTALL):
            blocks.append(m.group(1))

    if not blocks:
        return None

    decl_re = re.compile(
        r"^\s*(?:noncomputable\s+)?(?:theorem|lemma|def|example|structure|class|inductive|instance|axiom|abbrev|opaque)\b",
        re.MULTILINE,
    )

    # Prefer blocks that contain a declaration and "sorry" (typical termination).
    for b in reversed(blocks):
        if decl_re.search(b) and ("sorry" in b):
            return b.strip()

    # Fall back to last block with a declaration.
    for b in reversed(blocks):
        if decl_re.search(b):
            return b.strip()

    # If nothing matches, return the last block anyway.
    return blocks[-1].strip()


def _extract_lean_fallback(raw: str) -> Optional[str]:
    """
    Fallback extraction for Lean code when standard code fence extraction fails.

    Handles cases where LLM outputs:
    - Theorem without code fences
    - Mixed format with explanations
    - Code with markdown artifacts
    """
    # Remove code fence markers
    cleaned = re.sub(r"```lean\n?", "", raw)
    cleaned = re.sub(r"```\n?", "", cleaned)

    # Remove EVOLVE-BLOCK markers (may be concatenated with content)
    cleaned = re.sub(r"--\s*EVOLVE-BLOCK-START\s*", "\n", cleaned)
    cleaned = re.sub(r"--\s*EVOLVE-BLOCK-END\s*", "\n", cleaned)
    cleaned = re.sub(r"#\s*EVOLVE-BLOCK-START\s*", "\n", cleaned)
    cleaned = re.sub(r"#\s*EVOLVE-BLOCK-END\s*", "\n", cleaned)

    lines = cleaned.splitlines()

    header_idx: Optional[int] = None
    decl_idx: Optional[int] = None

    decl_re = re.compile(
        r"^\s*(?:noncomputable\s+)?(?:theorem|lemma|def|example|structure|class|inductive|instance|axiom|abbrev|opaque)\b"
    )
    for i, line in enumerate(lines):
        stripped = line.strip()
        if header_idx is None and (
            stripped.startswith("import ")
            or stripped.startswith("open scoped ")
            or stripped.startswith("open ")
            or stripped.startswith("set_option ")
        ):
            header_idx = i
        if decl_idx is None and decl_re.search(line):
            decl_idx = i
            break

    if decl_idx is None:
        return None

    start_idx = header_idx if header_idx is not None and header_idx < decl_idx else decl_idx

    out_lines: list[str] = []
    saw_decl = False
    for line in lines[start_idx:]:
        out_lines.append(line)
        if not saw_decl and decl_re.search(line):
            saw_decl = True
        if saw_decl and "sorry" in line:
            break

    return "\n".join(out_lines).strip()


def apply_full_patch(
    patch_str: str,
    original_str: Optional[str] = None,
    patch_dir: Optional[Union[str, Path]] = None,
    original_path: Optional[Union[str, Path]] = None,
    language: str = "python",
    verbose: bool = True,
) -> tuple[str, int, Optional[Path], Optional[str], Optional[str], Optional[Path]]:
    if original_str is None and original_path is None:
        raise ValueError("Either original_str or original_path must be provided")
    if original_str is None:
        if original_path is None:
            raise ValueError("original_path cannot be None")
        og_path = Path(original_path)
        original = og_path.read_text("utf-8")
    else:
        original = original_str

    error_message: Optional[str] = None
    # Init with original content and 0 applied patches in case of error
    updated_content: str = original
    num_applied: int = 0
    output_path: Optional[Path] = None

    # Extract code from language fences
    if language == "lean":
        extracted_code = _extract_last_fenced_block_for_lean(patch_str)
    else:
        extracted_code = extract_between(
            patch_str,
            f"```{language}",
            "```",
            False,
        )

    # Handle the case where extract_between returns None, dict, or "none"
    if (
        extracted_code is None
        or isinstance(extracted_code, dict)
        or extracted_code == "none"
    ):
        # Fallback for Lean: try to extract theorem directly from raw output
        if language == "lean":
            fallback_code = _extract_lean_fallback(patch_str)
            if fallback_code:
                logger.info("apply_full_patch: Used Lean fallback extraction")
                extracted_code = fallback_code
            else:
                error_message = "Could not extract code from patch string (Lean fallback also failed)"
                return original, 0, None, error_message, None, None
        else:
            error_message = "Could not extract code from patch string"
            return original, 0, None, error_message, None, None

    patch_code = str(extracted_code)

    # Normalize marker boundaries to avoid glued START/END lines (Lean bug class).
    patch_code = _ensure_evolve_marker_newlines(patch_code)

    # For Lean: validate that extracted code contains a declaration (not just header)
    if language == "lean":
        keyword_pattern = re.compile(
            r"(?:noncomputable\s+)?(?:theorem|lemma|def|example|structure|class|inductive|instance|axiom|abbrev|opaque)\b",
        )
        if not keyword_pattern.search(patch_code):
            # Extracted code doesn't contain any declaration keyword
            # Try Lean fallback extraction on the raw patch string
            fallback_code = _extract_lean_fallback(patch_str)
            if fallback_code:
                logger.info("apply_full_patch: Extracted code was header-only, used Lean fallback")
                patch_code = fallback_code
            else:
                error_message = "Extracted code contains no Lean declaration (header-only output)"
                return original, 0, None, error_message, None, None

    if patch_dir is not None:
        patch_dir = Path(patch_dir)
        patch_dir.mkdir(parents=True, exist_ok=True)
        # Store the raw patch content
        patch_path = patch_dir / "rewrite.txt"
        patch_path.write_text(patch_code, "utf-8")

    try:
        # Get mutable ranges from original content
        mutable_ranges = _mutable_ranges(original)

        if not mutable_ranges:
            # No EVOLVE-BLOCK regions found, treat as error for full patch
            msg = "No EVOLVE-BLOCK regions found in original content"
            error_message = msg
            return original, 0, None, error_message, None, None

        # Build updated content by preserving immutable parts
        # and replacing mutable parts
        updated_content = ""
        last_end = 0

        # Detect EVOLVE markers presence in the patch content
        patch_has_start = EVOLVE_START.search(patch_code) is not None
        patch_has_end = EVOLVE_END.search(patch_code) is not None
        patch_has_both = patch_has_start and patch_has_end
        patch_has_none = not patch_has_start and not patch_has_end

        def _find_start_marker_for_range(text: str, range_start: int) -> Optional[re.Match]:
            """Find the EVOLVE_START match whose payload begins at `range_start`."""
            for m in EVOLVE_START.finditer(text):
                nl = text.find("\n", m.end())
                payload_start = (nl + 1) if nl != -1 else len(text)
                if payload_start == range_start:
                    return m
            return None

        def _find_end_marker_for_range(text: str, range_end: int) -> Optional[re.Match]:
            """Find the EVOLVE_END match whose marker starts at `range_end`."""
            for m in EVOLVE_END.finditer(text):
                if m.start() == range_end:
                    return m
            return None

        if patch_has_both:
            # Patch contains both EVOLVE-BLOCK markers, extract from them
            patch_mutable_ranges = _mutable_ranges(patch_code)
            # Patch contains EVOLVE-BLOCK markers, extract from them
            for i, (start, end) in enumerate(mutable_ranges):
                # Add immutable part before this mutable range
                updated_content += original[last_end:start]

                # Get corresponding mutable content from patch
                if i < len(patch_mutable_ranges):
                    patch_start, patch_end = patch_mutable_ranges[i]
                    replacement_content = patch_code[patch_start:patch_end]
                else:
                    # Not enough mutable regions in patch, keep original
                    replacement_content = original[start:end]

                updated_content += replacement_content
                last_end = end
        elif patch_has_none:
            # Patch doesn't contain EVOLVE-BLOCK markers
            # Assume entire patch content should replace all mutable regions
            if len(mutable_ranges) == 1:
                # Single mutable region. If the patch appears to be a full-file
                # rewrite that omitted EVOLVE markers, safely extract only the
                # content intended for the evolve block by matching immutable
                # prefix/suffix from the original file.
                start, end = mutable_ranges[0]

                # Immutable portions that remain outside the evolve block
                immutable_prefix = original[:start]
                immutable_suffix = original[end:]

                # Also compute the portions strictly outside the marker lines
                # to detect full-file patches that omitted EVOLVE markers.
                # Find the start and end marker line boundaries.
                start_match = _find_start_marker_for_range(original, start)
                end_match = _find_end_marker_for_range(original, end)

                prefix_outside = (
                    original[: start_match.start()] if start_match else immutable_prefix
                )
                suffix_outside = (
                    original[end_match.end() :] if end_match else immutable_suffix
                )

                # Heuristic: if patch includes the same immutable prefix/suffix
                # outside the markers, treat the middle part as the evolve-block
                # replacement. Be tolerant to a missing trailing newline in the
                # footer by checking both versions.
                suffix_opts = (suffix_outside, suffix_outside.rstrip("\r\n"))
                if patch_code.startswith(prefix_outside) and any(
                    patch_code.endswith(sfx) for sfx in suffix_opts
                ):
                    mid_start = len(prefix_outside)
                    # choose the matching suffix option to compute end
                    sfx = next(sfx for sfx in suffix_opts if patch_code.endswith(sfx))
                    mid_end = len(patch_code) - len(sfx)
                    replacement_content = patch_code[mid_start:mid_end]
                    # Ensure marker boundaries stay on their own lines.
                    # Add a leading newline only if there is a START marker.
                    if (
                        start_match is not None
                        and replacement_content
                        and not replacement_content.startswith("\n")
                    ):
                        replacement_content = "\n" + replacement_content
                    # Add a trailing newline only if there is an END marker.
                    if (
                        end_match is not None
                        and replacement_content
                        and not replacement_content.endswith("\n")
                    ):
                        replacement_content = replacement_content + "\n"
                    updated_content = (
                        immutable_prefix + replacement_content + immutable_suffix
                    )
                else:
                    # Otherwise, assume the patch_code represents only the
                    # evolve-block payload and insert it directly between markers.
                    # Ensure proper newline handling around the patch content.
                    payload = patch_code
                    if (
                        start_match is not None
                        and payload
                        and not payload.startswith("\n")
                    ):
                        payload = "\n" + payload
                    if end_match is not None and payload and not payload.endswith("\n"):
                        payload = payload + "\n"
                    updated_content = immutable_prefix + payload + immutable_suffix
            else:
                # Multiple EVOLVE-BLOCK regions found, ambiguous without markers
                error_message = (
                    "Multiple EVOLVE-BLOCK regions found but patch "
                    "doesn't specify which to replace"
                )
                return original, 0, None, error_message, None, None
        else:
            # Patch contains exactly one marker (START xor END).
            # Only safe to apply when original has a single evolve region.
            if len(mutable_ranges) != 1:
                error_message = (
                    "Patch contains only one EVOLVE-BLOCK marker, but the original "
                    f"has {len(mutable_ranges)} editable regions; cannot determine target"
                )
                return original, 0, None, error_message, None, None

            # Single target region in original
            start, end = mutable_ranges[0]
            immutable_prefix = original[:start]
            immutable_suffix = original[end:]

            # Find exact marker locations in original for newline policy
            start_match = _find_start_marker_for_range(original, start)
            end_match = _find_end_marker_for_range(original, end)

            # Compute outside-of-markers prefix/suffix from original
            prefix_outside = (
                original[: start_match.start()] if start_match else immutable_prefix
            )
            suffix_outside = (
                original[end_match.end() :] if end_match else immutable_suffix
            )

            # Extract payload based on which single marker is present in patch
            if patch_has_start and not patch_has_end:
                m = EVOLVE_START.search(patch_code)
                payload = patch_code[m.end() :] if m else patch_code
                # Trim footer if the patch included it
                for sfx in (suffix_outside, suffix_outside.rstrip("\r\n")):
                    if sfx and payload.endswith(sfx):
                        payload = payload[: -len(sfx)]
                        break
            elif patch_has_end and not patch_has_start:
                m = EVOLVE_END.search(patch_code)
                payload = patch_code[: m.start()] if m else patch_code
                # Trim header if the patch included it
                for pfx in (prefix_outside, prefix_outside.rstrip("\r\n")):
                    if pfx and payload.startswith(pfx):
                        payload = payload[len(pfx) :]
                        break
            else:
                payload = patch_code

            # Normalize newlines so markers remain on their own lines
            if start_match is not None and payload and not payload.startswith("\n"):
                payload = "\n" + payload
            if end_match is not None and payload and not payload.endswith("\n"):
                payload = payload + "\n"

            updated_content = immutable_prefix + payload + immutable_suffix

        # Add remaining immutable content after last mutable range
        if patch_has_both and mutable_ranges:
            updated_content += original[mutable_ranges[-1][1] :]

        num_applied = 1

    except Exception as e:
        error_message = f"Error applying full patch: {str(e)}"
        return original, 0, None, error_message, None, None

    if language == "python":
        suffix = ".py"
    elif language == "cpp":
        suffix = ".cpp"
    elif language == "cuda":
        suffix = ".cu"
    elif language == "rust":
        suffix = ".rs"
    elif language == "swift":
        suffix = ".swift"
    elif language in ["json", "json5"]:
        suffix = ".json"
    elif language == "lean":
        suffix = ".lean"
    else:
        raise ValueError(f"Language {language} not supported")

    # If successful, proceed to write files if patch_dir is specified
    if patch_dir is not None:
        # Store the original string as a backup file
        backup_path = patch_dir / f"original{suffix}"
        backup_path.write_text(original, "utf-8")

        # Write the updated file
        output_path = patch_dir / f"main{suffix}"
        output_path.write_text(updated_content, "utf-8")

        # Write the git diff if requested
        diff_path = patch_dir / "edit.diff"
        write_git_diff(
            original,
            updated_content,
            filename=backup_path.name,
            out_path=diff_path,
        )
        patch_txt = diff_path.read_text("utf-8")
        # Print the patch file
        if verbose:
            logger.info(f"Patch file written to: {diff_path}")
            logger.info(f"Patch file content:\n{patch_txt}")
        return (
            updated_content,
            num_applied,
            output_path,
            error_message,
            patch_txt,
            diff_path,
        )
    else:
        return updated_content, num_applied, None, error_message, None, None
