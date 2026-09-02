import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTOFORMAL_DIR = ROOT / "examples" / "icml2026_autoformalization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOFORMAL_DIR))


class TestEvolASTSafeModeLetInType(unittest.TestCase):
    def test_safe_mode_does_not_corrupt_let_goal_type(self) -> None:
        from evolast_lean import apply_evolast_to_lean_code

        code = """-- EVOLVE-BLOCK-START
import Mathlib

theorem foo : let x := 1; x = 1 := by sorry
-- EVOLVE-BLOCK-END
"""
        out, info = apply_evolast_to_lean_code(code, mode="safe")
        self.assertTrue(bool(info.get("ok", False)))
        self.assertIn("let x := 1; x = 1", out)
        self.assertIn(":= by sorry", out)
        # Regression guard: previously mis-split at the `:=` inside `let`.
        self.assertNotIn("(let x) := 1", out)

    def test_safe_mode_handles_let_binding_with_by(self) -> None:
        from evolast_lean import apply_evolast_to_lean_code

        code = """-- EVOLVE-BLOCK-START
import Mathlib

theorem foo : let x := by trivial; True := by sorry
-- EVOLVE-BLOCK-END
"""
        out, info = apply_evolast_to_lean_code(code, mode="safe")
        self.assertTrue(bool(info.get("ok", False)))
        self.assertIn("let x := by trivial; True", out)
        self.assertIn(":= by sorry", out)
        self.assertNotIn("(let x) := by", out)


if __name__ == "__main__":
    unittest.main()

