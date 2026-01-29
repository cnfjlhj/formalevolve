import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTOFORMAL_DIR = ROOT / "examples" / "icml2026_autoformalization"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOFORMAL_DIR))


class TestEvolASTAggressiveBinderParensRegression(unittest.TestCase):
    def test_aggressive_mode_does_not_strip_nested_parens_in_binder_types(self) -> None:
        from evolast_lean import apply_evolast_to_lean_code

        code = """-- EVOLVE-BLOCK-START
import Mathlib

theorem t (s : Finset (EuclideanSpace ℝ (Fin 2))) : True := by sorry
-- EVOLVE-BLOCK-END
"""
        out, info = apply_evolast_to_lean_code(
            code,
            mode="aggressive",
            p=0.0,                                                      
            seed=0,
            max_rewrites=0,
        )
        self.assertTrue(bool(info.get("ok", False)))
                                                                                  
                                                                           
        self.assertIn("(s : Finset (EuclideanSpace ℝ (Fin 2)))", out)
        self.assertIn("Fin 2)))", out)


if __name__ == "__main__":
    unittest.main()

