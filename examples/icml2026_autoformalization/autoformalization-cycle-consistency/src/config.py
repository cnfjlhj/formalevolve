"""
Configuration for Cycle Consistency Autoformalization

Based on: "Improving autoformalization via cycle consistency and
incremental type-checking using language-model probabilistic programs"
(MATH-AI 2025)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration"""
                                                                  
    formalizer_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    formalizer_base_url: Optional[str] = None                     

                                                                      
                                       
    informalizer_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    informalizer_base_url: Optional[str] = None

                           
    temperature: float = 0.7
    max_tokens: int = 512
    num_candidates: int = 5                                                  


@dataclass
class CycleConsistencyConfig:
    """Cycle consistency specific configuration"""
                                                                      
    use_log_prob: bool = True

                                                                    
    normalize_by_length: bool = True

                                                           
    selection_temperature: float = 1.0


@dataclass
class PromptConfig:
    """Prompt templates"""
                                   
    formalize_system: str = """You are an expert at translating natural language mathematics into Lean 4 formal statements.
Given a natural language mathematical statement, output ONLY the Lean 4 theorem statement.
Do not include any explanation or proof."""

    formalize_template: str = """Natural language version: {informal_statement}

Translate to Lean 4:
```lean4
import Mathlib

theorem formalized"""

                                     
    informalize_system: str = """You are an expert at translating formal Lean 4 mathematics into natural language.
Given a Lean 4 theorem statement, output a clear natural language description of what the theorem states."""

    informalize_template: str = """Translate this Lean 4 statement to natural language:

```lean4
{formal_statement}
```

Natural language version:"""


@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    cycle_consistency: CycleConsistencyConfig = field(default_factory=CycleConsistencyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

                     
    verbose: bool = True
    save_all_candidates: bool = True


                         
DEFAULT_CONFIG = Config()
