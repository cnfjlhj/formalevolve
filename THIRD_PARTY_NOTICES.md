# Third-party notices

FormalEvolve contains adapted software and benchmark snapshots from the projects below. The upstream license texts are preserved in [`third_party/licenses/`](third_party/licenses/).

## ShinkaEvolve

- Upstream: https://github.com/SakanaAI/ShinkaEvolve
- Upstream component: the evolutionary-search engine adapted under `shinka/`
- Copyright: Copyright 2025 Sakana AI
- License: Apache License 2.0
- Local license copy: [`third_party/licenses/ShinkaEvolve-LICENSE`](third_party/licenses/ShinkaEvolve-LICENSE)

FormalEvolve modifies and extends this component for Lean autoformalization, symbolic AST rewriting, fixed-budget search, semantic filtering, and paper-specific evaluation.

## ProofNet

- Upstream: https://github.com/zhangir-azerbayev/proofnet
- Upstream component: benchmark items adapted into the ProofNet JSONL snapshots under `examples/formalevolve_autoformalization/benchmark/`
- Copyright: Copyright (c) 2022 zhangir-azerbayev
- License: MIT License
- Local license copy: [`third_party/licenses/ProofNet-LICENSE`](third_party/licenses/ProofNet-LICENSE)

The bundled files may contain local preprocessing and Lean-version adaptations. They are not presented as an unmodified upstream release.

## CombiBench

- Upstream: https://github.com/MoonshotAI/CombiBench
- Upstream component: benchmark items adapted into the CombiBench JSONL snapshots under `examples/formalevolve_autoformalization/benchmark/`
- Copyright: Copyright (c) 2025 Moonshot AI and Project Numina
- License: MIT License
- Local license copy: [`third_party/licenses/CombiBench-LICENSE`](third_party/licenses/CombiBench-LICENSE)

The bundled files may contain local preprocessing and Lean-version adaptations. They are not presented as an unmodified upstream release.
