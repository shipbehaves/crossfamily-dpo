# crossfamily-dpo (P1)

Eval-driven **DPO** on small open models, **across families** (Qwen + Mistral). The first real
post-training result. Public data, reproducible, and the writeup **leads with failure analysis**.

## The question
Take the same DPO recipe and run it on Qwen3-4B and a small Mistral. Does it transfer across
families? Where does each improve, where does each regress (especially honesty vs over-refusal)?

## Plan (small steps)
1. **smoke** (free, laptop): `uv run python src/smoke.py` proves the loop end to end on a 0.5B model.
2. **baseline eval**: measure the base models on IFEval + TruthfulQA before any training.
3. **real DPO** (HF Jobs, ~$5-15 each): DPO on Qwen3-4B and a small Mistral, multiple seeds.
4. **after eval + analysis**: deltas, regressions, what broke, what mattered.
5. **publish**: HF model + dataset + this writeup (failure-analysis first).

## Reproducibility
`uv` + `uv.lock` pin every dependency. Seeds fixed. Artifacts pushed to the Hub. `make`-style steps
documented so a stranger can re-run them.

## Status
Scaffolded 2026-06-18. Next: `uv sync`, then the smoke run.
