# crossfamily-dpo

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
`uv` + `uv.lock` pin every dependency. Seeds fixed. Artifacts pushed to the Hub. Each step is a
single documented command (see the reproduce lines under each result).

## Result (2026-06-18)

Same LoRA DPO recipe on UltraFeedback, two families, measured with the **correct DPO metric**
(reference-relative implicit-reward accuracy, see below). DPO lifts held-out preference accuracy
from chance to ~0.67 on both:

| model | implicit-reward acc | old absolute metric (broken) | length-norm | margin |
|---|---|---|---|---|
| [Qwen3-4B-Instruct](https://huggingface.co/yavuz-ai/qwen3-4b-dpo-ultrafeedback-lora) | **0.666** | 0.498 | 0.592 | +6.30 |
| [Ministral-8B-Instruct](https://huggingface.co/yavuz-ai/ministral-8b-dpo-ultrafeedback-lora) | **0.672** | 0.514 | 0.640 | +6.53 |

### The failure behind the result

Run #1 first looked like a **total null** (0.498, no movement). It wasn't. The eval measured the
policy's *absolute* sum-of-token-logprob, but DPO optimizes the *reference-relative* implicit
reward `logp_policy(y) - logp_ref(y)`. Scoring the wrong quantity buried a real effect under length
and content variance. Re-scoring the **unchanged** run #1 checkpoint with the correct metric
(policy = LoRA on, reference = LoRA off) moved it 0.498 -> 0.666, with no retraining (~$1). The
training margin had been moving the right way (+6.4 nats) the whole time. Full teardown, including
a length-confound hypothesis that did **not** survive the data, in [`NOTES.md`](./NOTES.md).

**Lesson:** match the eval to the training objective. On held-out data the DPO objective *is*
reward accuracy, not absolute likelihood.

## Status
Both legs trained, evaluated, and published to the Hub. Writeup in `NOTES.md` (findings #1-#2).
Reproduce: `hf jobs uv run src/train_dpo.py --flavor a100-large --secrets HF_TOKEN --env BASE_MODEL=<model> --env OUTPUT_REPO=<repo>`.
