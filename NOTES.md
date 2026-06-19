# Lab notebook - crossfamily-dpo

Running log of findings. The writeup leads with these, not with the happy path.

---

## Finding 1 - DPO diverges at an SFT-scale learning rate (2026-06-18)

**Where:** `src/smoke_real.py`, real UltraFeedback slice (24 pairs), Qwen2.5-0.5B, 12 steps, batch 1, CPU/MPS.

**Symptom:** with `learning_rate=5e-5` the run did not just train badly, it diverged. Loss
swung from ~0 to ~6.8; reward margins went from +10.5 (model slams chosen up) to -6.8 (model
flips to preferring the rejected answer). Final train_loss 1.78.

**Diagnosis:** `5e-5` is an SFT-scale rate. DPO optimises a log-sigmoid of the reward margin,
which is far more sensitive: large updates overshoot, and with batch size 1 every step chases a
single noisy example, so the model whiplashes between preferring chosen and rejected.

**Fix + proof (one variable changed):** dropped LR 10x to `5e-6` and added `warmup_ratio=0.1`.
Loss stayed in a 0.43-0.73 band, margins stayed small and mostly positive, final train_loss 0.63.
The explosion disappeared. That isolates the learning rate as the cause.

| step | loss (5e-5) | margin (5e-5) | loss (5e-6) | margin (5e-6) |
|---|---|---|---|---|
| 5  | 0.0009  | +7.0  | 0.68 | +0.02 |
| 9  | 0.00003 | +10.5 | 0.73 | -0.07 |
| 10 | 5.24    | -5.2  | 0.57 | +0.27 |
| 12 | 6.80    | -6.8  | 0.69 | +0.01 |

**Carry into the real run:** real DPO uses an even lower rate (~5e-7 for full fine-tuning of the
4B), warmup, and a larger effective batch via gradient accumulation to cut the batch-1 noise.
Watch reward margins and loss for the same divergence signature at scale.

**Cost of this lesson:** $0. Caught locally before any paid GPU run. That is the point of
the failure-first smoke.

---

## Finding 2 - a working DPO run read as a null under the wrong metric (2026-06-18)

**Where:** run #1 held-out eval (Qwen3-4B-Instruct, LoRA DPO, 5k UltraFeedback pairs), then
confirmed by re-score and cross-family replication.

**Symptom:** the first real run looked like a total null. Held-out "preference accuracy" went
0.490 -> 0.498 (no movement). First instinct was "DPO did nothing" or "the model has no headroom".

**False start (recorded on purpose):** I almost reported "length confound" as the cause. Checked
it: chosen answers are longer on average (+32.5 tokens) but longer in only 52.4% of pairs, so
length is a contributor, not the clean driver. The hypothesis did not survive contact with data.

**Diagnosis (correct):** the metric measured the wrong quantity. It used the policy's ABSOLUTE
sum-of-token-logprob (no reference model), but DPO never optimizes logp(chosen) > logp(rejected)
in absolute terms. It optimizes the REFERENCE-RELATIVE implicit reward:
`s(y) = logp_policy(y) - logp_ref(y)`, correct iff `s(chosen) > s(rejected)`. The absolute metric
leaves in ~25-30 nats of length/content variance that cancels in the implicit reward, burying the
real signal. The run's own training margin had moved +6.4 nats the right way the whole time.

**Fix + proof (no retraining):** re-scored the EXACT run #1 checkpoint with implicit-reward
accuracy (policy = adapter on, reference = adapter off). Result on the same held-out 500 pairs:

| metric | run #1 checkpoint |
|---|---|
| old absolute sum-logprob acc (broken) | 0.498 |
| **implicit-reward acc (correct)** | **0.666** |
| length-normalized acc (robustness) | 0.592 |
| mean implicit margin | +6.30 |

The reconstruction predicted it: `sigmoid(beta * margin) = sigmoid(0.1 * 6.4) ~= 0.65`. Cost: ~$1.

**Cross-family confirmation (same recipe, correct metric):**

| model | implicit-reward acc | old (broken) | length-norm | margin |
|---|---|---|---|---|
| Qwen3-4B-Instruct | 0.666 | 0.498 | 0.592 | +6.30 |
| Ministral-8B-Instruct | 0.672 | 0.514 | 0.640 | +6.53 |

DPO lifts preference accuracy from chance to ~0.67 on both families, and the broken metric shows
~0.50 on both. (Transformers warns it loads Ministral via the `mistral` class; verified all 327
weight shards load with no re-initialization, so the warning is a benign config-type alias.)

**Lesson:** match the eval to the training objective. On held-out data the DPO objective IS reward
accuracy (reference-relative), not absolute likelihood. A real effect can be fully present and
invisible under a plausible-looking but mismatched metric. The single highest-value exhibit is the
before/after re-score of one unchanged checkpoint: it isolates a measurement error from a training
error cleanly.
