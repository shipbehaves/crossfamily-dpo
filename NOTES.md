# Lab notebook — crossfamily-dpo

Running log of findings. The writeup leads with these, not with the happy path.

---

## Finding 1 — DPO diverges at an SFT-scale learning rate (2026-06-18)

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

**Cost of this lesson:** $0. Caught on the laptop before any paid GPU run. That is the point of
the failure-first smoke.
