---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
datasets:
  - trl-lib/ultrafeedback_binarized
tags:
  - dpo
  - lora
  - trl
  - preference-optimization
pipeline_tag: text-generation
---

# qwen3-4b-dpo-ultrafeedback-lora

LoRA DPO adapter for Qwen3-4B-Instruct-2507, trained on UltraFeedback. one leg of
[`crossfamily-dpo`](https://github.com/shipbehaves/crossfamily-dpo): the same recipe run across
two model families (Qwen + Mistral), writeup first.

## result

held-out preference accuracy, measured with the DPO implicit-reward metric (reference-relative,
so length cancels). n = 500 UltraFeedback pairs, disjoint from training.

| metric | value |
|---|---|
| implicit-reward accuracy | **0.666** |
| old absolute-logprob metric (broken baseline) | 0.498 |
| length-normalized accuracy | 0.592 |
| mean implicit-reward margin | +6.30 |

## the failure analysis (the point)

the first eval read 0.498 and looked like a null. it was a measurement bug, not a training bug. it
scored absolute log-probability, but DPO optimizes the reference-relative implicit reward
`logp_policy(y) - logp_ref(y)`. re-scoring the unchanged checkpoint with the correct metric moved it
0.498 -> 0.666 with no retraining. full teardown (incl. a length-confound hypothesis that did not
survive the data) in the repo's `NOTES.md`.

## training

- method: LoRA DPO (TRL); r=16, alpha=32, dropout=0.05, target_modules=all-linear
- data: trl-lib/ultrafeedback_binarized; 5000 pairs; 1 epoch
- optim: lr=5e-5, cosine, warmup 0.1, beta=0.1; effective batch 16; bf16; seed=42
- reference: adapter-disabled base (ref_model=None), no second model in memory
- hardware: 1x A100 (HF Jobs)

## use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "yavuz-ai/qwen3-4b-dpo-ultrafeedback-lora")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
```

## reproduce

```
hf jobs uv run src/train_dpo.py --flavor a100-large --secrets HF_TOKEN \
  --env BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
  --env OUTPUT_REPO=yavuz-ai/qwen3-4b-dpo-ultrafeedback-lora
```

## scope

the base is already preference-aligned, so this is a clean reference-relative preference lift on
held-out data, not a state-of-the-art model. the value is the method and the failure analysis,
reproduced identically on a Mistral model: [ministral-8b-dpo-ultrafeedback-lora](https://huggingface.co/yavuz-ai/ministral-8b-dpo-ultrafeedback-lora).
