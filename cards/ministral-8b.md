---
license: other
license_name: mrl
base_model: mistralai/Ministral-8B-Instruct-2410
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

# ministral-8b-dpo-ultrafeedback-lora

LoRA DPO adapter for Ministral-8B-Instruct-2410, trained on UltraFeedback. one leg of
[`crossfamily-dpo`](https://github.com/shipbehaves/crossfamily-dpo): the same recipe run across
two model families (Qwen + Mistral), writeup first.

inherits the Mistral Research License (research / non-commercial) from the base model.

## result

held-out preference accuracy, measured with the DPO implicit-reward metric (reference-relative,
so length cancels). n = 500 UltraFeedback pairs, disjoint from training.

| metric | value |
|---|---|
| implicit-reward accuracy | **0.672** |
| old absolute-logprob metric (broken baseline) | 0.514 |
| length-normalized accuracy | 0.640 |
| mean implicit-reward margin | +6.53 |

same recipe on Qwen3-4B-Instruct lands 0.666, so the effect holds across families.

## the failure analysis (the point)

the first run of this pipeline (on the Qwen leg) read 0.498 and looked like a null. it was a
measurement bug, not a training bug: it scored absolute log-probability, but DPO optimizes the
reference-relative implicit reward `logp_policy(y) - logp_ref(y)`. the corrected metric is what is
reported above. full teardown in the repo's `NOTES.md`.

## training

- method: LoRA DPO (TRL); r=16, alpha=32, dropout=0.05, target_modules=all-linear
- data: trl-lib/ultrafeedback_binarized; 5000 pairs; 1 epoch
- optim: lr=5e-5, cosine, warmup 0.1, beta=0.1; effective batch 16; bf16; seed=42
- reference: adapter-disabled base (ref_model=None), no second model in memory
- hardware: 1x A100 (HF Jobs)
- note: transformers loads Ministral via the `mistral` class; verified all weight shards load with
  no re-initialization, so the warning is a benign config-type alias.

## use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
model = PeftModel.from_pretrained(base, "yavuz-ai/ministral-8b-dpo-ultrafeedback-lora")
tok = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
```

## reproduce

```
hf jobs uv run src/train_dpo.py --flavor a100-large --secrets HF_TOKEN \
  --env BASE_MODEL=mistralai/Ministral-8B-Instruct-2410 \
  --env OUTPUT_REPO=yavuz-ai/ministral-8b-dpo-ultrafeedback-lora
```

## scope

the base is already preference-aligned, so this is a clean reference-relative preference lift on
held-out data, not a state-of-the-art model. the value is the method and the failure analysis,
reproduced identically on a Qwen model: [qwen3-4b-dpo-ultrafeedback-lora](https://huggingface.co/yavuz-ai/qwen3-4b-dpo-ultrafeedback-lora).
