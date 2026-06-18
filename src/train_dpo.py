# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers==5.12.1",
#   "trl==1.6.0",
#   "datasets",
#   "peft",
#   "accelerate",
#   "huggingface_hub",
#   "hf_transfer",
# ]
# ///
"""
train_dpo.py — the first REAL run (P1). Built to run on HF Jobs GPU via:

    hf jobs uv run src/train_dpo.py --flavor a100-large --secrets HF_TOKEN

It does three things and then publishes:
  1. eval BEFORE: on a held-out slice, how often does the BASE model already prefer the
     chosen answer over the rejected one? (preference accuracy = the DPO-native metric)
  2. train: LoRA DPO on a UltraFeedback subset, with the recipe corrected by finding #1
     (sane LR + warmup + cosine + gradient accumulation to cut batch-size-1 noise).
  3. eval AFTER: same held-out slice, now with the trained adapter. The delta is the result.
  4. push the adapter + a results.json + a short card to the Hub.

Everything is env-overridable so the SAME script runs the Mistral leg (cross-family) by
changing BASE_MODEL + OUTPUT_REPO. Nothing here assumes a specific family.

Why LoRA: a 4B full fine-tune needs an optimizer state far larger than the weights. LoRA
trains a small adapter instead, so it fits one GPU cheaply. With PEFT, TRL uses the
adapter-disabled model as the frozen reference (ref_model=None) — no second copy in memory.
"""
import json
import os
from itertools import islice

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ---- config (env-overridable; defaults = the Qwen leg of run #1) --------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DATASET = os.environ.get("DATASET", "trl-lib/ultrafeedback_binarized")
N_TRAIN = int(os.environ.get("N_TRAIN", "5000"))   # training pairs (1 epoch over this slice)
N_EVAL = int(os.environ.get("N_EVAL", "500"))      # held-out pairs for before/after accuracy
LR = float(os.environ.get("LR", "5e-5"))           # LoRA tolerates a higher LR than full FT
BETA = float(os.environ.get("BETA", "0.1"))        # DPO strength
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "")    # e.g. yavuz-ai/qwen3-4b-dpo-ultrafeedback-lora
SEED = int(os.environ.get("SEED", "42"))

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def split_pair(row):
    """UltraFeedback binarized rows are [user, assistant] lists. Return (prompt_msgs, chosen, rejected)."""
    prompt_msgs = row["chosen"][:-1]            # the shared user turn(s)
    chosen = row["chosen"][-1]["content"]       # assistant answer (preferred)
    rejected = row["rejected"][-1]["content"]   # assistant answer (dispreferred)
    return prompt_msgs, chosen, rejected


@torch.no_grad()
def seq_logprob(model, tok, prompt_msgs, response_text):
    """Total log-probability the model assigns to `response_text` after `prompt_msgs`."""
    prompt_ids = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, return_tensors="pt")
    resp_ids = tok(response_text, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([prompt_ids, resp_ids], dim=1)[:, :MAX_LEN].to(model.device)
    logits = model(input_ids).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    n_prompt = prompt_ids.shape[1]
    return token_lp[:, n_prompt - 1:].sum().item()   # sum over response tokens only


def pref_accuracy(model, tok, eval_rows):
    """Fraction of held-out pairs where the model scores chosen above rejected, + mean margin."""
    model.eval()
    wins, margins = 0, []
    for row in eval_rows:
        prompt_msgs, chosen, rejected = split_pair(row)
        lp_c = seq_logprob(model, tok, prompt_msgs, chosen)
        lp_r = seq_logprob(model, tok, prompt_msgs, rejected)
        wins += int(lp_c > lp_r)
        margins.append(lp_c - lp_r)
    return wins / len(eval_rows), sum(margins) / len(margins)


def main():
    print(f"device={DEVICE} base={BASE_MODEL} n_train={N_TRAIN} n_eval={N_EVAL} lr={LR} beta={BETA}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to(DEVICE)

    # stream the slice we need; hold out the LAST N_EVAL pairs so train/eval never overlap
    stream = load_dataset(DATASET, split="train", streaming=True)
    rows = list(islice(stream, N_TRAIN + N_EVAL))
    train_rows, eval_rows = rows[:N_TRAIN], rows[N_TRAIN:]
    train_ds = Dataset.from_list([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in train_rows])
    print(f"loaded {len(train_rows)} train + {len(eval_rows)} held-out eval pairs")

    # 1) BEFORE: base-model preference accuracy on the held-out set
    acc_before, marg_before = pref_accuracy(model, tok, eval_rows)
    print(f"BEFORE  pref_acc={acc_before:.3f}  mean_margin={marg_before:.3f}")

    # 2) TRAIN: LoRA DPO with the finding-#1-corrected recipe
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    args = DPOConfig(
        output_dir="out/dpo", num_train_epochs=1,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,  # effective batch 16
        learning_rate=LR, warmup_ratio=0.1, lr_scheduler_type="cosine", beta=BETA,
        max_length=MAX_LEN, bf16=True, gradient_checkpointing=True,
        logging_steps=10, save_strategy="no", report_to="none", seed=SEED,
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=train_ds, processing_class=tok, peft_config=lora,
    )
    trainer.train()

    # 3) AFTER: same held-out set, now with the trained adapter live
    acc_after, marg_after = pref_accuracy(trainer.model, tok, eval_rows)
    print(f"AFTER   pref_acc={acc_after:.3f}  mean_margin={marg_after:.3f}")
    print(f"DELTA   pref_acc {acc_after - acc_before:+.3f}   mean_margin {marg_after - marg_before:+.3f}")

    results = {
        "base_model": BASE_MODEL, "dataset": DATASET,
        "n_train": N_TRAIN, "n_eval": N_EVAL, "lr": LR, "beta": BETA, "seed": SEED,
        "pref_acc_before": round(acc_before, 4), "pref_acc_after": round(acc_after, 4),
        "pref_acc_delta": round(acc_after - acc_before, 4),
        "mean_margin_before": round(marg_before, 4), "mean_margin_after": round(marg_after, 4),
    }
    os.makedirs("out/dpo", exist_ok=True)
    with open("out/dpo/results.json", "w") as f:
        json.dump(results, f, indent=2)
    trainer.save_model("out/dpo")

    # 4) PUBLISH (only if a repo was named and a write token is present)
    if OUTPUT_REPO:
        api = HfApi()
        api.create_repo(OUTPUT_REPO, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path="out/dpo", repo_id=OUTPUT_REPO, repo_type="model")
        print(f"pushed adapter + results to https://huggingface.co/{OUTPUT_REPO}")
    print("RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main()
