# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers==5.12.1",
#   "peft",
#   "datasets",
#   "huggingface_hub",
#   "hf_transfer",
# ]
# ///
"""
rescore.py — re-evaluate run #1's EXISTING checkpoint with the CORRECT metric. No retraining.

Run #1 reported preference accuracy with the wrong ruler: absolute sum-of-token-logprob,
no reference model. DPO optimizes the REFERENCE-RELATIVE implicit reward, so the right
held-out metric is DPO implicit-reward accuracy:

    s_chosen   = logp_policy(chosen)   - logp_ref(chosen)
    s_rejected = logp_policy(rejected) - logp_ref(rejected)
    correct iff s_chosen > s_rejected

policy = base + the published LoRA adapter (enabled); ref = the SAME model with the adapter
DISABLED (the exact frozen reference TRL used in training). Length cancels because each
sequence pays the same length tax under policy and ref.

We report three numbers on the identical held-out slice run #1 used:
  1. OLD absolute sum-logprob accuracy   (should reproduce ~0.49 -> proves the artifact)
  2. NEW implicit-reward accuracy         (the correct headline)
  3. length-normalized per-token accuracy (robustness column)

Run:  hf jobs uv run src/rescore.py --flavor a100-large --secrets HF_TOKEN
"""
import os
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
ADAPTER = os.environ.get("ADAPTER_REPO", "yavuz-ai/qwen3-4b-dpo-ultrafeedback-lora")
N_EVAL = int(os.environ.get("N_EVAL", "500"))
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def logprob(model, tok, prompt_msgs, response_text):
    enc = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"]
    resp_ids = tok(response_text, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([prompt_ids, resp_ids], dim=1)[:, :MAX_LEN].to(model.device)
    logits = model(input_ids).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    n_prompt = prompt_ids.shape[1]
    resp_lp = tok_lp[:, n_prompt - 1:]
    return resp_lp.sum().item(), resp_lp.shape[1]   # (sum logprob, n response tokens)


def main():
    print(f"device={DEVICE} base={BASE} adapter={ADAPTER} n_eval={N_EVAL}")
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEVICE)
    model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE)   # adapter ENABLED = policy
    model.eval()

    eval_rows = load_dataset("trl-lib/ultrafeedback_binarized", split=f"train[5000:{5000+N_EVAL}]")

    old_correct = new_correct = norm_correct = 0
    margins, adapter_moved = [], 0
    for row in eval_rows:
        pm = row["chosen"][:-1]
        c, r = row["chosen"][-1]["content"], row["rejected"][-1]["content"]
        # policy (adapter on)
        pc, lc = logprob(model, tok, pm, c)
        pr, lr = logprob(model, tok, pm, r)
        # reference (adapter off) — the exact training reference
        with model.disable_adapter():
            rc, _ = logprob(model, tok, pm, c)
            rr, _ = logprob(model, tok, pm, r)
        s_c, s_r = pc - rc, pr - rr           # implicit rewards (length-canceling)
        new_correct += int(s_c > s_r)
        old_correct += int(pc > pr)            # the broken metric (no reference)
        norm_correct += int((pc / max(lc, 1)) > (pr / max(lr, 1)))
        margins.append(s_c - s_r)
        adapter_moved += int(abs(pc - rc) > 1e-3)   # sanity: adapter actually changes logprob

    n = len(eval_rows)
    mean_margin = sum(margins) / n
    print(f"\nheld-out pairs: {n}  (adapter changed logprob on {adapter_moved}/{n} — sanity)")
    print(f"1) OLD absolute sum-logprob acc : {old_correct/n:.3f}   (run #1 reported ~0.498)")
    print(f"2) NEW implicit-reward acc      : {new_correct/n:.3f}   <-- correct headline")
    print(f"3) length-normalized acc        : {norm_correct/n:.3f}   (robustness)")
    print(f"   mean implicit-reward margin  : {mean_margin:+.3f}")
    print("RESCORE " + str({"old_abs_acc": round(old_correct/n,4), "implicit_reward_acc": round(new_correct/n,4),
                            "len_norm_acc": round(norm_correct/n,4), "mean_implicit_margin": round(mean_margin,4), "n": n}))


if __name__ == "__main__":
    main()
