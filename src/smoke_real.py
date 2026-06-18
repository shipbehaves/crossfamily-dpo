"""
smoke_real.py — same walking skeleton, but on REAL preference data.

smoke.py proved the loop on 8 hand-written toy pairs. This proves the next unknown:
that a real, public preference dataset (UltraFeedback) loads, formats, and trains
through the SAME pipeline. Still tiny + free + on the laptop (small model, few steps,
a small streamed slice). The only thing changing vs smoke.py is the data source.

Data: trl-lib/ultrafeedback_binarized — conversational pairs (chosen/rejected are each
a [user, assistant] message list). TRL extracts the shared prompt and applies the chat
template automatically, so we pass the rows straight in.

Run:  cd ~/Projects/crossfamily-dpo && uv run python src/smoke_real.py
Expect: streams ~24 real pairs, runs ~12 DPO steps, prints "REAL-DATA SMOKE OK".
"""
import os
from itertools import islice
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # still tiny on purpose; the real run uses Qwen3-4B / a Mistral
N = 24                                 # a small streamed slice — enough to prove the data path

# DPO LESSON (learned from this very smoke): an SFT-scale learning rate (5e-5) makes DPO
# diverge — the loss explodes and reward margins flip negative. DPO wants a much lower rate
# plus warmup. These are env-overridable so we can demonstrate the failure vs the fix.
LR = float(os.environ.get("LR", "5e-6"))      # ~10x lower than the broken 5e-5; real 4B run goes lower still
WARMUP = float(os.environ.get("WARMUP", "0.1"))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # stream the first N real pairs (avoids downloading the whole dataset for a smoke)
    stream = load_dataset("trl-lib/ultrafeedback_binarized", split="train", streaming=True)
    rows = list(islice(stream, N))
    ds = Dataset.from_list([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    print(f"loaded {len(ds)} real UltraFeedback pairs (conversational format)")

    args = DPOConfig(
        output_dir="out/smoke_real",
        max_steps=12,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        warmup_ratio=WARMUP,
        logging_steps=1,
        report_to="none",
        bf16=False, fp16=False,           # CPU-safe for the smoke
        max_length=512,                   # real prompts are longer than the toy ones
    )

    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model("out/smoke_real")
    print("\nREAL-DATA SMOKE OK — UltraFeedback ran through the DPO loop and saved to out/smoke_real")

if __name__ == "__main__":
    main()
