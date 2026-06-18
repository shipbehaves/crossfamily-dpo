"""
smoke.py — the walking skeleton for P1.

Goal: prove the ENTIRE DPO loop runs end to end (load model -> load preference pairs ->
a few DPO steps -> save), on a tiny model, for free, on the laptop. This is NOT a real
experiment. It exists to catch plumbing bugs before we spend a cent on HF Jobs.

Run:  cd ~/Projects/crossfamily-dpo && uv run python src/smoke.py
Expect: it downloads a ~0.5B model once, runs 8 DPO steps on 8 toy pairs, prints "SMOKE OK".
If it errors, that error IS the first lesson (we read it and fix it; that is the job).
"""
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # tiny on purpose; the real run uses Qwen3-4B / a Mistral
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "smoke_prefs.jsonl")

def main():
    # 1. the model we will nudge, and its tokenizer (turns text into tokens)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # 2. the preference data: each row is (prompt, chosen, rejected). DPO teaches the model
    #    to make "chosen" more likely and "rejected" less likely, relative to a frozen copy.
    ds = load_dataset("json", data_files=DATA, split="train")

    # 3. tiny DPO config: 8 steps, batch 1, no GPU assumptions. This only proves the loop.
    args = DPOConfig(
        output_dir="out/smoke",
        max_steps=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=1,
        report_to="none",
        bf16=False, fp16=False,           # CPU-safe for the smoke
        max_length=256,                   # TRL 1.6 dropped max_prompt_length; max_length covers it
    )

    # 4. the trainer builds its own frozen reference model automatically.
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model("out/smoke")
    print("\nSMOKE OK — the DPO loop ran end to end and saved to out/smoke")

if __name__ == "__main__":
    main()
