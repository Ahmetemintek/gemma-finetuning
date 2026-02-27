"""
Evaluation script for QLoRA fine-tuned Turkish-Gemma-9B on medical QA.

Compares base model (no adapter) vs fine-tuned model (with LoRA)
on the validation set using greedy decoding, EM, and token-level F1.

"""

import gc

import yaml
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel

from model_utils import load_tokenizer, load_quantized_model
from formatting import format_prompt


# ── Configuration ─────────────────────────────────────────────────────────────

with open("config/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
ADAPTER_DIR = config["training"]["output_dir"]
VAL_FILE = config["data"]["val_file"]
MAX_NEW_TOKENS = 128
NUM_SAMPLES_TO_PRINT = 5


# ── Step 1: Load validation dataset ──────────────────────────────────────────

print("Loading validation dataset...")
val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")
print(f"Validation samples: {len(val_dataset)}")


# ── Step 2: Load tokenizer ───────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = load_tokenizer(MODEL_NAME)


# ── Step 3: Metrics ──────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Step 4: Inference loop ───────────────────────────────────────────────────

def run_evaluation(model, label: str):
    """Run greedy inference on validation set and return metrics + samples."""
    print(f"\nRunning inference: {label}...")

    em_scores = []
    f1_scores = []
    samples = []

    for i, example in enumerate(tqdm(val_dataset, desc=label)):
        prompt = format_prompt(example["context"], example["question"])
        reference = example["answer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        em_scores.append(exact_match(prediction, reference))
        f1_scores.append(token_f1(prediction, reference))

        if i < NUM_SAMPLES_TO_PRINT:
            samples.append((example["question"], reference, prediction))

    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    return avg_em, avg_f1, samples


# ── Step 5: Evaluate base model ──────────────────────────────────────────────

print("Loading quantized model (4-bit)...")
base_model = load_quantized_model(MODEL_NAME)
base_model.eval()

base_em, base_f1, base_samples = run_evaluation(base_model, "Base Model")

del base_model
gc.collect()
torch.cuda.empty_cache()


# ── Step 6: Evaluate fine-tuned model ────────────────────────────────────────

print("\nLoading quantized model (4-bit)...")
finetuned_model = load_quantized_model(MODEL_NAME)

print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
finetuned_model = PeftModel.from_pretrained(finetuned_model, ADAPTER_DIR)
finetuned_model.eval()

ft_em, ft_f1, ft_samples = run_evaluation(finetuned_model, "Fine-tuned Model")

del finetuned_model
gc.collect()
torch.cuda.empty_cache()


# ── Step 7: Print comparison ─────────────────────────────────────────────────

print(f"\nValidation samples: {len(val_dataset)}\n")

results_df = pd.DataFrame({
    "Metric": ["Exact Match (EM)", "Token F1"],
    "Base Model": [f"{base_em:.2f}%", f"{base_f1:.2f}%"],
    "Fine-tuned (LoRA)": [f"{ft_em:.2f}%", f"{ft_f1:.2f}%"],
    "Delta": [f"{ft_em - base_em:+.2f}%", f"{ft_f1 - base_f1:+.2f}%"],
})
print(results_df.to_string(index=False))

samples_df = pd.DataFrame(
    [
        {
            "Model": model_name,
            "Soru": q[:80],
            "Reference": r[:80],
            "Predicted": p[:80],
        }
        for model_name, sample_list in [("Base", base_samples), ("Fine-tuned", ft_samples)]
        for q, r, p in sample_list
    ]
)
print(f"\nSample predictions ({NUM_SAMPLES_TO_PRINT} per model):\n")
print(samples_df.to_string(index=False))
