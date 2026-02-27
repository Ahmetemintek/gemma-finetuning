"""
Evaluation script for QLoRA fine-tuned Turkish-Gemma-9B on medical QA.

Loads the base model in 4-bit, applies saved LoRA adapter,
runs greedy inference on the validation set, and computes EM / F1.

"""

import yaml
import torch
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


# ── Step 3: Load base model (4-bit) + LoRA adapter ──────────────────────────

print("Loading quantized model (4-bit)...")
model = load_quantized_model(MODEL_NAME)

print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()


# ── Step 4: Metrics ──────────────────────────────────────────────────────────

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


# ── Step 5: Run inference and evaluate ───────────────────────────────────────

print("Running inference on validation set...")

em_scores = []
f1_scores = []
samples = []

for i, example in enumerate(tqdm(val_dataset, desc="Evaluating")):
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

    em = exact_match(prediction, reference)
    f1 = token_f1(prediction, reference)
    em_scores.append(em)
    f1_scores.append(f1)

    if i < NUM_SAMPLES_TO_PRINT:
        samples.append((example["question"], reference, prediction))


# ── Step 6: Print results ────────────────────────────────────────────────────

avg_em = sum(em_scores) / len(em_scores) * 100
avg_f1 = sum(f1_scores) / len(f1_scores) * 100

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Validation samples : {len(val_dataset)}")
print(f"Exact Match (EM)   : {avg_em:.2f}%")
print(f"Token F1           : {avg_f1:.2f}%")
print("=" * 60)

print(f"\nSample predictions ({NUM_SAMPLES_TO_PRINT}):\n")
for i, (question, reference, prediction) in enumerate(samples):
    print(f"--- Sample {i + 1} ---")
    print(f"  Soru      : {question[:100]}")
    print(f"  Reference : {reference[:100]}")
    print(f"  Predicted : {prediction[:100]}")
    print()
