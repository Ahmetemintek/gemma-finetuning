"""
Push LoRA adapter and tokenizer to Hugging Face Hub.

Uploads only the adapter weights — not the full model.
The base model must be loaded separately by the user.
"""

import os

from huggingface_hub import HfApi, login


# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
ADAPTER_DIR = "outputs/checkpoints"
REPO_ID = "Ahmetemintek/turkish-gemma-9b-medical-qlora"

BASE_MODEL = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
DATASET = "MedTurkQuAD"


# ── Step 1: Authenticate ─────────────────────────────────────────────────────

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set.")

login(token=HF_TOKEN)
print("Authenticated with Hugging Face Hub.")


# ── Step 2: Create repository ────────────────────────────────────────────────

api = HfApi()
api.create_repo(repo_id=REPO_ID, exist_ok=True)
print(f"Repository ready: https://huggingface.co/{REPO_ID}")


# ── Step 3: Generate model card ──────────────────────────────────────────────

MODEL_CARD = f"""\
---
library_name: peft
base_model: {BASE_MODEL}
tags:
  - qlora
  - medical-qa
  - turkish
  - gemma
  - lora
license: apache-2.0
language:
  - tr
datasets:
  - {DATASET}
---

# Turkish-Gemma-9B Medical QA (QLoRA Adapter)

Parameter-efficient fine-tuned **LoRA adapter** for medical question answering in Turkish.

**+50 EM and +49 F1 improvement using QLoRA with only 1.05% trainable parameters.**

## Model Details

| Property | Value |
|---|---|
| Base model | [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) |
| Method | QLoRA (4-bit NF4, double quantization) |
| Compute dtype | float16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 54M / 5.1B (1.05%) |
| Dataset | {DATASET} |
| Epochs | 2 |
| Optimizer | paged_adamw_8bit |

## Evaluation Results

| Metric | Base Model | Fine-tuned (LoRA) | Delta |
|---|---|---|---|
| Exact Match (EM) | 4.63% | 54.76% | +50.12% |
| Token F1 | 25.80% | 75.39% | +49.58% |

Evaluated on 820 validation samples with greedy decoding.

## Usage

This repository contains only the LoRA adapter weights. Load the base model separately:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_name = "{BASE_MODEL}"
adapter_name = "{REPO_ID}"

tokenizer = AutoTokenizer.from_pretrained(adapter_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, adapter_name)
model.eval()
```

## Prompt Format

```
Bağlam:
{{context}}

Soru:
{{question}}

Cevap:
```
"""

model_card_path = os.path.join(ADAPTER_DIR, "README.md")
with open(model_card_path, "w") as f:
    f.write(MODEL_CARD)
print("Model card generated.")


# ── Step 4: Upload adapter + tokenizer ───────────────────────────────────────

api.upload_folder(
    folder_path=ADAPTER_DIR,
    repo_id=REPO_ID,
    ignore_patterns=["checkpoint-*", "optimizer.pt", "scheduler.pt",
                     "scaler.pt", "rng_state.pth", "training_args.bin"],
)

print(f"\nAdapter pushed to: https://huggingface.co/{REPO_ID}")
