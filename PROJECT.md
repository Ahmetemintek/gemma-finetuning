# Project: QLoRA Fine-Tuning of Turkish-Gemma-9B for Medical QA

## 1. Project Overview

This project demonstrates parameter-efficient fine-tuning (QLoRA) of a 9B-parameter Turkish large language model on a curated Turkish medical question-answering dataset.

The goal is to showcase:

- Domain adaptation using QLoRA
- Task alignment for generative QA
- Proper evaluation methodology
- Clear documentation of limitations
- Clean, reproducible LLM training pipeline

This is a portfolio-oriented applied LLM project.

---

## 2. Base Model

Model:
https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1

Model Type:
- Causal Language Model
- Turkish-focused
- ~9B parameters

We will NOT modify the base weights directly.
We will use QLoRA for parameter-efficient fine-tuning.

---

## 3. Dataset

Dataset:
https://www.kaggle.com/datasets/incidelen/medturkquad

Dataset Type:
- Turkish Medical QA
- Originally extractive QA (SQuAD-style)

We will convert it into generative instruction format:

Format example:

Soru: <question>
Cevap: <answer>

The model will be trained to generate the answer autoregressively.

---

## 4. Objectives

### Primary Objective
Demonstrate effective domain adaptation via QLoRA.

### Secondary Objectives
- Compare base vs fine-tuned model
- Measure EM and F1
- Perform qualitative error analysis
- Document limitations and risks

---

## 5. Technical Stack

Environment:
- Google Colab (GPU required: T4 16GB minimum)

Core Libraries:
- transformers
- peft
- bitsandbytes
- accelerate
- datasets
- evaluate
- torch

Optional:
- wandb (if logging needed)
- matplotlib (for simple visualizations)

---

## 6. Training Strategy

Method:
- QLoRA (4-bit quantization)
- LoRA adapters applied to attention and MLP projection layers

Quantization:
- 4-bit NF4
- Double quantization enabled

LoRA Targets:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

Training:
- Instruction-style generative fine-tuning
- Causal LM objective
- Small batch size with gradient accumulation
- 1–2 epochs (portfolio demonstration)

We are NOT:
- Training from scratch
- Modifying model architecture
- Adding new heads

---

## 7. Evaluation Plan

We will evaluate:

### Automatic Metrics
- Exact Match (EM)
- Token-level F1

### Controlled Comparison
- Base model vs Fine-tuned model
- Same evaluation prompts
- Structured comparison table

### Qualitative Analysis
- Hallucination analysis
- Terminology correctness
- Failure cases
- Overfitting signals

---

## 8. Repository Structure

root/
│
├── project.md
├── requirements.txt
├── config/
│   └── training_config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_processing.py
│   ├── formatting.py
│   ├── model_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── notebooks/
│   └── eda.ipynb
|   └── colab_training.ipynb
|
└── outputs/
    ├── checkpoints/
    └── evaluation_results/

All scripts must be modular and reusable.
No monolithic notebook-only pipelines.

---

## 9. Engineering Principles

1. Reproducibility first.
2. Deterministic seeds where possible.
3. Modular functions.
4. Clear separation of concerns:
   - Data processing
   - Model loading
   - Training
   - Evaluation
5. No hardcoded paths.
6. Config-driven parameters.
7. Minimal but clean logging.

---

## 10. AI-Assisted Coding Guidelines (Cursor)

When generating code:

- Always explain architectural decisions briefly in comments.
- Prefer clarity over cleverness.
- Avoid unnecessary abstractions.
- Keep functions short and focused.
- Include docstrings.
- Validate assumptions (assert shapes, check keys).
- Fail loudly, not silently.
- Use typing hints where reasonable.
- Avoid deprecated APIs.
- Follow HuggingFace best practices.

Never:
- Mix data preprocessing inside training loop.
- Hardcode GPU assumptions.
- Skip evaluation.

If uncertain about API usage:
- Reference official HuggingFace documentation patterns.

---

## 11. Constraints

- Limited GPU memory (T4 16GB)
- Must use QLoRA
- Training time must be reasonable (<3 hours)
- Project should remain minimal but rigorous

---

## 12. Out of Scope

- Full hyperparameter search
- RLHF / DPO
- Production deployment
- Multi-GPU training
- Full-scale medical validation

---

## 13. Definition of Done

Project is complete when:

- Model successfully fine-tuned using QLoRA
- Evaluation metrics computed
- Base vs fine-tuned comparison documented
- Clear limitations section written
- Repository is clean and reproducible
- Results suitable for portfolio presentation

---

## 14. Expected Deliverables

- Fine-tuned LoRA adapter
- Evaluation report
- Clean GitHub repository
- Short technical write-up explaining:
  - Why QLoRA
  - Why generative QA format
  - Observed improvements
  - Failure cases
