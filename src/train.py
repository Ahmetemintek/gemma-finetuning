"""
QLoRA fine-tuning script for Turkish-Gemma-9B on medical QA.

Orchestrates: model loading, dataset processing, tokenization,
label masking, training, and LoRA adapter saving.

Run: python src/train.py
"""

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from model_utils import (
    load_tokenizer,
    load_quantized_model,
    apply_lora_adapters,
    print_trainable_parameters,
)
from formatting import format_prompt


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
OUTPUT_DIR = "outputs/checkpoints"
MAX_SEQ_LENGTH = 512
SEED = 42

# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# Logging and checkpointing
LOGGING_STEPS = 25
SAVE_STEPS = 100
EVAL_STEPS = 100


# ── Step 1: Load datasets ────────────────────────────────────────────────────

print("Loading datasets...")
train_dataset = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
val_dataset = load_dataset("json", data_files="data/processed/validation.jsonl", split="train")
print(f"Train: {len(train_dataset)} examples | Validation: {len(val_dataset)} examples")


# ── Step 2: Load tokenizer and quantized model ───────────────────────────────

print("Loading tokenizer...")
tokenizer = load_tokenizer(MODEL_NAME)

print("Loading quantized model (4-bit)...")
model = load_quantized_model(MODEL_NAME)


# ── Step 3: Apply LoRA adapters ───────────────────────────────────────────────

print("Applying LoRA adapters...")
model = apply_lora_adapters(model)
print_trainable_parameters(model)


# ── Step 4: Tokenize with label masking ───────────────────────────────────────

def tokenize_and_mask(example):
    """
    Tokenize full formatted text and mask prompt tokens in labels.

    Masking strategy:
    - Tokenize full text (prompt + answer + EOS)
    - Tokenize prompt only (up to and including "Cevap:\\n")
    - Set labels to -100 for all prompt tokens
    - Only answer tokens contribute to loss
    """
    full_text = example["formatted_text"] + tokenizer.eos_token
    prompt = format_prompt(example["context"], example["question"])

    # Tokenize both to find masking boundary
    full_enc = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH)
    prompt_enc = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH)

    prompt_len = len(prompt_enc["input_ids"])

    # Mask prompt tokens, keep answer tokens for loss
    labels = full_enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len

    full_enc["labels"] = labels
    return full_enc


print("Tokenizing datasets...")
train_dataset = train_dataset.map(
    tokenize_and_mask, remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    tokenize_and_mask, remove_columns=val_dataset.column_names
)


# ── Step 5: Data collator ────────────────────────────────────────────────────

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)


# ── Step 6: Training arguments ───────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
    dataloader_pin_memory=False,
)


# ── Step 7: Trainer ──────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)


# ── Step 8: Train ────────────────────────────────────────────────────────────

print("Starting training...")
trainer.train()


# ── Step 9: Save LoRA adapter and tokenizer ───────────────────────────────────

print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. LoRA adapter saved.")
