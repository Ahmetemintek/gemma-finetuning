"""
Qualitative inference comparison: base model vs fine-tuned LoRA model.

Runs the same demo prompts through both models to visually demonstrate
the effect of QLoRA fine-tuning on medical QA behavior.

"""

import gc

import yaml
import torch
import pandas as pd
from peft import PeftModel

from model_utils import load_tokenizer, load_quantized_model
from formatting import format_prompt


# ── Configuration ─────────────────────────────────────────────────────────────

with open("config/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
ADAPTER_DIR = config["training"]["output_dir"]
MAX_NEW_TOKENS = 128


# ── Demo examples ────────────────────────────────────────────────────────────

EXAMPLES = [
    {
        "context": (
            "Verem, Mycobacterium tuberculosis adlı bakteri tarafından neden olunan "
            "bakteriyel ve bulaşıcı bir hastalıktır. Verem hastasının çevreye tükürdüğü "
            "balgamı ya da öksürdüğünde saçılan basil yüklü damlacıklarla bulaşır. "
            "Belirtileri arasında ateş, titreme, gece terlemesi, iştahsızlık, kilo kaybı "
            "ve yorgunluk sayılabilir."
        ),
        "question": "Veremin belirtileri nelerdir?",
    },
    {
        "context": (
            "Tip 2 diyabet, vücudun insülini etkili kullanamadığı kronik bir metabolizma "
            "hastalığıdır. Tedavide yaşam tarzı değişiklikleri, oral antidiyabetik ilaçlar "
            "ve gerektiğinde insülin tedavisi uygulanır. Kontrol altına alınmazsa böbrek "
            "yetmezliği, görme kaybı ve kardiyovasküler hastalıklara yol açabilir."
        ),
        "question": "Tip 2 diyabet kontrol altına alınmazsa hangi komplikasyonlara yol açabilir?",
    },
    {
        "context": (
            "Astım, hava yollarının kronik iltihaplanması sonucu oluşan bir solunum yolu "
            "hastalığıdır. Nöbetler sırasında hırıltılı solunum, nefes darlığı, göğüste "
            "sıkışma hissi ve öksürük görülür. Tedavide inhaler kortikosteroidler ve "
            "bronkodilatörler kullanılır."
        ),
        "question": "Astım tedavisinde hangi ilaçlar kullanılır?",
    },
]


# ── Inference helper ─────────────────────────────────────────────────────────

def generate_answers(model, examples):
    """Generate answers for all examples using greedy decoding."""
    answers = []
    for ex in examples:
        prompt = format_prompt(ex["context"], ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        answers.append(answer)
    return answers


# ── Step 1: Load tokenizer ───────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = load_tokenizer(MODEL_NAME)


# ── Step 2: Evaluate base model ──────────────────────────────────────────────

print("Loading quantized model (4-bit)...")
base_model = load_quantized_model(MODEL_NAME)
base_model.eval()

print("Generating base model answers...")
base_answers = generate_answers(base_model, EXAMPLES)

del base_model
gc.collect()
torch.cuda.empty_cache()


# ── Step 3: Evaluate fine-tuned model ────────────────────────────────────────

print("Loading quantized model (4-bit)...")
ft_model = load_quantized_model(MODEL_NAME)

print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
ft_model.eval()

print("Generating fine-tuned model answers...")
ft_answers = generate_answers(ft_model, EXAMPLES)

del ft_model
gc.collect()
torch.cuda.empty_cache()


# ── Step 4: Build comparison table ────────────────────────────────────────────

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

comparison_df = pd.DataFrame({
    "Context": [ex["context"] for ex in EXAMPLES],
    "Question": [ex["question"] for ex in EXAMPLES],
    "Base Model Answer": base_answers,
    "Fine-tuned Answer": ft_answers,
})

print("\nQUALITATIVE COMPARISON: BASE vs FINE-TUNED (QLoRA)\n")
print(comparison_df.to_string(index=False))
