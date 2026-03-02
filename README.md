# QLoRA Fine-Tuning: Turkish-Gemma-9B for Medical QA

Parameter-efficient fine-tuning (QLoRA) of [Turkish-Gemma-9b-v0.1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1) on [MedTurkQuAD](https://www.kaggle.com/datasets/incidelen/medturkquad), a Turkish medical QA dataset, to adapt the model for generative question-answering in the medical domain.

**[🤗 Model on Hugging Face](https://huggingface.co/Ahmetemintek/turkish-gemma-9b-medical-qlora)**

---

## Overview

Traditional medical NLP pipelines often rely on multiple task-specific encoder models (e.g., separate NER, relation extraction, and QA systems). While effective, this approach increases system complexity and maintenance overhead.

This project explores an alternative paradigm: adapting a single decoder-based LLM to perform context-grounded medical QA using parameter-efficient fine-tuning (QLoRA). Instead of explicitly extracting spans via token-level classification, the model learns to generate structured answers conditioned on context and question prompts.

The experiment investigates whether generative fine-tuning can approximate extractive behavior in a domain-specific setting while training only **1.05% of model parameters** on a single T4 GPU.

## Results

| Metric | Base Model | Fine-tuned (LoRA) | Delta |
|--------|------------|-------------------|-------|
| Exact Match (EM) | 4.63% | 54.76% | **+50.12%** |
| Token F1 | 25.80% | 75.39% | **+49.58%** |

*Evaluated on 820 validation samples with greedy decoding.*

The results demonstrate substantial improvement over the base model, suggesting that efficient generative adaptation can serve as a flexible alternative to traditional task-specific pipelines.

## Qualitative Comparison

**Example: Complication Identification**

> **Context:** Tip 2 diyabet, vücudun insülini etkili kullanamadığı kronik bir metabolizma hastalığıdır. Tedavide yaşam tarzı değişiklikleri, oral antidiyabetik ilaçlar ve gerektiğinde insülin tedavisi uygulanır. Kontrol altına alınmazsa böbrek yetmezliği, görme kaybı ve kardiyovasküler hastalıklara yol açabilir.
>
> **Question:** Tip 2 diyabet kontrol altına alınmazsa hangi komplikasyonlara yol açabilir?

| | Answer |
|---|---|
| **Base Model** | Tip 2 diyabet kontrol altına alınmazsa **böbrek yetmezliği**, **görme kaybı** (retinopati) ve **kardiyovasküler hastalıklar** gibi ciddi komplikasyonlar gelişebilir. Ayrıca nöropati (sinir hasarı), ayak ülseri ve enfeksiyon riski de artar. |
| **Fine-tuned (LoRA)** | böbrek yetmezliği, görme kaybı ve kardiyovasküler hastalıklar |

The base model generates a verbose, partially hallucinated response that extends beyond the provided context. The fine-tuned model extracts a concise, context-grounded answer — approximating extractive behavior through generative fine-tuning.

## Project Structure

```
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
│
├── config/
│   └── training_config.yaml # Hyperparameters and training settings
│
├── data/
│   ├── raw/                # Raw dataset before processing
│   └── processed/          # Formatted data ready for training
│
├── src/
│   ├── data_processing.py  # Load and preprocess the dataset
│   ├── formatting.py      # Convert data to instruction (Soru/Cevap) format
│   ├── model_utils.py      # Model loading, QLoRA setup
│   ├── train.py           # Training loop and checkpointing
│   ├── evaluate.py        # Metrics (EM, F1) and evaluation
│   └── inference.py       # Generate answers for prompts
|   └── push_to_hub.py     # Push model to huggingface hub
│
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis
│   ├── colab_training.ipynb   # Training notebook for Google Colab
│   └── evaluation.ipynb       # Evaluation and inference notebook for Google Colab
│
└── outputs/
    ├── checkpoints/           # Saved model checkpoints
    └── evaluation_results/    # Evaluation logs and results
```

## Training Details

| Property | Value |
|----------|-------|
| Base model | Turkish-Gemma-9b-v0.1 (9B parameters) |
| Method | QLoRA (4-bit NF4, double quantization) |
| Compute dtype | float16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Trainable parameters | 54M / 5.1B (1.05%) |
| Training samples | 6,560 |
| Validation samples | 820 |
| Epochs | 2 |
| Optimizer | paged_adamw_8bit |
| Hardware | Single T4 GPU (15GB VRAM) |
| Training time | ~9 hours |

## Usage

### Training

Run training on Google Colab using the provided notebook:

1. Open `notebooks/colab_training.ipynb` in Colab
2. Set `HF_TOKEN` in Colab Secrets
3. Upload processed data to Google Drive at `MyDrive/gemma-finetuning/data/processed/`
4. Run all cells

The notebook clones the repo, installs dependencies, syncs data from Drive, runs training, and saves the adapter back to Drive.

### Evaluation

Run evaluation and qualitative inference:

1. Open `notebooks/evaluation.ipynb` in Colab
2. Ensure adapter is saved in Drive at `MyDrive/gemma-finetuning/outputs/checkpoints/`
3. Run all cells

The notebook computes EM/F1 metrics and generates qualitative comparisons between base and fine-tuned models.

### Inference

For standalone inference with custom prompts:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_name = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
adapter_name = "Ahmetemintek/turkish-gemma-9b-medical-qlora"

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

prompt = """Bağlam:
Verem, Mycobacterium tuberculosis adlı bakteri tarafından neden olunan bakteriyel ve bulaşıcı bir hastalıktır.

Soru:
Vereme ne neden olur?

Cevap:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
```

## Limitations

- The model is trained with a generative objective and may paraphrase or slightly extend answers rather than strictly extracting spans.
- It does not provide token-level offsets like traditional encoder-based extractive QA systems.
- Training was performed on ~6.5k samples with a 512-token context limit; performance may degrade on long or out-of-distribution medical texts.
- This project is an experimental demonstration of parameter-efficient domain adaptation and is not intended for clinical deployment.

## Citation

If you use this work, please cite:

```bibtex
@misc{turkish-gemma-medical-qlora,
  author = {Ahmet Emin Tek},
  title = {QLoRA Fine-Tuning: Turkish-Gemma-9B for Medical QA},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Ahmetemintek/gemma-finetuning}
}
```

## License

This project is licensed under the Apache 2.0 License. The base model and dataset have their own respective licenses.