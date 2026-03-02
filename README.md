# QLoRA Fine-Tuning: Turkish-Gemma-9B for Medical QA

Parameter-efficient fine-tuning (QLoRA) of [Turkish-Gemma-9b-v0.1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1) on [MedTurkQuAD](https://www.kaggle.com/datasets/incidelen/medturkquad), a Turkish medical QA dataset, to adapt the model for generative question-answering in the medical domain.

---

## Motivation & Architectural Perspective

Traditional medical NLP pipelines often rely on multiple task-specific encoder models (e.g., separate NER, relation extraction, and QA systems). While effective, this approach increases system complexity and maintenance overhead.

In this project, we explore a different paradigm: adapting a single decoder-based large language model to perform context-grounded medical QA using parameter-efficient fine-tuning (QLoRA). Instead of explicitly extracting spans via token-level classification, the model learns to generate structured answers conditioned on context and question prompts.

This experiment investigates whether generative fine-tuning can approximate extractive behavior in a domain-specific setting while training only 1.05% of model parameters on a single T4 GPU. The results show substantial improvement over the base model (+50 EM, +49 F1), suggesting that efficient generative adaptation can serve as a flexible alternative to traditional task-specific pipelines.


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
│   ├── eda.ipynb          # Exploratory data analysis
│   └── colab_training.ipynb # Training notebook for Google Colab
|   └── evaluation.ipynb        # Evaluation notebook for Google Colab
│
└── outputs/
    ├── checkpoints/       # Saved model checkpoints
    └── evaluation_results/ # Evaluation logs and results
```
