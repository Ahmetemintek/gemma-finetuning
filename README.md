# QLoRA Fine-Tuning: Turkish-Gemma-9B for Medical QA

Parameter-efficient fine-tuning (QLoRA) of [Turkish-Gemma-9b-v0.1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-v0.1) on [MedTurkQuAD](https://www.kaggle.com/datasets/incidelen/medturkquad), a Turkish medical QA dataset, to adapt the model for generative question-answering in the medical domain.

---

## Project Structure

```
├── project.md              # Full project specification and design
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (keys, paths)
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
│
├── notebooks/
│   ├── eda.ipynb          # Exploratory data analysis
│   └── colab_training.ipynb # Training notebook for Google Colab
│
└── outputs/
    ├── checkpoints/       # Saved model checkpoints
    └── evaluation_results/ # Evaluation logs and results
```
