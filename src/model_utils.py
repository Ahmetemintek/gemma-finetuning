"""
Model setup utilities for QLoRA fine-tuning.

This module provides clean abstractions for:
- Loading tokenizer
- Loading base model with 4-bit quantization
- Applying LoRA adapters for parameter-efficient fine-tuning

Nothing else.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load tokenizer for the given model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_quantized_model(model_name: str) -> PreTrainedModel:
    """
    Load base causal language model with 4-bit quantization.
    
    Uses:
    - 4-bit NF4 quantization
    - Double quantization for memory efficiency
    - float16 compute dtype (T4 compatible)
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Quantized base model
    """
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model


def apply_lora_adapters(
    model: PreTrainedModel,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> PreTrainedModel:
    """
    Apply LoRA adapters to attention and MLP projection layers.
    
    Targets:
    - Attention: q_proj, k_proj, v_proj, o_proj
    - MLP: gate_proj, up_proj, down_proj
    
    Args:
        model: Base quantized model
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        
    Returns:
        Model with LoRA adapters applied
    """
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    return model


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Print summary of trainable vs total parameters.
    
    Useful for verifying LoRA setup.
    
    Args:
        model: Model to inspect
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_ratio = 100 * trainable_params / all_params
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable%: {trainable_ratio:.2f}%")
