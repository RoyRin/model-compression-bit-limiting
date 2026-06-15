#!/usr/bin/env python3
"""
Train a LoRA adapter on a single cluster of chat data.

Usage:
    python scripts/train_lora_cluster.py \
        --cluster-dir /path/to/lmsys-clustered/clusters/cluster_000 \
        --output-dir /path/to/loras/cluster_000
"""

import argparse
import json
import os
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Avoid tokenizer parallelism issues with dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def load_cluster_data(cluster_dir: Path) -> list[str]:
    """Load training texts from a cluster directory."""
    train_path = cluster_dir / "train.json"

    if not train_path.exists():
        raise FileNotFoundError(f"No train.json found in {cluster_dir}")

    with open(train_path, 'r') as f:
        data = json.load(f)

    return data['texts']


def format_texts_for_training(texts: list[str],
                              tokenizer,
                              max_length: int = 2048) -> Dataset:
    """Format texts into a HuggingFace Dataset for training."""

    def tokenize_function(examples):
        # Tokenize - DataCollatorForLanguageModeling will handle labels
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_special_tokens_mask=True,
        )
        return result

    dataset = Dataset.from_dict({'text': texts})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=1,  # Avoid multiprocessing issues
    )

    return tokenized


def train_lora(
    cluster_dir: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    max_samples: int = None,
    save_steps: int = 500,
    logging_steps: int = 50,
):
    """Train a LoRA adapter on cluster data."""
    cluster_dir = Path(cluster_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training LoRA for cluster: {cluster_dir.name}")
    print(f"Base model: {base_model}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load training data
    print("Loading training data...")
    texts = load_cluster_data(cluster_dir)
    print(f"Loaded {len(texts)} training samples")

    if max_samples and max_samples < len(texts):
        import random
        random.shuffle(texts)
        texts = texts[:max_samples]
        print(f"Using {max_samples} samples")

    # Load tokenizer
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = format_texts_for_training(texts, tokenizer, max_length)
    print(f"Dataset size: {len(train_dataset)}")

    # Load base model
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,  # bfloat16 is more stable for training
        device_map={"": 0},  # Load to single GPU
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Enable gradient checkpointing before adding LoRA
    model.gradient_checkpointing_enable()

    # Configure LoRA
    print(f"Configuring LoRA (rank={lora_rank}, alpha={lora_alpha})")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj"
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Ensure LoRA parameters require gradients
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=False,
        bf16=True,  # Use bf16 to match model dtype
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":
                                       False},  # Required for PEFT
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        'cluster_dir': str(cluster_dir),
        'base_model': base_model,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'num_samples': len(texts),
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
    }
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter on a single cluster")

    parser.add_argument("--cluster-dir",
                        type=str,
                        required=True,
                        help="Cluster directory containing train.json")
    parser.add_argument("--output-dir",
                        type=str,
                        required=True,
                        help="Output directory for LoRA adapter")

    # Model config
    parser.add_argument("--base-model",
                        type=str,
                        default=DEFAULT_BASE_MODEL,
                        help=f"Base model (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--lora-rank",
                        type=int,
                        default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha",
                        type=int,
                        default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout",
                        type=float,
                        default=0.05,
                        help="LoRA dropout (default: 0.05)")

    # Training config
    parser.add_argument("--learning-rate",
                        type=float,
                        default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--gradient-accumulation-steps",
                        type=int,
                        default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max-length",
                        type=int,
                        default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--max-samples",
                        type=int,
                        default=None,
                        help="Max training samples (default: all)")
    parser.add_argument("--save-steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--logging-steps",
                        type=int,
                        default=50,
                        help="Log every N steps (default: 50)")

    args = parser.parse_args()

    train_lora(
        cluster_dir=args.cluster_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_samples=args.max_samples,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
