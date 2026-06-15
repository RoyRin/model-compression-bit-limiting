#!/usr/bin/env python3
"""
Two-stage knowledge distillation with saved teacher logits.

Stage 1: Generate teacher logits and save to disk
Stage 2: Train student using saved logits (teacher not in memory)
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Optional


class TextDataset(Dataset):
    """Dataset for tokenizing text."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text,
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding="max_length",
                                 return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "idx": idx
        }


class LogitsDataset(Dataset):
    """Dataset that loads pre-computed teacher logits."""

    def __init__(self,
                 logits_dir: str,
                 texts: List[str],
                 tokenizer,
                 max_length: int = 512):
        self.logits_dir = Path(logits_dir)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Load saved logits
        logits_file = self.logits_dir / f"logits_{idx}.pt"
        if not logits_file.exists():
            raise FileNotFoundError(f"Teacher logits not found: {logits_file}")

        teacher_logits = torch.load(logits_file, map_location="cpu")

        # Validate logits
        if not torch.isfinite(teacher_logits).all():
            raise ValueError(
                f"Non-finite values in teacher logits for sample {idx}")

        # Tokenize text
        text = self.texts[idx]
        encoded = self.tokenizer(text,
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding="max_length",
                                 return_tensors="pt")

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "teacher_logits": teacher_logits
        }


def generate_and_save_teacher_logits(teacher_model_name: str,
                                     data_file: str,
                                     output_dir: str,
                                     batch_size: int = 8,
                                     max_length: int = 512,
                                     device: str = "cuda"):
    """Stage 1: Generate teacher logits and save to disk."""

    print("=" * 80)
    print("STAGE 1: Generating Teacher Logits")
    print("=" * 80)

    # Load texts
    with open(data_file, 'r') as f:
        texts = [
            line.strip().replace(' \\n ', '\n') for line in f if line.strip()
        ]
    print(f"Loaded {len(texts)} texts from {data_file}")

    # Create output directory
    logits_dir = Path(output_dir) / "teacher_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    # Check if logits already exist
    existing = sum(1 for _ in logits_dir.glob("logits_*.pt"))
    if existing == len(texts):
        print(
            f"✓ All {len(texts)} teacher logits already exist in {logits_dir}")
        return str(logits_dir), texts

    # Load teacher model
    print(f"\nLoading teacher model: {teacher_model_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, torch_dtype=torch.float16).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model.eval()

    # Create dataset and dataloader
    dataset = TextDataset(texts, teacher_tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Generate and save logits
    print(f"\nGenerating teacher logits...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating logits"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["idx"]

            # Get teacher logits
            outputs = teacher_model(input_ids=input_ids,
                                    attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Save each sample's logits
            for i, idx in enumerate(indices):
                logits_file = logits_dir / f"logits_{idx}.pt"
                sample_logits = logits[i].cpu()

                # Check for numerical issues before saving
                if not torch.isfinite(sample_logits).all():
                    print(
                        f"Warning: Non-finite values in teacher logits for sample {idx}, skipping"
                    )
                    continue

                # Save as float16 to save disk space
                torch.save(sample_logits.half(), logits_file)

    # Clean up teacher model
    del teacher_model
    torch.cuda.empty_cache()

    print(f"\n✓ Saved {len(texts)} teacher logits to {logits_dir}")

    # Save metadata
    metadata = {
        "teacher_model": teacher_model_name,
        "num_samples": len(texts),
        "max_length": max_length,
        "data_file": data_file
    }
    with open(logits_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(logits_dir), texts


def train_student_with_saved_logits(student_model_name: str,
                                    logits_dir: str,
                                    texts: List[str],
                                    save_dir: str,
                                    num_epochs: int = 3,
                                    batch_size: int = 4,
                                    learning_rate: float = 5e-5,
                                    temperature: float = 2.0,
                                    max_length: int = 512,
                                    device: str = "cuda"):
    """Stage 2: Train student using pre-computed teacher logits."""

    print("\n" + "=" * 80)
    print("STAGE 2: Training Student with Saved Logits")
    print("=" * 80)

    # Load student model
    print(f"\nLoading student model: {student_model_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name, torch_dtype=torch.float16).to(device)
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # Create dataset with saved logits
    dataset = LogitsDataset(logits_dir, texts, student_tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_steps //
                                                10,
                                                num_training_steps=total_steps)

    print(f"\nTraining Configuration:")
    print(f"  Student: {student_model_name}")
    print(f"  Training samples: {len(texts)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Temperature: {temperature}")
    print(f"  Total steps: {total_steps}")
    print()

    # Training loop
    student_model.train()
    global_step = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        progress_bar = tqdm(dataloader,
                            desc=f"Training Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(
                device).float()  # Convert to float32 for stability

            # Get student logits
            student_outputs = student_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Shift logits and labels for next-token prediction
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
            shift_attention_mask = attention_mask[:, 1:].contiguous()

            # Apply temperature
            student_log_probs = F.log_softmax(shift_logits / temperature,
                                              dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / temperature,
                                      dim=-1)

            # KL divergence loss
            kl_loss = F.kl_div(student_log_probs,
                               teacher_probs,
                               reduction='none').sum(dim=-1)  # Sum over vocab

            # Mask padding tokens
            kl_loss = kl_loss * shift_attention_mask
            num_valid_tokens = shift_attention_mask.sum()

            # Check for numerical issues
            if num_valid_tokens == 0:
                continue  # Skip batch with no valid tokens

            loss = kl_loss.sum() / num_valid_tokens

            # Check for nan/inf
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected, skipping batch")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}\n")

        # Save checkpoint after each epoch
        checkpoint_dir = Path(save_dir) / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(checkpoint_dir)
        student_tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    final_dir = Path(save_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(final_dir)
    student_tokenizer.save_pretrained(final_dir)
    print(f"\n✓ Training complete! Final model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage distillation with saved teacher logits")

    # Model arguments
    parser.add_argument("--teacher-model",
                        type=str,
                        required=True,
                        help="Teacher model name or path")
    parser.add_argument("--student-model",
                        type=str,
                        required=True,
                        help="Student model name or path")

    # Data arguments
    parser.add_argument("--data-file",
                        type=str,
                        required=True,
                        help="Path to text file with training data")
    parser.add_argument("--max-length",
                        type=int,
                        default=512,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="Batch size for student training")
    parser.add_argument("--teacher-batch-size",
                        type=int,
                        default=8,
                        help="Batch size for teacher logits generation")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate")
    parser.add_argument("--temperature",
                        type=float,
                        default=2.0,
                        help="Distillation temperature")

    # Output arguments
    parser.add_argument("--save-dir",
                        type=str,
                        required=True,
                        help="Directory to save student model")
    parser.add_argument("--skip-logits-generation",
                        action="store_true",
                        help="Skip stage 1 if logits already exist")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Stage 1: Generate teacher logits (if not skipped)
    if not args.skip_logits_generation or not (Path(args.save_dir) /
                                               "teacher_logits").exists():
        logits_dir, texts = generate_and_save_teacher_logits(
            teacher_model_name=args.teacher_model,
            data_file=args.data_file,
            output_dir=args.save_dir,
            batch_size=args.teacher_batch_size,
            max_length=args.max_length,
            device=device)
    else:
        print("Skipping teacher logits generation (already exists)")
        logits_dir = str(Path(args.save_dir) / "teacher_logits")
        with open(args.data_file, 'r') as f:
            texts = [
                line.strip().replace(' \\n ', '\n') for line in f
                if line.strip()
            ]

    # Stage 2: Train student with saved logits
    train_student_with_saved_logits(student_model_name=args.student_model,
                                    logits_dir=logits_dir,
                                    texts=texts,
                                    save_dir=args.save_dir,
                                    num_epochs=args.num_epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate,
                                    temperature=args.temperature,
                                    max_length=args.max_length,
                                    device=device)


if __name__ == "__main__":
    main()
