#!/usr/bin/env python3
"""
Knowledge Distillation Script
Distills knowledge from a large teacher model to a smaller student model
using KL divergence on top-k token probabilities.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, List
import json
import os
from datetime import datetime


class TextDataset(Dataset):
    """Dataset for text generation or teacher-forced training."""

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
            "attention_mask": encoded["attention_mask"].squeeze()
        }


class KnowledgeDistiller:
    """Main class for knowledge distillation."""

    def __init__(self,
                 teacher_model_name: str,
                 student_model_name: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 top_k: int = 100,
                 top_k_weight: int = 10,
                 top_k_extra_weight: float = 2.0):
        self.device = device
        self.top_k = top_k
        self.top_k_weight = top_k_weight
        self.top_k_extra_weight = top_k_extra_weight

        # Load models
        print(f"Loading teacher model: {teacher_model_name}")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name).to(device)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name)
        self.teacher_model.eval()  # Teacher stays in eval mode

        print(f"Loading student model: {student_model_name}")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name).to(device)
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_name)

        # Ensure tokenizers are compatible
        if self.teacher_tokenizer.vocab_size != self.student_tokenizer.vocab_size:
            raise ValueError(
                f"Teacher vocab size ({self.teacher_tokenizer.vocab_size}) != "
                f"Student vocab size ({self.student_tokenizer.vocab_size})")

        # Set pad token if not set
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token

    def generate_teacher_data(self,
                              num_sequences: int = 1000,
                              max_length: int = 512) -> List[str]:
        """Generate text using the teacher model."""
        print(f"Generating {num_sequences} sequences from teacher model...")
        generated_texts = []

        with torch.no_grad():
            for _ in tqdm(range(num_sequences),
                          desc="Generating sequences",
                          unit="seq"):
                # Start with a random prompt or empty
                prompt_tokens = torch.tensor(
                    [[self.teacher_tokenizer.bos_token_id]]).to(self.device)

                outputs = self.teacher_model.generate(
                    prompt_tokens,
                    max_length=max_length,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=self.teacher_tokenizer.pad_token_id)

                text = self.teacher_tokenizer.decode(outputs[0],
                                                     skip_special_tokens=True)
                generated_texts.append(text)

        return generated_texts

    def compute_kl_loss(self,
                        teacher_logits: torch.Tensor,
                        student_logits: torch.Tensor,
                        attention_mask: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Compute weighted KL divergence loss between teacher and student distributions.
        
        Args:
            teacher_logits: [batch_size, seq_len, vocab_size]
            student_logits: [batch_size, seq_len, vocab_size]
            attention_mask: [batch_size, seq_len]
            temperature: Temperature for softening distributions
        """
        # Apply temperature and get probabilities
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        # Get top-k indices from teacher
        top_k_values, top_k_indices = torch.topk(teacher_probs,
                                                 k=self.top_k,
                                                 dim=-1)

        # Initialize loss tensor
        batch_size, seq_len, _ = teacher_logits.shape
        kl_loss = torch.zeros(batch_size, seq_len).to(self.device)

        # Compute KL divergence for top-k tokens
        for i in range(self.top_k):
            teacher_prob = top_k_values[:, :, i]
            indices = top_k_indices[:, :, i]
            student_log_prob = torch.gather(student_log_probs, -1,
                                            indices.unsqueeze(-1)).squeeze(-1)

            # Weight factor: extra weight for top-10
            weight = self.top_k_extra_weight if i < self.top_k_weight else 1.0

            # KL divergence: p * log(p/q) = p * (log(p) - log(q))
            kl_term = teacher_prob * (torch.log(teacher_prob + 1e-8) -
                                      student_log_prob)
            kl_loss += weight * kl_term

        # Apply attention mask
        kl_loss = kl_loss * attention_mask

        # Average over valid tokens
        return kl_loss.sum() / attention_mask.sum()

    def train_step(self, batch, optimizer, temperature: float = 1.0) -> float:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids,
                                                 attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Get student predictions
        student_outputs = self.student_model(input_ids=input_ids,
                                             attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Compute KL loss
        loss = self.compute_kl_loss(teacher_logits, student_logits,
                                    attention_mask, temperature)

        # Backward pass
        loss.backward()

        return loss.item()

    def train(self,
              train_texts: List[str],
              val_texts: Optional[List[str]] = None,
              batch_size: int = 4,
              num_epochs: int = 3,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              gradient_accumulation_steps: int = 1,
              temperature: float = 3.0,
              save_dir: str = "distilled_models",
              save_every_n_steps: int = 1000):
        """Main training loop."""
        # Create datasets
        train_dataset = TextDataset(train_texts, self.teacher_tokenizer)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)

        val_loader = None
        if val_texts:
            val_dataset = TextDataset(val_texts, self.teacher_tokenizer)
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.student_model.parameters(),
                                      lr=learning_rate)

        total_steps = len(
            train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)

        # Training loop
        global_step = 0
        best_val_loss = float('inf')

        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            self.student_model.train()
            train_loss = 0
            optimizer.zero_grad()

            progress_bar = tqdm(train_loader,
                                desc=f"Training Epoch {epoch+1}/{num_epochs}",
                                unit="batch")
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch, optimizer, temperature)
                loss = loss / gradient_accumulation_steps
                train_loss += loss

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Update progress bar with more info
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{loss * gradient_accumulation_steps:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step
                    })

                    # Save checkpoint
                    if global_step % save_every_n_steps == 0:
                        self.save_checkpoint(save_dir, global_step)

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader, temperature)
                print(f"Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_dir, "best")

        # Save final model
        self.save_checkpoint(save_dir, "final")

    def evaluate(self, dataloader, temperature: float = 3.0) -> float:
        """Evaluate on validation set."""
        self.student_model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get predictions
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask)
                student_outputs = self.student_model(
                    input_ids=input_ids, attention_mask=attention_mask)

                # Compute loss
                loss = self.compute_kl_loss(teacher_outputs.logits,
                                            student_outputs.logits,
                                            attention_mask, temperature)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, save_dir: str, suffix: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-{suffix}")
        self.student_model.save_pretrained(checkpoint_dir)
        self.student_tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Language Models")
    parser.add_argument("--teacher-model",
                        type=str,
                        required=True,
                        help="Teacher model name or path")
    parser.add_argument("--student-model",
                        type=str,
                        required=True,
                        help="Student model name or path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "supervised"],
        default="generate",
        help=
        "Mode: 'generate' text with teacher or 'supervised' with existing data"
    )
    parser.add_argument("--data-file",
                        type=str,
                        help="Path to text file for supervised mode")
    parser.add_argument("--num-sequences",
                        type=int,
                        default=1000,
                        help="Number of sequences to generate")
    parser.add_argument("--max-length",
                        type=int,
                        default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=3,
                        help="Number of epochs")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate")
    parser.add_argument("--temperature",
                        type=float,
                        default=3.0,
                        help="Distillation temperature")
    parser.add_argument("--top-k",
                        type=int,
                        default=100,
                        help="Top-k tokens to consider")
    parser.add_argument("--top-k-weight",
                        type=int,
                        default=10,
                        help="Number of top tokens to weight extra")
    parser.add_argument("--top-k-extra-weight",
                        type=float,
                        default=2.0,
                        help="Extra weight for top tokens")
    parser.add_argument("--gradient-accumulation-steps",
                        type=int,
                        default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--save-dir",
                        type=str,
                        default="distilled_models",
                        help="Directory to save models")
    parser.add_argument("--val-split",
                        type=float,
                        default=0.1,
                        help="Validation split ratio")

    args = parser.parse_args()

    # Initialize distiller
    distiller = KnowledgeDistiller(teacher_model_name=args.teacher_model,
                                   student_model_name=args.student_model,
                                   top_k=args.top_k,
                                   top_k_weight=args.top_k_weight,
                                   top_k_extra_weight=args.top_k_extra_weight)

    # Get training data
    if args.mode == "generate":
        # Generate data from teacher
        all_texts = distiller.generate_teacher_data(
            num_sequences=args.num_sequences, max_length=args.max_length)
    else:
        # Load from file
        if not args.data_file:
            raise ValueError("--data-file required for supervised mode")

        with open(args.data_file, 'r') as f:
            all_texts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(all_texts)} texts from {args.data_file}")

    # Split into train/val
    val_size = int(len(all_texts) * args.val_split)
    train_texts = all_texts[val_size:]
    val_texts = all_texts[:val_size] if val_size > 0 else None

    print(f"Training set: {len(train_texts)} texts")
    if val_texts:
        print(f"Validation set: {len(val_texts)} texts")

    print(f"\nDistillation Configuration:")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  Student: {args.student_model}")
    print(
        f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})"
    )
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Top-k: {args.top_k} (extra weight on top-{args.top_k_weight})")
    print("-" * 80)

    # Train
    distiller.train(
        train_texts=train_texts,
        val_texts=val_texts,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_dir=args.save_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
