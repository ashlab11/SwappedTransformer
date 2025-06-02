"""
Main training script for the model. Conducts masked-token prediction on two datasets:
1. The WikiText-103 dataset, which is a large-scale language modeling dataset.
2. OpenWebtext, a dataset of web text for language modeling.
This script initializes the model, sets up the training loop, and evaluates the model on the validation set.
"""

import transformers, accelerate, datasets, evaluate, torch
from Encoders import Encoder
import argparse
import os
from datasets import load_dataset
import torch
import torch.nn as nn
import evaluate, accelerate

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# --- Loading and preparing the dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

max_length = 256
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=False, max_length=256)

# mapping tokenization function
ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8 if device == 'mps' else None,
)

swapped = False
model = Encoder(
    vocab_size = len(tokenizer),
    embed_dim = 256, 
    ff_dim = 1024,
    n_layers = 4,
    n_heads = 16,
    activation_function=nn.ReLU,
    swapped=swapped,  # Set to True if you want to use the swapped architecture
    dropout=0.1,
    layer_norm_eps=1e-5
)
print(f"Total trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

model.to(device)

# --- Training arguments ----
training_args = TrainingArguments(
    output_dir = f"./models/swapped_{swapped}", 
    num_train_epochs = 3, 
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    gradient_accumulation_steps = 1,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_dir = "./logs",
    logging_steps = 50, 
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

# --- Trainer setup ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    data_collator=collator,
)
# --- Training the model ----
trainer.train()