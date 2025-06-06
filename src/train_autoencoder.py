"""Training script for the autoencoder model."""
import torch
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from models import AutoEncoder
import torch.nn as nn


device = (
    'mps' if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)

# --- Load dataset ---
try:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
except Exception as e:
    print(f"Could not download dataset: {e}")
    fallback_text = [
        "hello world",
        "this is a tiny dataset",
        "used when the real dataset can't be downloaded"
    ] * 100
    ds = Dataset.from_dict({"text": fallback_text})

# remove empty texts
ds = ds.filter(lambda x: len(x["text"]) > 0)

# --- Simple character-level tokenizer to avoid external downloads ---
all_text = "".join(ds["text"])
chars = sorted(set(all_text))
vocab = {c: i + 4 for i, c in enumerate(chars)}
vocab["[PAD]"] = 0
vocab["[UNK]"] = 1
vocab["[CLS]"] = 2
vocab["[SEP]"] = 3
inv_vocab = {i: c for c, i in vocab.items()}

def encode(text):
    return [vocab.get(c, vocab["[UNK]"]) for c in text][:256]

def tokenize_function(examples):
    ids = [encode(t) for t in examples["text"]]
    attn = [[1] * len(i) for i in ids]
    return {"input_ids": ids, "labels": ids, "attention_mask": attn}

ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])

def collator(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

vocab_size = len(vocab)

model = AutoEncoder(
    vocab_size=vocab_size,
    embed_dim=256,
    slot_dim=512,
    ff_dim=1024,
    n_layers_enc=4,
    n_layers_dec=4,
    n_heads=8,
    n_slots=16,
    activation_function=nn.GELU,
    dropout=0.1,
    layer_norm_eps=1e-5,
)

print(
    "Total trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)
model.to(device)

# Optimizer groups
no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
optim_groups = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.1,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

training_args = TrainingArguments(
    output_dir="models/test_run",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    learning_rate=6e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw_torch",
    logging_strategy="steps",
    logging_steps=5,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=collator,
    optimizers=(torch.optim.AdamW(optim_groups, lr=6e-4), None),
)

trainer.train()
