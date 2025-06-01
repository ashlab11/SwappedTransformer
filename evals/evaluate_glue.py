#!/usr/bin/env python3
# glue.py
#
# A script to fine-tune a BERT-like model on any of the 9 GLUE classification tasks,
# using locally downloaded GLUE data. After training, it runs predictions on the test split
# and writes out predicted labels to a file in the output directory.


import os
import argparse
import numpy as np

import torch
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main(
    model_name_or_path = "huawei-noah/TinyBERT_General_4L_312D", 
    task = 'cola',
    num_train_epochs=3,
    max_seq_length=128,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    seed=42,
):
    
    output_dir = f"results/{task}/{model_name_or_path.split('/')[-1]}"
    # List of GLUE tasks that are classification
    glue_tasks = {
        "cola": "cola",
        "mnli": "mnli",
        "mrpc": "mrpc",
        "qnli": "qnli",
        "qqp": "qqp",
        "rte": "rte",
        "sst2": "sst2",
        "wnli": "wnli",
        "stsb": "stsb",  # STS-B is regression; we include for completeness but handle separately
    }
    if task not in glue_tasks:
        raise ValueError(f"Task {task} not found. Must be one of {list(glue_tasks.keys())}.")
    
    # For STS-B (regression), we'll still load, but limit to regression metrics later
    raw_datasets = load_dataset(
        "glue",
        task,
    ) 
    
    # --------------------------- 
    # 2. Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=1 if task == "stsb" else raw_datasets["train"].features["label"].num_classes,
        finetuning_task=task,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    
    #Print total parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in the model: {total_params}")
    # ---------------------------
    # 3. Preprocess function: tokenize inputs
    # ---------------------------
    # Identify whether the task has sentence pairs or single sentences
    sentence1_key, sentence2_key = None, None
    if task in ["cola", "sst2"]:
        sentence1_key = "sentence"
    else:
        # All other GLUE tasks are sentence pairs
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"

    def preprocess_function(examples):
        # Tokenize, using sentence2 if available
        args_to_pass = (
            (examples[sentence1_key], examples[sentence2_key])
            if sentence2_key
            else (examples[sentence1_key],)
        )
        result = tokenizer(
            *args_to_pass,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )
        return result

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in raw_datasets["train"].column_names if c not in ["label"]],
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"].remove_columns("label") if "label" in tokenized_datasets["test"].column_names else tokenized_datasets["test"]

    # ---------------------------
    # 4. Data collator
    # ---------------------------
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    
    # ---------------------------
    # 5. Load evaluation metric
    # ---------------------------
    metric = load("glue", task)

    def compute_metrics(p):
        preds = p.predictions
        if task == "stsb":
            preds = preds.squeeze()
            return metric.compute(predictions=preds, references=p.label_ids)
        else:
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)
        
    # ---------------------------
    # 6. TrainingArguments & Trainer
    # ---------------------------
    
    if task == "cola":
        best_metric = "matthews_correlation"
    elif task == "stsb":
        best_metric = "pearson"
    else:
        best_metric = "accuracy"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        seed=seed,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )    
    
    trainer.train()
    
    # ---------------------------
    # 8. Evaluate on validation
    # ---------------------------
    print("*** Validation Results ***")
    eval_result = trainer.evaluate(eval_dataset)
    for k, v in eval_result.items():
        print(f"{k}: {v:.4f}")
    
main()