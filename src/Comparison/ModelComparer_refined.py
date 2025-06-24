
# 📁 File: ModelComparer.py
# Purpose: Compare multiple transformer-based NER models on Amharic Telegram messages

import pandas as pd
import numpy as np
import torch
import time
import os
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch import nn

# -------------------------------
# Step 1: Load Labeled CoNLL Dataset
# -------------------------------

def parse_conll(file_path):
    tokens, labels = [], []
    current_tokens, current_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                current_tokens.append(parts[0])
                current_labels.append(parts[1])
    if current_tokens:
        tokens.append(current_tokens)
        labels.append(current_labels)
    return Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

# -------------------------------
# Step 2: Label Mapping
# -------------------------------

label2id = {
    "O": 0,
    "B-PRODUCT": 1,
    "I-PRODUCT": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}
id2label = {v: k for k, v in label2id.items()}

def encode_labels(example):
    return {"labels": [label2id.get(label.strip().upper(), 0) for label in example["ner_tags"]]}

# -------------------------------
# Step 3: Tokenization Utility
# -------------------------------

class TokenizerWrapper:
    _cache = {}
    @classmethod
    def get_tokenizer(cls, model_name):
        if model_name not in cls._cache:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._cache[model_name] = tokenizer
        return cls._cache[model_name]

def tokenize_and_align_labels(examples, model_name):
    tokenizer = TokenizerWrapper.get_tokenizer(model_name)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# -------------------------------
# Step 4: Metric Computation
# -------------------------------

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    true_preds = [[id2label[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[id2label[l] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    return {
        "accuracy": accuracy_score(sum(true_labels, []), sum(true_preds, [])),
        "f1": f1_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "report": classification_report(true_labels, true_preds, output_dict=True)
    }

# -------------------------------
# Step 5: Training and Evaluation
# -------------------------------

def train_and_evaluate(config, train_dataset, eval_dataset):
    print(f"🚀 Training {config['nickname']}")
    model = AutoModelForTokenClassification.from_pretrained(
        config["name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = TokenizerWrapper.get_tokenizer(config["name"])

    tokenized_train = train_dataset.map(lambda x: tokenize_and_align_labels(x, config["name"]), batched=True)
    tokenized_eval = eval_dataset.map(lambda x: tokenize_and_align_labels(x, config["name"]), batched=True)

    training_args = TrainingArguments(
        output_dir=f"./outputs/{config['nickname'].replace(' ', '_')}",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    trainer.train()
    results = trainer.evaluate()
    return {
        "model": config["nickname"],
        "f1": results.get("eval_f1", 0),
        "accuracy": results.get("eval_accuracy", 0),
        "precision": results.get("eval_precision", 0),
        "recall": results.get("eval_recall", 0)
    }

# -------------------------------
# Step 6: Main Comparison Routine
# -------------------------------

if __name__ == "__main__":
    dataset = parse_conll("ner_auto_labels_fixed.conll")
    dataset = dataset.map(encode_labels)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]

    model_configs = [
        {"name": "xlm-roberta-base", "nickname": "XLM-RoBERTa"},
        {"name": "bert-base-multilingual-cased", "nickname": "mBERT"},
        {"name": "distilbert-base-multilingual-cased", "nickname": "DistilBERT"}
    ]

    results = []
    for config in model_configs:
        result = train_and_evaluate(config, train_dataset, eval_dataset)
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison_summary.csv", index=False)
    print("\n📊 Model Comparison Summary:")
    print(results_df)
