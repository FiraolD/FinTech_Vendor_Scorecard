import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, classification_report

# -------------------------------
# 📄 Step 1: Parse CoNLL Format
# -------------------------------

def parse_conll_to_dataset(file_path):
    tokens, labels = [], []
    token_seq, label_seq = [], []

    valid_labels = {"O", "B-PRODUCT", "I-PRODUCT", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if token_seq:
                    tokens.append(token_seq)
                    labels.append(label_seq)
                    token_seq, label_seq = [], []
                continue
            if "\t" in line:
                token, label = line.split("\t")
                label = label.strip().upper()
                if label in valid_labels:
                    token_seq.append(token.strip())
                    label_seq.append(label)
        # Add last sequence
        if token_seq:
            tokens.append(token_seq)
            labels.append(label_seq)

    return Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

dataset = parse_conll_to_dataset("Data/ner_auto_labels_fixed.conll")
print("✅ CoNLL dataset loaded with", len(dataset), "sequences.")

# -------------------------------
# 🏷️ Step 2: Define Label Mapping
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
    return {"labels": [label2id[label] for label in example["ner_tags"]]}

encoded_dataset = dataset.map(encode_labels)

# -------------------------------
# ✂️ Step 3: Train-Test Split
# -------------------------------

split_dataset = encoded_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# -------------------------------
# 🤖 Step 4: Tokenization & Alignment
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",   # Ensures same length for batching
        max_length=128,         # You can adjust based on your data
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
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

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "ner_tags", "labels"])
tokenized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "ner_tags", "labels"])


# -------------------------------
# ⚙️ Step 5: Load Model
# -------------------------------

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# -------------------------------
# ⚙️ Step 6: TrainingArguments
# -------------------------------

training_args = TrainingArguments(
    output_dir="./amharic_ner_model_conll",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False
)

# -------------------------------
# 📊 Step 7: Metrics
# -------------------------------

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=2)
    labels = p.label_ids

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return {
        "accuracy": accuracy_score(sum(true_labels, []), sum(true_predictions, [])),
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions)
    }

# -------------------------------
# 🏋️ Step 8: Training
# -------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

print("🚀 Starting training...")
trainer.train()

# -------------------------------
# 📊 Step 9: Evaluation
# -------------------------------

print("📊 Evaluating model performance...")
results = trainer.evaluate()
print(f"Validation Results:\n{results}")

output = trainer.predict(tokenized_eval)
metrics = compute_metrics(output)

print("\n🔍 Classification Report:")
print(metrics['report'])

# -------------------------------
# 💾 Step 10: Save Model
# -------------------------------

model.save_pretrained("./amharic_ner_model_conll")
tokenizer.save_pretrained("./XML_ROBERTA_amharic_ner_model_conll")
print("✅ Model saved to ./XML_ROBERTA_amharic_ner_model")
