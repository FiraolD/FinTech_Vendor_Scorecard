# train_distilbert-base-multilingual-cased
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, classification_report

# ----------------------------
# Load and Parse CoNLL
# ----------------------------
def parse_conll(file_path):
    tokens, labels = [], []
    token_seq, label_seq = [], []
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
                token_seq.append(token.strip())
                label_seq.append(label.strip().upper())
        if token_seq:
            tokens.append(token_seq)
            labels.append(label_seq)
    return Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

dataset = parse_conll("Data/ner_auto_labels_fixed.conll")

label2id = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2, "B-PRICE": 3, "I-PRICE": 4, "B-LOC": 5, "I-LOC": 6}
id2label = {v: k for k, v in label2id.items()}

def encode_labels(example):
    return {"labels": [label2id[label] for label in example["ner_tags"]]}

encoded = dataset.map(encode_labels)
split = encoded.train_test_split(test_size=0.2, seed=42)
train_dataset, eval_dataset = split["train"], split["test"]

# Replace this line only:
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-multilingual-cased", num_labels=len(label2id), id2label=id2label, label2id=label2id
)


def tokenize_and_align_labels(model_name, examples):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(i)
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

# Tokenize train and eval
tokenized_train = train_dataset.map(lambda x: tokenize_and_align_labels("distilbert-base-multilingual-cased", x), batched=True)
tokenized_eval = eval_dataset.map(lambda x: tokenize_and_align_labels("distilbert-base-multilingual-cased", x), batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=len(label2id), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./mbert_amharic_ner",
    eval_strategy="epoch",
    learning_rate=2e-5,
    save_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    true_preds = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[id2label[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    return {"accuracy": accuracy_score(true_labels, true_preds), "f1": f1_score(true_labels, true_preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

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

model.save_pretrained("./DistilBert_amharic_ner")
tokenizer.save_pretrained("./DistilBert_amharic_ner")
