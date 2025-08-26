import argparse
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


# -----------------
# Language - file mapping
# -----------------
LANG_FILE_MAP = {
    "english":   ("english/train_en.tsv", "english/dev_test_en.tsv"),
    "german":    ("german/train_de.tsv", "german/dev_test_de.tsv"),
    "italian":   ("italian/train_it.tsv", "italian/dev_test_it.tsv"),
    "arabic":    ("arabic/train_ar.tsv", "arabic/dev_test_ar.tsv"),
    "bulgarian": ("bulgarian/train_bg.tsv", "bulgarian/dev_test_bg.tsv"),
}


# -----------------
# Metrics
# -----------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(labels, preds, labels=[0], zero_division=0)

    return {
        'macro-F1': m_f1,
        'macro-P': m_prec,
        'macro-R': m_rec,
        'SUBJ-F1': p_f1[0],
        'SUBJ-P': p_prec[0],
        'SUBJ-R': p_rec[0],
        'accuracy': acc
    }


# -----------------
# Load dataset
# -----------------
def load_dataset(train_lang, test_lang, base_path="./data"):
    if train_lang not in LANG_FILE_MAP or test_lang not in LANG_FILE_MAP:
        raise ValueError(f"Languages must be in {list(LANG_FILE_MAP.keys())}")

    train_path = f"{base_path}/{LANG_FILE_MAP[train_lang][0]}"
    test_path = f"{base_path}/{LANG_FILE_MAP[test_lang][1]}"

    print(f"\nLoading data:")
    print(f"  Train language: {train_lang}")
    print(f"  Test  language: {test_lang}")

    train_df = pd.read_csv(train_path, sep='\t', quoting=csv.QUOTE_NONE)[['sentence', 'label']].rename(columns={"sentence": "text"})
    test_df = pd.read_csv(test_path, sep='\t', quoting=csv.QUOTE_NONE)[['sentence', 'label']].rename(columns={"sentence": "text"})

    label_mapping = {'SUBJ': 0, 'OBJ': 1}
    train_df['label'] = train_df['label'].map(label_mapping)
    test_df['label'] = test_df['label'].map(label_mapping)

    train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_split.reset_index(drop=True)),
        "val": datasets.Dataset.from_pandas(val_split.reset_index(drop=True)),
        "test": datasets.Dataset.from_pandas(test_df.reset_index(drop=True))
    })
    return dataset


# -----------------
# Main training function
# -----------------
def main(train_lang, test_lang):
    dataset = load_dataset(train_lang, test_lang)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # ---- Baseline training ----
    baseline_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./results/{train_lang}_to_{test_lang}_baseline",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=baseline_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    test_results1 = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"\n=== Final Test Evaluation (baseline {train_lang}→{test_lang}) ===")
    print(test_results1)

    # ---- Optimized training ----
    optimized_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=2)

    optimized_training_args = TrainingArguments(
        output_dir=f"./results/{train_lang}_to_{test_lang}_optimized",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.02,
        learning_rate=3e-5,
        seed=1111,
    )

    optimized_trainer = Trainer(
        model=optimized_model,
        args=optimized_training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    optimized_trainer.train()
    test_results2 = optimized_trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"\n=== Final Test Evaluation (optimized {train_lang}→{test_lang}) ===")
    print(test_results2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate multilingual model.")
    parser.add_argument("--train_lang", type=str, required=True, help="Language for training dataset")
    parser.add_argument("--test_lang", type=str, required=True, help="Language for test dataset")
    args = parser.parse_args()

    main(args.train_lang, args.test_lang)
