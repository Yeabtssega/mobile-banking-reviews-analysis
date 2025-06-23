import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# 1. Load the dataset from your local .conll files
data_files = {
    "train": r"C:\Users\HP\EthioMart-NER\train.conll",
    "validation": r"C:\Users\HP\EthioMart-NER\valid.conll"
}

# Use 'conll2003' script to load conll files if formatted correctly
dataset = load_dataset("conll2003", data_files=data_files, trust_remote_code=True)

# 2. Extract label list and number of labels
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
print("Labels:", label_list)

# 3. Load model and tokenizer
model_name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# 4. Tokenize and align labels function
label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False  # Padding will be handled by data collator
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 5. Tokenize the datasets
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 6. Load seqeval metric for evaluation
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "eval_precision": results["overall_precision"],
        "eval_recall": results["overall_recall"],
        "eval_f1": results["overall_f1"],
        "eval_accuracy": results["overall_accuracy"],
    }

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./amharic-ner-model",
    eval_strategy="epoch",             # updated from deprecated 'evaluation_strategy'
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    logging_dir="./logs",
    logging_steps=10,
)

# 8. Create data collator to pad inputs and labels together
data_collator = DataCollatorForTokenClassification(tokenizer)

# 9. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 10. Start training
trainer.train()

# 11. Save final model and tokenizer
trainer.save_model("./amharic-ner-model")
tokenizer.save_pretrained("./amharic-ner-model")

print("✅ Fine-tuning complete. Model saved in './amharic-ner-model'")
