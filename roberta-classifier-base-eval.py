from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel, TrainingArguments, Trainer, DataCollatorWithPadding, RobertaForSequenceClassification, TrainerCallback
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
from torch import nn
import json
from copy import deepcopy
import matplotlib.pyplot as plt

class ComputeTrainMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def extract_accuracies(log_history):
    train_accuracies = []
    dev_accuracies = []
    
    for log in log_history:
        if "train_accuracy" in log:
            train_accuracies.append(log["train_accuracy"])
        if "eval_accuracy" in log:
            dev_accuracies.append(log["eval_accuracy"])
    
    return train_accuracies, dev_accuracies

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    train_data = load_dataset("stanfordnlp/sst2", split="train")
    test_data = load_dataset("stanfordnlp/sst2", split="validation")
    
    train_test_split = train_data.train_test_split(test_size = 0.2)
    
    tokenizer = RobertaTokenizer.from_pretrained("./result-roberta-bitfit/checkpoint-8420")
    model = RobertaForSequenceClassification.from_pretrained("./result-roberta-bitfit/checkpoint-8420", num_labels=2)
    
    def tokenize_func(example):
        return tokenizer(example["sentence"], truncation = True)
    
    train_dataset = train_test_split["train"].map(tokenize_func, batched=True)
    dev_dataset = train_test_split["test"].map(tokenize_func, batched=True)
    test_dataset = test_data.map(tokenize_func, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    training_args = TrainingArguments(
        per_device_eval_batch_size=16,
        output_dir = "./result-roberta-lora-eval",
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        eval_dataset = test_dataset,
        data_collator=data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
    )
    
    # Test set
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Resumed from checkpoint lora - test set metrics: {test_metrics}")