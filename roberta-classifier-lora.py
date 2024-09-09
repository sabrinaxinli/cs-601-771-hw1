from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    TrainerCallback)
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
from torch import nn
import json
from copy import deepcopy
import matplotlib.pyplot as plt
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel)

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

def plot_accuracies(train_accuracies, eval_accuracies):
    train_epochs = list(range(1, len(train_accuracies) + 1))
    
    # Train Accuracy
    plt.figure()
    plt.plot(train_epochs, train_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy Over Epochs")
    plt.legend()
    plt.savefig("./result-roberta-lora/train_accuracy_plot.png")
    plt.show()
    
    # Val Accuracy
    dev_epochs = list(range(1, len(eval_accuracies) + 1))
    plt.figure()
    plt.plot(dev_epochs, eval_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.savefig("./result-roberta-lora/dev_accuracy_plot.png")
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    train_data = load_dataset("stanfordnlp/sst2", split="train")
    test_data = load_dataset("stanfordnlp/sst2", split="validation")
    
    train_test_split = train_data.train_test_split(test_size = 0.2)
    
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # model = RobertaForSST2Classification("roberta-base", 2) 
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    total_trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    
    print(f"Number of trainable parameters in roberta-base for sequence classification: {total_trainable_params}")
    
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
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        num_train_epochs = 30,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        eval_strategy="epoch",
        learning_rate=1e-4,
        save_strategy="epoch",
        output_dir = "./result-roberta-lora",
        logging_dir = "./logs-roberta-lora",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        resume_from_checkpoint = True
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = dev_dataset,
        data_collator=data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
    )
    
    trainer.add_callback(ComputeTrainMetricsCallback(trainer))
    
    trained = trainer.train(resume_from_checkpoint=True)
    
    # Test set
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Test set metrics: {test_metrics}")
    
    train_acc, dev_acc = extract_accuracies(trainer.state.log_history)
    plot_accuracies(train_acc, dev_acc)