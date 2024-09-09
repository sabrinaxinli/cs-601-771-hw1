from transformers import TrainerState
import matplotlib.pyplot as plt

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
    plt.savefig("train_accuracy_plot.png")
    plt.show()
    
    # Val Accuracy
    dev_epochs = list(range(1, len(eval_accuracies) + 1))
    plt.figure()
    plt.plot(dev_epochs, eval_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.savefig("dev_accuracy_plot.png")
    plt.show()
    
trainer_state = TrainerState.load_from_json("./result-roberta-bitfit/checkpoint-8420/trainer_state.json")

print(trainer_state.log_history)

train_acc, dev_acc = extract_accuracies(trainer_state.log_history)
print(f"train acc: {train_acc}")
print(f"dev acc: {dev_acc}")
plot_accuracies(train_acc, dev_acc)