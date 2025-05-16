import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from torchvision.datasets import GTSRB
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def load_config(config_path="config/config_efficientnet_dgx-1.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataloaders(config, transform):
    dataset = GTSRB(root="./data", split="train", transform=transform, download=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_dataset = GTSRB(root="./data", split="test", transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader


def init_model(config, device):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config["num_classes"])
    return model.to(device), weights.transforms()


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), correct / len(loader.dataset), all_preds, all_labels

def save_hyperparameters(config, run_dir):
    """Guarda los hiperparámetros del archivo YAML como TXT legible"""
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def save_metrics_and_confusion(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    pd.DataFrame(cm).to_csv(f"{prefix}_confusion_matrix_raw.csv", index=False)
    pd.DataFrame(cm_norm).to_csv(f"{prefix}_confusion_matrix_norm.csv", index=False)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.savefig(f"{prefix}_confusion_matrix_raw.png")
    plt.close()

    ConfusionMatrixDisplay(confusion_matrix=cm_norm).plot(cmap="Blues")
    plt.savefig(f"{prefix}_confusion_matrix_norm.png")
    plt.close()


def save_training_curves(history, path_prefix):
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.legend(), plt.grid(), plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(df["train_acc"], label="Train Acc")
    plt.plot(df["val_acc"], label="Val Acc")
    plt.legend(), plt.grid(), plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(f"{path_prefix}_training_curves.png")
    plt.close()


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join("runs", f"{config['model_name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    save_hyperparameters(config, run_dir)
    best_model_path = os.path.join(run_dir, "best_model.pth")

    model, transform = init_model(config, device)
    train_loader, val_loader, test_loader = prepare_dataloaders(config, transform)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "metrics_history_partial.csv"), index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            save_training_curves(history, os.path.join(run_dir, "best"))
            save_metrics_and_confusion(val_labels, val_preds, os.path.join(run_dir, "best"))

            with open(os.path.join(run_dir, "best_model_metrics.txt"), "w") as f:
                f.write(f"Accuracy: {val_acc:.4f}\n")
                f.write(f"F1 Score: {f1_score(val_labels, val_preds, average='weighted'):.4f}\n")
                f.write(f"Precision: {precision_score(val_labels, val_preds, average='weighted'):.4f}\n")
                f.write(f"Recall: {recall_score(val_labels, val_preds, average='weighted'):.4f}\n")

    # Evaluación en test
    model.load_state_dict(torch.load(best_model_path))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    save_metrics_and_confusion(test_labels, test_preds, os.path.join(run_dir, "test"))

    with open(os.path.join(run_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"F1 Score: {f1_score(test_labels, test_preds, average='weighted'):.4f}\n")
        f.write(f"Precision: {precision_score(test_labels, test_preds, average='weighted'):.4f}\n")
        f.write(f"Recall: {recall_score(test_labels, test_preds, average='weighted'):.4f}\n")

    print(f"\nEntrenamiento finalizado. Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
