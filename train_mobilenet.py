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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def load_config(config_path="config/config_mobilenet_dgx-1.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_hyperparameters(config, run_dir):
    with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def prepare_dataloaders(config, transform):
    dataset = GTSRB(root="./data", split="train", transform=transform, download=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    test_dataset = GTSRB(root="./data", split="test", transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader


def init_model(config, device):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config["num_classes"])
    return model.to(device), weights.transforms()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct = 0.0, 0
    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_training_curves(history, prefix):
    df = pd.DataFrame(history)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
    plt.subplot(1,2,2)
    plt.plot(df["train_acc"], label="Train Acc")
    plt.plot(df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"{prefix}_training_curves.png")
    plt.close()


def save_confusion(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    pd.DataFrame(cm).to_csv(f"{prefix}_confusion_matrix_raw.csv", index=False)
    pd.DataFrame(cm_norm).to_csv(f"{prefix}_confusion_matrix_norm.csv", index=False)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{prefix}_confusion_matrix_raw.png")
    plt.close()
    ConfusionMatrixDisplay(cm_norm).plot(cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(f"{prefix}_confusion_matrix_norm.png")
    plt.close()


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join("runs", f"{config['model_name']}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    save_hyperparameters(config, run_dir)
    best_model_path = os.path.join(run_dir, "best_model.pth")

    model, transform = init_model(config, device)
    train_loader, val_loader, test_loader = prepare_dataloaders(config, transform)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # dentro de main(), justo antes del bucle de entrenamiento:
    patience = config.get("early_stopping_patience", 3)
    min_delta = config.get("min_delta", 0.0)
    no_improve_epochs = 0

    for epoch in range(1, config["num_epochs"] + 1):
        tloss, tacc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vloss, vacc, vpreds, vlabels = evaluate(model, val_loader, criterion, device)

        # registro histórico
        history["train_loss"].append(tloss)
        history["train_acc"].append(tacc)
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        pd.DataFrame(history).to_csv(
            os.path.join(run_dir, "metrics_history_partial.csv"), index=False
        )

        # Early Stopping / guardado mejor modelo
        if vacc > best_val_acc + min_delta:
            best_val_acc = vacc
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            save_training_curves(history, os.path.join(run_dir, "best"))
            save_confusion(vlabels, vpreds, os.path.join(run_dir, "best"))
            # guardar métricas del mejor modelo...
            with open(os.path.join(run_dir, "best_model_metrics.txt"), "w") as f:
                f.write(f"Val Accuracy: {vacc:.4f}\n")
                f.write(f"Val F1 Score: {f1_score(vlabels, vpreds, average='weighted'):.4f}\n")
                f.write(f"Val Precision: {precision_score(vlabels, vpreds, average='weighted'):.4f}\n")
                f.write(f"Val Recall: {recall_score(vlabels, vpreds, average='weighted'):.4f}\n")
            print(f"✔️ Mejor modelo guardado en epoch {epoch} (Val Acc: {vacc:.4f})")
        else:
            no_improve_epochs += 1
            print(f"⚠️ Sin mejora en validación: {no_improve_epochs}/{patience} epochs")

        # si superamos patience, detenemos entrenamiento
        if no_improve_epochs >= patience:
            print(f"⏹️ Early stopping tras {patience} épocas sin mejora.")
            break

    model.load_state_dict(torch.load(best_model_path))
    _, test_acc, tpreds, tlabels = evaluate(model, test_loader, criterion, device)
    save_confusion(tlabels, tpreds, os.path.join(run_dir, "test"))
    with open(os.path.join(run_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {f1_score(tlabels, tpreds, average='weighted'):.4f}\n")
        f.write(f"Test Precision: {precision_score(tlabels, tpreds, average='weighted'):.4f}\n")
        f.write(f"Test Recall: {recall_score(tlabels, tpreds, average='weighted'):.4f}\n")

    print(f"\nEntrenamiento completo — Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
