import os
import json
import time
import torch
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import GTSRB
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    resnet50, ResNet50_Weights,
    swin_t, Swin_T_Weights
)


def load_txt_metrics(file_path):
    with open(file_path, "r") as f:
        return {line.split(":")[0].strip(): float(line.split(":")[1].strip()) for line in f if ":" in line}

def load_hyperparameters(file_path):
    with open(file_path, "r") as f:
        return {line.split(":")[0].strip(): line.split(":")[1].strip() for line in f if ":" in line}

def get_model_and_transform(model_dir):
    name = model_dir.name
    if "ConvNeXt" in name:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 43)
    elif "EfficientNet" in name:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 43)
    elif "MobileNetV3" in name:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 43)
    elif "ResNet50" in name:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 43)
    elif "SwinT" in name:
        weights = Swin_T_Weights.DEFAULT
        model = swin_t(weights=weights)
        model.head = torch.nn.Linear(model.head.in_features, 43)
    else:
        raise ValueError(f"Model not recognized in {name}")
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location="cpu"))
    model.eval()
    return model, weights.transforms()

def get_test_sample(transform, idx=0):
    dataset = GTSRB(root="./data", split="test", download=True)
    image, _ = dataset[idx]
    return transform(image).unsqueeze(0)

def measure_inference(model, input_tensor):
    process = psutil.Process()
    start_mem = process.memory_info().rss
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    end_time = time.time()
    end_mem = process.memory_info().rss
    return round(end_time - start_time, 4), round((end_mem - start_mem) / (1024**2), 2)


if __name__ == "__main__":

    # === Paths ===
    BASE_DIR = "gtsrb_recognition_runs"
    OUT_DIR = "comparative_analysis"
    os.makedirs(OUT_DIR, exist_ok=True)

    # === Processing ===
    summary = []
    for model_dir in sorted(Path(BASE_DIR).iterdir()):
        if not model_dir.is_dir():
            continue
        metrics = load_txt_metrics(model_dir / "test_metrics.txt")
        hparams = load_hyperparameters(model_dir / "hyperparameters.txt")
        model, transform = get_model_and_transform(model_dir)
        input_tensor = get_test_sample(transform)
        inf_time, inf_mem = measure_inference(model, input_tensor)
        model_size = round(os.path.getsize(model_dir / "best_model.pth") / (1024**2), 2)
        summary.append({
            "Model": model_dir.name.split("_")[0],
            "Accuracy": metrics.get("Test Accuracy", 0),
            "F1": metrics.get("Test F1 Score", 0),
            "Precision": metrics.get("Test Precision", 0),
            "Recall": metrics.get("Test Recall", 0),
            "InferenceTime(s)": inf_time,
            "InferenceRAM(MB)": inf_mem,
            "ModelSize(MB)": model_size,
            "Hyperparams": json.dumps(hparams)
        })

    # === Save CSV ===
    df = pd.DataFrame(summary)
    df.to_csv(f"{OUT_DIR}/comparative_summary.csv", index=False)

    # === Figures ===
    metrics = ["Accuracy", "F1", "InferenceTime(s)", "InferenceRAM(MB)", "ModelSize(MB)"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(df["Model"], df[metric], color="skyblue")
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.title(f"{metric} por Modelo")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{metric}_by_model.png")
        plt.close()

    # 2D Comparison: Accuracy vs Inference Time
    plt.figure(figsize=(8, 6))
    plt.scatter(df["InferenceTime(s)"], df["Accuracy"], s=100, color="green")
    for i, label in enumerate(df["Model"]):
        plt.annotate(label, (df["InferenceTime(s)"][i], df["Accuracy"][i]))
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.title("Accuracy vs Inference Time")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/accuracy_vs_inference_time.png")
    plt.close()
