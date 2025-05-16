import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from torchvision.datasets import GTSRB
from torch.utils.data import random_split
from datetime import datetime

# ==== Configuración ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_dir = os.path.join("runs", f"GTSRB_Distribution_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# ==== Dataset sin transformaciones ====
dataset = GTSRB(root='./data', split='train', download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = GTSRB(root='./data', split='test', download=True)

# ==== Función para graficar y guardar ====
def plot_and_save_class_distribution(dataset, title, filename_prefix, run_dir, class_names=None):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    # Gráfico
    plt.figure(figsize=(12, 5))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel("Clase")
    plt.ylabel("Número de imágenes")
    plt.title(title)
    plt.xticks(classes, rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()

    img_path = os.path.join(run_dir, f"{filename_prefix}_class_distribution.png")
    plt.savefig(img_path)
    plt.close()
    print(f"📊 Guardado gráfico: {img_path}")

    # CSV con nombres de clase
    class_labels = [class_names[c] if class_names else f"Clase_{c}" for c in classes]
    df = pd.DataFrame({"class_index": classes, "class_name": class_labels, "count": counts})
    csv_path = os.path.join(run_dir, f"{filename_prefix}_class_distribution.csv")
    df.to_csv(csv_path, index=False)
    print(f"📁 Guardado CSV: {csv_path}")


# ==== Ejecutar ====
class_names = dataset.classes

plot_and_save_class_distribution(train_dataset, "Distribución de clases - Entrenamiento", "train", run_dir, class_names)
plot_and_save_class_distribution(val_dataset, "Distribución de clases - Validación", "val", run_dir, class_names)
plot_and_save_class_distribution(test_dataset, "Distribución de clases - Test", "test", run_dir, class_names)
