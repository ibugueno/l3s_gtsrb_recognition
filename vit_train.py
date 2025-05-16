import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import GTSRB
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

# ==== Carpeta de guardado con timestamp ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_dir = os.path.join("runs", f"ViT_GTSRB_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
print(f"📁 Guardando resultados en: {run_dir}")

# ==== Configuración ====
image_size = 224
batch_size = 4
num_classes = 43
num_epochs = 10
save_path = os.path.join(run_dir, 'best_vit_gtsrb.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Transformaciones ====
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==== Dataset y DataLoaders ====
dataset = GTSRB(root='./data', split='train', transform=transform, download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataset = GTSRB(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==== Modelo ====
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# ==== Funciones ====
def train(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc='Train'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Eval', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ==== Entrenamiento ====
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1:02d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✔️ Modelo mejorado guardado (Val Acc: {val_acc:.4f})")

# ==== Evaluación final ====
model.load_state_dict(torch.load(save_path))
model.eval()
test_loss, test_acc = evaluate(model, test_loader)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

# ==== Curvas de entrenamiento ====
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "training_curves.png"))
plt.show()

# ==== Matriz de Confusión ====
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation=90)
plt.title("Matriz de confusión")
plt.savefig(os.path.join(run_dir, "confusion_matrix_raw.png"))
plt.show()

# ==== Matriz de Confusión Normalizada ====
cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
disp_norm.plot(cmap='Blues', xticks_rotation=90)
plt.title("Matriz de confusión normalizada")
plt.savefig(os.path.join(run_dir, "confusion_matrix_norm.png"))
plt.show()

# ==== Exportar métricas a CSV ====
df_metrics = pd.DataFrame(history)
df_metrics.to_csv(os.path.join(run_dir, "metrics_history.csv"), index=False)
