"""
train_swin.py
-------------
Training pipeline for the Swin Transformer DR classifier.

Architecture : swin_base_patch4_window7_224  (timm)
Dataset      : APTOS 2019 Blindness Detection
Task         : 5-class ordinal classification (DR grades 0–4)

Key techniques
--------------
  * CLAHE preprocessing on the LAB lightness channel
  * Weighted Random Sampler  → handles class imbalance at the batch level
  * Class-weighted CrossEntropyLoss → handles imbalance at the loss level
  * Mixed-precision training (torch.amp)
  * Cosine Annealing LR scheduler
  * Best model checkpoint by Quadratic Weighted Kappa (QWK)
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from timm import create_model
from torch.amp import GradScaler, autocast
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================
TRAIN_IMG_DIR = "data/train_images"
VAL_IMG_DIR   = "data/val_images"
TEST_IMG_DIR  = "data/test_images"

TRAIN_CSV = "data/train.csv"
VAL_CSV   = "data/valid.csv"
TEST_CSV  = "data/test.csv"

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 0          # set > 0 on Linux for faster loading
EPOCHS      = 10
NUM_CLASSES = 5
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH   = "models/swin_aptos_best.pth"

CLASS_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild NPDR)",
    "Grade 2 (Moderate NPDR)",
    "Grade 3 (Severe NPDR)",
    "Grade 4 (Proliferative DR)",
]


# =============================================================================
# 1. CLAHE PREPROCESSING
# =============================================================================
def apply_clahe(image_path: str, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> Image.Image:
    """
    Apply CLAHE to the LAB lightness channel of a fundus image.
    Enhances local contrast without over-amplifying noise.
    Returns a PIL RGB image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_ch  = clahe.apply(l_ch)

    img = cv2.merge((l_ch, a_ch, b_ch))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img)


# =============================================================================
# 2. DATA LOADING & CLEANING
# =============================================================================
def load_and_clean(csv_path: str, img_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"id_code": "image", "diagnosis": "level"})
    df = df[["image", "level"]]
    df["image_path"] = df["image"].apply(lambda x: os.path.join(img_dir, str(x) + ".png"))

    before = len(df)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing images.")
    return df


print("Loading datasets...")
train_df = load_and_clean(TRAIN_CSV, TRAIN_IMG_DIR)
val_df   = load_and_clean(VAL_CSV,   VAL_IMG_DIR)
test_df  = load_and_clean(TEST_CSV,  TEST_IMG_DIR)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
print(f"\nClass distribution (train):\n{train_df['level'].value_counts().sort_index()}\n")


# =============================================================================
# 3. CLASS WEIGHTS & WEIGHTED SAMPLER
# =============================================================================
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["level"].values,
)
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

sample_weights = [class_weights_np[lbl] for lbl in train_df["level"].values]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

print("Class weights for loss:")
for i, w in enumerate(class_weights_np):
    print(f"  {CLASS_NAMES[i]}: {w:.4f}")


# =============================================================================
# 4. DATASET
# =============================================================================
class DRDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, use_clahe: bool = True):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row["level"])
        image = apply_clahe(row["image_path"]) if self.use_clahe else Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# 5. TRANSFORMS
# =============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# 6. DATA LOADERS
# =============================================================================
train_dataset = DRDataset(train_df, transform=train_transforms, use_clahe=True)
val_dataset   = DRDataset(val_df,   transform=val_transforms,   use_clahe=True)
test_dataset  = DRDataset(test_df,  transform=val_transforms,   use_clahe=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,    num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,      num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,      num_workers=NUM_WORKERS, pin_memory=True)


# =============================================================================
# 7. MODEL, LOSS, OPTIMISER, SCHEDULER
# =============================================================================
print("-" * 50)
print(f"Device   : {DEVICE}")
print(f"Epochs   : {EPOCHS}")
print(f"Img size : {IMG_SIZE}x{IMG_SIZE}")
print("-" * 50)

os.makedirs("models", exist_ok=True)

model     = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = GradScaler("cuda")


# =============================================================================
# 8. TRAINING LOOP
# =============================================================================
best_kappa = -1.0
history    = {"train_loss": [], "val_loss": [], "val_kappa": [], "val_acc": []}

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, labels in train_loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * images.size(0)
        train_loop.set_postfix(loss=loss.item())

    train_loss /= len(train_dataset)
    scheduler.step()

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast("cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_dataset)
    kappa     = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    val_acc   = accuracy_score(all_labels, all_preds)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_kappa"].append(kappa)
    history["val_acc"].append(val_acc)

    print(
        f"\nEpoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2%} | "
        f"Kappa: {kappa:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if kappa > best_kappa:
        best_kappa = kappa
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ New best model saved — Kappa: {kappa:.4f}")

print(f"\nTraining complete.  Best Val Kappa: {best_kappa:.4f}")


# =============================================================================
# 9. FINAL EVALUATION ON TEST SET
# =============================================================================
print("\nRunning test evaluation...")
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.numpy())

test_kappa = cohen_kappa_score(test_labels, test_preds, weights="quadratic")
test_acc   = accuracy_score(test_labels, test_preds)

print(f"\nTest Accuracy : {test_acc:.2%}")
print(f"Test QW Kappa : {test_kappa:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))


# =============================================================================
# 10. CONFUSION MATRIX
# =============================================================================
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("DRBot – Swin Transformer Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/confusion_matrix.png", dpi=150)
print("Confusion matrix saved to assets/confusion_matrix.png")
