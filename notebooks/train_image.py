# notebooks/train_image.py
"""
Fine-tune EfficientNet-B0 on CIFAKE + FaceForensics++ for image forgery detection.
Run from project root:  python notebooks/train_image.py

Dataset setup:
  - CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
    Place in:  data/cifake/REAL/  and  data/cifake/FAKE/
  - FaceForensics++: request access at https://github.com/ondyari/FaceForensics
    Place in:  data/ff++/REAL/  and  data/ff++/FAKE/

The script merges both datasets into a single DataLoader.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.datasets as D
import timm
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

SAVE_PATH = "models/efficientnet_forgery"
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Transforms ──────────────────────────────────────────────────────────────

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_datasets():
    datasets_train, datasets_val = [], []

    for data_dir in ["data/cifake", "data/ff++"]:
        p = Path(data_dir)
        if not p.exists():
            print(f"Warning: {data_dir} not found — skipping.")
            continue

        train_p = p / "train"
        val_p   = p / "val"

        # If no train/val split exists, use full folder (80/20 split manually)
        if train_p.exists():
            datasets_train.append(D.ImageFolder(str(train_p), transform=train_transform))
            if val_p.exists():
                datasets_val.append(D.ImageFolder(str(val_p), transform=val_transform))
        else:
            full = D.ImageFolder(str(p), transform=train_transform)
            n = len(full)
            n_train = int(0.8 * n)
            train_set, val_set = torch.utils.data.random_split(full, [n_train, n - n_train])
            # Re-apply val transform to val split
            val_set.dataset.transform = val_transform
            datasets_train.append(train_set)
            datasets_val.append(val_set)

    if not datasets_train:
        raise RuntimeError("No image datasets found. See docstring for setup instructions.")

    return ConcatDataset(datasets_train), ConcatDataset(datasets_val)


def make_balanced_sampler(dataset):
    """Weighted sampler so REAL and FAKE are seen equally regardless of class imbalance."""
    labels = []
    for ds in dataset.datasets:
        if hasattr(ds, "dataset"):           # Subset (from random_split)
            for idx in ds.indices:
                labels.append(ds.dataset.targets[idx])
        else:
            labels.extend(ds.targets)

    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train():
    train_ds, val_ds = load_datasets()
    sampler = make_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # EfficientNet-B0 with 2-class head
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model = model.to(DEVICE)

    # Freeze backbone for first 2 epochs (feature extraction), then unfreeze
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze all layers after epoch 2
        if epoch == 3:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR / 5)
            print("All layers unfrozen for fine-tuning.")

        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(lbls.numpy())

        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.4f} | AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ New best AUC={auc:.4f} — model saved to {SAVE_PATH}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()