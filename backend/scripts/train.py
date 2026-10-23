"""
DeepGuard AI — EfficientNet-B0 Fine-Tuning Script
==================================================
Trains a genuine binary image classifier on a labeled real-versus-fake dataset.

Dataset Structure Expected:
dataset/
  train/
    REAL/  (authentic photos)
    FAKE/  (AI-generated / manipulated images)
  val/
    REAL/
    FAKE/

Usage:
    python backend/scripts/train.py --dataset_dir ./dataset --epochs 15 --batch_size 32
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure backend root is in path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import MODELS_DIR, MODEL_WEIGHTS_PATH, NUM_CLASSES
from app.model import build_efficientnet_b0
from app.transforms import get_training_transforms, get_inference_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for Real vs Fake Detection")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root directory (containing train/ and val/)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--save_path", type=str, default=str(MODEL_WEIGHTS_PATH), help="Destination path for best model weights")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()

def main():
    args = parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    # Dataset paths
    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir = os.path.join(args.dataset_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"[Error] Dataset directory must contain 'train' and 'val' subfolders. Given: {args.dataset_dir}")
        sys.exit(1)

    # Transforms (strictly identical preprocessing)
    train_transform = get_training_transforms()
    val_transform = get_inference_transforms()

    print("[Training] Loading dataset splits...")
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)

    print(f"[Training] Classes detected: {train_dataset.classes}")
    print(f"[Training] Samples: {len(train_dataset)} train, {len(val_dataset)} validation")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Build model with ImageNet pretraining
    print("[Training] Initializing EfficientNet-B0 with ImageNet pretrained weights...")
    model = build_efficientnet_b0(pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f" -> Best model saved to {args.save_path} (Val Acc: {best_val_acc * 100:.2f}%)")

    print(f"\n[Training Complete] Peak Validation Accuracy: {best_val_acc * 100:.2f}%. Model saved at: {args.save_path}")

if __name__ == "__main__":
    main()
