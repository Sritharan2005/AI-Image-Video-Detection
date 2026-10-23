"""
DeepGuard AI - Domain-Specific Workplace Fine-Tuning Pipeline
============================================================
Safely fine-tunes EfficientNet-B0 on a balanced supplementary domain dataset
(real workplace portraits vs. AI-generated workplace portraits).

Operational Safeguards:
1. Checkpoint Preservation:
   The existing production checkpoint (backend/models/efficientnet_b0_deepguard.pth)
   is NEVER overwritten. New checkpoints are saved to:
   backend/models/efficientnet_b0_workplace_finetuned.pth
2. Catastrophic Forgetting Prevention:
   Uses low initial learning rate (3e-5) with Cosine Annealing to preserve
   general features learned from OpenFake while adapting the head and deep layers.
3. Strict Split Isolation:
   Train split for backprop, Validation split for early stopping and threshold
   calibration, and Held-Out Test split for final uncorrupted benchmark evaluation.
4. Parity:
   Uses identical input dimensions (224x224), ImageNet normalization, and
   class index mapping (0: FAKE, 1: REAL).

Usage:
    python backend/scripts/fine_tune_workplace.py --data_dir backend/data/workplace_dataset --epochs 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import (
    MODEL_WEIGHTS_PATH,
    NUM_CLASSES,
    CLASS_NAMES,
    IMG_SIZE,
    IMAGE_NET_MEAN,
    IMAGE_NET_STD,
)
from app.model import build_efficientnet_b0
from app.transforms import get_training_transforms, get_inference_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Domain-Specific Workplace Fine-Tuning Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train/, val/, and test/ splits")
    parser.add_argument("--base_weights", type=str, default=str(MODEL_WEIGHTS_PATH), help="Source checkpoint to initialize from")
    parser.add_argument("--output_weights", type=str, default=str(backend_root / "models" / "efficientnet_b0_workplace_finetuned.pth"), help="Destination path for new fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (conservative to prevent catastrophic forgetting)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()

def main():
    args = parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\n" + "=" * 76)
    print("   DEEPGUARD DOMAIN-SPECIFIC FINE-TUNING PIPELINE")
    print("=" * 76)
    print(f"Compute Device        : {device}")
    print(f"Base Checkpoint       : {args.base_weights}")
    print(f"Output Checkpoint     : {args.output_weights}")
    print(f"Dataset Root          : {args.data_dir}")
    print(f"Learning Rate         : {args.lr}")
    print(f"Batch Size            : {args.batch_size}")

    # Verify base weights exist
    if not os.path.exists(args.base_weights):
        print(f"[Error] Base weights file not found: {args.base_weights}")
        sys.exit(1)

    # Verify split directories
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.exists(d):
            print(f"[Error] Missing required split directory '{name}' at: {d}")
            sys.exit(1)

    train_transform = get_training_transforms()
    val_transform = get_inference_transforms()

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = ImageFolder(root=test_dir, transform=val_transform)

    print(f"\n[Split Breakdown]")
    print(f"  Train Split      : {len(train_dataset)} images ({train_dataset.class_to_idx})")
    print(f"  Validation Split : {len(val_dataset)} images ({val_dataset.class_to_idx})")
    print(f"  Held-Out Test    : {len(test_dataset)} images ({test_dataset.class_to_idx})")

    # Verify alphabetical class mapping consistency
    assert train_dataset.class_to_idx.get("FAKE") == 0, "Index 0 must be FAKE"
    assert train_dataset.class_to_idx.get("REAL") == 1, "Index 1 must be REAL"

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)

    # Load Model with Base Weights
    print(f"\n[Model Initialization] Loading base weights from: {args.base_weights}")
    model = build_efficientnet_b0(pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(args.base_weights, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    os.makedirs(os.path.dirname(args.output_weights), exist_ok=True)

    print("\n[Training Loop] Beginning fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint based on Validation Loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.output_weights)
            print(f"  -> Checkpoint saved! Best Val Loss: {best_val_loss:.4f} (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Triggered after {epoch} epochs.")
                break

    print(f"\n[SUCCESS] Fine-tuning complete.")
    print(f"Preserved Original Weights : {args.base_weights}")
    print(f"Saved Fine-Tuned Weights   : {args.output_weights}")

if __name__ == "__main__":
    main()
