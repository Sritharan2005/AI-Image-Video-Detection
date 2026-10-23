"""
DeepGuard AI — Model Evaluation Pipeline
=========================================
Evaluates a fine-tuned EfficientNet-B0 checkpoint on a held-out test dataset split.
Computes genuine, verifiable performance metrics (Accuracy, Precision, Recall, F1-Score).

Usage:
    python backend/scripts/evaluate.py --test_dir ./dataset/test --weights_path ./backend/models/efficientnet_b0_deepguard.pth
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure backend root is in path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import MODEL_WEIGHTS_PATH, NUM_CLASSES, CLASS_NAMES
from app.model import build_efficientnet_b0
from app.transforms import get_inference_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard Model on Test Dataset")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to held-out test directory (with REAL/ and FAKE/ subfolders)")
    parser.add_argument("--weights_path", type=str, default=str(MODEL_WEIGHTS_PATH), help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()

def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Evaluation] Device: {device}")

    if not os.path.exists(args.weights_path):
        print(f"[Error] Weights file not found at: {args.weights_path}")
        print("Please train the model first or provide the correct --weights_path.")
        sys.exit(1)

    if not os.path.exists(args.test_dir):
        print(f"[Error] Test dataset directory not found at: {args.test_dir}")
        sys.exit(1)

    # Transform (strictly identical to inference)
    test_transform = get_inference_transforms()
    test_dataset = ImageFolder(root=args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[Evaluation] Test samples loaded: {len(test_dataset)}")
    print(f"[Evaluation] Classes: {test_dataset.classes}")

    # Load Model
    model = build_efficientnet_b0(pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(args.weights_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("[Evaluation] Running evaluation loop...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    # Dynamic Class indexing from ImageFolder (FAKE: 0, REAL: 1)
    fake_idx = test_dataset.class_to_idx.get("FAKE", 0)
    real_idx = test_dataset.class_to_idx.get("REAL", 1)

    tp = sum(1 for p, t in zip(all_preds, all_targets) if p == fake_idx and t == fake_idx)  # True Fake
    tn = sum(1 for p, t in zip(all_preds, all_targets) if p == real_idx and t == real_idx)  # True Real
    fp = sum(1 for p, t in zip(all_preds, all_targets) if p == fake_idx and t == real_idx)  # False Fake (Real predicted as Fake)
    fn = sum(1 for p, t in zip(all_preds, all_targets) if p == real_idx and t == fake_idx)  # False Real (Fake predicted as Real / Missed Fake)

    total = len(all_targets)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 50)
    print("      DEEPGUARD MODEL EVALUATION REPORT       ")
    print("=" * 50)
    print(f"Total Test Images Analyzed : {total}")
    print(f"Accuracy                   : {accuracy * 100:.2f}%")
    print(f"Precision (Fake class)     : {precision * 100:.2f}%")
    print(f"Recall (Fake class)        : {recall * 100:.2f}%")
    print(f"F1-Score                   : {f1:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  True FAKE  : {tp:5d}  |  False REAL (Missed Fake) : {fn:5d}")
    print(f"  False FAKE : {fp:5d}  |  True REAL                : {tn:5d}")
    print("=" * 50)

if __name__ == "__main__":
    main()
