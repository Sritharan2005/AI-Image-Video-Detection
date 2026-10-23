"""
DeepGuard AI - Fine-Tuning Smoke Test Pipeline
==============================================
Validates the end-to-end training and inference execution loop:
1. One training batch (Forward pass -> Loss computation -> Backpropagation -> Optimizer step).
2. One validation batch (Forward pass -> Validation loss & accuracy computation).
3. Checkpoint persistence: saves checkpoint to temporary file, then loads it back.
4. Checkpoint integrity: asserts that backend/models/efficientnet_b0_deepguard.pth is untouched.
5. Inference execution: runs an end-to-end forward prediction on the loaded model.
6. Environment logging: logs seeds, PyTorch version, torchvision version, and compute device.
"""

import hashlib
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
)
from app.model import build_efficientnet_b0
from app.transforms import get_training_transforms, get_inference_transforms

SMOKE_DATASET_DIR = backend_root / "data" / "workplace_dataset"
SMOKE_CHECKPOINT_PATH = backend_root / "models" / "scratch_smoke_checkpoint.pth"

def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_smoke_test():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    print("\n" + "=" * 76)
    print("   DEEPGUARD FINE-TUNING SMOKE TEST")
    print("=" * 76)

    # 1. Environment & Package Logging
    SEED = 42
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Random Seed           : {SEED}")
    print(f"PyTorch Version       : {torch.__version__}")
    print(f"Torchvision Version   : {torchvision.__version__}")
    print(f"Compute Device        : {device}")
    print(f"Production Checkpoint : {MODEL_WEIGHTS_PATH}")

    # Check that production weights exist and record initial hash
    prod_path = Path(MODEL_WEIGHTS_PATH)
    assert prod_path.exists(), f"Production weights not found at {prod_path}"
    initial_prod_sha = get_file_sha256(prod_path)
    print(f"Production SHA-256    : {initial_prod_sha}")

    # 2. Dataset Split Check
    train_dir = SMOKE_DATASET_DIR / "train"
    val_dir = SMOKE_DATASET_DIR / "val"
    assert train_dir.exists() and val_dir.exists(), "Dataset splits not initialized."

    train_tf = get_training_transforms()
    val_tf = get_inference_transforms()

    train_ds = ImageFolder(root=str(train_dir), transform=train_tf)
    val_ds = ImageFolder(root=str(val_dir), transform=val_tf)

    print(f"Train Dataset Samples : {len(train_ds)} (Classes: {train_ds.class_to_idx})")
    print(f"Val Dataset Samples   : {len(val_ds)} (Classes: {val_ds.class_to_idx})")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # 3. Model Initialization from Base Weights
    print("\n[Step 1/5] Loading architecture and base weights...")
    model = build_efficientnet_b0(pretrained=False, num_classes=NUM_CLASSES)
    sd = torch.load(prod_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    # 4. Training Batch Smoke Pass
    print("[Step 2/5] Executing 1 mini-batch training forward/backward pass...")
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"  -> Batch Loss computed: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss computation produced NaN!"

    # 5. Validation Batch Smoke Pass
    print("[Step 3/5] Executing 1 mini-batch validation pass...")
    model.eval()
    with torch.no_grad():
        v_images, v_labels = next(iter(val_loader))
        v_images, v_labels = v_images.to(device), v_labels.to(device)
        v_outputs = model(v_images)
        v_loss = criterion(v_outputs, v_labels)
        v_probs = torch.softmax(v_outputs, dim=1)
        _, v_preds = torch.max(v_outputs, 1)
        v_acc = (v_preds == v_labels).sum().item() / v_labels.size(0)

    print(f"  -> Val Loss: {v_loss.item():.4f} | Val Accuracy: {v_acc*100:.1f}%")

    # 6. Checkpoint Save and Load
    print("[Step 4/5] Testing checkpoint save/load to temporary scratch file...")
    SMOKE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SMOKE_CHECKPOINT_PATH)
    assert SMOKE_CHECKPOINT_PATH.exists(), "Failed to save smoke checkpoint."
    print(f"  -> Checkpoint successfully saved ({SMOKE_CHECKPOINT_PATH.stat().st_size / 1024:.1f} KB)")

    loaded_model = build_efficientnet_b0(pretrained=False, num_classes=NUM_CLASSES)
    loaded_sd = torch.load(SMOKE_CHECKPOINT_PATH, map_location=device)
    loaded_model.load_state_dict(loaded_sd)
    loaded_model.to(device).eval()
    print("  -> Checkpoint successfully reloaded and verified.")

    # 7. Post-Load Inference Verification
    print("[Step 5/5] Running inference on reloaded model...")
    with torch.no_grad():
        test_out = loaded_model(v_images)
        test_probs = torch.softmax(test_out, dim=1)
    assert test_probs.shape == (v_images.size(0), 2), "Unexpected output probability tensor shape."
    print(f"  -> Inference completed successfully on batch of {v_images.size(0)} images.")

    # Verify Production Checkpoint Unchanged
    final_prod_sha = get_file_sha256(prod_path)
    assert initial_prod_sha == final_prod_sha, "CRITICAL ERROR: Production checkpoint was modified!"
    print(f"[VERIFIED] Production checkpoint preserved untouched (SHA-256 match).")

    # Clean up scratch smoke checkpoint
    if SMOKE_CHECKPOINT_PATH.exists():
        SMOKE_CHECKPOINT_PATH.unlink()
        print("  -> Temporary smoke checkpoint cleaned up.")

    print("\n" + "=" * 76)
    print("   ALL 5 SMOKE TEST STEPS PASSED SUCCESSFULLY!")
    print("=" * 76)

if __name__ == "__main__":
    run_smoke_test()
