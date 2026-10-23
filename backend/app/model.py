"""
DeepGuard Model Architecture: EfficientNet-B0 Binary Classifier
"""
from typing import Optional
from app.config import NUM_CLASSES

def build_efficientnet_b0(pretrained: bool = False, num_classes: int = NUM_CLASSES):
    """
    Builds an EfficientNet-B0 backbone adapted for binary classification (REAL vs FAKE).
    
    Args:
        pretrained: If True, loads official ImageNet pretrained feature extraction weights.
                    Set to True when initializing for fine-tuning.
        num_classes: Number of target categories (default: 2 for REAL and FAKE).
    """
    try:
        import torch.nn as nn
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        # Replace classification head with binary classification linear layer
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        return model
    except ImportError as e:
        raise RuntimeError(
            "PyTorch and torchvision are required to build the model. "
            "Please install them using: pip install torch torchvision"
        ) from e
