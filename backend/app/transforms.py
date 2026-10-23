"""
Data Preprocessing Transforms
Ensures strictly identical preprocessing across training, evaluation, and live inference.
"""
from app.config import IMG_SIZE, RESIZE_SIZE, IMAGE_NET_MEAN, IMAGE_NET_STD

def get_inference_transforms():
    """
    Standard evaluation and inference transform for EfficientNet-B0:
    1. Resize to 256
    2. Center crop to 224x224 (preserves aspect ratio without distortion)
    3. Convert to PyTorch Tensor [0.0, 1.0]
    4. Normalize using ImageNet mean & std
    """
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])
    except ImportError:
        return None

def get_training_transforms():
    """
    Data augmentation transforms for training:
    1. Random resized crop to 224x224
    2. Random horizontal flip (photometric consistency)
    3. Mild color jitter
    4. Convert to Tensor
    5. Normalize using ImageNet mean & std
    """
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])
    except ImportError:
        return None
