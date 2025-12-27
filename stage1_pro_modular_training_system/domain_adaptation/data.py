"""
Unlabeled road image dataset for ExPLoRA domain adaptation (Phase 4)

Loads unlabeled images from NATIX extras and SDXL synthetics directories.
No labels needed (self-supervised learning).
"""

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List
from ..data.transforms import TimmStyleAugmentation


class UnlabeledRoadDataset(Dataset):
    """
    Unlabeled road image dataset for ExPLoRA pretraining.
    
    Phase 4.1: Load images from NATIX extras and SDXL synthetics directories.
    Returns images only (no labels).
    """
    
    def __init__(
        self,
        natix_extras_dir: Optional[str] = None,
        sdxl_synthetics_dir: Optional[str] = None,
        augment: bool = True
    ):
        """
        Initialize unlabeled dataset.
        
        Args:
            natix_extras_dir: Path to NATIX extras directory (optional)
            sdxl_synthetics_dir: Path to SDXL synthetics directory (optional)
            augment: Whether to apply augmentation (default: True for pretraining)
        """
        self.image_paths = []
        self.augment = augment
        
        # Phase 4.1: Load from NATIX extras directory
        if natix_extras_dir and os.path.exists(natix_extras_dir):
            natix_paths = self._collect_images(natix_extras_dir)
            self.image_paths.extend(natix_paths)
            print(f"✅ Loaded {len(natix_paths)} images from NATIX extras")
        
        # Phase 4.1: Load from SDXL synthetics directory
        if sdxl_synthetics_dir and os.path.exists(sdxl_synthetics_dir):
            sdxl_paths = self._collect_images(sdxl_synthetics_dir)
            self.image_paths.extend(sdxl_paths)
            print(f"✅ Loaded {len(sdxl_paths)} images from SDXL synthetics")
        
        if len(self.image_paths) == 0:
            raise ValueError("No unlabeled images found. Provide at least one of natix_extras_dir or sdxl_synthetics_dir")
        
        print(f"✅ Total unlabeled images: {len(self.image_paths)}")
        
        # Apply same transforms as training (augmentation)
        if self.augment:
            self.transform = TimmStyleAugmentation(img_size=224, scale=(0.8, 1.0))
        else:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
    
    def _collect_images(self, directory: str) -> List[str]:
        """Collect all image file paths from directory recursively"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Return image only (no label).
        
        Args:
            idx: Index
        
        Returns:
            pixel_values: Image tensor [3, 224, 224]
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        pixel_tensor = self.transform(image)
        
        # ImageNet normalization
        if self.augment:
            # TimmStyleAugmentation already returns tensor
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pixel_values = (pixel_tensor - mean) / std
        else:
            # Manual normalization for val transform
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pixel_values = (pixel_tensor - mean) / std
        
        return pixel_values
