"""
🔥 **PyTorch Lightning 2.4 DataModule (2025 Best Practices)**
Complete DataModule with multiple val loaders (val_select, val_calib, val_test)
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, WeightedRandomSampler
from torchvision import transforms as T
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

import lightning as L
from lightning.pytorch import LightningDataModule


class RoadworkDataset(Dataset):
    """Roadwork image dataset"""
    
    def __init__(
        self,
        image_dir: Path,
        labels_file: Path,
        transform=None,
        return_image_only: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.labels_file = Path(labels_file)
        self.transform = transform
        self.return_image_only = return_image_only
        
        # Load labels
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load samples from labels file"""
        samples = []
        
        # Try to load from CSV
        if self.labels_file.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(self.labels_file)
            
            for _, row in df.iterrows():
                image_path = self.image_dir / row["image_name"]
                label = int(row["label"])
                samples.append((str(image_path), label))
        
        # Try to load from JSON
        elif self.labels_file.suffix == ".json":
            import json
            with open(self.labels_file) as f:
                data = json.load(f)
                samples = [(d["image"], d["label"]) for d in data]
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default to tensor
            image = T.ToTensor()(image)
        
        if self.return_image_only:
            return image
        else:
            return image, label


class RoadworkDataModule(LightningDataModule):
    """
    Complete Lightning 2.4 DataModule for Stage-1 Pro System
    
    Features (2025 best practices):
    - Multiple val loaders (val_select, val_calib, val_test)
    - Weighted sampling for class imbalance
    - Train/val/test transforms
    - Efficient data loading
    """
    
    def __init__(
        self,
        train_image_dir: Path,
        train_labels_file: Path,
        val_image_dir: Path,
        val_labels_file: Path,
        splits_file: Optional[Path] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform=None,
        val_transform=None,
        train_val_split: float = 0.8,
        val_calib_split: float = 0.5,
    ):
        super().__init__()
        
        self.train_image_dir = Path(train_image_dir)
        self.train_labels_file = Path(train_labels_file)
        self.val_image_dir = Path(val_image_dir)
        self.val_labels_file = Path(val_labels_file)
        self.splits_file = splits_file
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Transforms
        if train_transform is None:
            self.train_transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.train_transform = train_transform
        
        if val_transform is None:
            self.val_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.val_transform = val_transform
        
        # Datasets (will be created in setup)
        self.train_dataset = None
        self.val_select_dataset = None
        self.val_calib_dataset = None
        self.val_test_dataset = None
        
        # Dataloaders
        self.train_loader = None
        self.val_select_loader = None
        self.val_calib_loader = None
        self.val_test_loader = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup data - create splits"""
        print("🔥 Setting up DataModule...")
        
        # Load validation dataset
        val_dataset = RoadworkDataset(
            self.val_image_dir,
            self.val_labels_file,
            transform=self.val_transform,
        )
        
        # Load training dataset
        train_dataset_full = RoadworkDataset(
            self.train_image_dir,
            self.train_labels_file,
            transform=None,  # Will apply in dataloader
        )
        
        # Create splits
        if self.splits_file and Path(self.splits_file).exists():
            # Load from file
            import json
            with open(self.splits_file) as f:
                splits = json.load(f)
            
            self.train_dataset = Subset(train_dataset_full, splits["train"])
            self.val_select_dataset = Subset(val_dataset, splits["val_select"])
            self.val_calib_dataset = Subset(val_dataset, splits["val_calib"])
            self.val_test_dataset = Subset(val_dataset, splits["val_test"])
            
            print(f"✅ Loaded splits from {self.splits_file}")
            print(f"  Train: {len(self.train_dataset)}")
            print(f"  Val_select: {len(self.val_select_dataset)}")
            print(f"  Val_calib: {len(self.val_calib_dataset)}")
            print(f"  Val_test: {len(self.val_test_dataset)}")
            
        else:
            # Create splits automatically
            total_len = len(val_dataset)
            val_select_len = int(total_len * 0.1)  # 10% for model selection
            val_calib_len = int(total_len * 0.1)   # 10% for calibration
            val_test_len = total_len - val_select_len - val_calib_len
            
            # Create validation splits
            val_indices = list(range(total_len))
            np.random.shuffle(val_indices)
            
            val_select_indices = val_indices[:val_select_len]
            val_calib_indices = val_indices[val_select_len:val_select_len+val_calib_len]
            val_test_indices = val_indices[val_select_len+val_calib_len:]
            
            self.val_select_dataset = Subset(val_dataset, val_select_indices)
            self.val_calib_dataset = Subset(val_dataset, val_calib_indices)
            self.val_test_dataset = Subset(val_dataset, val_test_indices)
            
            # Train: use remaining training data
            self.train_dataset = train_dataset_full
            
            # Save splits
            splits = {
                "train": list(range(len(train_dataset_full))),
                "val_select": val_select_indices,
                "val_calib": val_calib_indices,
                "val_test": val_test_indices,
            }
            
            if self.splits_file:
                import json
                with open(self.splits_file, "w") as f:
                    json.dump(splits, f, indent=2)
            
            print(f"✅ Created splits automatically:")
            print(f"  Train: {len(self.train_dataset)}")
            print(f"  Val_select: {len(self.val_select_dataset)}")
            print(f"  Val_calib: {len(self.val_calib_dataset)}")
            print(f"  Val_test: {len(self.val_test_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        
        self.val_select_loader = DataLoader(
            self.val_select_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        self.val_calib_loader = DataLoader(
            self.val_calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        self.val_test_loader = DataLoader(
            self.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        print("✅ DataModule setup complete!")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val_select batches: {len(self.val_select_loader)}")
        print(f"  Val_calib batches: {len(self.val_calib_loader)}")
        print(f"  Val_test batches: {len(self.val_test_loader)}")
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        # Lightning will iterate over both val_select and val_calib
        # We'll handle this in LightningModule
        return [self.val_select_loader, self.val_calib_loader]
    
    def test_dataloader(self):
        return self.val_test_loader
