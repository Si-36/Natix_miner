"""
═════════════════════════════════════════════════════════
NATIX ROADWORK DETECTION DATASET (8,549 TRAIN + 251 TEST)
═══════════════════════════════════════════════════════════

Dataset: natix-network-org/roadwork
- Training: 8,549 images (10.5 GB Parquet)
- Test: 251 images (public test set)
- Image size: 4032×3024 pixels
- Metadata: GPS (100%), Weather (60%), Daytime (60%), Scene (60%), Text (60%)

Features:
- Loads from HuggingFace Datasets
- Handles NULL metadata (40% missing in test)
- Returns 12-view extracted images
- Returns pre-encoded metadata
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
import torchvision.transforms as transforms


class NATIXRoadworkDataset(Dataset):
    """
    ══════════════════════════════════════════════════════════
    NATIX ROADWORK DATASET WITH METADATA
    ══════════════════════════════════════════════════════════
    
    Dataset Structure:
    - Training: 8,549 images with labels
    - Test: 251 images (labels hidden in competition)
    - Image size: 4032×3024 (12 MP)
    - Format: RGB color, 8-bit per channel
    
    Metadata Availability:
    - GPS: 100% available (lat, lon)
    - Weather: 60% available (40% NULL)
    - Daytime: 60% available (40% NULL)
    - Scene: 60% available (40% NULL)
    - Text: 60% available (40% NULL)
    
    Vocabularies:
    - Weather: ['sunny', 'rainy', 'foggy', 'cloudy', 'clear', 'overcast', 'snowy', 'NULL']
    - Daytime: ['day', 'night', 'dawn', 'dusk', 'light', 'NULL']
    - Scene: ['urban', 'highway', 'residential', 'rural', 'industrial', 'commercial', 'NULL']
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Any] = None,
        multi_view_extractor: Optional[Any] = None
    ):
        """
        Initialize NATIX dataset
        
        Args:
            split: 'train' or 'test'
            transform: Augmentation transform (None for val/test)
            multi_view_extractor: Multi-view extraction function (optional)
        """
        super().__init__()
        
        self.split = split
        self.transform = transform
        self.multi_view_extractor = multi_view_extractor
        
        # Vocabularies
        self.weather_vocab = ['sunny', 'rainy', 'foggy', 'cloudy',
                             'clear', 'overcast', 'snowy', 'NULL']
        self.daytime_vocab = ['day', 'night', 'dawn', 'dusk',
                             'light', 'NULL']
        self.scene_vocab = ['urban', 'highway', 'residential',
                           'rural', 'industrial', 'commercial', 'NULL']
        
        # Load from HuggingFace
        print(f"\n📦 Loading {split} split from HuggingFace...")
        self.dataset = load_dataset("natix-network-org/roadwork", split=split)
        print(f"✅ Loaded {len(self.dataset)} images ({split} split)\n")
        
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.dataset)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load image from path
        
        Args:
            image_path: Path to image file
        
        Returns:
            image: PIL Image (RGB)
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            # If image is already a tensor
            image = image_path
        
        return image
    
    def _encode_weather(self, weather: Any) -> int:
        """
        Encode weather to index
        
        Args:
            weather: Weather value (string or None)
        
        Returns:
            index: Weather vocabulary index (0-7), 7 for NULL
        """
        if weather is None or str(weather).lower() == 'null' or str(weather).strip() == '':
            return 7  # NULL index
        elif weather in self.weather_vocab:
            return self.weather_vocab.index(weather)
        else:
            return 7  # Default to NULL
        
    def _encode_daytime(self, daytime: Any) -> int:
        """
        Encode daytime to index
        
        Args:
            daytime: Daytime value (string or None)
        
        Returns:
            index: Daytime vocabulary index (0-5), 5 for NULL
        """
        if daytime is None or str(daytime).lower() == 'null' or str(daytime).strip() == '':
            return 5  # NULL index
        elif daytime in self.daytime_vocab:
            return self.daytime_vocab.index(daytime)
        else:
            return 5  # Default to NULL
    
    def _encode_scene(self, scene: Any) -> int:
        """
        Encode scene to index
        
        Args:
            scene: Scene value (string or None)
        
        Returns:
            index: Scene vocabulary index (0-6), 6 for NULL
        """
        if scene is None or str(scene).lower() == 'null' or str(scene).strip() == '':
            return 6  # NULL index
        elif scene in self.scene_vocab:
            return self.scene_vocab.index(scene)
        else:
            return 6  # Default to NULL
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get single sample
        
        Args:
            idx: Sample index
        
        Returns:
            sample: Dict with keys:
                - 'image': [12, 3, 518, 518] or original image
                - 'label': torch.Tensor (0 or 1)
                - 'metadata': Dict with GPS, weather, daytime, scene, text
                - 'index': int (original index)
        """
        # Get sample from HuggingFace dataset
        sample = self.dataset[idx]
        
        # Load image
        image = self._load_image(sample['image'])
        original_image = image
        
        # Get label (only for training split)
        if self.split == 'train':
            label = int(sample['label'])  # 0 or 1
        else:
            # Test set: label is hidden in competition
            # Use -1 as placeholder
            label = -1
        
        # Extract metadata
        gps = [
            float(sample.get('latitude', 0.0)),
            float(sample.get('longitude', 0.0))
        ]
        
        weather = sample.get('weather', 'NULL')
        weather_idx = self._encode_weather(weather)
        
        daytime = sample.get('daytime', 'NULL')
        daytime_idx = self._encode_daytime(daytime)
        
        scene = sample.get('scene', 'NULL')
        scene_idx = self._encode_scene(scene)
        
        text = sample.get('description', '')
        
        # Build metadata dictionary
        metadata = {
            'gps': gps,  # [lat, lon]
            'weather': weather_idx,  # 0-7 (7=NULL)
            'daytime': daytime_idx,  # 0-5 (5=NULL)
            'scene': scene_idx,  # 0-6 (6=NULL)
            'text': text  # String
        }
        
        # Apply augmentation (train only)
        if self.transform is not None and self.split == 'train':
            image = self.transform(image)
        
        # Apply multi-view extraction if available
        if self.multi_view_extractor is not None:
            # Extract 12 views from 4032×3024 image
            image = self.multi_view_extractor(image)  # [12, 3, 518, 518]
        
        # Build sample dictionary
        sample_dict = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': metadata,
            'index': idx
        }
        
        return sample_dict
    
    @staticmethod
    def get_gps_coordinates(dataset: Dataset, split: str = 'test') -> np.ndarray:
        """
        Extract all GPS coordinates from dataset
        
        Args:
            dataset: NATIXRoadworkDataset instance
            split: 'train' or 'test'
        
        Returns:
            gps_coords: [N, 2] array of (lat, lon) coordinates
        """
        print(f"\n📍 Extracting GPS coordinates from {split} split...")
        
        gps_coords = []
        num_samples = len(dataset)
        
        for i in range(num_samples):
            sample = dataset[i]
            lat = sample['metadata']['gps'][0]
            lon = sample['metadata']['gps'][1]
            gps_coords.append([lat, lon])
        
        gps_coords = np.array(gps_coords, dtype=np.float32)
        print(f"✅ Extracted {len(gps_coords)} GPS coordinates\n")
        
        return gps_coords
    
    @staticmethod
    def get_statistics(dataset: Dataset) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Args:
            dataset: NATIXRoadworkDataset instance
        
        Returns:
            stats: Dict with dataset statistics
        """
        print("\n📊 Computing dataset statistics...")
        
        num_samples = len(dataset)
        weather_counts = {w: 0 for w in dataset.weather_vocab}
        daytime_counts = {d: 0 for d in dataset.daytime_vocab}
        scene_counts = {s: 0 for s in dataset.scene_vocab}
        
        label_counts = {0: 0, 1: 0}
        gps_available = 0
        weather_available = 0
        daytime_available = 0
        scene_available = 0
        text_available = 0
        
        for i in range(num_samples):
            sample = dataset[i]
            metadata = sample['metadata']
            label = sample['label'].item()
            
            # Count labels
            if label >= 0:
                label_counts[label] += 1
            
            # Count weather
            weather = dataset.weather_vocab[metadata['weather']]
            weather_counts[weather] += 1
            if metadata['weather'] != 7:  # Not NULL
                weather_available += 1
            
            # Count daytime
            daytime = dataset.daytime_vocab[metadata['daytime']]
            daytime_counts[daytime] += 1
            if metadata['daytime'] != 5:  # Not NULL
                daytime_available += 1
            
            # Count scene
            scene = dataset.scene_vocab[metadata['scene']]
            scene_counts[scene] += 1
            if metadata['scene'] != 6:  # Not NULL
                scene_available += 1
            
            # Count GPS
            if metadata['gps'][0] != 0.0 or metadata['gps'][1] != 0.0:
                gps_available += 1
            
            # Count text
            if metadata['text'] and metadata['text'].strip():
                text_available += 1
        
        # Build statistics dictionary
        stats = {
            'num_samples': num_samples,
            'label_distribution': label_counts,
            'weather_distribution': weather_counts,
            'daytime_distribution': daytime_counts,
            'scene_distribution': scene_counts,
            'availability': {
                'gps': gps_available / num_samples,
                'weather': weather_available / num_samples,
                'daytime': daytime_available / num_samples,
                'scene': scene_available / num_samples,
                'text': text_available / num_samples
            }
        }
        
        # Print statistics
        print("="*60)
        print("📊 NATIX DATASET STATISTICS")
        print("="*60)
        print(f"Total Samples: {stats['num_samples']}")
        print(f"\nLabel Distribution:")
        print(f"  No Roadwork (0): {stats['label_distribution'][0]} ({100 * stats['label_distribution'][0] / num_samples:.1f}%)")
        print(f"  Roadwork (1): {stats['label_distribution'][1]} ({100 * stats['label_distribution'][1] / num_samples:.1f}%)")
        
        print(f"\nWeather Distribution:")
        for weather, count in stats['weather_distribution'].items():
            print(f"  {weather:15s}: {count:4d} ({100 * count / num_samples:5.1f}%)")
        
        print(f"\nDaytime Distribution:")
        for daytime, count in stats['daytime_distribution'].items():
            print(f"  {daytime:15s}: {count:4d} ({100 * count / num_samples:5.1f}%)")
        
        print(f"\nScene Distribution:")
        for scene, count in stats['scene_distribution'].items():
            print(f"  {scene:15s}: {count:4d} ({100 * count / num_samples:5.1f}%)")
        
        print(f"\nMetadata Availability:")
        print(f"  GPS: {100 * stats['availability']['gps']:5.1f}%")
        print(f"  Weather: {100 * stats['availability']['weather']:5.1f}%")
        print(f"  Daytime: {100 * stats['availability']['daytime']:5.1f}%")
        print(f"  Scene: {100 * stats['availability']['scene']:5.1f}%")
        print(f"  Text: {100 * stats['availability']['text']:5.1f}%")
        print("="*60 + "\n")
        
        return stats


def create_dataloaders(
    config: Dict[str, Any],
    multi_view_extractor: Optional[Any] = None
) -> tuple:
    """
    Create train, val, and test dataloaders
    
    Args:
        config: Configuration dictionary
        multi_view_extractor: Multi-view extraction function
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    from torch.utils.data import DataLoader, random_split
    
    # Load training dataset
    train_dataset = NATIXRoadworkDataset(
        split='train',
        transform=None,  # Will be applied in augmentation pipeline
        multi_view_extractor=multi_view_extractor
    )
    
    # Split train into train + val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test dataset
    test_dataset = NATIXRoadworkDataset(
        split='test',
        transform=None,
        multi_view_extractor=multi_view_extractor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print("="*60)
    print("📊 DATALOADER SUMMARY")
    print("="*60)
    print(f"Train: {len(train_subset)} samples")
    print(f"Validation: {len(val_subset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Batch Size: {config.get('batch_size', 8)}")
    print(f"Num Workers: {config.get('num_workers', 4)}")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("🧠 Testing NATIXRoadworkDataset...\n")
    
    # Create dataset
    train_dataset = NATIXRoadworkDataset(split='train')
    
    # Get sample
    sample = train_dataset[0]
    
    print("\n📋 Sample Information:")
    print(f"  Image shape: {sample['image'].shape if torch.is_tensor(sample['image']) else 'PIL Image'}")
    print(f"  Label: {sample['label'].item()}")
    print(f"  GPS: {sample['metadata']['gps']}")
    print(f"  Weather: {train_dataset.weather_vocab[sample['metadata']['weather']]}")
    print(f"  Daytime: {train_dataset.daytime_vocab[sample['metadata']['daytime']]}")
    print(f"  Scene: {train_dataset.scene_vocab[sample['metadata']['scene']]}")
    print(f"  Text: {sample['metadata']['text'][:50]}..." if len(sample['metadata']['text']) > 50 else f"  Text: {sample['metadata']['text']}")
    
    # Get statistics
    stats = NATIXRoadworkDataset.get_statistics(train_dataset)
    
    print("\n✅ Dataset test passed!\n")

