"""
Multi-View Extractor (12 views from 4032x3024 images)
Latest 2026 implementation with efficient tiling

Strategy:
- Extract 12 overlapping 518x518 views from high-res image
- 3 global views (center, full resize, random crop)
- 9 local views (3x3 grid with 25% overlap)
- LANCZOS interpolation for highest quality
- GPU-accelerated resizing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Tuple, List, Optional
import math
import logging

logger = logging.getLogger(__name__)


class MultiViewExtractor(nn.Module):
    """
    Extract 12 views from high-resolution images
    
    Input: [B, 3, 3024, 4032] (NATIX dataset dimensions)
    Output: [B, 12, 3, 518, 518] (12 views per image)
    
    View composition:
    - 3 global views: center crop, full resize, random crop
    - 9 local views: 3x3 overlapping grid (25% overlap)
    """
    
    def __init__(
        self,
        original_height: int = 3024,
        original_width: int = 4032,
        num_views: int = 12,
        view_size: int = 518,
        tile_size: int = 1344,
        overlap: float = 0.25,
        interpolation: str = 'lanczos',
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        **kwargs
    ):
        super().__init__()
        
        self.original_height = original_height
        self.original_width = original_width
        self.num_views = num_views
        self.view_size = view_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.interpolation = interpolation
        
        # ImageNet normalization (for DINOv3)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
        # Compute stride for overlapping tiles
        self.stride = int(tile_size * (1 - overlap))
        
        # Pre-compute tile coordinates for 3x3 grid
        self.tile_coords = self._compute_tile_coordinates()
        
        logger.info(f"MultiViewExtractor initialized:")
        logger.info(f"  Input: {original_height}x{original_width}")
        logger.info(f"  Output: {num_views} views of {view_size}x{view_size}")
        logger.info(f"  Tile size: {tile_size}x{tile_size}, Overlap: {overlap*100:.0f}%, Stride: {self.stride}")
    
    def _compute_tile_coordinates(self) -> List[Tuple[int, int, int, int]]:
        """
        Pre-compute coordinates for 3x3 overlapping grid
        
        Returns:
            List of (y1, y2, x1, x2) coordinates for each tile
        """
        coords = []
        
        # 3x3 grid with overlap
        for row in range(3):
            for col in range(3):
                y1 = row * self.stride
                x1 = col * self.stride
                y2 = y1 + self.tile_size
                x2 = x1 + self.tile_size
                
                # Ensure within bounds
                y2 = min(y2, self.original_height)
                x2 = min(x2, self.original_width)
                y1 = max(0, y2 - self.tile_size)
                x1 = max(0, x2 - self.tile_size)
                
                coords.append((y1, y2, x1, x2))
        
        return coords
    
    def _resize_lanczos(self, images: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        High-quality LANCZOS resizing (GPU-accelerated)
        
        Args:
            images: [B, C, H, W]
            size: (height, width)
        
        Returns:
            Resized images [B, C, size[0], size[1]]
        """
        if self.interpolation == 'lanczos':
            # Use TorchVision's high-quality resize
            return TF.resize(
                images,
                size=list(size),
                interpolation=TF.InterpolationMode.LANCZOS,
                antialias=True
            )
        elif self.interpolation == 'bicubic':
            return F.interpolate(
                images,
                size=size,
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
        else:  # bilinear
            return F.interpolate(
                images,
                size=size,
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
    
    def _extract_center_crop(self, images: torch.Tensor) -> torch.Tensor:
        """
        View 1: Center crop (1344x1344) → resize to 518x518
        Best for: Main object in center (common in NATIX dataset)
        """
        batch_size = images.size(0)
        
        # Center coordinates
        center_y = self.original_height // 2
        center_x = self.original_width // 2
        half_tile = self.tile_size // 2
        
        # Extract center crop
        y1 = center_y - half_tile
        y2 = center_y + half_tile
        x1 = center_x - half_tile
        x2 = center_x + half_tile
        
        crops = images[:, :, y1:y2, x1:x2]
        
        # Resize to view_size
        return self._resize_lanczos(crops, (self.view_size, self.view_size))
    
    def _extract_full_resize(self, images: torch.Tensor) -> torch.Tensor:
        """
        View 2: Full image resized to 518x518
        Best for: Global context, scene understanding
        """
        return self._resize_lanczos(images, (self.view_size, self.view_size))
    
    def _extract_random_crop(self, images: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        View 3: Random crop (1344x1344) → resize to 518x518
        Best for: Data augmentation during training
        During inference: Use deterministic center-right crop
        """
        batch_size = images.size(0)
        
        if training:
            # Random crop per image in batch
            crops = []
            for i in range(batch_size):
                # Random top-left corner
                max_y = self.original_height - self.tile_size
                max_x = self.original_width - self.tile_size
                y1 = torch.randint(0, max_y + 1, (1,)).item() if max_y > 0 else 0
                x1 = torch.randint(0, max_x + 1, (1,)).item() if max_x > 0 else 0
                y2 = y1 + self.tile_size
                x2 = x1 + self.tile_size
                
                crop = images[i:i+1, :, y1:y2, x1:x2]
                crops.append(crop)
            
            crops = torch.cat(crops, dim=0)
        else:
            # Deterministic: center-right crop
            center_y = self.original_height // 2
            right_x = self.original_width - self.tile_size
            half_tile = self.tile_size // 2
            
            y1 = center_y - half_tile
            y2 = center_y + half_tile
            x1 = right_x
            x2 = x1 + self.tile_size
            
            crops = images[:, :, y1:y2, x1:x2]
        
        # Resize to view_size
        return self._resize_lanczos(crops, (self.view_size, self.view_size))
    
    def _extract_local_grid(self, images: torch.Tensor) -> torch.Tensor:
        """
        Views 4-12: 3x3 overlapping grid (9 local views)
        Best for: Fine-grained details, local patterns
        
        Grid layout:
        [View 4] [View 5] [View 6]
        [View 7] [View 8] [View 9]
        [View 10] [View 11] [View 12]
        """
        batch_size = images.size(0)
        local_views = []
        
        for (y1, y2, x1, x2) in self.tile_coords:
            # Extract tile
            tile = images[:, :, y1:y2, x1:x2]
            
            # Resize to view_size
            tile_resized = self._resize_lanczos(tile, (self.view_size, self.view_size))
            local_views.append(tile_resized)
        
        # Stack: [B, 9, 3, 518, 518]
        return torch.stack(local_views, dim=1)
    
    def forward(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        return_coords: bool = False
    ) -> torch.Tensor:
        """
        Extract 12 views from images
        
        Args:
            images: [B, 3, H, W] input images (0-1 range or 0-255)
            normalize: If True, apply ImageNet normalization
            return_coords: If True, return tile coordinates
        
        Returns:
            views: [B, 12, 3, 518, 518] extracted views
            coords: (optional) List of tile coordinates
        """
        batch_size = images.size(0)
        
        # Ensure images are in [0, 1] range
        if images.max() > 1.0:
            images = images / 255.0
        
        # Extract views
        all_views = []
        
        # Global views (3 views)
        view1 = self._extract_center_crop(images)  # Center crop
        view2 = self._extract_full_resize(images)  # Full resize
        view3 = self._extract_random_crop(images, training=self.training)  # Random/deterministic crop
        
        all_views.extend([view1, view2, view3])
        
        # Local views (9 views)
        local_views = self._extract_local_grid(images)  # [B, 9, 3, 518, 518]
        
        # Unpack local views
        for i in range(9):
            all_views.append(local_views[:, i])
        
        # Stack all views: [B, 12, 3, 518, 518]
        views = torch.stack(all_views, dim=1)
        
        # Apply ImageNet normalization
        if normalize:
            # Reshape for normalization: [B*12, 3, 518, 518]
            views_flat = views.reshape(batch_size * self.num_views, 3, self.view_size, self.view_size)
            views_flat = (views_flat - self.mean) / self.std
            # Reshape back: [B, 12, 3, 518, 518]
            views = views_flat.reshape(batch_size, self.num_views, 3, self.view_size, self.view_size)
        
        if return_coords:
            coords = {
                'center_crop': 'center',
                'full_resize': 'full',
                'random_crop': 'random',
                'local_grid': self.tile_coords
            }
            return views, coords
        
        return views
    
    def get_view_names(self) -> List[str]:
        """Get descriptive names for all 12 views"""
        return [
            'center_crop',
            'full_resize',
            'random_crop',
            'grid_top_left',
            'grid_top_center',
            'grid_top_right',
            'grid_mid_left',
            'grid_mid_center',
            'grid_mid_right',
            'grid_bot_left',
            'grid_bot_center',
            'grid_bot_right'
        ]


# ══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Multi-View Extractor...")
    
    # Create extractor
    extractor = MultiViewExtractor(
        original_height=3024,
        original_width=4032,
        num_views=12,
        view_size=518,
        tile_size=1344,
        overlap=0.25
    )
    
    if torch.cuda.is_available():
        extractor = extractor.cuda()
    
    # Test input (batch of 2 images)
    batch_size = 2
    images = torch.randn(batch_size, 3, 3024, 4032)
    if torch.cuda.is_available():
        images = images.cuda()
    
    # Extract views
    views = extractor(images, normalize=True)
    print(f"✅ Extracted views: {views.shape}")  # [2, 12, 3, 518, 518]
    
    # Get view names
    view_names = extractor.get_view_names()
    print(f"✅ View names: {view_names}")
    
    # Test with coordinates
    views, coords = extractor(images, normalize=True, return_coords=True)
    print(f"✅ Views with coords: {views.shape}")
    print(f"   Grid coords: {len(coords['local_grid'])} tiles")
    
    print("\n🎉 Multi-View Extractor test passed!")
