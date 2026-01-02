"""
GPS Sinusoidal Encoding
2026 implementation for geographic coordinates

Features:
- Sinusoidal encoding for latitude/longitude
- NULL-safe (handles missing GPS)
- 32 frequency bands
- 128-dim output (64 lat + 64 lon)
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class GPSSinusoidalEncoding(nn.Module):
    """
    Sinusoidal encoding for GPS coordinates
    
    Input: [B, 2] - (latitude, longitude) pairs
    Output: [B, 128] - Encoded GPS features
    """
    
    def __init__(
        self,
        num_freqs: int = 32,
        output_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        
        self.num_freqs = num_freqs
        self.output_dim = output_dim
        
        # Frequency bands (log-spaced)
        freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # Projection to output_dim
        self.projection = nn.Linear(num_freqs * 4, output_dim, bias=False)
        
        logger.info(f"GPS encoding: {num_freqs} freqs → {output_dim}-dim")
    
    def forward(self, gps_coords: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            gps_coords: [B, 2] - (lat, lon)
            mask: [B] - 1 if GPS available, 0 if NULL
        Returns:
            encoded: [B, 128]
        """
        batch_size = gps_coords.size(0)
        
        # Normalize GPS to [-1, 1]
        lat = gps_coords[:, 0:1] / 90.0  # Latitude: -90 to 90
        lon = gps_coords[:, 1:2] / 180.0  # Longitude: -180 to 180
        
        # Sinusoidal encoding
        lat_encoded = self._encode_coordinate(lat)  # [B, num_freqs*2]
        lon_encoded = self._encode_coordinate(lon)  # [B, num_freqs*2]
        
        # Concatenate
        encoded = torch.cat([lat_encoded, lon_encoded], dim=-1)  # [B, num_freqs*4]
        
        # Project to output_dim
        encoded = self.projection(encoded)  # [B, output_dim]
        
        # Handle NULL GPS
        if mask is not None:
            # Zero out features for NULL GPS
            encoded = encoded * mask.unsqueeze(-1)
        
        return encoded
    
    def _encode_coordinate(self, coord: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coord: [B, 1]
        Returns:
            encoded: [B, num_freqs*2]
        """
        # Expand: [B, 1] → [B, num_freqs]
        coord_expanded = coord * self.freq_bands.unsqueeze(0)
        
        # Sin and cos
        sin_encoding = torch.sin(coord_expanded * math.pi)
        cos_encoding = torch.cos(coord_expanded * math.pi)
        
        # Concatenate: [B, num_freqs*2]
        return torch.cat([sin_encoding, cos_encoding], dim=-1)


if __name__ == "__main__":
    print("Testing GPS Encoding...")
    encoder = GPSSinusoidalEncoding(num_freqs=32, output_dim=128)
    gps = torch.tensor([[35.6895, 51.3890], [40.7128, -74.0060]])  # Tehran, NYC
    mask = torch.tensor([1.0, 1.0])  # Both available
    encoded = encoder(gps, mask)
    print(f"✅ Input: {gps.shape}, Output: {encoded.shape}")
    print("🎉 GPS encoding test passed!")
