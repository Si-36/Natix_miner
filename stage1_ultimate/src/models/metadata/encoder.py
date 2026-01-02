"""
Metadata Encoder (5 fields, NULL-safe)
2026 implementation

Fields:
1. GPS (lat, lon) - 100% available
2. Weather - 60% available (40% NULL)
3. Daytime - 60% available (40% NULL)
4. Scene - 60% available (40% NULL)
5. Text description - 60% available (40% NULL)

Total output: 704-dim
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging

from .gps_sinusoidal import GPSSinusoidalEncoding

logger = logging.getLogger(__name__)


class MetadataEncoder(nn.Module):
    """
    NULL-safe metadata encoder
    
    Output: 704-dim vector
    - GPS: 128-dim
    - Weather: 64-dim
    - Daytime: 64-dim  
    - Scene: 64-dim
    - Text: 384-dim
    """
    
    def __init__(
        self,
        gps_freqs: int = 32,
        gps_dim: int = 128,
        weather_vocab: int = 8,
        weather_dim: int = 64,
        daytime_vocab: int = 6,
        daytime_dim: int = 64,
        scene_vocab: int = 7,
        scene_dim: int = 64,
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_dim: int = 384,
        text_frozen: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # GPS encoding
        self.gps_encoder = GPSSinusoidalEncoding(gps_freqs, gps_dim)
        
        # Categorical embeddings (with learnable NULL token at index 0)
        self.weather_embedding = nn.Embedding(weather_vocab, weather_dim)
        self.daytime_embedding = nn.Embedding(daytime_vocab, daytime_dim)
        self.scene_embedding = nn.Embedding(scene_vocab, scene_dim)
        
        # Text encoder (Sentence-BERT)
        self.text_encoder_name = text_encoder
        self.text_model = AutoModel.from_pretrained(text_encoder)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        
        if text_frozen:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Text adapter (project 384 → 384 with learnable params)
        self.text_adapter = nn.Linear(text_dim, text_dim)
        
        self.total_dim = gps_dim + weather_dim + daytime_dim + scene_dim + text_dim
        
        logger.info(f"MetadataEncoder: {self.total_dim}-dim output")
    
    def forward(
        self,
        gps_coords: torch.Tensor,
        weather_ids: torch.Tensor,
        daytime_ids: torch.Tensor,
        scene_ids: torch.Tensor,
        text_descriptions: list,
        gps_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            gps_coords: [B, 2] - (lat, lon)
            weather_ids: [B] - 0=NULL, 1-7=weather types
            daytime_ids: [B] - 0=NULL, 1-5=daytime types
            scene_ids: [B] - 0=NULL, 1-6=scene types
            text_descriptions: List[str] - Text descriptions
            gps_mask: [B] - 1 if GPS available
        Returns:
            metadata_features: [B, 704]
        """
        batch_size = gps_coords.size(0)
        
        # GPS encoding
        gps_features = self.gps_encoder(gps_coords, gps_mask)  # [B, 128]
        
        # Categorical embeddings
        weather_features = self.weather_embedding(weather_ids)  # [B, 64]
        daytime_features = self.daytime_embedding(daytime_ids)  # [B, 64]
        scene_features = self.scene_embedding(scene_ids)  # [B, 64]
        
        # Text encoding
        text_features = self._encode_text(text_descriptions)  # [B, 384]
        text_features = self.text_adapter(text_features)
        
        # Concatenate all
        metadata_features = torch.cat([
            gps_features,
            weather_features,
            daytime_features,
            scene_features,
            text_features
        ], dim=-1)  # [B, 704]
        
        return metadata_features
    
    @torch.no_grad()
    def _encode_text(self, texts: list) -> torch.Tensor:
        """Encode text with Sentence-BERT"""
        # Tokenize
        inputs = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.text_adapter.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        outputs = self.text_model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [B, 384]
        
        return embeddings


if __name__ == "__main__":
    print("Testing MetadataEncoder...")
    encoder = MetadataEncoder()
    gps = torch.tensor([[35.6895, 51.3890], [40.7128, -74.0060]])
    weather = torch.tensor([1, 0])  # 1=sunny, 0=NULL
    daytime = torch.tensor([2, 0])  # 2=morning, 0=NULL
    scene = torch.tensor([3, 0])  # 3=urban, 0=NULL
    texts = ["Road construction", ""]
    gps_mask = torch.tensor([1.0, 1.0])
    
    features = encoder(gps, weather, daytime, scene, texts, gps_mask)
    print(f"✅ Output: {features.shape}")  # [2, 704]
    print("🎉 MetadataEncoder test passed!")
