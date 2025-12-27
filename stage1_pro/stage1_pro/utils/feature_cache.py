import torch
from typing import Optional


def save_features(features: torch.Tensor, labels: torch.Tensor, path: str):
    """Save extracted features and labels."""
    torch.save({"features": features, "labels": labels}, path)


def load_features(path: str) -> tuple:
    """Load extracted features and labels."""
    data = torch.load(path)
    return data["features"], data["labels"]


class CachedDataset(torch.utils.data.Dataset):
    """Dataset that uses pre-extracted features."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
