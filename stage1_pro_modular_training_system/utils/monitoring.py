"""
Monitoring utilities for training progress and GPU usage.
"""

import torch
from tqdm import tqdm
from typing import Optional


class ProgressMonitor:
    """Progress bar monitor using tqdm"""
    
    def __init__(self, total: int, desc: str = ""):
        self.pbar = tqdm(total=total, desc=desc)
    
    def update(self, n: int = 1, **kwargs):
        """Update progress bar with metrics"""
        self.pbar.update(n)
        if kwargs:
            self.pbar.set_postfix(kwargs)
    
    def close(self):
        """Close progress bar"""
        self.pbar.close()


class GPUMonitor:
    """GPU memory monitoring"""
    
    @staticmethod
    def get_memory_usage(device: str = "cuda") -> dict:
        """
        Get current GPU memory usage.
        
        Args:
            device: Device to monitor
        
        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
        reserved = torch.cuda.memory_reserved(device) / 1e9  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9  # GB
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "available": True
        }
    
    @staticmethod
    def reset_peak_stats(device: str = "cuda"):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
