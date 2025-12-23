#!/usr/bin/env python3
"""
Cascade Smoke Test - Test full cascade pipeline on 20-50 sample images

Purpose: Prove that the full cascade pipeline works and fits A6000 memory (48GB).
Tests all stages: DINOv3 → RF-DETR/YOLO → GLM/Molmo → Florence-2

Usage:
    python test_cascade_small.py --test_images data/natix_official/val/images/ --num_samples 30
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

import torch
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from scripts.inference.cascade_pipeline import (
        CascadePipeline,
        CascadeResult,
        StageResult
    )
except ImportError:
    # Alternative import if running from scripts directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.inference.cascade_pipeline import (
        CascadePipeline,
        CascadeResult,
        StageResult
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cascade_test")


def get_test_images(image_dir: str, num_samples: int = 30) -> List[str]:
    """Get list of test image paths"""
    image_dir = Path(image_dir)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = [
        str(p) for p in image_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Sample random images if we have more than needed
    if len(all_images) > num_samples:
        import random
        all_images = random.sample(all_images, num_samples)
    
    logger.info(f"Found {len(all_images)} test images")
    return sorted(all_images[:num_samples])


def test_cascade(
    image_paths: List[str],
    cascade: CascadePipeline,
    log_file: str = "cascade_smoke_test.log"
) -> List[Dict[str, Any]]:
    """Run cascade inference on test images and collect metrics"""
    
    results = []
    
    logger.info(f"Testing cascade on {len(image_paths)} images...")
    
    for idx, img_path in enumerate(image_paths):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Run cascade
            start_time = time.perf_counter()
            result: CascadeResult = cascade.predict(image)
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Get GPU memory usage
            gpu_memory_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            gpu_memory_max_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            
            # Collect stage results
            stage_details = []
            for stage_result in result.stage_results:
                stage_details.append({
                    'stage': stage_result.stage,
                    'decision': stage_result.decision.value,
                    'confidence': stage_result.confidence,
                    'latency_ms': stage_result.latency_ms,
                    'details': stage_result.details
                })
            
            # Store result
            sample_result = {
                'image_id': Path(img_path).name,
                'image_path': str(img_path),
                'exit_stage': result.exit_stage,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'total_latency_ms': total_time,
                'gpu_memory_mb': gpu_memory_mb,
                'gpu_memory_max_mb': gpu_memory_max_mb,
                'stage_results': stage_details
            }
            
            results.append(sample_result)
            
            logger.info(
                f"[{idx+1}/{len(image_paths)}] {Path(img_path).name} | "
                f"Stage {result.exit_stage} | "
                f"Conf: {result.confidence:.3f} | "
                f"Time: {total_time:.1f}ms | "
                f"VRAM: {gpu_memory_max_mb:.0f}MB"
            )
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            results.append({
                'image_id': Path(img_path).name,
                'image_path': str(img_path),
                'error': str(e)
            })
    
    # Save results to log file
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Cascade test complete! Results saved to {log_file}")
    
    # Print summary statistics
    print_summary(results)
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics"""
    
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) == 0:
        logger.warning("No valid results to summarize")
        return
    
    # Stage exit distribution
    exit_stages = [r['exit_stage'] for r in valid_results]
    stage_counts = {}
    for stage in exit_stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    # Latency statistics
    latencies = [r['total_latency_ms'] for r in valid_results]
    stage_latencies = {}
    for r in valid_results:
        for sr in r['stage_results']:
            stage = sr['stage']
            if stage not in stage_latencies:
                stage_latencies[stage] = []
            stage_latencies[stage].append(sr['latency_ms'])
    
    # GPU memory
    gpu_memories = [r['gpu_memory_max_mb'] for r in valid_results if r['gpu_memory_max_mb'] > 0]
    
    print("\n" + "="*80)
    print("CASCADE SMOKE TEST SUMMARY")
    print("="*80)
    print(f"\nTotal Images Tested: {len(valid_results)}")
    
    print(f"\nExit Stage Distribution:")
    for stage in sorted(stage_counts.keys()):
        count = stage_counts[stage]
        pct = 100 * count / len(valid_results)
        print(f"  Stage {stage}: {count} images ({pct:.1f}%)")
    
    print(f"\nLatency Statistics (ms):")
    print(f"  Overall: mean={np.mean(latencies):.1f}, median={np.median(latencies):.1f}, "
          f"min={np.min(latencies):.1f}, max={np.max(latencies):.1f}")
    
    for stage in sorted(stage_latencies.keys()):
        stage_lats = stage_latencies[stage]
        print(f"  Stage {stage}: mean={np.mean(stage_lats):.1f}, median={np.median(stage_lats):.1f}")
    
    if gpu_memories:
        print(f"\nGPU Memory Usage:")
        print(f"  Max VRAM: {max(gpu_memories):.0f} MB ({max(gpu_memories)/1024:.1f} GB)")
        print(f"  Mean VRAM: {np.mean(gpu_memories):.0f} MB ({np.mean(gpu_memories)/1024:.1f} GB)")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cascade smoke test on sample images")
    parser.add_argument(
        '--test_images',
        type=str,
        default='data/natix_official/val/images/',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=30,
        help='Number of images to test (default: 30)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cascade_config.yaml',
        help='Path to cascade config file'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='cascade_smoke_test.log',
        help='Output log file path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Get test images
    test_images = get_test_images(args.test_images, args.num_samples)
    
    # Load cascade pipeline
    logger.info("Loading cascade pipeline...")
    try:
        # Get models directory (assume we're in streetvision_cascade/)
        models_dir = Path(__file__).parent / "models"
        cascade = CascadePipeline(
            config_path=args.config,
            models_dir=str(models_dir),
            device=args.device
        )
        cascade.load()
        logger.info("✅ Cascade pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load cascade: {e}")
        logger.error("Make sure all models are downloaded and config is correct")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run tests
    results = test_cascade(test_images, cascade, args.log_file)
    
    logger.info(f"✅ Smoke test complete! Check {args.log_file} for detailed results")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

