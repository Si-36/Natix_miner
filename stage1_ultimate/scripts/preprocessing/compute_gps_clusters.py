#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════
GPS CLUSTERING WITH HDBSCAN (2026 SOTA)
═════════════════════════════════════════════════════════

2026 Best Practice: HDBSCAN over K-Means for GPS data

Why HDBSCAN (2026 standard):
- Handles varying densities (real-world GPS has clusters of different sizes)
- No need to pre-specify K (unlike K-Means)
- Hierarchical clustering with stability scores
- Better noise handling (outliers detected automatically)
- Built-in Haversine distance support (geospatial-aware!)

Paper: "Comparing DBSCAN and HDBSCAN for Geospatial Clustering" (2024)
Finding: HDBSCAN adapts to varying densities, superior for real GPS data

Benefits over K-Means:
✅ No K parameter tuning (HDBSCAN finds optimal clusters)
✅ Robust to outliers (noise points handled separately)
✅ Density-adaptive (handles rural vs urban distribution)
✅ Stability scores (cluster persistence)

Expected Results:
- 4-6 optimal clusters (test set distribution)
- 1-3% noise points (outliers)
- Stability scores > 0.7 (high-quality clusters)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging

# HDBSCAN (2026 geospatial clustering standard)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️  hdbscan not installed. Install with: pip install hdbscan>=0.8.38")

# Geospatial utilities
from geopy.distance import geodesic

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset.natix_base import NATIXRoadworkDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Compute Haversine distance between two GPS coordinates
    
    Args:
        lat1, lon1: First coordinate (degrees)
        lat2, lon2: Second coordinate (degrees)
    
    Returns:
        Distance in kilometers
    """
    return geodesic((lat1, lon1), (lat2, lon2)).km


def compute_gps_clusters_hdbscan_2026(
    test_gps: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 3,
    cluster_selection_method: str = 'eom',
    prediction_data: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute GPS clusters using HDBSCAN (2026 SOTA)
    
    Args:
        test_gps: [N, 2] array of GPS coordinates (lat, lon)
        min_cluster_size: Minimum samples in a valid cluster
        min_samples: Core point threshold
        cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
        prediction_data: Enable soft clustering (probabilities)
        output_dir: Path to save cluster visualizations
    
    Returns:
        Dictionary with:
            - 'labels': [N] cluster labels (-1 for noise)
            - 'probabilities': [N] membership probabilities
            - 'persistence': [K] stability scores
            - 'cluster_centers': [K, 2] cluster centers
            - 'num_clusters': number of clusters found
            - 'num_noise': number of noise points
            - 'cluster_stats': per-cluster statistics
    """
    
    logger.info("="*60)
    logger.info("🔍 HDBSCAN Clustering (2026 SOTA)")
    logger.info("="*60)
    logger.info(f"   Input: {test_gps.shape[0]} GPS coordinates")
    logger.info(f"   min_cluster_size: {min_cluster_size}")
    logger.info(f"   min_samples: {min_samples}")
    logger.info(f"   cluster_selection_method: {cluster_selection_method}")
    
    if not HDBSCAN_AVAILABLE:
        raise ImportError(
            "HDBSCAN not available. Install with: pip install hdbscan>=0.8.38"
        )
    
    # Convert lat/lon to radians for Haversine
    coords_rad = np.radians(test_gps)
    
    # HDBSCAN clustering (2026 method)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine',  # Built-in Haversine for GPS!
        cluster_selection_method=cluster_selection_method,
        prediction_data=prediction_data,
        algorithm='boruvka_kdtree'  # Fast implementation
    )
    
    logger.info("   Running HDBSCAN...")
    
    # Fit HDBSCAN
    clusterer.fit(coords_rad)
    
    # Extract results
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    persistence = clusterer.cluster_persistence_
    
    # Get cluster statistics
    unique_labels = set(labels)
    noise_label = -1
    num_noise = (labels == noise_label).sum()
    num_clusters = len(unique_labels) - (1 if noise_label in unique_labels else 0)
    
    # Compute cluster centers
    cluster_centers = []
    cluster_stats = {}
    
    for label in unique_labels:
        if label == noise_label:
            continue
        
        # Get all points in cluster
        cluster_points = test_gps[labels == label]
        
        # Compute centroid (mean lat, lon)
        centroid = cluster_points.mean(axis=0)
        cluster_centers.append(centroid)
        
        # Compute cluster statistics
        num_points = len(cluster_points)
        
        # Compute pairwise distances within cluster
        distances = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = compute_haversine_distance(
                    cluster_points[i, 0], cluster_points[i, 1],
                    cluster_points[j, 0], cluster_points[j, 1]
                )
                distances.append(dist)
        
        cluster_radius = np.max(distances) if distances else 0.0
        cluster_std = np.std(distances) if distances else 0.0
        
        # Get stability score
        cluster_idx = list(unique_labels).index(label)
        stability = persistence[cluster_idx] if cluster_idx < len(persistence) else 0.0
        
        cluster_stats[int(label)] = {
            'center': centroid,
            'num_points': num_points,
            'radius_km': cluster_radius,
            'std_km': cluster_std,
            'stability_score': stability,
            'prob_mean': probabilities[labels == label].mean()
        }
    
    cluster_centers = np.array(cluster_centers)
    
    # Print summary
    logger.info("\n✅ HDBSCAN Results:")
    logger.info(f"   Number of clusters: {num_clusters}")
    logger.info(f"   Noise points: {num_noise} ({100 * num_noise / len(labels):.1f}%)")
    logger.info(f"   Clusters stability: {persistence.mean():.3f} (mean)")
    
    # Per-cluster statistics
    logger.info("\n📊 Per-Cluster Statistics:")
    for label, stats in cluster_stats.items():
        logger.info(f"   Cluster {label}:")
        logger.info(f"      Center: ({stats['center'][0]:.4f}, {stats['center'][1]:.4f})")
        logger.info(f"      Points: {stats['num_points']}")
        logger.info(f"      Radius: {stats['radius_km']:.2f} km")
        logger.info(f"      Std: {stats['std_km']:.2f} km")
        logger.info(f"      Stability: {stats['stability_score']:.3f}")
        logger.info(f"      Prob mean: {stats['prob_mean']:.3f}")
    
    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save labels
        labels_file = output_dir / "gps_cluster_labels.npy"
        np.save(labels_file, labels)
        logger.info(f"\n💾 Saved labels to {labels_file}")
        
        # Save cluster centers
        centers_file = output_dir / "gps_cluster_centers.npy"
        np.save(centers_file, cluster_centers)
        logger.info(f"💾 Saved cluster centers to {centers_file}")
        
        # Save metadata
        metadata = {
            'num_clusters': num_clusters,
            'num_noise': int(num_noise),
            'cluster_persistence': persistence.tolist(),
            'cluster_stats': cluster_stats
        }
        metadata_file = output_dir / "gps_cluster_metadata.json"
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata to {metadata_file}")
    
    return {
        'labels': labels,
        'probabilities': probabilities,
        'persistence': persistence,
        'cluster_centers': cluster_centers,
        'num_clusters': num_clusters,
        'num_noise': int(num_noise),
        'cluster_stats': cluster_stats
    }


def visualize_clusters(
    test_gps: np.ndarray,
    labels: np.ndarray,
    output_dir: Path
):
    """
    Visualize GPS clusters (optional)
    
    Args:
        test_gps: [N, 2] GPS coordinates
        labels: [N] cluster labels
        output_dir: Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("⚠️  matplotlib/seaborn not available, skipping visualization")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'latitude': test_gps[:, 0],
        'longitude': test_gps[:, 1],
        'cluster': labels
    })
    
    # Color map
    unique_clusters = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        color = 'black' if cluster_id == -1 else colors[i]
        label = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
        
        plt.scatter(
            cluster_data['longitude'],
            cluster_data['latitude'],
            c=[color],
            label=label,
            alpha=0.6,
            s=50
        )
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Clusters (HDBSCAN 2026)', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "gps_clusters_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Saved visualization to {output_path}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Compute GPS clusters using HDBSCAN (2026 SOTA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='test',
        choices=['train', 'test', 'val'],
        help='Dataset split to cluster'
    )
    
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=10,
        help='Minimum samples in a valid cluster'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=3,
        help='Core point threshold'
    )
    
    parser.add_argument(
        '--cluster-selection-method',
        type=str,
        default='eom',
        choices=['eom', 'leaf'],
        help='Cluster selection method (eom=Excess of Mass)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/gps_analysis',
        help='Output directory for cluster results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate cluster visualization'
    )
    
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset loading, use saved GPS file'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🔍 GPS CLUSTERING WITH HDBSCAN (2026 SOTA)")
    print("="*70)
    
    # Load GPS coordinates
    if args.skip_dataset:
        logger.info("Loading saved GPS coordinates...")
        gps_file = Path(args.output_dir) / "test_gps_coordinates.npy"
        if not gps_file.exists():
            raise FileNotFoundError(
                f"GPS file not found: {gps_file}. "
                "Run without --skip-dataset first."
            )
        test_gps = np.load(gps_file)
    else:
        logger.info(f"Loading {args.dataset} dataset...")
        dataset = NATIXRoadworkDataset(
            split=args.dataset,
            transform=None
        )
        
        # Extract GPS coordinates
        test_gps = []
        for i in tqdm(range(len(dataset)), desc="Extracting GPS"):
            sample = dataset[i]
            gps = sample['metadata']['gps']
            test_gps.append(gps)
        
        test_gps = np.array(test_gps)
        
        # Save for reuse
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gps_file = output_dir / f"{args.dataset}_gps_coordinates.npy"
        np.save(gps_file, test_gps)
        logger.info(f"💾 Saved GPS coordinates to {gps_file}")
    
    # Compute clusters
    result = compute_gps_clusters_hdbscan_2026(
        test_gps=test_gps,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        output_dir=Path(args.output_dir)
    )
    
    # Visualize if requested
    if args.visualize:
        visualize_clusters(
            test_gps=test_gps,
            labels=result['labels'],
            output_dir=Path(args.output_dir)
        )
    
    print("\n" + "="*70)
    print("✅ GPS CLUSTERING COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Input: {test_gps.shape[0]} GPS coordinates")
    print(f"   Clusters found: {result['num_clusters']}")
    print(f"   Noise points: {result['num_noise']} ({100 * result['num_noise'] / len(result['labels']):.1f}%)")
    print(f"   Mean stability: {result['persistence'].mean():.3f}")
    print(f"   Output saved to: {args.output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

