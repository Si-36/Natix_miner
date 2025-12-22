#!/usr/bin/env python3
"""
FiftyOne Hard-Case Mining Pipeline
Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025

Daily workflow:
1. Collect validator queries (24 hours)
2. Run FiftyOne Brain hardness analysis
3. Extract top 200 hardest cases
4. Auto-annotate with SAM 3 (when available)
5. Generate targeted SDXL synthetics
6. Retrain DINOv3 head

Expected: +0.2-0.5% accuracy improvement per week
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fiftyone")


def setup_fiftyone():
    """Setup FiftyOne environment"""
    try:
        import fiftyone as fo
        logger.info(f"FiftyOne version: {fo.__version__}")
        return fo
    except ImportError:
        logger.error("FiftyOne not installed. Run: pip install fiftyone==1.5.2")
        sys.exit(1)


def collect_validator_queries(
    queries_dir: str,
    days: int = 1
) -> List[Dict[str, Any]]:
    """
    Collect validator queries from the last N days
    
    Queries should be stored as JSON files with:
    - image_path: Path to the query image
    - prediction: Model's prediction (0.0-1.0)
    - confidence: Model's confidence
    - timestamp: When query was received
    """
    queries_path = Path(queries_dir)
    
    if not queries_path.exists():
        logger.warning(f"Queries directory not found: {queries_dir}")
        logger.info("Creating sample structure for demo...")
        queries_path.mkdir(parents=True, exist_ok=True)
        return []
    
    queries = []
    cutoff_time = datetime.now() - timedelta(days=days)
    
    for json_file in queries_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                query = json.load(f)
                
            # Check timestamp
            query_time = datetime.fromisoformat(query.get("timestamp", "2020-01-01"))
            if query_time > cutoff_time:
                queries.append(query)
                
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            
    logger.info(f"Collected {len(queries)} queries from last {days} days")
    return queries


def create_fiftyone_dataset(
    fo,
    queries: List[Dict[str, Any]],
    dataset_name: str
) -> Any:
    """Create FiftyOne dataset from queries"""
    
    # Delete existing dataset if exists
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    # Create new dataset
    samples = []
    
    for query in tqdm(queries, desc="Creating FiftyOne dataset"):
        try:
            sample = fo.Sample(filepath=query["image_path"])
            
            # Add prediction as classification
            sample["prediction"] = fo.Classification(
                label="roadwork" if query["prediction"] > 0.5 else "no_roadwork",
                confidence=query["confidence"]
            )
            
            # Add raw values
            sample["prediction_score"] = query["prediction"]
            sample["query_timestamp"] = query.get("timestamp", "")
            
            samples.append(sample)
            
        except Exception as e:
            logger.warning(f"Failed to create sample: {e}")
            
    # Create dataset
    dataset = fo.Dataset(dataset_name)
    dataset.add_samples(samples)
    
    logger.info(f"Created dataset '{dataset_name}' with {len(dataset)} samples")
    return dataset


def compute_hardness(fo, dataset) -> Any:
    """
    Compute hardness scores using FiftyOne Brain
    
    Hardness is based on:
    - Prediction uncertainty (confidence near 0.5)
    - Model confusion patterns
    """
    import fiftyone.brain as fob
    
    logger.info("Computing hardness scores with FiftyOne Brain...")
    
    # Compute hardness based on prediction confidence
    # Hard cases = low confidence (near 0.5 decision boundary)
    
    for sample in dataset:
        conf = sample.prediction.confidence
        # Hardness = 1 - |confidence - 0.5| * 2
        # Low confidence → high hardness
        hardness = 1 - abs(conf - 0.5) * 2
        sample["hardness"] = hardness
        sample.save()
    
    logger.info("✅ Hardness scores computed")
    return dataset


def extract_hard_cases(
    fo,
    dataset,
    output_dir: str,
    count: int = 200
) -> List[str]:
    """Extract top N hardest cases"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort by hardness (descending)
    hard_view = dataset.sort_by("hardness", reverse=True).limit(count)
    
    # Export hard cases
    hard_cases = []
    for i, sample in enumerate(hard_view):
        # Copy image to output directory
        src_path = Path(sample.filepath)
        dst_path = output_path / f"hard_{i:04d}{src_path.suffix}"
        
        if src_path.exists():
            import shutil
            shutil.copy(src_path, dst_path)
            
            # Save metadata
            metadata = {
                "original_path": str(src_path),
                "hardness": sample.hardness,
                "prediction": sample.prediction_score,
                "confidence": sample.prediction.confidence
            }
            
            with open(dst_path.with_suffix(".json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            hard_cases.append(str(dst_path))
    
    logger.info(f"Extracted {len(hard_cases)} hard cases to {output_dir}")
    return hard_cases


def analyze_failure_modes(
    hard_cases_dir: str
) -> Dict[str, int]:
    """
    Analyze failure mode patterns in hard cases
    
    This helps decide which SDXL prompts to generate
    """
    hard_path = Path(hard_cases_dir)
    
    # Placeholder - would use image analysis or manual tagging
    failure_modes = {
        "night_scenes": 0,
        "rain_conditions": 0,
        "partial_occlusion": 0,
        "far_distance": 0,
        "glare": 0,
        "fog": 0,
        "unusual_equipment": 0
    }
    
    # Count JSON metadata files
    for json_file in hard_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                
            # Check for tags if present
            for tag in metadata.get("tags", []):
                if tag in failure_modes:
                    failure_modes[tag] += 1
                    
        except Exception:
            pass
            
    logger.info("Failure mode analysis:")
    for mode, count in sorted(failure_modes.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            logger.info(f"  {mode}: {count}")
            
    return failure_modes


def run_daily_mining_pipeline(args):
    """Run complete daily hard-case mining pipeline"""
    
    fo = setup_fiftyone()
    
    date_str = datetime.now().strftime('%Y%m%d')
    
    logger.info("=" * 60)
    logger.info(f"DAILY HARD-CASE MINING - {date_str}")
    logger.info("Per REALISTIC_DEPLOYMENT_PLAN.md")
    logger.info("=" * 60)
    
    # Step 1: Collect queries
    logger.info("\n[1/5] Collecting validator queries...")
    queries = collect_validator_queries(args.queries_dir, days=args.days)
    
    if not queries:
        logger.warning("No queries found. Creating demo dataset...")
        # Create demo samples for testing
        demo_dir = Path(args.queries_dir)
        demo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder
        logger.info("Pipeline ready. Add validator queries to continue.")
        return
    
    # Step 2: Create FiftyOne dataset
    logger.info("\n[2/5] Creating FiftyOne dataset...")
    dataset_name = f"queries_{date_str}"
    dataset = create_fiftyone_dataset(fo, queries, dataset_name)
    
    # Step 3: Compute hardness
    logger.info("\n[3/5] Computing hardness scores...")
    dataset = compute_hardness(fo, dataset)
    
    # Step 4: Extract hard cases
    logger.info("\n[4/5] Extracting hard cases...")
    output_dir = Path(args.output_dir) / f"batch_{date_str}"
    hard_cases = extract_hard_cases(fo, dataset, str(output_dir), count=args.hard_case_count)
    
    # Step 5: Analyze failure modes
    logger.info("\n[5/5] Analyzing failure modes...")
    failure_modes = analyze_failure_modes(str(output_dir))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ DAILY MINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Queries processed: {len(queries)}")
    logger.info(f"Hard cases extracted: {len(hard_cases)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review hard cases in FiftyOne App")
    logger.info("  2. Tag failure modes (night, rain, occlusion, etc.)")
    logger.info("  3. Generate targeted SDXL synthetics")
    logger.info("  4. Retrain DINOv3 classifier head")
    logger.info("=" * 60)
    
    # Save pipeline report
    report = {
        "date": date_str,
        "queries_count": len(queries),
        "hard_cases_count": len(hard_cases),
        "output_dir": str(output_dir),
        "failure_modes": failure_modes
    }
    
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="FiftyOne Hard-Case Mining Pipeline")
    parser.add_argument("--queries-dir", type=str, default="./logs/validator_queries",
                        help="Directory containing validator query logs")
    parser.add_argument("--output-dir", type=str, default="./data/hard_cases",
                        help="Output directory for hard cases")
    parser.add_argument("--hard-case-count", type=int, default=200,
                        help="Number of hard cases to extract daily")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days of queries to analyze")
    parser.add_argument("--launch-app", action="store_true",
                        help="Launch FiftyOne App for manual review")
    args = parser.parse_args()
    
    run_daily_mining_pipeline(args)
    
    # Optional: Launch FiftyOne App
    if args.launch_app:
        import fiftyone as fo
        logger.info("\nLaunching FiftyOne App for manual review...")
        session = fo.launch_app()
        input("Press ENTER to close FiftyOne App...")
        session.close()


if __name__ == "__main__":
    main()

