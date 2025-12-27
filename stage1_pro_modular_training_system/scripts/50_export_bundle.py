"""
Phase 1.10: Complete Bundle Export (Dec 2025 Best Practice)

Exports production bundle with all required artifacts.
Enforces mutual exclusivity: ONLY ONE policy file per bundle (Phase 1: thresholds.json).
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def validate_bundle_files(
    bundle_dir: str,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Validate all required bundle files exist (Phase 1.10).
    
    Required files for Phase 1:
    - model_best.pth
    - thresholds.json (EXACTLY ONE policy file!)
    - splits.json
    - metrics.csv (training_log.csv)
    - config.json
    - bundle.json (manifest)
    
    CRITICAL: Enforce mutual exclusivity - ONLY ONE policy file!
    - Phase 1: thresholds.json (NO gateparams.json, NO scrcparams.json)
    - Phase 3: gateparams.json (NO thresholds.json, NO scrcparams.json)
    - Phase 6: scrcparams.json (NO thresholds.json, NO gateparams.json)
    
    Args:
        bundle_dir: Bundle directory
        verbose: Print status messages
    
    Returns:
        Dict with validation results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 1.10: BUNDLE VALIDATION")
        print(f"{'='*80}")
    
    results = {
        'model_best.pth': False,
        'thresholds.json': False,
        'gateparams.json': False,
        'scrcparams.json': False,
        'splits.json': False,
        'metrics.csv': False,
        'config.json': False,
        'bundle.json': False,
        'valid': False,
        'policy_count': 0,
        'active_exit_policy': None
    }
    
    # Check required files
    required_files = [
        'model_best.pth',
        'splits.json',
        'metrics.csv',
        'config.json'
    ]
    
    for filename in required_files:
        filepath = os.path.join(bundle_dir, filename)
        if os.path.exists(filepath):
            results[filename] = True
            if verbose:
                print(f"✅ {filename}: Exists")
        else:
            if verbose:
                print(f"❌ {filename}: NOT FOUND")
    
    # Check policy files (MUTUAL EXCLUSIVITY)
    policy_files = ['thresholds.json', 'gateparams.json', 'scrcparams.json']
    
    for policy_file in policy_files:
        filepath = os.path.join(bundle_dir, policy_file)
        if os.path.exists(filepath):
            results[policy_file] = True
            results['policy_count'] += 1
            
            # Determine active_exit_policy
            if policy_file == 'thresholds.json':
                results['active_exit_policy'] = 'softmax'
            elif policy_file == 'gateparams.json':
                results['active_exit_policy'] = 'gate'
            elif policy_file == 'scrcparams.json':
                results['active_exit_policy'] = 'scrc'
            
            if verbose:
                print(f"✅ {policy_file}: Policy file found")
    
    # CRITICAL: Check mutual exclusivity
    if results['policy_count'] != 1:
        if verbose:
            print(f"❌ MUTUAL EXCLUSIVITY VIOLATION")
            print(f"   Found {results['policy_count']} policy files (expected 1)")
            print(f"   Policy files: {[f for f in policy_files if results[f]]}")
        results['valid'] = False
        return results
    else:
        if verbose:
            print(f"✅ Mutual exclusivity: {results['policy_count']} policy file")
            print(f"   Active exit policy: {results['active_exit_policy']}")
    
    # Validate Phase 1 specific requirements
    if results['active_exit_policy'] != 'softmax':
        if verbose:
            print(f"❌ Phase 1 requires 'softmax' exit policy")
            print(f"   Found: {results['active_exit_policy']}")
        results['valid'] = False
        return results
    
    # Check bundle.json (will create if missing)
    bundle_json_path = os.path.join(bundle_dir, 'bundle.json')
    if os.path.exists(bundle_json_path):
        results['bundle.json'] = True
        if verbose:
            print(f"✅ bundle.json: Exists")
    else:
        if verbose:
            print(f"⚠️  bundle.json: Will create")
    
    # Final validation
    results['valid'] = all([
        results['model_best.pth'],
        results['splits.json'],
        results['metrics.csv'],
        results['config.json'],
        results['policy_count'] == 1,
        results['active_exit_policy'] == 'softmax'
    ])
    
    if verbose:
        print(f"\n{'='*80}")
        if results['valid']:
            print(f"✅ BUNDLE VALID")
        else:
            print(f"❌ BUNDLE INVALID")
        print(f"{'='*80}")
    
    return results


def create_bundle_json(
    bundle_dir: str,
    output_dir: str,
    config_path: str,
    verbose: bool = True
) -> Dict:
    """
    Create bundle.json manifest (Phase 1.10).
    
    Contains:
    - bundle_name
    - timestamp
    - git_commit
    - active_exit_policy (softmax/gate/scrc)
    - artifacts: List of required files
    - version: 1.0
    
    Args:
        bundle_dir: Bundle directory
        output_dir: Output directory
        config_path: Path to config.json
        verbose: Print status messages
    
    Returns:
        bundle.json dict
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get git commit
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=output_dir
        )
        git_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        git_commit = "unknown"
    
    # Build bundle.json
    bundle_json = {
        'bundle_name': f"stage1_phase1_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'git_commit': git_commit,
        'active_exit_policy': 'softmax',  # Phase 1: always softmax
        'phase': 1,
        'version': '1.0',
        'artifacts': {
            'model': 'model_best.pth',
            'policy': 'thresholds.json',
            'splits': 'splits.json',
            'metrics': 'metrics.csv',
            'config': 'config.json'
        },
        'requirements': {
            'min_pytorch_version': '2.0.0',
            'min_transformers_version': '4.40.0',
            'min_timm_version': '1.0.0',
            'min_scikit_learn_version': '1.5.0'
        }
    }
    
    # Save bundle.json
    bundle_json_path = os.path.join(bundle_dir, 'bundle.json')
    
    with open(bundle_json_path, 'w') as f:
        json.dump(bundle_json, f, indent=2)
    
    if verbose:
        print(f"\nPhase 1.10: Created bundle.json")
        print(f"   Path: {bundle_json_path}")
        print(f"   Bundle Name: {bundle_json['bundle_name']}")
        print(f"   Active Exit Policy: {bundle_json['active_exit_policy']}")
        print(f"   Phase: {bundle_json['phase']}")
        print(f"   Version: {bundle_json['version']}")
        print(f"   Git Commit: {bundle_json['git_commit']}")
    
    return bundle_json


def export_bundle(
    output_dir: str,
    bundle_name: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Export complete production bundle (Phase 1.10).
    
    Creates tarball with all required artifacts.
    Enforces mutual exclusivity: ONLY ONE policy file!
    
    Args:
        output_dir: Output directory containing artifacts
        bundle_name: Custom bundle name (optional)
        verbose: Print status messages
    
    Returns:
        Path to exported bundle tarball
    """
    # Validate bundle files
    validation_results = validate_bundle_files(output_dir, verbose=verbose)
    
    if not validation_results['valid']:
        raise ValueError("Bundle validation failed. See validation results above.")
    
    # Create bundle.json
    bundle_json = create_bundle_json(
        bundle_dir=output_dir,
        output_dir=output_dir,
        config_path=os.path.join(output_dir, 'config.json'),
        verbose=verbose
    )
    
    # Bundle name
    if bundle_name is None:
        bundle_name = bundle_json['bundle_name']
    
    # Create tarball
    bundle_tarball = f"{bundle_name}.tar.gz"
    bundle_tarball_path = os.path.join(output_dir, bundle_tarball)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 1.10: EXPORTING BUNDLE")
        print(f"{'='*80}")
        print(f"Bundle Name: {bundle_name}")
        print(f"Tarball Path: {bundle_tarball_path}")
    
    # Create tarball (include all required files)
    with tarfile.open(bundle_tarball_path, "w:gz") as tar:
        # Add required files
        required_files = [
            'model_best.pth',
            'thresholds.json',
            'splits.json',
            'metrics.csv',
            'config.json',
            'bundle.json'
        ]
        
        for filename in required_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                tar.add(filepath, arcname=filename)
                if verbose:
                    print(f"   Added: {filename}")
    
    if verbose:
        # Get tarball size
        tarball_size_mb = os.path.getsize(bundle_tarball_path) / (1024 * 1024)
        print(f"\n✅ Bundle exported successfully!")
        print(f"   Tarball: {bundle_tarball_path}")
        print(f"   Size: {tarball_size_mb:.2f} MB")
        print(f"   Files: {len(required_files)}")
        print(f"   Active Exit Policy: {validation_results['active_exit_policy']}")
        print(f"{'='*80}")
    
    return bundle_tarball_path


def main():
    import tarfile
    
    parser = argparse.ArgumentParser(description="Phase 1.10: Export Production Bundle")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory containing artifacts")
    parser.add_argument("--bundle-name", type=str, default=None, help="Custom bundle name")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print status messages")
    
    args = parser.parse_args()
    
    # Export bundle
    bundle_tarball_path = export_bundle(
        output_dir=args.output_dir,
        bundle_name=args.bundle_name,
        verbose=args.verbose
    )
    
    print(f"\n✅ PHASE 1.10 COMPLETE")
    print(f"   Bundle exported to: {bundle_tarball_path}")


if __name__ == "__main__":
    main()
