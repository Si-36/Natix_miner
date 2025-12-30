#!/usr/bin/env python3
"""
Verify ALL metric computation uses streetvision/eval (eval gateway)

This prevents MCC drift by ensuring only centralized eval functions are used.
"""

import sys
import re
from pathlib import Path

def check_file_for_forbidden_patterns(filepath):
    """Check if file uses forbidden metric computation patterns"""

    forbidden_patterns = [
        # Direct sklearn imports (should use streetvision.eval instead)
        (r'from sklearn\.metrics import.*matthews', 'matthews_corrcoef should come from streetvision.eval.compute_mcc'),
        (r'matthews_corrcoef\(', 'Use streetvision.eval.compute_mcc instead'),

        # Direct confusion matrix computation (should use streetvision.eval)
        (r'confusion_matrix\([^)]*\)(?!.*#.*OK)', 'Use streetvision.eval.compute_confusion instead'),

        # Direct precision/recall/f1 (should use streetvision.eval)
        (r'from sklearn\.metrics import precision_score', 'Use streetvision.eval.compute_precision'),
        (r'from sklearn\.metrics import recall_score', 'Use streetvision.eval.compute_recall'),
        (r'from sklearn\.metrics import f1_score', 'Use streetvision.eval.compute_f1'),
    ]

    issues = []

    with open(filepath, 'r') as f:
        content = f.read()

        for pattern, message in forbidden_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'file': filepath,
                    'line': line_num,
                    'pattern': pattern,
                    'message': message,
                })

    return issues

def main():
    # Files to check (production steps)
    files_to_check = [
        'src/streetvision/pipeline/steps/train_baseline.py',
        'src/streetvision/pipeline/steps/sweep_thresholds.py',
        'src/streetvision/pipeline/steps/train_explora.py',
        'src/streetvision/pipeline/steps/calibrate_scrc.py',
        'src/streetvision/pipeline/steps/export_bundle.py',
        'scripts/train_cli_v2.py',
    ]

    print("=" * 80)
    print("EVAL GATEWAY VERIFICATION")
    print("=" * 80)
    print("\nChecking that all metric computation uses streetvision/eval...")
    print()

    all_issues = []

    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  {filepath}: File not found (skipping)")
            continue

        issues = check_file_for_forbidden_patterns(path)
        if issues:
            all_issues.extend(issues)
            print(f"❌ {filepath}: {len(issues)} issue(s) found")
            for issue in issues:
                print(f"   Line {issue['line']}: {issue['message']}")
        else:
            print(f"✅ {filepath}: OK")

    print()
    print("=" * 80)

    if all_issues:
        print(f"❌ FAILED: {len(all_issues)} eval gateway violation(s) found")
        print()
        print("FIX: All metric computation must use streetvision/eval:")
        print("  - compute_mcc()")
        print("  - compute_accuracy()")
        print("  - compute_precision()")
        print("  - compute_recall()")
        print("  - compute_f1()")
        print("  - compute_confusion()")
        print("  - compute_all_metrics()")
        print()
        sys.exit(1)
    else:
        print("✅ PASSED: All files use centralized eval gateway")
        print()
        print("This ensures:")
        print("  - No MCC drift across phases")
        print("  - Consistent metric computation")
        print("  - Single source of truth")
        print()
        sys.exit(0)

if __name__ == "__main__":
    main()
