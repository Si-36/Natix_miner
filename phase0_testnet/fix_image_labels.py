#!/usr/bin/env python3
"""
Fix image metadata by adding 'label' field.
All images in Roadwork dataset should have label=1 (Roadwork present).
"""

import json
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "natix" / "Roadwork" / "image"

print(f"Fixing metadata in: {cache_dir}")

json_files = list(cache_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files")

fixed_count = 0
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if label is missing or needs to be set
        if 'label' not in metadata:
            # All images in Roadwork dataset are roadwork images (label=1)
            metadata['label'] = 1
            
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            fixed_count += 1
            if fixed_count <= 5:  # Show first 5
                print(f"✅ Fixed: {json_file.name}")
    
    except Exception as e:
        print(f"❌ Error processing {json_file.name}: {e}")

print(f"\n✅ Fixed {fixed_count} metadata files")
print(f"✅ Total files: {len(json_files)}")


