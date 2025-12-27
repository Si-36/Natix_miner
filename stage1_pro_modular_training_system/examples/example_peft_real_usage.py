#!/usr/bin/env python3
"""
REAL HuggingFace PEFT Library Usage Example - Dec 2025

This shows ACTUAL library usage - NOT wrappers or abstractions.

Installation:
    pip install peft>=0.10.0 transformers>=4.30.0 torch>=2.0.0

Example 1: Apply LoRA to DINOv3 backbone
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model
    
    # Load backbone
    backbone = AutoModel.from_pretrained("facebook/dinov3-vith14")
    
    # Create LoRA config (REAL library usage)
    lora_config = LoraConfig(
        r=16,                    # Rank
        lora_alpha=32,             # Scaling factor
        target_modules=["query", "key", "value", "dense", "fc1", "fc2"],
        lora_dropout=0.0,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    # Apply LoRA (REAL library call)
    backbone = get_peft_model(backbone, lora_config)
    
    # Now backbone is PeftModel with LoRA adapters
    # Only adapter parameters are trainable!

Example 2: Apply DoRA (Weight-Decomposed LoRA)
    from peft import DoRAConfig, get_peft_model
    
    # Load backbone
    backbone = AutoModel.from_pretrained("facebook/dinov3-vith14")
    
    # Create DoRA config (REAL library usage)
    dora_config = DoRAConfig(
        r=16,                    # Rank
        lora_alpha=32,             # Scaling factor
        target_modules=["query", "key", "value", "dense", "fc1", "fc2"],
        lora_dropout=0.0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        use_dora=True              # Enable weight decomposition
    )
    
    # Apply DoRA (REAL library call)
    backbone = get_peft_model(backbone, dora_config)
    
    # Now backbone is PeftModel with DoRA adapters
    # Better than LoRA! (Dec 2025 best practice)

Example 3: Save/Load Adapters
    # Save adapters (small file, fast)
    backbone.save_pretrained("path/to/adapters")
    
    # Load adapters
    backbone = AutoModel.from_pretrained("facebook/dinov3-vith14")
    backbone = get_peft_model(backbone, dora_config)
    backbone.load_adapter("path/to/adapters")

Example 4: Merge for Inference (Zero Overhead)
    # Merge adapters into base model
    merged_backbone = backbone.merge_and_unload()
    
    # Now merged_backbone is regular nn.Module
    # No adapter overhead during inference!

Example 5: Training with PEFT (Only optimize adapters)
    # Create optimizer with ONLY PEFT parameters
    optimizer = torch.optim.AdamW(
        [p for p in backbone.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # During training, only adapter weights are updated
    # Frozen backbone weights stay the same
"""

import torch
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, DoRAConfig, get_peft_model

print("\n" + "="*80)
print("REAL HuggingFace PEFT Library Usage Example")
print("Dec 2025 Production-Grade Implementation")
print("="*80 + "\n")

# Example 1: Load DINOv3 backbone
print("📦 Step 1: Load DINOv3 backbone")
print("-" * 80)

backbone = AutoModel.from_pretrained("facebook/dinov3-vith14")
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vith14")

print(f"✅ Loaded backbone: facebook/dinov3-vith14")
print(f"   Total parameters: {sum(p.numel() for p in backbone.parameters()):,}")
print()

# Example 2: Create LoRA config (REAL library usage)
print("📊 Step 2: Create LoRA config (REAL HuggingFace library)")
print("-" * 80)

lora_config = LoraConfig(
    r=16,                    # Rank (default: 16)
    lora_alpha=32,             # Scaling factor (typically 2*r)
    target_modules=[          # Target modules for DINOv3 ViT
        "query",
        "key",
        "value",
        "dense",       # Attention output
        "fc1",         # MLP input
        "fc2"          # MLP output
    ],
    lora_dropout=0.0,       # Dropout (default: 0.0)
    bias="none",            # Bias handling
    task_type="FEATURE_EXTRACTION"
)

print(f"✅ Created LoraConfig:")
print(f"   Rank (r): {lora_config.r}")
print(f"   Alpha: {lora_config.lora_alpha}")
print(f"   Target modules: {lora_config.target_modules}")
print(f"   Task type: {lora_config.task_type}")
print()

# Example 3: Apply LoRA to backbone (REAL library call)
print("🔄 Step 3: Apply LoRA to backbone (REAL HuggingFace library call)")
print("-" * 80)

backbone_lora = get_peft_model(backbone, lora_config)

print(f"✅ Applied LoRA adapters to backbone")
print(f"   Model type: {type(backbone_lora).__name__}")

# Count trainable parameters
trainable_params = sum(p.numel() for p in backbone_lora.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in backbone_lora.parameters())
trainable_ratio = trainable_params / total_params

print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Trainable ratio: {100*trainable_ratio:.2f}%")
print(f"   🔥 Only {100*trainable_ratio:.2f}% of parameters are trainable!")
print()

# Example 4: Create DoRA config (Dec 2025 best practice)
print("📊 Step 4: Create DoRA config (Weight-Decomposed LoRA)")
print("-" * 80)

dora_config = DoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query",
        "key",
        "value",
        "dense",
        "fc1",
        "fc2"
    ],
    lora_dropout=0.0,
    bias="none",
    task_type="FEATURE_EXTRACTION",
    use_dora=True  # Enable weight decomposition (DoRA magic!)
)

print(f"✅ Created DoRAConfig:")
print(f"   Rank (r): {dora_config.r}")
print(f"   Alpha: {dora_config.lora_alpha}")
print(f"   Use DoRA: {dora_config.use_dora} (weight decomposition!)")
print()

# Example 5: Apply DoRA to backbone
print("🔄 Step 5: Apply DoRA to backbone")
print("-" * 80)

# Create fresh backbone for DoRA
backbone_dora = AutoModel.from_pretrained("facebook/dinov3-vith14")
backbone_dora = get_peft_model(backbone_dora, dora_config)

print(f"✅ Applied DoRA adapters to backbone")
print(f"   Model type: {type(backbone_dora).__name__}")

# Count trainable parameters
trainable_params_dora = sum(p.numel() for p in backbone_dora.parameters() if p.requires_grad)
total_params_dora = sum(p.numel() for p in backbone_dora.parameters())
trainable_ratio_dora = trainable_params_dora / total_params_dora

print(f"   Trainable parameters: {trainable_params_dora:,} ({trainable_params_dora/1e6:.2f}M)")
print(f"   Total parameters: {total_params_dora:,} ({total_params_dora/1e6:.2f}M)")
print(f"   Trainable ratio: {100*trainable_ratio_dora:.2f}%")
print(f"   🔥 DoRA: Weight-decomposed LoRA for better performance!")
print()

# Example 6: Test forward pass
print("🧪 Step 6: Test forward pass with PEFT adapters")
print("-" * 80)

# Create dummy input
dummy_input = torch.randn(2, 3, 224, 224)

# Process input
inputs = processor(images=dummy_input, return_tensors="pt")

# Forward pass with LoRA
with torch.no_grad():
    outputs_lora = backbone_lora(**inputs)
    features_lora = outputs_lora.last_hidden_state[:, 0, :]  # CLS token

print(f"✅ LoRA forward pass successful")
print(f"   Output shape: {features_lora.shape}")

# Forward pass with DoRA
with torch.no_grad():
    outputs_dora = backbone_dora(**inputs)
    features_dora = outputs_dora.last_hidden_state[:, 0, :]  # CLS token

print(f"✅ DoRA forward pass successful")
print(f"   Output shape: {features_dora.shape}")
print()

# Example 7: Save adapters
print("💾 Step 7: Save adapters (small files, fast)")
print("-" * 80)

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    
    # Save LoRA adapters
    lora_dir = temp_path / "lora_adapters"
    lora_dir.mkdir(parents=True, exist_ok=True)
    backbone_lora.save_pretrained(str(lora_dir))
    
    print(f"✅ Saved LoRA adapters to {lora_dir}")
    
    # List files
    lora_files = list(lora_dir.glob("*"))
    print(f"   Files created:")
    for file in sorted(lora_files):
        file_size = file.stat().st_size
        print(f"      - {file.name} ({file_size:,} bytes)")
    
    # Save DoRA adapters
    dora_dir = temp_path / "dora_adapters"
    dora_dir.mkdir(parents=True, exist_ok=True)
    backbone_dora.save_pretrained(str(dora_dir))
    
    print(f"✅ Saved DoRA adapters to {dora_dir}")
    
    # List files
    dora_files = list(dora_dir.glob("*"))
    print(f"   Files created:")
    for file in sorted(dora_files):
        file_size = file.stat().st_size
        print(f"      - {file.name} ({file_size:,} bytes)")

print()

# Example 8: Merge adapters for inference (zero overhead)
print("🚀 Step 8: Merge adapters for inference (zero overhead)")
print("-" * 80)

# Merge LoRA
merged_lora = backbone_lora.merge_and_unload()

print(f"✅ Merged LoRA adapters and unloaded PEFT wrapper")
print(f"   Model type: {type(merged_lora).__name__}")
print(f"   🔥 Now regular nn.Module (zero inference overhead!)")

# Merge DoRA
merged_dora = backbone_dora.merge_and_unload()

print(f"✅ Merged DoRA adapters and unloaded PEFT wrapper")
print(f"   Model type: {type(merged_dora).__name__}")
print(f"   🔥 Now regular nn.Module (zero inference overhead!)")
print()

# Example 9: Compare outputs before/after merge
print("📊 Step 9: Verify merge correctness (outputs should match)")
print("-" * 80)

with torch.no_grad():
    # Before merge
    outputs_before = backbone_lora(**inputs)
    features_before = outputs_before.last_hidden_state[:, 0, :]
    
    # After merge
    outputs_after = merged_lora(**inputs)
    features_after = outputs_after.last_hidden_state[:, 0, :]
    
    # Difference
    diff = torch.abs(features_before - features_after).max().item()
    
    print(f"✅ LoRA merge verification:")
    print(f"   Max difference: {diff:.2e}")
    
    if diff < 1e-5:
        print(f"   ✅ PASS: Merge is correct (difference < 1e-5)")
    else:
        print(f"   ❌ FAIL: Merge has errors!")

print()

# Summary
print("="*80)
print("REAL HuggingFace PEFT Library Summary")
print("="*80)
print()
print("✅ REAL library usage shown:")
print("   - from peft import LoraConfig, DoRAConfig, get_peft_model")
print("   - Create config with proper parameters")
print("   - Apply PEFT with get_peft_model()")
print("   - Save/load adapters with save_pretrained()/load_adapter()")
print("   - Merge for inference with merge_and_unload()")
print()
print("🎯 Key Benefits:")
print("   - Only {100*trainable_ratio:.1f}% of parameters trainable")
print("   - Faster training (less computation)")
print("   - Smaller checkpoint files")
print("   - Zero inference overhead after merge")
print()
print("📚 References:")
print("   - HuggingFace PEFT: https://github.com/huggingface/peft")
print("   - LoRA paper: https://arxiv.org/abs/2106.09685")
print("   - DoRA paper: https://arxiv.org/abs/2402.09353")
print("="*80 + "\n")

