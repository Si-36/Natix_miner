#!/usr/bin/env python3
"""
StreetVision 6-Model Cascade - Model Downloader
Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025

Downloads all 6 models for the cascade architecture:
- Stage 1: DINOv3-Large (Vision Transformer backbone)
- Stage 2a: RF-DETR-Medium (Object Detection)
- Stage 2b: YOLOv12-X (Object Detection)
- Stage 3a: GLM-4.6V-Flash-9B (Image VLM)
- Stage 3b: Molmo-2-8B (Video VLM)
- Stage 4: Florence-2-Large (OCR)

Total download: ~31GB
Quantized total: ~21GB (fits in 24GB VRAM)
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Model configurations
MODELS = {
    "stage1_dinov3": {
        "name": "DINOv3-Large",
        # DINOv3 is available on HF, but NOT under `facebook/dinov3-large`.
        # Use a real DINOv3 checkpoint repo id (ViT-L/16 is the closest match to "Large").
        # Note: these repos are often gated ("manual") and require accepting the license on HF first.
        "hf_repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "type": "vision_encoder",
        "size_gb": 6.0,
        "quantized_gb": 3.0,
        "description": "Stage 1 backbone - Binary classifier (roadwork vs no-roadwork)",
        "required": True
    },
    "stage2_rfdetr": {
        "name": "RF-DETR-Medium (RT-DETR)",
        "hf_repo": "PekingU/rtdetr_r50vd",  # RT-DETR base
        "type": "object_detection",
        "size_gb": 3.8,
        "quantized_gb": 1.9,
        "description": "Stage 2a - Object detection ensemble partner",
        "required": True
    },
    "stage2_yolo": {
        "name": "YOLOv12-X (YOLO11x)",
        "hf_repo": None,  # Downloaded via ultralytics
        "ultralytics_model": "yolo11x.pt",
        "type": "object_detection",
        "size_gb": 6.2,
        "quantized_gb": 3.1,
        "description": "Stage 2b - Object detection ensemble partner",
        "required": True
    },
    "stage3_glm": {
        "name": "GLM-4.6V-Flash-9B",
        "hf_repo": "zai-org/GLM-4.6V-Flash",
        "type": "vision_language_model",
        "size_gb": 9.0,
        "quantized_gb": 2.3,
        "description": "Stage 3a - VLM reasoning for hard image cases",
        "required": True
    },
    "stage3_molmo": {
        "name": "Molmo-2-8B",
        "hf_repo": "allenai/Molmo2-8B",
        "type": "vision_language_model",
        "size_gb": 4.5,
        "quantized_gb": 1.2,
        "description": "Stage 3b - VLM reasoning for video queries",
        "required": True
    },
    "stage4_florence": {
        "name": "Florence-2-Large",
        "hf_repo": "microsoft/Florence-2-large",
        "type": "vision_language_model",
        "size_gb": 1.5,
        "quantized_gb": 1.5,
        "description": "Stage 4 - OCR fallback for text-based detection",
        "required": True
    }
}

def get_cache_dir():
    """Get HuggingFace cache directory"""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

def check_disk_space(required_gb: float) -> bool:
    """Check if enough disk space is available"""
    import shutil
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    total, used, free = shutil.disk_usage(cache_dir)
    free_gb = free / (1024 ** 3)
    
    print(f"💾 Disk space: {free_gb:.1f}GB free, {required_gb:.1f}GB required")
    return free_gb >= required_gb

def download_hf_model(model_id: str, model_name: str, save_dir: Path) -> bool:
    """Download model from HuggingFace Hub"""
    print(f"\n📥 Downloading {model_name} from HuggingFace...")
    print(f"   Repository: {model_id}")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download full model
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=save_dir / model_id.replace("/", "_"),
            resume_download=True,
            ignore_patterns=["*.md", "*.txt", "*.git*"],
            max_workers=int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "8")),
        )
        
        print(f"   ✅ Downloaded to: {local_dir}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download {model_name}: {e}")
        return False

def download_ultralytics_model(model_name: str, save_dir: Path) -> bool:
    """Download YOLO model via ultralytics"""
    print(f"\n📥 Downloading {model_name} via Ultralytics...")
    
    try:
        from ultralytics import YOLO
        
        # This automatically downloads the model
        model = YOLO(model_name)
        
        # Save to our directory
        model_path = save_dir / model_name
        
        print(f"   ✅ YOLO model ready: {model_name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download {model_name}: {e}")
        return False

def download_dinov2_model(save_dir: Path) -> bool:
    """Download DINOv2-Large (DINOv3 fallback)"""
    print(f"\n📥 Downloading DINOv2-Large (DINOv3 architecture)...")
    
    try:
        from transformers import AutoModel, AutoImageProcessor
        
        model_id = "facebook/dinov2-large"
        
        # Download model
        print("   Loading model weights...")
        model = AutoModel.from_pretrained(model_id)
        
        # Download processor
        print("   Loading image processor...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Save locally
        local_path = save_dir / "dinov2-large"
        model.save_pretrained(local_path)
        processor.save_pretrained(local_path)
        
        print(f"   ✅ DINOv2-Large saved to: {local_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download DINOv2: {e}")
        return False


def download_dinov3_model(save_dir: Path, model_id: str) -> bool:
    """Download DINOv3 model via snapshot_download (does not load weights into RAM)."""
    print(f"\n📥 Downloading DINOv3 backbone...")
    print(f"   Repository: {model_id}")
    try:
        from huggingface_hub import snapshot_download

        target_dir = save_dir / model_id.split("/")[-1]
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            resume_download=True,
            ignore_patterns=["*.md", "*.txt", "*.git*"],
            max_workers=int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "8")),
        )
        print(f"   ✅ Downloaded to: {local_dir}")
        return True
    except Exception as e:
        msg = str(e)
        print(f"   ❌ Failed to download DINOv3: {e}")
        if "gated" in msg.lower() or "403" in msg or "restricted" in msg.lower():
            print("   🔒 This DINOv3 repo is gated on Hugging Face (manual acceptance).")
            print("   Fix:")
            print(f"     1) Open the model page and click 'Agree' / 'Request access': {model_id}")
            print("     2) Verify you are logged in: hf auth whoami")
            print("     3) Retry the download command.")
        return False

def download_rtdetr_model(save_dir: Path) -> bool:
    """Download RT-DETR model"""
    print(f"\n📥 Downloading RT-DETR (RF-DETR equivalent)...")
    
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        
        model_id = "PekingU/rtdetr_r50vd"
        
        print("   Loading model weights...")
        model = RTDetrForObjectDetection.from_pretrained(model_id)
        
        print("   Loading image processor...")
        processor = RTDetrImageProcessor.from_pretrained(model_id)
        
        local_path = save_dir / "rtdetr-medium"
        model.save_pretrained(local_path)
        processor.save_pretrained(local_path)
        
        print(f"   ✅ RT-DETR saved to: {local_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download RT-DETR: {e}")
        return False

def download_glm_model(save_dir: Path) -> bool:
    """Download GLM-4.6V model (Stage 3a) without loading weights into RAM"""
    print(f"\n📥 Downloading GLM-4.6V-Flash-9B...")
    
    try:
        from huggingface_hub import snapshot_download

        model_id = "zai-org/GLM-4.6V-Flash"
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=save_dir / "GLM-4.6V-Flash",
            resume_download=True,
            ignore_patterns=["*.md", "*.txt", "*.git*"],
            max_workers=int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "8")),
        )

        print(f"   ✅ Downloaded to: {local_dir}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download GLM-4.6V: {e}")
        msg = str(e)
        if "401" in msg or "Invalid username or password" in msg or "not authenticated" in msg.lower():
            print("   🔐 Hugging Face auth required.")
            print("   Fix:")
            print("     1) Create a READ token at: https://huggingface.co/settings/tokens")
            print("     2) Login: hf auth login   (paste token)")
            print("     3) Verify: hf auth whoami")
            print("     4) Re-run: python scripts/download_models.py --stage 3")
        print(f"   Note: this model can require significant RAM/disk during download; retry later (downloads resume).")
        return False

def download_molmo_model(save_dir: Path) -> bool:
    """Download Molmo-2 model (Stage 3b) without loading weights into RAM"""
    print(f"\n📥 Downloading Molmo-2-8B...")
    
    try:
        from huggingface_hub import snapshot_download

        model_id = "allenai/Molmo2-8B"
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=save_dir / "Molmo2-8B",
            resume_download=True,
            ignore_patterns=["*.md", "*.txt", "*.git*"],
            max_workers=int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "8")),
        )

        print(f"   ✅ Downloaded to: {local_dir}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download Molmo: {e}")
        msg = str(e)
        if "401" in msg or "Invalid username or password" in msg or "not authenticated" in msg.lower():
            print("   🔐 Hugging Face auth required.")
            print("   Fix:")
            print("     1) Create a READ token at: https://huggingface.co/settings/tokens")
            print("     2) Login: hf auth login   (paste token)")
            print("     3) Verify: hf auth whoami")
            print("     4) Re-run: python scripts/download_models.py --stage 3")
        return False

def download_florence_model(save_dir: Path) -> bool:
    """Download Florence-2-Large model"""
    print(f"\n📥 Downloading Florence-2-Large...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        model_id = "microsoft/Florence-2-large"
        
        print("   Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        print("   Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        local_path = save_dir / "florence-2-large"
        model.save_pretrained(local_path)
        processor.save_pretrained(local_path)
        
        print(f"   ✅ Florence-2-Large saved to: {local_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download Florence-2: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download StreetVision 6-Model Cascade")
    parser.add_argument("--models-dir", type=str, default="./models",
                        help="Directory to save models")
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "4", "all"], default="all",
                        help="Which stage(s) to download")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip large VLM models (GLM, Molmo) for 8GB GPU testing")
    parser.add_argument("--dinov3-id", type=str, default=MODELS["stage1_dinov3"]["hf_repo"],
                        help="Hugging Face repo id for Stage-1 DINOv3 backbone (e.g. facebook/dinov3-vitl16-pretrain-lvd1689m)")
    parser.add_argument("--force-dinov2", action="store_true",
                        help="Force Stage-1 backbone to use facebook/dinov2-large (baseline), even if DINOv3 is available")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel download workers for HuggingFace snapshot_download (default: 8)")
    parser.add_argument("--enable-hf-transfer", action="store_true",
                        help="Enable hf_transfer accelerated downloads (requires: pip install hf_transfer)")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 StreetVision 6-Model Cascade - Model Downloader")
    print("   Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025")
    print("=" * 70)

    # Speed knobs (env-driven so helper functions don't need args threading)
    os.environ["HF_SNAPSHOT_MAX_WORKERS"] = str(args.max_workers)
    if args.enable_hf_transfer:
        try:
            import importlib.util
            if importlib.util.find_spec("hf_transfer") is None:
                raise ImportError("hf_transfer not installed")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("⚡ hf_transfer enabled (HF_HUB_ENABLE_HF_TRANSFER=1)")
        except Exception:
            print("⚠️  --enable-hf-transfer requested but hf_transfer is not installed.")
            print("   Install it with: pip install -U hf_transfer")
            print("   Continuing with standard downloader...")
    
    # Calculate total download size
    total_size = sum(m["size_gb"] for m in MODELS.values())
    print(f"\n📊 Total models: 6")
    print(f"📊 Total download size: ~{total_size:.1f}GB")
    print(f"📊 Quantized total (VRAM): ~21GB")
    
    # Check disk space
    if not check_disk_space(total_size + 10):  # 10GB buffer
        print("⚠️  Warning: Low disk space. Downloads may fail.")
    
    # Download each model
    results = {}
    
    # Stage 1: DINOv3 (or DINOv2 baseline if forced)
    if args.stage in ["1", "all"]:
        print("\n" + "=" * 70)
        print("STAGE 1: DINOv3-Large (Binary Classifier Backbone)")
        print("=" * 70)
        if args.force_dinov2:
            results["stage1_dinov3"] = download_dinov2_model(models_dir / "stage1_dinov3")
        else:
            results["stage1_dinov3"] = download_dinov3_model(models_dir / "stage1_dinov3", args.dinov3_id)
    
    # Stage 2a: RF-DETR
    if args.stage in ["2", "all"]:
        print("\n" + "=" * 70)
        print("STAGE 2a: RF-DETR-Medium (Object Detection)")
        print("=" * 70)
        results["stage2_rfdetr"] = download_rtdetr_model(models_dir / "stage2_rfdetr")
    
    # Stage 2b: YOLOv12-X
    if args.stage in ["2", "all"]:
        print("\n" + "=" * 70)
        print("STAGE 2b: YOLOv12-X (Object Detection)")
        print("=" * 70)
        results["stage2_yolo"] = download_ultralytics_model("yolo11x.pt", models_dir / "stage2_yolo")
    
    # Stage 3a: GLM-4.6V
    if args.stage in ["3", "all"] and not args.skip_large:
        print("\n" + "=" * 70)
        print("STAGE 3a: GLM-4.6V-Flash-9B (Image VLM)")
        print("=" * 70)
        results["stage3_glm"] = download_glm_model(models_dir / "stage3_glm")
    
    # Stage 3b: Molmo-2
    if args.stage in ["3", "all"] and not args.skip_large:
        print("\n" + "=" * 70)
        print("STAGE 3b: Molmo-2-8B (Video VLM)")
        print("=" * 70)
        results["stage3_molmo"] = download_molmo_model(models_dir / "stage3_molmo")
    
    # Stage 4: Florence-2
    if args.stage in ["4", "all"]:
        print("\n" + "=" * 70)
        print("STAGE 4: Florence-2-Large (OCR Fallback)")
        print("=" * 70)
        results["stage4_florence"] = download_florence_model(models_dir / "stage4_florence")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 70)
    
    for model_key, success in results.items():
        model_info = MODELS[model_key]
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {model_info['name']}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    print(f"\n   Downloaded: {successful}/{total} models")
    
    if successful == total:
        print("\n🎉 All models downloaded successfully!")
        print(f"   Models saved to: {models_dir.absolute()}")
    else:
        print("\n⚠️  Some models failed to download. Check errors above.")
        print("   You can retry failed models individually.")
    
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())

