#!/usr/bin/env python3
"""
StreetVision 4-Stage Cascade Inference Pipeline
Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025

Stage Flow:
    Input Query (224×224 image or video)
            |
            v
    ┌─────────────────────────────┐
    │ STAGE 1: DINOv3-Large       │
    │ Threshold: p ≥ 0.88 or ≤0.12│
    │ Exit: 60% of queries        │
    │ Latency: 18-25ms            │
    └─────────────────────────────┘
            |
            | 40% continue
            v
    ┌─────────────────────────────┐
    │ STAGE 2: RF-DETR + YOLOv12  │
    │ Exit: Both agree (0 or ≥3)  │
    │ Exit: 25-30% of queries     │
    │ Latency: 35-50ms            │
    └─────────────────────────────┘
            |
            | 10-15% continue
            v
    ┌─────────────────────────────┐
    │ STAGE 3: GLM-4.6V or Molmo-2│
    │ VLM reasoning for hard cases│
    │ Exit: 8-10% of queries      │
    │ Latency: 120-200ms          │
    └─────────────────────────────┘
            |
            | 2-5% continue
            v
    ┌─────────────────────────────┐
    │ STAGE 4: Florence-2-Large   │
    │ OCR keyword search fallback │
    │ Exit: 2-5% of queries       │
    │ Latency: 80-100ms           │
    └─────────────────────────────┘
            |
            v
       Final Answer
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cascade")


class CascadeDecision(Enum):
    """Cascade routing decisions"""
    EXIT_POSITIVE = "EXIT_POSITIVE"
    EXIT_NEGATIVE = "EXIT_NEGATIVE"
    CONTINUE = "CONTINUE"


@dataclass
class StageResult:
    """Result from a cascade stage"""
    decision: CascadeDecision
    confidence: float
    stage: int
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeResult:
    """Final cascade prediction result"""
    prediction: float  # 0.0 = no roadwork, 1.0 = roadwork
    confidence: float
    exit_stage: int
    total_latency_ms: float
    stage_results: list = field(default_factory=list)


class Stage1DINOv3:
    """
    Stage 1: DINOv3-Large Binary Classifier
    
    - Frozen DINOv3-Large backbone (1.3B params frozen)
    - Trainable MLP classifier head (300K params)
    - Exit threshold: p >= 0.88 or p <= 0.12 (60% exit rate)
    - Target latency: 18-25ms
    - Target accuracy on exits: 99.2%
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        positive_threshold: float = 0.88,
        negative_threshold: float = 0.12
    ):
        self.device = device
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.model = None
        self.processor = None
        self.classifier = None
        self.model_path = model_path
        
    def load(self):
        """Load DINOv3 model and classifier head"""
        logger.info("Loading Stage 1: DINOv3-Large...")
        
        from transformers import AutoModel, AutoImageProcessor
        
        # Load backbone
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Create classifier head (or load trained weights)
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            # Infer hidden size from a dummy forward pass (robust across DINOv2/DINOv3 variants)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device).half()
                try:
                    out = self.model(pixel_values=dummy)
                except TypeError:
                    out = self.model(dummy)
                if hasattr(out, "last_hidden_state"):
                    hidden_size = int(out.last_hidden_state.shape[-1])
                elif hasattr(out, "pooler_output"):
                    hidden_size = int(out.pooler_output.shape[-1])
                elif isinstance(out, torch.Tensor):
                    hidden_size = int(out.shape[-1])
                else:
                    raise RuntimeError(f"Cannot infer backbone hidden size from output type: {type(out)}")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 2)  # Binary: roadwork vs no-roadwork
        ).to(self.device).half()
        
        # Load trained classifier weights if available
        classifier_path = Path(self.model_path) / "classifier_head.pth"
        if classifier_path.exists():
            self.classifier.load_state_dict(torch.load(classifier_path))
            logger.info("   Loaded trained classifier head")
        else:
            logger.warning("   Using untrained classifier head (random weights)")
        
        self.classifier.eval()
        logger.info("   ✅ Stage 1 loaded")
        
    def predict(self, image: Image.Image) -> StageResult:
        """Run Stage 1 prediction"""
        start_time = time.perf_counter()
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device).half() for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            # Get DINOv3 features
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0]  # CLS token
            
            # Classify
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=1)
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract probabilities
        p_no_roadwork = probs[0, 0].item()
        p_roadwork = probs[0, 1].item()
        
        # Decision logic per plan
        if p_roadwork >= self.positive_threshold:
            decision = CascadeDecision.EXIT_POSITIVE
            confidence = p_roadwork
        elif p_roadwork <= self.negative_threshold:  # Equivalent to p_no_roadwork >= 0.88
            decision = CascadeDecision.EXIT_NEGATIVE
            confidence = p_no_roadwork
        else:
            decision = CascadeDecision.CONTINUE
            confidence = max(p_roadwork, p_no_roadwork)
        
        return StageResult(
            decision=decision,
            confidence=confidence,
            stage=1,
            latency_ms=latency_ms,
            details={
                "p_roadwork": p_roadwork,
                "p_no_roadwork": p_no_roadwork,
                "threshold_positive": self.positive_threshold,
                "threshold_negative": self.negative_threshold
            }
        )


class Stage2Detectors:
    """
    Stage 2: RF-DETR + YOLOv12 Detection Ensemble
    
    - Two detectors run in parallel
    - Exit if both agree (0 objects OR >= 3 objects)
    - Continue if disagreement or 1-2 objects (ambiguous)
    - Target latency: 35-50ms (parallel)
    - Target accuracy: 97%
    """
    
    def __init__(
        self,
        rfdetr_path: str,
        yolo_path: str,
        device: str = "cuda",
        detection_threshold: float = 0.4,
        agreement_threshold: int = 3
    ):
        self.device = device
        self.detection_threshold = detection_threshold
        self.agreement_threshold = agreement_threshold
        self.rfdetr_path = rfdetr_path
        self.yolo_path = yolo_path
        
        self.rfdetr_model = None
        self.rfdetr_processor = None
        self.yolo_model = None
        
        # Roadwork-related class IDs (will be populated based on model)
        self.roadwork_classes = {
            "construction", "cone", "traffic_cone", "barrier", 
            "construction_sign", "excavator", "worker", "person"
        }
        
    def load(self):
        """Load both detection models"""
        logger.info("Loading Stage 2: RF-DETR + YOLOv12...")
        
        # Load RT-DETR
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        
        self.rfdetr_model = RTDetrForObjectDetection.from_pretrained(
            self.rfdetr_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.rfdetr_model.eval()
        
        self.rfdetr_processor = RTDetrImageProcessor.from_pretrained(self.rfdetr_path)
        logger.info("   ✅ RT-DETR loaded")
        
        # Load YOLO
        from ultralytics import YOLO
        self.yolo_model = YOLO(self.yolo_path)
        logger.info("   ✅ YOLOv12 loaded")
        
    def _count_roadwork_objects(self, detections: list, class_names: dict) -> int:
        """Count roadwork-related objects in detections"""
        count = 0
        for det in detections:
            class_name = class_names.get(det.get("class_id", -1), "").lower()
            if any(rw in class_name for rw in self.roadwork_classes):
                count += 1
        return count
        
    def predict(self, image: Image.Image) -> StageResult:
        """Run Stage 2 detection ensemble"""
        start_time = time.perf_counter()
        
        # Run RT-DETR
        rfdetr_inputs = self.rfdetr_processor(images=image, return_tensors="pt")
        rfdetr_inputs = {k: v.to(self.device) for k, v in rfdetr_inputs.items()}
        
        with torch.no_grad():
            rfdetr_outputs = self.rfdetr_model(**rfdetr_inputs)
        
        # Post-process RT-DETR
        target_sizes = torch.tensor([[image.height, image.width]]).to(self.device)
        rfdetr_results = self.rfdetr_processor.post_process_object_detection(
            rfdetr_outputs, 
            threshold=self.detection_threshold,
            target_sizes=target_sizes
        )[0]
        rfdetr_count = len(rfdetr_results["boxes"])
        
        # Run YOLO
        yolo_results = self.yolo_model(image, conf=self.detection_threshold, verbose=False)
        yolo_count = len(yolo_results[0].boxes) if yolo_results else 0
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Agreement logic per plan
        if rfdetr_count == 0 and yolo_count == 0:
            # Both agree: no roadwork objects
            decision = CascadeDecision.EXIT_NEGATIVE
            confidence = 0.95
        elif rfdetr_count >= self.agreement_threshold and yolo_count >= self.agreement_threshold:
            # Both agree: many roadwork objects
            decision = CascadeDecision.EXIT_POSITIVE
            confidence = 0.95
        elif abs(rfdetr_count - yolo_count) > 2:
            # Major disagreement → need VLM
            decision = CascadeDecision.CONTINUE
            confidence = 0.5
        elif 1 <= rfdetr_count <= 2 or 1 <= yolo_count <= 2:
            # Ambiguous (few objects) → need VLM
            decision = CascadeDecision.CONTINUE
            confidence = 0.6
        else:
            # Default: trust average
            avg_count = (rfdetr_count + yolo_count) / 2
            if avg_count >= 2:
                decision = CascadeDecision.EXIT_POSITIVE
                confidence = 0.8
            else:
                decision = CascadeDecision.EXIT_NEGATIVE
                confidence = 0.7
        
        return StageResult(
            decision=decision,
            confidence=confidence,
            stage=2,
            latency_ms=latency_ms,
            details={
                "rfdetr_count": rfdetr_count,
                "yolo_count": yolo_count,
                "agreement_threshold": self.agreement_threshold
            }
        )


class Stage3VLM:
    """
    Stage 3: GLM-4.6V-Flash (images) / Molmo-2 (video)
    
    - VLM reasoning for hard cases that passed Stage 1-2
    - Image queries → GLM-4.6V
    - Video queries → Molmo-2
    - AWQ 4-bit quantization for VRAM efficiency
    - Target latency: 120-200ms
    - Target accuracy: 95%
    """
    
    def __init__(
        self,
        glm_path: str,
        molmo_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.75
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.glm_path = glm_path
        self.molmo_path = molmo_path
        
        self.glm_model = None
        self.glm_tokenizer = None
        self.molmo_model = None
        self.molmo_processor = None
        
    def load(self):
        """Load VLM models (load on-demand to save VRAM)"""
        logger.info("Loading Stage 3: VLM models...")
        logger.info("   (Models loaded on-demand to save VRAM)")
        
    def _load_glm(self):
        """Load GLM model on-demand"""
        if self.glm_model is not None:
            return
            
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("   Loading GLM-4V...")
        self.glm_tokenizer = AutoTokenizer.from_pretrained(
            self.glm_path, 
            trust_remote_code=True
        )
        self.glm_model = AutoModelForCausalLM.from_pretrained(
            self.glm_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.glm_model.eval()
        logger.info("   ✅ GLM-4V loaded")
        
    def _unload_glm(self):
        """Unload GLM to free VRAM"""
        if self.glm_model is not None:
            del self.glm_model
            del self.glm_tokenizer
            self.glm_model = None
            self.glm_tokenizer = None
            torch.cuda.empty_cache()
        
    def predict_image(self, image: Image.Image) -> StageResult:
        """Run Stage 3 VLM prediction on image"""
        start_time = time.perf_counter()
        
        self._load_glm()
        
        # Prepare prompt
        prompt = """Is there roadwork construction visible in this image? 
Consider: orange cones, barriers, construction workers, equipment.
Answer only 'yes' or 'no'."""
        
        # This is a simplified version - actual GLM-4V inference would use its chat interface
        # For now, return placeholder
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Placeholder logic (replace with actual VLM inference)
        decision = CascadeDecision.CONTINUE
        confidence = 0.5
        
        return StageResult(
            decision=decision,
            confidence=confidence,
            stage=3,
            latency_ms=latency_ms,
            details={"model": "GLM-4V", "query_type": "image"}
        )
        

class Stage4Florence:
    """
    Stage 4: Florence-2-Large OCR Fallback
    
    - OCR to find roadwork-related text in image
    - Keywords: "road work", "construction", "lane closed", etc.
    - Last resort for hardest cases
    - Target latency: 80-100ms
    - Target accuracy: 85-90%
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        self.keywords = [
            "road work", "construction", "lane closed", "detour",
            "caution", "workers ahead", "slow", "men working"
        ]
        
    def load(self):
        """Load Florence-2 model"""
        logger.info("Loading Stage 4: Florence-2-Large...")
        
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        
        logger.info("   ✅ Stage 4 loaded")
        
    def predict(self, image: Image.Image) -> StageResult:
        """Run Stage 4 OCR-based prediction"""
        start_time = time.perf_counter()
        
        # Run OCR task
        prompt = "<OCR>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Search for keywords
        text_lower = generated_text.lower()
        found_keywords = [kw for kw in self.keywords if kw.lower() in text_lower]
        
        # Decision logic per plan
        if len(found_keywords) >= 2:
            decision = CascadeDecision.EXIT_POSITIVE
            confidence = 0.85
        elif len(found_keywords) == 1:
            decision = CascadeDecision.EXIT_POSITIVE
            confidence = 0.70
        else:
            decision = CascadeDecision.EXIT_NEGATIVE
            confidence = 0.60
        
        return StageResult(
            decision=decision,
            confidence=confidence,
            stage=4,
            latency_ms=latency_ms,
            details={
                "ocr_text": generated_text[:200],
                "found_keywords": found_keywords
            }
        )


class CascadePipeline:
    """
    Complete 4-Stage Cascade Pipeline
    
    Orchestrates all stages with proper routing and early exits.
    """
    
    def __init__(
        self,
        config_path: str,
        models_dir: str,
        device: str = "cuda"
    ):
        self.device = device
        self.models_dir = Path(models_dir)
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize stages (lazy loading)
        self.stage1 = None
        self.stage2 = None
        self.stage3 = None
        self.stage4 = None
        
        self._loaded = False
        
    def load(self):
        """Load all cascade stages"""
        if self._loaded:
            return
            
        logger.info("=" * 60)
        logger.info("Loading StreetVision 4-Stage Cascade Pipeline")
        logger.info("=" * 60)
        
        # Stage 1: DINOv3 (prefer local snapshot if present)
        stage1_id = self.config["stage1"]["model"]["name"]
        stage1_local = self.models_dir / "stage1_dinov3" / stage1_id.split("/")[-1]
        stage1_path = str(stage1_local) if stage1_local.exists() else stage1_id
        self.stage1 = Stage1DINOv3(
            model_path=stage1_path,
            device=self.device,
            positive_threshold=self.config["stage1"]["thresholds"]["positive_exit"],
            negative_threshold=self.config["stage1"]["thresholds"]["negative_exit"]
        )
        self.stage1.load()
        
        # Stage 2: Detectors
        self.stage2 = Stage2Detectors(
            rfdetr_path=str(self.models_dir / "stage2_rfdetr" / "rtdetr-medium"),
            yolo_path="yolo11x.pt",  # Downloaded by ultralytics
            device=self.device
        )
        self.stage2.load()
        
        # Stage 3: VLM (lazy loaded) - prefer local snapshots if present
        glm_id = self.config["stage3"]["models"]["glm_image"]["name"]
        molmo_id = self.config["stage3"]["models"]["molmo_video"]["name"]
        glm_local = self.models_dir / "stage3_glm" / glm_id.split("/")[-1]
        molmo_local = self.models_dir / "stage3_molmo" / molmo_id.split("/")[-1]
        glm_path = str(glm_local) if glm_local.exists() else glm_id
        molmo_path = str(molmo_local) if molmo_local.exists() else molmo_id
        self.stage3 = Stage3VLM(
            glm_path=glm_path,
            molmo_path=molmo_path,
            device=self.device
        )
        self.stage3.load()
        
        # Stage 4: Florence OCR
        self.stage4 = Stage4Florence(
            model_path=str(self.models_dir / "stage4_florence" / "florence-2-large"),
            device=self.device
        )
        self.stage4.load()
        
        self._loaded = True
        logger.info("=" * 60)
        logger.info("✅ Cascade Pipeline Ready")
        logger.info("=" * 60)
        
    def predict(self, image: Image.Image) -> CascadeResult:
        """
        Run full cascade prediction
        
        Returns probability of roadwork detection [0.0, 1.0]
        """
        if not self._loaded:
            self.load()
            
        stage_results = []
        total_start = time.perf_counter()
        
        # STAGE 1: DINOv3 Binary Classifier
        result1 = self.stage1.predict(image)
        stage_results.append(result1)
        
        if result1.decision == CascadeDecision.EXIT_POSITIVE:
            return self._build_result(1.0, result1.confidence, 1, stage_results, total_start)
        elif result1.decision == CascadeDecision.EXIT_NEGATIVE:
            return self._build_result(0.0, result1.confidence, 1, stage_results, total_start)
        
        # STAGE 2: Detection Ensemble
        result2 = self.stage2.predict(image)
        stage_results.append(result2)
        
        if result2.decision == CascadeDecision.EXIT_POSITIVE:
            return self._build_result(0.95, result2.confidence, 2, stage_results, total_start)
        elif result2.decision == CascadeDecision.EXIT_NEGATIVE:
            return self._build_result(0.1, result2.confidence, 2, stage_results, total_start)
        
        # STAGE 3: VLM Reasoning
        result3 = self.stage3.predict_image(image)
        stage_results.append(result3)
        
        if result3.decision == CascadeDecision.EXIT_POSITIVE:
            return self._build_result(0.85, result3.confidence, 3, stage_results, total_start)
        elif result3.decision == CascadeDecision.EXIT_NEGATIVE:
            return self._build_result(0.15, result3.confidence, 3, stage_results, total_start)
        
        # STAGE 4: OCR Fallback
        result4 = self.stage4.predict(image)
        stage_results.append(result4)
        
        if result4.decision == CascadeDecision.EXIT_POSITIVE:
            return self._build_result(0.75, result4.confidence, 4, stage_results, total_start)
        else:
            return self._build_result(0.2, result4.confidence, 4, stage_results, total_start)
    
    def _build_result(
        self,
        prediction: float,
        confidence: float,
        exit_stage: int,
        stage_results: list,
        start_time: float
    ) -> CascadeResult:
        """Build final cascade result"""
        total_latency = (time.perf_counter() - start_time) * 1000
        
        return CascadeResult(
            prediction=prediction,
            confidence=confidence,
            exit_stage=exit_stage,
            total_latency_ms=total_latency,
            stage_results=stage_results
        )


def main():
    """Test cascade pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Cascade Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--config", type=str, default="./configs/cascade_config.yaml")
    parser.add_argument("--models-dir", type=str, default="./models")
    args = parser.parse_args()
    
    # Load pipeline
    pipeline = CascadePipeline(
        config_path=args.config,
        models_dir=args.models_dir
    )
    pipeline.load()
    
    # Load test image
    image = Image.open(args.image).convert("RGB")
    
    # Run prediction
    result = pipeline.predict(image)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Prediction (roadwork): {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Exit Stage: {result.exit_stage}")
    print(f"Total Latency: {result.total_latency_ms:.1f}ms")
    print("\nStage Details:")
    for sr in result.stage_results:
        print(f"  Stage {sr.stage}: {sr.decision.value} (conf={sr.confidence:.3f}, {sr.latency_ms:.1f}ms)")


if __name__ == "__main__":
    main()

