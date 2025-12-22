#!/usr/bin/env python3
"""
SDXL Synthetic Data Generator (FREE)
Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025

Generates FREE synthetic roadwork images using Stable Diffusion XL
Saves $40 compared to using Cosmos

Output:
- 500 positive images (roadwork scenes)
- 500 negative images (normal roads)
- Total: 1,000 images for FREE

Generation time: 3-4 hours on RTX 3090 (slower on smaller GPUs)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("sdxl")

# Roadwork prompts (positive class)
ROADWORK_PROMPTS = [
    "construction workers with orange safety vests on highway, photorealistic dashcam photo, daytime, 4K",
    "orange traffic cones on urban street, road construction, realistic photo, natural lighting",
    "excavator digging on road, construction site, photorealistic, dashcam view",
    "road barrier with construction sign, highway, daytime, realistic photo",
    "construction crew working on street, orange vests, realistic dashcam image",
    "roadwork ahead sign on highway, traffic cones, photorealistic photo",
    "asphalt paving machine on road, construction, realistic image, daylight",
    "construction site with yellow excavator, urban road, photorealistic",
    "orange safety barriers on street, road work, realistic dashcam photo",
    "workers in hard hats on highway, construction, realistic photo, bright daylight",
    "lane closure with orange cones, highway construction, photorealistic dashcam",
    "road construction equipment, urban setting, daytime, realistic photo",
    "construction zone with workers and barriers, highway, photorealistic",
    "roadwork machinery on suburban street, realistic dashcam photo",
    "orange construction barriers on residential street, daytime, photorealistic",
    "road repair crew with equipment, urban road, realistic dashcam image",
    "construction warning signs on highway, orange cones, photorealistic",
    "workers laying asphalt, road construction, realistic photo, daylight",
    "road closure with construction vehicles, urban, photorealistic dashcam",
    "construction site entrance on highway, realistic photo, daytime",
]

# Normal road prompts (negative class)
NORMAL_ROAD_PROMPTS = [
    "empty highway with no construction, daytime, photorealistic dashcam photo",
    "urban street with parked cars, no construction, realistic photo, daylight",
    "residential street with trees, no roadwork, photorealistic dashcam image",
    "highway with moving traffic, no construction, realistic photo, clear day",
    "city intersection with traffic lights, no construction, photorealistic",
    "suburban road with houses, no roadwork, realistic dashcam photo, daytime",
    "country road with fields, no construction, photorealistic image",
    "downtown street with shops, no roadwork, realistic photo, daylight",
    "freeway with cars, no construction zone, photorealistic dashcam",
    "quiet street at sunset, no construction, realistic photo",
    "parking lot entrance, no roadwork, photorealistic dashcam image",
    "tree-lined avenue, no construction, realistic photo, daytime",
    "highway overpass, no roadwork, photorealistic dashcam photo",
    "rural road with farmland, no construction, realistic image",
    "beach road with ocean view, no roadwork, photorealistic dashcam",
    "mountain road with scenic view, no construction, realistic photo",
    "industrial area street, no roadwork, photorealistic dashcam image",
    "shopping center parking, no construction, realistic photo, daytime",
    "school zone street, no roadwork, photorealistic dashcam photo",
    "airport access road, no construction, realistic image, daylight",
]


def load_sdxl_pipeline(device: str = "cuda"):
    """Load Stable Diffusion XL pipeline"""
    logger.info("Loading Stable Diffusion XL pipeline...")
    
    from diffusers import StableDiffusionXLPipeline
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    logger.info("✅ SDXL pipeline loaded")
    return pipe


def generate_images(
    pipe,
    prompts: list,
    output_dir: Path,
    num_images: int,
    inference_steps: int = 30,
    seed: int = 42
):
    """Generate images from prompts"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    for i in tqdm(range(num_images), desc=f"Generating to {output_dir.name}"):
        prompt = prompts[i % len(prompts)]
        
        # Add variation to seed
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        
        # Generate
        image = pipe(
            prompt=prompt,
            num_inference_steps=inference_steps,
            generator=generator,
            guidance_scale=7.5
        ).images[0]
        
        # Save
        image_path = output_dir / f"{i:06d}.png"
        image.save(image_path)
        
        # Log progress
        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{num_images} images")


def main():
    parser = argparse.ArgumentParser(description="Generate SDXL Synthetic Data")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic_sdxl",
                        help="Output directory")
    parser.add_argument("--positive-count", type=int, default=500,
                        help="Number of positive (roadwork) images")
    parser.add_argument("--negative-count", type=int, default=500,
                        help="Number of negative (normal road) images")
    parser.add_argument("--inference-steps", type=int, default=30,
                        help="SDXL inference steps (30 recommended)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("SDXL Synthetic Data Generator (FREE)")
    logger.info("Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025")
    logger.info("=" * 60)
    logger.info(f"Positive images: {args.positive_count}")
    logger.info(f"Negative images: {args.negative_count}")
    logger.info(f"Total images: {args.positive_count + args.negative_count}")
    logger.info(f"Cost: $0 (FREE vs $40 for Cosmos)")
    logger.info("=" * 60)
    
    # Check GPU memory
    if args.device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU memory: {gpu_mem:.1f}GB")
        
        if gpu_mem < 8:
            logger.warning("⚠️ Low GPU memory. Generation may be slow or fail.")
            logger.warning("Consider using --device cpu (slower but works)")
    
    # Load pipeline
    pipe = load_sdxl_pipeline(args.device)
    
    # Generate positive images (roadwork)
    logger.info("\n" + "=" * 60)
    logger.info("Generating POSITIVE images (roadwork scenes)")
    logger.info("=" * 60)
    
    generate_images(
        pipe=pipe,
        prompts=ROADWORK_PROMPTS,
        output_dir=output_dir / "positive",
        num_images=args.positive_count,
        inference_steps=args.inference_steps,
        seed=args.seed
    )
    
    # Generate negative images (normal roads)
    logger.info("\n" + "=" * 60)
    logger.info("Generating NEGATIVE images (normal roads)")
    logger.info("=" * 60)
    
    generate_images(
        pipe=pipe,
        prompts=NORMAL_ROAD_PROMPTS,
        output_dir=output_dir / "negative",
        num_images=args.negative_count,
        inference_steps=args.inference_steps,
        seed=args.seed + 10000  # Different seed for negatives
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ SDXL GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Positive images: {args.positive_count}")
    logger.info(f"Negative images: {args.negative_count}")
    logger.info(f"Total: {args.positive_count + args.negative_count}")
    logger.info(f"Output: {output_dir.absolute()}")
    logger.info(f"Cost: $0 (FREE)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

