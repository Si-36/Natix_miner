 # NATIX Subnet 72 — Stage 2 + Stage 3 Master Plan (masterplan8)
 
 Scope:
 - Stage 2: compression + memory + token/KV optimization stack.
 - Stage 3: advanced throughput/latency optimizations + production hardening.
 
 Hard constraints:
 - No code changes in this phase; this is a planning document only.
 - Exactly 18 top-level TODOs (below). Each TODO contains actionable substeps.
 - Consolidated from: `plan_che_plani.md` … `plan_che_plani6.md` (ignore `plan_che_plani7.md`).
 
 Assumptions (edit if wrong):
 - Hardware: 2× H100 80GB (160GB total).
 - Baseline Stage 1 exists and is stable.
 - Goal: maximize Subnet 72 MCC/quality while hitting tight latency/throughput and maintaining robust failure modes.
 
 ---
 
 ## TODO 01 — Lock the Stage 2/3 success criteria + measurement harness (single source of truth)
 - Define target metrics and how they’re computed:
   - MCC (primary), plus AP for small objects, FN/FP rates by category, calibration metrics (ECE), and per-tier agreement rates.
   - Latency: p50/p90/p95 and “fast-path vs slow-path” split.
   - Throughput: images/sec at target batch size(s), including queueing overhead.
   - Memory: peak/steady VRAM per tier and per concurrent request.
 - Define evaluation slices (must be stable across all future tuning):
   - Weather: clear/rain/fog/snow/night.
   - Distance scale: small (cones/signs), medium, large.
   - Occlusion + motion blur.
   - Rare/unseen roadwork types.
 - Establish reproducible profiling methodology:
   - Warmup, fixed seeds where relevant, fixed input sizes, fixed concurrency.
   - GPU profiler collection plan (kernel timing, KV cache size, token counts).
 - Establish “acceptance gates” for each optimization:
   - Any change must pass: “<1% quality regression” (or explicit exception), plus latency/memory improvement target.
 - Lock the exact “Stage 2 done” and “Stage 3 done” deliverables (these are *gates*, not aspirations):
   - Stage 2 done means (all must be true on the frozen test suite):
     - Token reduction meets target (track per-tier):
       - Visual token reduction target (multi-view): aim 50%+ reduction at the vision encoder boundary.
       - KV cache footprint reduction target: aim 75–96% reduction for heavy tiers (depending on which techniques are enabled).
     - Latency improvement: measurable reduction on p50 and p95.
     - Quality: MCC and slice metrics not degraded beyond allowed threshold.
   - Stage 3 done means:
     - End-to-end throughput improvement is confirmed under production-like concurrency.
     - Tail latency is controlled (no “optimization-caused p95 explosions”).
     - Fast-path ratio achieved (target range from the plans: ~70–75% fast path).
     - Production safety gates and rollback paths are documented and testable.
 - Standardize the *single* evaluation harness you will re-use for every step:
   - Inputs:
     - A fixed, versioned dataset manifest (train/val/test) + a fixed slice manifest.
     - A fixed “prompt pack” for VLM tiers (roadwork descriptions are formulaic; keep them consistent).
   - Outputs (stored per run):
     - Predictions (final), intermediate tier outputs, routing decisions, confidences (raw + calibrated), token counts, KV sizes.
     - Per-slice scores + overall score.
   - Minimal reproducibility requirements:
     - Pin model weights (hash / version tag), config, runtime flags, and GPU type.
     - Record compiler/engine version if using TensorRT-style engines.
 - Define the performance targets (use as a living table; do not “optimize blind”):
   - Latency budget targets referenced in the consolidated plans:
     - Fast path: ~25–35ms average target.
     - Slow path: ~50–70ms average target.
   - Router targets:
     - Calibrated confidence thresholding only.
     - Keep explicit accounting of what percentage of cases hit each tier.
 - Define required artifact set per optimization (must exist before moving to next TODO):
   - “Before vs after” report:
     - token counts, KV sizes, latency (p50/p95), throughput, MCC, and slice deltas.
   - “Failure mode report”:
     - what breaks when this optimization is too aggressive (small objects, OCR/text, night).
   - “Rollback plan”:
     - toggle name(s), safe defaults, and the exact conditions that trigger fallback.
 
 ## TODO 02 — Finalize the tiered cascade architecture boundaries (what runs where + when)
 - Freeze the cascade contract (per-tier inputs/outputs):
   - Input: image(s) + optional metadata.
   - Output: roadwork presence + structured attributes + confidence + optional boxes/masks.
 - Define tier routing conditions (high-level):
   - Tier-0/1 detection ensemble produces initial detections.
   - Tier-2 zero-shot + segmentation validates novelty and improves localization.
   - Tier-3 fast VLM resolves ambiguous semantics/OCR.
   - Tier-4/5 power/precision used only for hard cases.
   - Tier-6 consensus invoked for rare disagreement or reward-critical cases.
 - Decide “what is authoritative” for final decision:
   - Prefer calibrated, weighted geometric-mean consensus over single-model outputs.
 - Set boundaries for what is Stage 2 vs Stage 3:
   - Stage 2 focuses on compression primitives (tokens/KV/quant/depth).
   - Stage 3 focuses on higher-level throughput methods (patching/compression across views, speculative decoding, distillation, batch DP, and production hardening).
 - Make the cascade *explicitly measurable* (this prevents silent drift):
   - For each tier, define:
     - Entry conditions (thresholds and required signals).
     - Exit conditions (what constitutes “resolved”).
     - Time budget target.
     - Output schema.
 - Define a canonical “routing feature vector” (from consolidated plans; keep it stable):
   - Detection ensemble signals:
     - confidence mean + variance across detectors.
     - count of objects + distribution of sizes.
     - disagreement rate across detectors.
   - Context signals:
     - weather class (clear/rain/fog/snow/night).
     - brightness/contrast, edge density.
   - Historical signals:
     - historical accuracy for “similar scenes” (nearest-neighbor or memory bank).
 - Decide the hard rules for escalation (example thresholds to be tuned, not guessed per-run):
   - If detection confidence is extremely high (e.g., ≥0.95 calibrated) and no slice-risk flags: stay on fast path.
   - If confidence is mid (e.g., 0.60–0.95) or high disagreement: go to fast VLM.
   - If confidence is low (<0.40) or novelty is detected: go to power/precision.
   - If tier outputs disagree (detectors vs VLM): trigger consensus or precision (depending on policy).
 - Define “conflict resolution” policy (must be deterministic):
   - If detectors say “yes” but VLM says “no” with high calibrated confidence:
     - prefer VLM for semantic hallucination filtering *only if* calibration is proven on that slice.
   - If VLM says “yes” but detectors say “no”:
     - treat as higher hallucination risk; require precision tier confirmation.
 - Lock the ensemble consensus math (from later plans):
   - Weighted geometric mean confidence aggregation rather than arithmetic mean.
   - Keep a stable weight table by model family and revise only via evaluation.
 
 ## TODO 03 — Stage 2.1: Implement Visual Token Sparsification (VASparse or equivalent)
 - Select token sparsification approach:
   - If multiple options exist, prefer the one with:
     - plug-and-play integration,
     - stable accuracy impact,
     - good compatibility with your backbone and downstream heads.
 - Configure NATIX-specific settings:
   - Multi-view redundancy aware masking.
   - Preserve critical tokens: small-object edges, text regions, and detection-relevant patches.
 - Validation plan:
   - Compare token counts, latency, and MCC on fixed evaluation slices.
   - Confirm no systematic FN increase on small objects.
 - Rollout plan:
   - Enable behind a feature flag.
   - Keep ability to bypass sparsification for “extreme difficulty” routed cases.
 - Concrete Stage 2.1 implementation plan (from the consolidated plans; translate into your stack specifics):
   - Baseline capture (before any token masking):
     - Collect:
       - per-tier visual token counts (before/after vision encoder),
       - KV cache sizes (if measurable at this stage),
       - latency/throughput and MCC/slices.
     - Save as: “Stage2.1-baseline”.
   - Install/integrate token sparsification module (planning only, no code changes here):
     - Confirm compatibility with:
       - vision encoder backbone (DINOv3 family in the plans),
       - downstream detection heads,
       - and any multi-view fusion used in Stage 1.
   - Define token masking policy knobs you will tune (keep them explicit):
     - Global sparsity ratio target: start at 0.50 visual token masking.
     - Attention thresholding:
       - choose a conservative threshold first; only increase once FN is stable.
     - Token preservation rules:
       - always preserve class/register tokens if the backbone uses them.
       - preserve “text/OCR candidate regions” tokens (signs).
       - preserve small-object edge tokens (cones/barriers).
     - Multi-view weighting:
       - if you have 6 views, weight the most informative view(s) higher for token retention.
   - NATIX roadwork-specific guardrails (must be in place before tuning):
     - “Small-object protection”:
       - if object size estimate is < ~32 px region, reduce masking aggressiveness in that region.
     - “Text protection”:
       - if OCR-like patterns are detected, preserve higher token density.
     - “Night/fog/snow protection”:
       - for these slices, masking must be more conservative.
   - Validation expectations (from plans; treat as targets, not promises):
     - Visual tokens: ~50% reduction.
     - Latency: measurable reduction (plans referenced ~35% latency reduction for this step in isolation).
     - Quality: MCC regression should be negligible and within the acceptance gate.
   - Stepwise tuning process (to avoid overshooting):
     - Pass 1: 0.50 mask ratio with conservative thresholds.
     - Pass 2: only if FN stable, adjust threshold upward for more reduction.
     - Pass 3: if quality regresses, add region-preservation rules rather than lowering all sparsity.
   - Rollback plan (must be decided before enabling):
     - Global kill-switch: disable token sparsification.
     - Per-slice bypass: disable on “extreme difficulty” bins or worst slices.
     - Per-tier bypass: e.g., disable for tiers that rely on dense tokens (OCR-heavy VLM prompts).
 
 
 ## TODO 04 — Stage 2.2: KV cache quantization (NVFP4 / FP8 / hybrid) for heavy VLMs
 - Identify which models use KV cache heavily (typically the larger VLM tiers).
 - Choose quantization granularity and calibration method:
   - KV-only quantization first (lower risk).
   - Per-channel calibration if available.
 - Engine strategy:
   - Build optimized inference engines where appropriate.
   - Keep fallbacks for correctness and debugging.
 - Validation gates:
   - Accuracy regression threshold (ideally <1% relative on key metrics).
   - Measure KV footprint reduction and impact on prefill/decode times.
 - Detailed Stage 2.2 execution plan (derived from the source plans; “latest wins” unless older is measurably better):
   - 2.2.1 Define the model list and KV hotspots
     - Primary (always-on) heavy tiers:
       - InternVL3.5-78B (or InternVL3-78B) — precision reasoning
       - Qwen3-VL-72B — precision validator / backup
     - Secondary (power tier / MoE, if actually used in your cascade):
       - Llama 4 Maverick
       - Qwen3-VL-30B
       - Ovis2-34B
     - Optional / on-demand / off-path (only load when forced by routing):
       - Qwen3-VL-235B
     - For each model, record:
       - Baseline KV cache size at typical sequence length(s)
       - Baseline prefill time and decode time
       - Baseline memory headroom under your target concurrency
 
   - 2.2.2 Decide the quantization scope (start KV-only)
     - First pass:
       - Quantize KV cache only (lowest regression risk).
       - Keep weights and activations in their existing stable precision.
     - Later passes (only if KV-only is stable):
       - Consider broader quantization per model, but treat it as a separate change with its own gates.
 
   - 2.2.3 Calibration protocol (must be repeatable)
     - Fix a calibration pack (versioned) that represents:
       - day/night
       - weather extremes
       - small-object scenes
       - OCR/sign scenes
       - “no roadwork” scenes
     - Calibration parameters emphasized in the plans:
       - calibration_samples: 128
       - calibration_method: minmax
       - per_channel: true
       - asymmetric: true
       - scope: KV cache only
     - Output artifacts per model:
       - quantization config used
       - calibration summary
       - “golden” before/after latency + accuracy report
 
   - 2.2.4 Engine build and runtime strategy
     - Build optimized engines for the quantized variants where applicable.
     - Require a fallback path:
       - If the engine build fails, fall back to the baseline runtime (no blocking).
     - Memory strategy:
       - Use paged KV cache if supported.
       - Reserve explicit buffers so tail latency doesn’t spike under load.
     - Concurrency strategy:
       - Test at the target concurrency, not just batch=1.
 
   - 2.2.5 Validation matrix (must pass before enabling broadly)
     - Accuracy:
       - MCC overall (frozen test suite)
       - MCC on worst slices: night / fog / snow
       - small-object slice metrics (cones, barriers, signs)
       - OCR/text slice metrics
     - Performance:
       - prefill latency (p50/p95)
       - decode/generation latency (p50/p95)
       - end-to-end request latency (p50/p95)
     - Memory:
       - peak KV cache per request
       - steady-state VRAM at target concurrency
       - headroom buffer remaining
 
   - 2.2.6 Target outcomes (directional; verify with your harness)
     - KV cache reduction target referenced in the plans:
       - ~75% vs FP16 (NVFP4 KV)
     - The “freed VRAM” must be explicitly allocated to one of:
       - higher batch buffers
       - more concurrent requests
       - larger safety buffers for p95 stability
 
   - 2.2.7 Rollback rules (non-negotiable)
     - Per-model rollback:
       - If a single model regresses (accuracy or tail latency), disable quantization for that model only.
     - Global rollback:
       - If p95/p99 becomes unstable under load, disable KV quantization globally.
     - Slice-triggered rollback:
       - If OCR/text or night slices regress beyond the acceptance gate, disable on those slices (routing-based bypass) while keeping it for easy slices.
 
   - 2.2.8 “Keep best even if older” rule (your instruction)
     - Default: NVFP4 KV path.
     - Exception: if an older FP8/hybrid setting is measurably better for a specific model (accuracy preserved and similar latency/memory), keep that setting for that model only.

 ## TODO 05 — Stage 2.3: Sparse attention KV compression (PureKV or equivalent) for multi-view/temporal
 - Apply learned or rule-guided sparsity for KV/prefill acceleration.
 - Define NATIX multi-view/temporal windows:
   - Spatial window sizes, temporal stride, and cross-view overlap handling.
 - Integration with your fusion strategy:
   - Compress before multi-view fusion when it reduces redundant tokens without harming cross-view consistency.
 - Validation:
   - Cross-view consistency metrics (IoU stability / object persistence across adjacent views).
   - Prefill speed improvement and KV compression factor.
 - Detailed Stage 2.3 execution plan (PureKV-style; keep best):
   - 2.3.1 When to use this technique (so you don’t add complexity for no gain)
     - Use for:
       - multi-view redundancy (6 cameras / 360 style),
       - temporal redundancy (sequential frames),
       - long-context VLM tiers where prefill dominates.
     - Skip for:
       - tiers that already run at low token counts and are not prefill-bound.
 
   - 2.3.2 Baseline capture (mandatory)
     - Record before enabling PureKV:
       - prefill latency (p50/p95)
       - KV cache footprint per request
       - token counts per tier (especially prefill token lengths)
       - cross-view persistence baseline:
         - matched object IoU stability across adjacent views
         - count consistency across adjacent views (no “missing cones” in side cameras)
 
   - 2.3.3 Initial configuration (start conservative; tune only after validation)
     - Spatial attention window:
       - window_size: 64
       - enable_cross_view: true
       - view_overlap_threshold: 0.30
       - preserve_object_tokens: true
       - edge_preservation_weight: 1.5
     - Temporal attention window:
       - window_size: 8
       - stride: 2
       - enable_motion_guided: true
       - optical_flow_assist: true (only if net latency improves)
     - Compression objective:
       - target_ratio: 5.0
       - learned_importance: true
       - sparsity_scheduler: cosine
 
   - 2.3.4 Integration rules with other systems (dedupe + keep best)
     - Order relative to Stage 2.1 token sparsification:
       - Apply token sparsification first, then PureKV, unless experiments show the reverse is more stable.
     - Order relative to multi-view fusion:
       - Prefer PureKV before fusion so redundant tokens are reduced earlier.
       - If fusion quality drops, restrict PureKV to:
         - temporal-only mode, or
         - only the heaviest tiers.
 
   - 2.3.5 Validation targets (directional, verify in harness)
     - KV compression target: ~5×.
     - Prefill acceleration target: ~3.16×.
     - Cross-view consistency target:
       - matched-object IoU > 0.95 on representative slices.
     - “No silent regressions” checks:
       - small-object slice FN must not increase.
       - night/fog slices must not produce unstable object counts.
 
   - 2.3.6 Known failure modes and fixes
     - Failure: small objects disappear due to sparsity.
       - Fix:
         - increase `edge_preservation_weight`
         - enforce `preserve_object_tokens`
         - lower sparsity only for small-object slices
     - Failure: motion-guided sparsity breaks in night/fog.
       - Fix:
         - disable `enable_motion_guided` for those slices
         - or increase temporal window/stride conservatism
     - Failure: overhead from optical flow outweighs wins.
       - Fix:
         - disable `optical_flow_assist`
         - keep motion-guided heuristics only if cheap
 
   - 2.3.7 Rollback rules
     - Roll back PureKV per tier if:
       - cross-view consistency drops below target,
       - small-object FN increases beyond gate,
       - tail latency worsens.
     - Keep a global kill-switch.

 ## TODO 06 — Stage 2.4: Dynamic depth routing (p-MoD / progressive MoD) for heavy tiers
 - Define the difficulty estimator contract:
   - Inputs: ensemble confidence variance, object count, weather class, brightness/contrast, edge density, historical similarity.
   - Output: difficulty bin (e.g., 5 bins) + calibrated probability.
 - Configure depth policy:
   - Min/max active layers per bin.
   - Penalty terms for skipping vs accuracy.
 - Train/fit the depth router using your validation set and difficulty slices.
 - Validate:
   - Distribution of bins (expected: majority easy).
   - Latency reduction weighted by real-world distribution.
   - Confirm hard-case accuracy does not regress.
 
 ## TODO 07 — Stage 2: Unify compression stack ordering + compatibility matrix
 - Decide the canonical order (typical safe order):
   - Token sparsification → KV quantization → sparse KV compression → dynamic depth.
 - Create a compatibility matrix:
   - Which tiers/models use which technique.
   - Known failure modes (e.g., sparsification + small text; aggressive KV quant + long context).
 - Add “escape hatches”:
   - Automatic fallback to less aggressive settings for detected edge cases.
 - Define the rollback strategy:
   - Per-technique toggle.
   - Per-tier bypass.
 
 ## TODO 08 — Stage 3.1: Adaptive patching / variable-resolution compute (APT or equivalent)
 - Goal:
   - Reduce unnecessary patch tokens on homogeneous regions while preserving small objects.
 - Configure patch policy:
   - Smaller patches for cones/signs/edges.
   - Larger patches for sky/road surface.
 - Minimal fine-tuning plan:
   - One-epoch retrofit if needed; keep the backbone stable.
 - Validation:
   - Token/patch count reduction.
   - Small object AP and overall MCC must not regress.
 
 ## TODO 09 — Stage 3.2: Progressive Visual Compression (PVC) for multi-view fusion
 - Goal:
   - Reduce multi-view redundancy (and optionally temporal redundancy) with controlled information retention.
 - Define view layout assumptions:
   - 6-view 360 layout (front/FL/FR/rear/RL/RR) or your actual camera topology.
 - Configure overlap matrix + view importance weights.
 - Validate:
   - Cross-view object persistence and consistency.
   - Latency/throughput improvements under realistic concurrency.
 
 ## TODO 10 — Stage 3.3: Speculative decoding for slow-path generation (SpecVLM / Eagle-style)
 - Identify where generation dominates latency:
   - Typically the precision tier when it produces verbose structured output.
 - Choose draft strategy:
   - Draft model size, draft length, tree width.
   - Acceptance threshold and calibration.
 - Training plan (if required):
   - Distill a small draft model from the main model on your roadwork-specific output distribution.
 - Validation:
   - Acceptance rate.
   - End-to-end latency reduction.
   - No quality regressions on hard cases.
 
 ## TODO 11 — Stage 3.4: Knowledge distillation for fast tiers (VL2Lite-style)
 - Define student targets:
   - Student should match or exceed baseline fast-tier performance at much lower latency.
 - Define teacher ensemble:
   - Use higher tiers as teachers with hard-case emphasis.
 - Distillation objectives:
   - Task loss + consistency + calibration.
 - Validation:
   - Net accuracy gain in fast tier.
   - Confirm no drift in confidence distribution.
 
 ## TODO 12 — Stage 3.5: Batch-level data parallelism + serving runtime optimization
 - Serving runtime decisions:
   - Batch scheduling policies (microbatch vs dynamic batch).
   - KV cache paging strategy.
   - Overlap compute/transfer where possible.
 - Concurrency strategy:
   - Separate fast-path queue from slow-path queue.
   - Priority rules for timeouts.
 - Validation:
   - Throughput scaling curves (batch size vs latency).
   - Tail latency (p95/p99) under contention.
 
 ## TODO 13 — Upgrade/lock the model roster for Stages 2/3 (avoid churn)
 - Freeze “recommended” models per role (examples from consolidated plans):
   - Foundation/backbone: DINOv3 ViT-H+/16 (or agreed variant) + Gram Anchoring.
   - Detection: YOLO26-X, RT-DETRv3, D-FINE (and minimal extra specialists).
   - Zero-shot: choose a coherent set (e.g., Anomaly-OV + AnomalyCLIP) with clear responsibilities.
   - Segmentation: SAM 3 integration only if it improves the target tasks and fits budgets.
   - Fast VLM: Qwen3-VL-4B / Molmo 2 (4B/8B) / Phi-4-Multimodal as routing options.
   - Precision: InternVL3.5-78B and/or Qwen3-VL-72B with strict routing criteria.
 - Explicitly remove/avoid duplication:
   - Do not keep multiple models with identical roles unless they contribute diversity (error independence).
 - Define the “on-demand loading” rules for heavy models.
 
 ## TODO 14 — Consensus + voting: implement a calibrated, weighted geometric-mean aggregator
 - Implement hierarchical consensus stages (conceptual, not code yet):
   - Binary agreement gate.
   - Weighted box/mask fusion.
   - Weighted geometric mean confidence aggregation.
 - Define weights by model family:
   - Detectors vs zero-shot vs fast VLM vs power vs precision.
 - Define thresholds:
   - A single global threshold is rarely optimal; define per-slice thresholds (night vs day, etc.) with guardrails.
 - Add disagreement resolution protocol:
   - If lower tiers disagree with VLM tiers, define escalation + override rules based on calibrated confidence.
 
 ## TODO 15 — Calibration + confidence reliability as a first-class deliverable
 - Calibrate each component output type:
   - Detection confidence calibration.
   - VLM score calibration.
   - Cross-model confidence normalization.
 - Choose post-hoc calibration toolkit:
   - Temperature scaling where applicable.
   - Platt scaling / isotonic regression depending on score behavior.
 - Define quality gates:
   - Target ECE threshold (e.g., <0.05) and slice-based calibration checks.
 - Integrate calibration into routing:
   - Routing decisions should be made on calibrated confidences only.
 
 ## TODO 16 — Data strategy for Stage 2/3: augmentation, coverage, and active learning loop
 - Define a roadwork-focused augmentation policy:
   - Geographic diversity.
   - Weather synthesis (rain/snow/fog).
   - Time-of-day conversion.
   - Compression artifacts, motion blur, occlusion.
 - Define evaluation-driven sampling:
   - Over-sample failure slices.
 - Define active learning:
   - What gets flagged (high disagreement, low confidence, novelty signals).
   - How it gets reviewed/relabeled.
   - How retraining or adapter updates are triggered.
 
 ## TODO 17 — Latency budgeting + production safety (graceful degradation)
 - Create an explicit time budget per stage:
   - Preprocess, detection ensemble, zero-shot/segmentation, routing, VLM, consensus, post-processing.
 - Define fast-path vs slow-path expectations:
   - Fast path should handle the majority of frames.
 - Define failure recovery:
   - If heavy GPU/tier unavailable, fall back to lower tiers with adjusted thresholds.
   - If a single model fails, remove it from ensemble and renormalize weights.
   - If memory pressure, disable optional tiers first and reduce batch size.
 - Define operational SLOs:
   - Max timeout, max queue depth, and drop policy.
 
 ## TODO 18 — Final consolidation pass + release checklist for Stage 2/3 readiness
 - Run a full dedupe pass across this plan:
   - Remove repeated techniques and unify terminology.
   - Ensure Stage 2 vs Stage 3 boundaries are consistent.
 - Produce the final “Stage 2 done” and “Stage 3 done” checklists:
   - Each checklist must reference the success criteria from TODO 01.
 - Produce a deployment readiness checklist:
   - Resource budgeting confirmed.
   - Monitoring requirements defined.
   - Rollback plan defined.
 - Confirm the constraint: exactly 18 top-level TODOs.
