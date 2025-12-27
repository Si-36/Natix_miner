---
name: ""
overview: ""
todos: []
---

---name: Stage-1 Pro Modular Training System (Phased with Gates)overview: Create comprehensive modular stage1_pro_modular_training_system/ training system implementing 2025 best practices in RISK-ORDERED PHASES. Phase 1 ships deploy bundle with baseline training + threshold sweep. Advanced features (SCRC, conformal risk, ExPLoRA, DoRAN, F-SAM) come in later phases ONLY after Phase 1 passes acceptance criteria. Baseline reference: train_stage1_head.py (preserve EXACT logic). Deploy bundle is the center - all inference loads from bundle files, no hard-coded thresholds.todos:

- id: config_dataclass

content: Create stage1_pro_modular_training_system/config.py with Stage1ProConfig dataclass including ALL fields from TrainingConfig in train_stage1_head.py (model_path, train_image_dir, train_labels_file, val_image_dir, val_labels_file, mode, cached_features_dir, use_extra_roadwork, roadwork_iccv_dir, roadwork_extra_dir, max_batch_size, fallback_batch_size, grad_accum_steps, epochs, warmup_epochs, lr_head, lr_backbone, weight_decay, dropout, label_smoothing, max_grad_norm, use_amp, use_ema, ema_decay, early_stop_patience, legacy_exit_threshold_for_logging, resume_checkpoint, output_dir, log_file). Phase 1: Add val_select_ratio=0.5, target_fnr_exit=0.02 (ONLY constraint, NO target_coverage), exit_policy='softmax'. Phase 2+: Add selective metrics flags. Phase 3+: Add gate_loss_weight, aux_weight, exit_policy can be 'gate'. Phase 4+: Add peft_type, peft_r, peft_blocks. Phase 5+: Add optimizer='fsam'. Phase 6+: Add calibration flags, exit_policy='scrc' (production default).status: pending

- id: config_new_fields

content: Add phase-specific fields to Stage1ProConfig: Phase 1 (target_fnr_exit=0.02 ONLY, val_select_ratio=0.33, val_calib_ratio=0.33, val_test_ratio=0.34, bootstrap_samples=1000, bootstrap_confidence=0.95, exit_policy='softmax'), Phase 2 (selective metrics flags), Phase 3 (gate_loss_weight=1.0, aux_weight=0.5, exit_policy can be 'gate'), Phase 4 (peft_type='doran', peft_r=16, peft_blocks=6), Phase 5 (optimizer='fsam'), Phase 6 (use_dirichlet=True, calibration_iters=300, exit_policy='scrc' as production default). CRITICAL: Do NOT add target_coverage - use single constraint (target_fnr_exit) and maximize coverage. Rename exit_threshold → legacy_exit_threshold_for_logging (monitoring only, NOT used for inference).status: pendingdependencies:

    - config_dataclass
- id: config_validation

content: Add config validation methods: validate_paths() checks all paths exist, validate_hyperparameters() checks ranges (dropout 0-1, lr > 0, etc.), validate_mode_compatibility() checks mode vs dataset flags, validate_dataset_availability() checks which datasets are availablestatus: pendingdependencies:

    - config_new_fields
- id: config_save_load

content: Implement config.save() and config.load() methods preserving exact JSON format from train_stage1_head.py. Include seed saving for reproducibility. Add timestamp and git commit hash (optional)status: pendingdependencies:

    - config_validation
- id: data_datasets

content: "Create stage1_pro_modular_training_system/data/datasets.py preserving EXACT logic from train_stage1_head.py: NATIXDataset, MultiRoadworkDataset with support for NATIX, ROADWork, Roboflow. Include header skip logic, path handling, dataset source tracking"status: pending

- id: data_splits

content: Create stage1_pro_modular_training_system/data/splits.py with deterministic hash-based split into train/val_select/val_calib/val_test (4-way split). CRITICAL: val_select for model selection/early stopping ONLY, val_calib for fitting calibrators/policies ONLY, val_test for final evaluation ONLY (no leakage). Save splits.json with indices, seed, metadata (class balance, dataset sources). Ensure reproducibilitystatus: pending

- id: data_transforms

content: Create stage1_pro_modular_training_system/data/transforms.py with TimmStyleAugmentation preserving existing scale/ratio settings. Support both aggressive (current) and moderate (2025 baseline) augmentation modesstatus: pending

- id: data_loaders

content: Create stage1_pro_modular_training_system/data/loaders.py with DataLoader creation, batch size auto-detection (pick_batch_size logic), gradient accumulation support, pin_memory, num_workers configurationstatus: pending

- id: model_backbone

content: Create stage1_pro_modular_training_system/model/backbone.py with DINOv3Backbone wrapper preserving freeze logic, feature extraction (CLS token), processor handling, model path loadingstatus: pending

- id: model_head

content: Create stage1_pro_modular_training_system/model/head.py with Stage1Head: Phase 1 single-head (matches train_stage1_head.py exactly), Phase 3+ extends to 3-head (cls_head, gate_head, aux_head). Shared trunk Linear(hidden_size,768)->ReLU->Dropout. Preserve torch.compile supportstatus: pending

- id: model_peft

content: Create stage1_pro_modular_training_system/model/peft.py with DoRAN implementation (DoRA fallback) - PHASE 4 ONLY. Apply to last N blocks (configurable), target QKV/attention output/MLP. Support both PEFT and full freeze modes. Phase 1-3: PEFT disabled.status: pending

- id: training_losses_selective

content: Create stage1_pro_modular_training_system/training/losses.py: Phase 1 preserves CrossEntropyLoss with class weights + label smoothing from train_stage1_head.py. Phase 3 adds SelectiveLoss and AuxiliaryLoss. Phase 6 adds ConformalRiskLoss.status: pending

- id: training_optimizers

content: Create stage1_pro_modular_training_system/training/optimizers.py: Phase 1-4 use AdamW preserving exact config from train_stage1_head.py (betas=(0.9,0.999), eps=1e-8, weight_decay). Phase 5+ adds F-SAM implementation (CVPR 2024) with gradient checkpointing support. Per-layer LR adaptation for Phase 4+.status: pending

- id: training_schedulers

content: Create stage1_pro_modular_training_system/training/schedulers.py with cosine annealing + warmup scheduler preserving exact lr_lambda logic from train_stage1_head.py. Support warmup_epochs configurationstatus: pending

- id: training_risk

content: Create stage1_pro_modular_training_system/training/risk_training.py with ConformalRiskTrainer implementing NeurIPS 2025 end-to-end conformal risk training - PHASE 6 ONLY. Batch splitting (pseudo-calib/pseudo-pred), FNR≤2% control, gradient through CRC. Phase 1-5: Not used.status: pending

- id: training_trainer_full

content: "Create stage1_pro_modular_training_system/training/trainer.py with Stage1Trainer class supporting ALL modes: extract_features, train_cached, train. Phase 1: Preserve EXACT logic from train_stage1_head.py (single-head, CrossEntropyLoss, AdamW, cosine scheduler, EMA). Phase 3+: Extend to 3-head with selective loss. Phase 6+: Add conformal risk training."status: pending

- id: training_ema

content: Create stage1_pro_modular_training_system/training/ema.py preserving EXACT EMA implementation from train_stage1_head.py (register, update, apply_shadow, restore, decay=0.9999)status: pending

- id: calibration_dirichlet

content: Create stage1_pro_modular_training_system/calibration/dirichlet.py with DirichletCalibrator (matrix scaling on logits), ODIRRegularizer (off-diagonal+intercept) - PHASE 6 ONLY. Fit on val_calib only. Save/load support. Phase 1-5: Not used.status: pending

- id: calibration_gate

content: Create stage1_pro_modular_training_system/calibration/gate_calib.py with PlattCalibrator (logistic) and IsotonicCalibrator for gate calibration - PHASE 3+ ONLY. Fit on val_calib gate logits. Save as separate artifact. Phase 1-2: Not used.status: pending

- id: calibration_scrc

content: Create stage1_pro_modular_training_system/calibration/scrc.py with SCRCCalibrator implementing Selective Conformal Risk Control (arXiv 2512.12844) - PHASE 6 ONLY. SCRC-I (calibration-only, default) and SCRC-T (transductive, optional). Two thresholds λ1 (selection), λ2 (set size). Output prediction sets {0},{1},{0,1}. Save scrcparams.json (NOT scrc_params.json, validate against scrcparams.schema.json). Phase 1-5: Not used.status: pending

- id: metrics_selective

content: Create stage1_pro_modular_training_system/metrics/selective.py with compute_risk_coverage (Risk@Coverage, Coverage@Risk), compute_augrc (AUGRC NeurIPS 2024), compute_selective_metrics suite. Support multi-threshold evaluationstatus: pending

- id: metrics_calibration

content: Create stage1_pro_modular_training_system/metrics/calibration.py preserving compute_ece from train_stage1_head.py PLUS compute_nll (Negative Log-Likelihood), compute_brier (Brier score), classwise ECE variantsstatus: pending

- id: metrics_exit

content: Create stage1_pro_modular_training_system/metrics/exit.py with exit coverage/accuracy computation preserving logic from train_stage1_head.py. Phase 1: Softmax-threshold exit only. Phase 3+: Add gate-based exit. Phase 6+: Add SCRC exit. CRITICAL: Always compute and report FNR_on_exited and coverage for active policy. All methods logged as metrics.status: pending

- id: metrics_bootstrap

content: Create stage1_pro_modular_training_system/metrics/bootstrap.py with bootstrap confidence interval computation. Implement bootstrap_resample() for sampling with replacement, compute_confidence_intervals() for percentile method (2.5th, 97.5th percentiles), apply to AUGRC, FNR_on_exited, coverage, risk-coverage curves. Default: 1000 bootstrap samples, 95% confidence level.status: pendingdependencies:

    - metrics_selective
    - metrics_exit
- id: explora_trainer

content: Create stage1_pro_modular_training_system/domain_adaptation/explora.py with ExPLoRATrainer for parameter-efficient extended pretraining - PHASE 4 ONLY. Unfreeze last 1-2 blocks, PEFT on rest. MAE-style masked modeling on unlabeled road images. Phase 1-3: Not used.status: pending

- id: explora_data

content: Create stage1_pro_modular_training_system/domain_adaptation/data.py for loading unlabeled road images (NATIX extras, SDXL synthetics) for ExPLoRA pretrainingstatus: pending

- id: feature_cache

content: Create stage1_pro_modular_training_system/utils/feature_cache.py preserving extract_features mode from train_stage1_head.py. Save train/val features to disk, support loading for train_cached modestatus: pending

- id: checkpointing

content: Create stage1_pro_modular_training_system/utils/checkpointing.py with save_checkpoint, load_checkpoint preserving exact format from train_stage1_head.py (model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_acc, patience_counter, ema_state_dict)status: pending

- id: logging

content: Create stage1_pro_modular_training_system/utils/logging.py with comprehensive logging preserving CSV format from train_stage1_head.py (Epoch, Train_Loss, Train_Acc, Val_Loss, Val_Acc, ECE, Exit_Coverage, Exit_Acc, Best_Val_Acc, LR). Phase 2+: Add selective metrics columns.status: pending

- id: script_splits

content: Create stage1_pro_modular_training_system/scripts/00_make_splits.py to create train/val_select/val_calib/val_test (4-way) splits deterministically. CRITICAL: Document usage rules (val_select=model selection, val_calib=calibration, val_test=evaluation). Save splits.json with full metadata including split purposes. Support all dataset sourcesstatus: pending

- id: script_explora

content: Create stage1_pro_modular_training_system/scripts/10_explora_pretrain.py to run ExPLoRA domain adaptation. Load unlabeled data, run pretraining, save backbone_explora.pth checkpointstatus: pending

- id: script_train_full

content: Create stage1_pro_modular_training_system/scripts/20_train.py (UNIFIED): Single training script with --phase {1..6} and --exit_policy {softmax,gate,scrc} arguments. Phase 1: Baseline training matching train_stage1_head.py exactly (single-head, CrossEntropyLoss). Phase 3: Train with gate head (3-head, SelectiveLoss). Phase 6: Train with conformal risk objective (3-head, ConformalRiskLoss). Validate phase supports exit_policy. All modes: extract_features, train_cached, train.status: pending

- id: script_calibrate

content: Create stage1_pro_modular_training_system/scripts/25_threshold_sweep.py for Phase 1: Threshold sweep matching cascade exit logic, save thresholds.json (validate against thresholds.schema.json). Create stage1_pro_modular_training_system/scripts/33_calibrate_gate.py for Phase 3: Fit gate calibrator, save gateparams.json (NOT thresholds.json, validate against gateparams.schema.json). Create stage1_pro_modular_training_system/scripts/36_calibrate_scrc.py for Phase 6: Fit Dirichlet calibrator, fit SCRC, save scrcparams.json (validate against scrcparams.schema.json). Each policy artifact is separate and mutually exclusive.status: pending

- id: script_eval

content: Create stage1_pro_modular_training_system/scripts/40_eval_selective.py: Load bundle.json first to determine active_exit_policy. Load exactly ONE policy file (thresholds.json OR gateparams.json OR scrcparams.json) according to bundle.json.active_exit_policy. Evaluate on val_test (NOT val_calib) to avoid optimistic results. Compute risk-coverage curves, AUGRC with bootstrap confidence intervals, FNR_on_exited and coverage for active policy with bootstrap CIs, proper scoring rules. Save metrics.csv with uncertainty estimates + plotsstatus: pending

- id: script_export

content: "Create stage1_pro_modular_training_system/scripts/50_export_bundle.py to package all artifacts: model_best.pth, exactly ONE policy file (thresholds.json OR gateparams.json OR scrcparams.json based on exit_policy), splits.json, metrics.csv, config.json, bundle.json manifest. Bundle.json declares active_exit_policy and file pointers. Validate all JSON files against schemas. Enforce mutual exclusivity: only one policy file per bundle."status: pending

- id: artifacts_structure

content: Create artifacts/stage1_pro_modular_training_system/runs/<run_id>/ directory structure. Ensure all scripts save to correct locations. Support run_id generation with timestampstatus: pending

- id: cli_interface

content: Create stage1_pro_modular_training_system/cli.py with comprehensive argparse preserving ALL arguments from train_stage1_head.py PLUS phase-specific options. Support mode routing (extract_features, train_cached, train). Add --phase flag to enable phase-specific features.status: pending

- id: tests_unit

content: Create tests/unit/ for unit tests of each module (config, datasets, splits, model components, losses, optimizers, metrics, calibration)status: pending

- id: tests_integration

content: Create tests/integration/ for full pipeline integration tests. Test extract_features->train_cached flow, train->calibrate->eval flow, checkpoint resumingstatus: pending

- id: docs_readme

content: Create stage1_pro_modular_training_system/README.md with comprehensive architecture docs, phase-by-phase guide, acceptance criteria for each phase, usage examples for all modes, migration guide from train_stage1_head.py, API reference, deploy bundle documentationstatus: pending

- id: docs_api

content: Create stage1_pro_modular_training_system/docs/API.md with detailed docstrings for all classes/functions. Include parameter descriptions, return types, usage examplesstatus: pending

- id: validation_config

content: "Add config validation to stage1_pro_modular_training_system/config.py: validate paths exist, hyperparameters in valid ranges, mode compatibility checks, dataset availability checks, phase compatibility (prevent Phase 3+ features in Phase 1). Validate exit_policy matches phase (Phase 1: only 'softmax', Phase 3+: 'softmax' or 'gate', Phase 6: all three). Validate target_fnr_exit > 0 and <= 0.1. Ensure legacy_exit_threshold_for_logging is only used for metrics, not inference."status: pending

- id: error_handling

content: "Add comprehensive error handling throughout: graceful degradation (fallback batch sizes, missing datasets), informative error messages, recovery from corrupted checkpoints"status: pending

- id: reproducibility

content: "Ensure full reproducibility: seed setting (random, numpy, torch, cudnn), deterministic operations, split persistence, config saving with all seeds"status: pending

- id: performance

content: "Optimize performance: preserve torch.compile, TF32 precision, efficient DataLoader (pin_memory, num_workers), gradient accumulation, mixed precision"status: pending

- id: monitoring

content: "Add monitoring utilities: progress bars (tqdm), training metrics visualization, real-time logging, checkpoint size monitoring, GPU memory tracking"status: pending

- id: data_natix_class

content: "Implement NATIXDataset class in datasets.py preserving EXACT **init** logic: image_dir, labels_file, processor, augment flag, header skip detection (if line.startswith('image,')), CSV parsing, label extraction, sample storage"status: pendingdependencies:

    - data_datasets
- id: data_natix_getitem

content: "Implement NATIXDataset.**getitem** preserving exact logic: load image with Image.open().convert('RGB'), apply augmentation or val_transform, ImageNet normalization (mean/std tensors), return pixel_values and label"status: pendingdependencies:

    - data_natix_class
- id: data_multiroadwork_class

content: "Implement MultiRoadworkDataset class preserving EXACT logic: dataset_configs list, processor, augment flag, sample/label/dataset_sources lists, CSV loading with header skip, path handling (absolute/relative), dataset statistics printing"status: pendingdependencies:

    - data_datasets
- id: data_multiroadwork_getitem

content: "Implement MultiRoadworkDataset.**getitem** preserving exact logic: load image from full_path, apply augmentation, ImageNet normalization, return pixel_values and label"status: pendingdependencies:

    - data_multiroadwork_class
- id: data_splits_hash

content: "Implement hash-based splitting in splits.py: hash(image_path + seed) % 100 < val_select_ratio*100 for deterministic assignment. Preserve class balance using stratified sampling"status: pendingdependencies:

    - data_splits
- id: data_splits_metadata

content: "Implement splits metadata collection: class distribution per split, dataset source distribution, total counts. Save to splits.json with full metadata"status: pendingdependencies:

    - data_splits_hash
- id: transforms_timm_aggressive

content: "Implement TimmStyleAugmentation aggressive mode preserving exact settings: RandomResizedCrop(scale=(0.7,1.0), ratio=(0.7,1.4)), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), RandomErasing(p=0.4, scale=(0.02,0.4))"status: pendingdependencies:

    - data_transforms
- id: transforms_timm_moderate

content: "Implement TimmStyleAugmentation moderate mode (2025 baseline): RandomResizedCrop(scale=(0.8,1.0), ratio=(0.75,1.33)), NO ColorJitter, RandomErasing(p=0.25, scale=(0.02,0.33))"status: pendingdependencies:

    - data_transforms
- id: transforms_val

content: "Implement validation transforms preserving exact logic: Resize(256, BICUBIC), CenterCrop(224), ToTensor()"status: pendingdependencies:

    - data_transforms
- id: loaders_batch_detection

content: "Implement pick_batch_size() preserving EXACT logic from train_stage1_head.py: try max_batch_size with dummy tensor, catch OOM, fallback to fallback_batch_size, torch.cuda.empty_cache()"status: pendingdependencies:

    - data_loaders
- id: loaders_creation

content: "Implement create_dataloaders() preserving exact settings: num_workers=4, pin_memory=True, drop_last=True for train, drop_last=False for val, shuffle=True for train"status: pendingdependencies:

    - loaders_batch_detection
- id: backbone_load

content: "Implement DINOv3Backbone.load() preserving exact logic: AutoModel.from_pretrained(model_path), AutoImageProcessor.from_pretrained(model_path), .to(device), .eval()"status: pendingdependencies:

    - model_backbone
- id: backbone_freeze

content: "Implement DINOv3Backbone.freeze() preserving exact logic: for param in backbone.parameters(): param.requires_grad = False. Count frozen parameters"status: pendingdependencies:

    - backbone_load
- id: backbone_extract

content: "Implement DINOv3Backbone.extract_features() preserving exact logic: outputs = backbone(pixel_values=images), features = outputs.last_hidden_state[:, 0, :] (CLS token)"status: pendingdependencies:

    - backbone_freeze
- id: head_init

content: "Implement Stage1Head.**init**() creating shared trunk (Linear->ReLU->Dropout) and three heads (cls Linear(768,2), gate Linear(768,1), aux Linear(768,2))"status: pendingdependencies:

    - model_head
- id: head_forward

content: "Implement Stage1Head.forward() returning tuple (logits, gate_logit, aux_logits). gate_logit should be squeezed to [B] dimension"status: pendingdependencies:

    - head_init
- id: head_compile

content: "Add torch.compile() support to Stage1Head preserving exact compilation logic from train_stage1_head.py: torch.compile(model, mode='default')"status: pendingdependencies:

    - head_forward
- id: peft_doran_impl

content: "Implement DoRAN adapter: create low-rank matrices with noise-based stabilization, auxiliary network for dynamic generation. Apply to QKV, attention output, MLP in last N blocks"status: pendingdependencies:

    - model_peft
- id: peft_dora_fallback

content: "Implement DoRA fallback: weight decomposition (magnitude + direction), low-rank update to direction. Use if DoRAN not available"status: pendingdependencies:

    - model_peft
- id: peft_apply

content: "Implement apply_peft() function: modify backbone parameters in-place, preserve frozen state for non-PEFT blocks, return modified backbone"status: pendingdependencies:

    - peft_doran_impl
    - peft_dora_fallback
- id: loss_selective_risk

content: "Implement SelectiveLoss selective_risk computation: compute error on accepted samples (where gate_prob >= threshold), return mean error"status: pendingdependencies:

    - training_losses_selective
- id: loss_coverage_penalty

content: "Implement SelectiveLoss: Control FNR ≤ target_fnr_exit, maximize coverage implicitly. NO target_coverage parameter. Selective risk = compute_error_on_accepted(gate_prob >= threshold). Coverage maximized by minimizing selective risk subject to FNR constraint."status: pendingdependencies:

    - training_losses_selective
- id: loss_conformal_split

content: "Implement ConformalRiskLoss batch splitting: split batch into calib (50%) and pred (50%) halves. Use random permutation for fairness"status: pendingdependencies:

    - training_losses_selective
- id: loss_conformal_threshold

content: "Implement ConformalRiskLoss threshold computation: compute conformal threshold on calib set to control FNR ≤ target_fnr. Use quantile-based method"status: pendingdependencies:

    - loss_conformal_split
- id: loss_conformal_backprop

content: "Implement ConformalRiskLoss gradient computation: differentiate through CRC objective on pred set. Return risk loss term"status: pendingdependencies:

    - loss_conformal_threshold
- id: loss_auxiliary

content: "Implement AuxiliaryLoss: standard CrossEntropyLoss on aux_logits with full coverage (all samples). Weight by aux_weight"status: pendingdependencies:

    - training_losses_selective
- id: loss_class_weighted

content: "Preserve existing CrossEntropyLoss with class weights: compute class_weights = total_samples / (num_classes * class_counts), create weight tensor, use in CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)"status: pendingdependencies:

    - training_losses_selective
- id: optimizer_fsam_forward

content: "Implement F-SAM forward step: compute loss at current params, compute adversarial perturbation (epsilon), compute loss at params + epsilon"status: pendingdependencies:

    - training_optimizers
- id: optimizer_fsam_backward

content: "Implement F-SAM backward step: compute gradient at perturbed params, update params using SAM update rule. Support gradient checkpointing"status: pendingdependencies:

    - optimizer_fsam_forward
- id: optimizer_adamw_fallback

content: "Implement AdamW fallback preserving exact config: betas=(0.9,0.999), eps=1e-8, weight_decay from config"status: pendingdependencies:

    - training_optimizers
- id: optimizer_per_layer_lr

content: "Implement per-layer LR adaptation: create separate param_groups for transformer (lower LR) and head (higher LR). Configurable LR ratios"status: pendingdependencies:

    - optimizer_fsam_backward
    - optimizer_adamw_fallback
- id: scheduler_warmup

content: "Implement warmup phase in scheduler preserving exact lr_lambda: if step < warmup_steps: return step / warmup_steps"status: pendingdependencies:

    - training_schedulers
- id: scheduler_cosine

content: "Implement cosine annealing phase preserving exact lr_lambda: progress = (step - warmup_steps) / (total_steps - warmup_steps), return 0.5 * (1 + cos(pi * progress))"status: pendingdependencies:

    - training_schedulers
- id: risk_trainer_init

content: "Implement ConformalRiskTrainer.**init**(): store model, target_fnr, device, batch_split_ratio"status: pendingdependencies:

    - training_risk
- id: risk_trainer_step

content: "Implement ConformalRiskTrainer.training_step(): split batch, compute threshold on calib, compute risk loss on pred, return total loss"status: pendingdependencies:

    - risk_trainer_init
- id: trainer_extract_features

content: "Implement Stage1Trainer.extract_features() mode preserving EXACT logic from train_stage1_head.py: load backbone, create datasets, extract CLS features, save to .pt files"status: pendingdependencies:

    - training_trainer_full
- id: trainer_train_cached

content: "Implement Stage1Trainer.train_cached() mode preserving EXACT logic: load cached features, create TensorDataset, train head only, support checkpoint resume"status: pendingdependencies:

    - training_trainer_full
- id: trainer_train_full

content: "Implement Stage1Trainer.train() mode: load backbone (with optional PEFT), create 3-head model, multi-dataset support, training loop with all losses, validation on val_select"status: pendingdependencies:

    - training_trainer_full
- id: trainer_training_loop

content: "Implement training loop preserving exact structure: train epoch (forward, backward, optimizer step, scheduler step, EMA update), val epoch (with EMA shadow), checkpoint saving, early stopping"status: pendingdependencies:

    - trainer_train_full
- id: trainer_grad_accum

content: "Implement gradient accumulation preserving exact logic: loss / grad_accum_steps, backward(), if (batch_idx + 1) % grad_accum_steps == 0: optimizer.step()"status: pendingdependencies:

    - trainer_training_loop
- id: trainer_amp

content: "Implement mixed precision preserving exact logic: with torch.amp.autocast('cuda'): forward pass, scaler.scale(loss).backward(), scaler.unscale(), scaler.step(), scaler.update()"status: pendingdependencies:

    - trainer_training_loop
- id: trainer_checkpoint_save

content: "Implement checkpoint saving preserving exact format: save model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_acc, patience_counter, ema_state_dict (if EMA enabled)"status: pendingdependencies:

    - trainer_training_loop
- id: trainer_checkpoint_resume

content: "Implement checkpoint resuming preserving exact logic: load checkpoint dict, restore model/optimizer/scheduler states, restore EMA shadow, set start_epoch, best_acc, patience_counter"status: pendingdependencies:

    - trainer_checkpoint_save
- id: trainer_val_logits

content: "Implement validation logits saving: collect all logits, gate_logits, labels during validation. Save to disk for calibration fitting"status: pendingdependencies:

    - trainer_training_loop
- id: ema_register

content: "Implement EMA.register() preserving exact logic: for name, param in model.named_parameters(): if param.requires_grad: shadow[name] = param.data.clone()"status: pendingdependencies:

    - training_ema
- id: ema_update

content: "Implement EMA.update() preserving exact logic: new_average = (1.0 - decay) * param.data + decay * shadow[name], shadow[name] = new_average.clone()"status: pendingdependencies:

    - training_ema
- id: ema_shadow

content: "Implement EMA.apply_shadow() and restore() preserving exact logic: backup params, replace with shadow, restore from backup"status: pendingdependencies:

    - training_ema
- id: calib_dirichlet_init

content: "Implement DirichletCalibrator.**init**(): create Linear(num_classes, num_classes, bias=True), initialize weight to identity, bias to zero"status: pendingdependencies:

    - calibration_dirichlet
- id: calib_dirichlet_forward

content: "Implement DirichletCalibrator.forward(): log_probs = log_softmax(logits), cal_logits = linear(log_probs), return cal_logits"status: pendingdependencies:

    - calib_dirichlet_init
- id: calib_dirichlet_fit

content: "Implement fit_dirichlet_calibrator(): create calibrator, use LBFGS optimizer (lr=0.5, max_iter=300), minimize CrossEntropyLoss, return fitted calibrator"status: pendingdependencies:

    - calib_dirichlet_forward
- id: calib_dirichlet_odir

content: "Implement ODIRRegularizer: compute penalty = lambda_odir * (||W - diag(W)||_F^2 + ||b||_2^2), add to loss during fitting"status: pendingdependencies:

    - calib_dirichlet_fit
- id: calib_gate_platt

content: "Implement PlattCalibrator: fit sigmoid(scale * gate_logit + bias) to correctness labels. Use sklearn LogisticRegression or PyTorch implementation"status: pendingdependencies:

    - calibration_gate
- id: calib_gate_isotonic

content: "Implement IsotonicCalibrator fallback: use sklearn IsotonicRegression for non-parametric calibration"status: pendingdependencies:

    - calibration_gate
- id: calib_scrc_selection

content: "Implement SCRC selection control: compute λ1 threshold on val_calib gate scores to control selection/acceptance. Use percentile method"status: pendingdependencies:

    - calibration_scrc
- id: calib_scrc_risk

content: "Implement SCRC risk control: compute λ2 threshold to control set size. Ensure FNR ≤ target_fnr on exited samples. Output prediction sets {0},{1},{0,1}"status: pendingdependencies:

    - calib_scrc_selection
- id: calib_scrc_inference

content: "Implement SCRC inference: given gate_score and class_logits, compute prediction set. If gate_score >= λ1 and class_conf >= λ2: singleton {y}, else: {0,1} (reject)"status: pendingdependencies:

    - calib_scrc_risk
- id: metrics_risk_coverage_compute

content: "Implement compute_risk_coverage(): sweep gate thresholds, compute (coverage, risk) pairs. Risk = error rate on accepted, Coverage = fraction accepted"status: pendingdependencies:

    - metrics_selective
- id: metrics_augrc_compute

content: "Implement compute_augrc(): compute Area Under Generalized Risk Curve using NeurIPS 2024 formula. Addresses AURC flaws"status: pendingdependencies:

    - metrics_selective
- id: metrics_selective_suite

content: "Implement compute_selective_metrics(): compute Risk@Coverage(c) for c in [0.5,0.6,0.7,0.8,0.9], Coverage@Risk(r) for r in [0.01,0.02,0.05,0.10], average set size, rejection rate"status: pendingdependencies:

    - metrics_risk_coverage_compute
    - metrics_augrc_compute
- id: metrics_ece_preserve

content: "Preserve compute_ece() EXACT implementation from train_stage1_head.py: 10 bins, confidence-accuracy matching, bin boundaries, weighted ECE computation"status: pendingdependencies:

    - metrics_calibration
- id: metrics_nll

content: "Implement compute_nll(): Negative Log-Likelihood = -mean(log(probs[labels])). Proper scoring rule"status: pendingdependencies:

    - metrics_calibration
- id: metrics_brier

content: "Implement compute_brier(): Brier score = mean((probs - one_hot_labels)^2). Mean squared error"status: pendingdependencies:

    - metrics_calibration
- id: metrics_exit_softmax

content: "Preserve exit coverage/accuracy computation for softmax threshold: exit_mask = (probs[:,1] >= threshold) | (probs[:,1] <= 1-threshold), compute coverage and accuracy"status: pendingdependencies:

    - metrics_exit
- id: metrics_exit_gate

content: "Implement exit coverage/accuracy for gate threshold: exit_mask = gate_prob >= gate_threshold, compute coverage and accuracy on exited samples"status: pendingdependencies:

    - metrics_exit
- id: explora_mae_objective

content: "Implement MAE-style masked image modeling: mask 75% of patches randomly, predict masked patches from visible patches, reconstruction loss (MSE)"status: pendingdependencies:

    - explora_trainer
- id: explora_peft_blocks

content: "Implement ExPLoRA block unfreezing: unfreeze last 1-2 transformer blocks, apply PEFT (DoRAN/LoRA) to remaining blocks"status: pendingdependencies:

    - explora_trainer
- id: explora_pretrain_loop

content: "Implement ExPLoRA pretraining loop: load unlabeled data, create dataloader, train for 5-10 epochs with MAE objective, save backbone_explora.pth"status: pendingdependencies:

    - explora_mae_objective
    - explora_peft_blocks
- id: explora_unlabeled_load

content: "Implement UnlabeledRoadDataset: load images from NATIX extras and SDXL synthetics directories, no labels needed, same transforms as training"status: pendingdependencies:

    - explora_data
- id: cache_extract_preserve

content: "Preserve extract_features() EXACT logic from train_stage1_head.py: load backbone, create datasets, extract CLS features with torch.no_grad(), save train_features.pt, train_labels.pt, val_features.pt, val_labels.pt"status: pendingdependencies:

    - feature_cache
- id: cache_load

content: "Implement load_cached_features(): load .pt files from cached_features_dir, return train_features, train_labels, val_features, val_labels tensors"status: pendingdependencies:

    - feature_cache
- id: checkpoint_save_format

content: "Preserve checkpoint save format EXACTLY from train_stage1_head.py: dict with epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_acc, patience_counter, ema_state_dict"status: pendingdependencies:

    - checkpointing
- id: checkpoint_load_validate

content: "Implement checkpoint loading with validation: check file exists, load dict, validate keys, handle missing keys gracefully, return checkpoint dict"status: pendingdependencies:

    - checkpointing
- id: logging_csv_header

content: "Preserve CSV logging header EXACTLY from train_stage1_head.py: 'Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,ECE,Exit_Coverage,Exit_Acc,Best_Val_Acc,LR'"status: pendingdependencies:

    - logging
- id: logging_csv_extend

content: "Extend CSV logging with new metrics: add Risk@Coverage, Coverage@Risk, AUGRC, GateCoverage, GateAcc columns. Maintain backward compatibility"status: pendingdependencies:

    - logging_csv_header
- id: script_splits_load_data

content: "Implement script 00_make_splits.py: load NATIX val dataset, call create_val_splits() to create 4-way split (train/val_select/val_calib/val_test), document usage rules in metadata, save splits.json with full metadata including split purposes"status: pendingdependencies:

    - script_splits
- id: script_explora_run

content: "Implement script 10_explora_pretrain.py: load config, load unlabeled data, create ExPLoRATrainer, run pretraining, save backbone_explora.pth"status: pendingdependencies:

    - script_explora
- id: script_train_extract_mode

content: "Implement script 20_train.py extract_features mode: route to Stage1Trainer.extract_features(). Support --phase and --exit_policy arguments."status: pendingdependencies:

    - script_train_full
- id: script_train_cached_mode

content: "Implement script 20_train.py train_cached mode: route to Stage1Trainer.train_cached(). Support --phase and --exit_policy arguments."status: pendingdependencies:

    - script_train_full
- id: script_train_full_mode

content: "Implement script 20_train.py train mode: Validate --phase and --exit_policy compatibility. Route to phase-specific training (Phase 1: baseline, Phase 3: gate, Phase 6: conformal risk). Load adapted backbone if exists, create Stage1Trainer, call train(), save model + val logits"status: pendingdependencies:

    - script_train_full
- id: script_calibrate_load

content: "Implement script 36_calibrate_scrc.py (Phase 6): load best checkpoint, load validation logits from training, prepare val_calib data. Fit Dirichlet calibrator, fit SCRC, save scrcparams.json (validate against schema)."status: pendingdependencies:

    - script_calibrate
- id: script_calibrate_fit_all

content: "Implement script 33_calibrate_gate.py (Phase 3): load gate logits, fit Platt calibrator, save gateparams.json (NOT thresholds.json, validate against gateparams.schema.json). Implement script 36_calibrate_scrc.py (Phase 6): fit Dirichlet calibrator, fit SCRC, save scrcparams.json (validate against scrcparams.schema.json)."status: pendingdependencies:

    - script_calibrate_load
- id: script_eval_load_all

content: "Implement script 40_eval_selective.py: Load bundle.json first to determine active_exit_policy. Load exactly ONE policy file (thresholds.json OR gateparams.json OR scrcparams.json) according to bundle.json.active_exit_policy. Load model + corresponding calibrators (if needed). Load val_test data (NOT val_calib, NOT val_select) for unbiased evaluation."status: pendingdependencies:

    - script_eval
    - bundle_manifest_create
- id: script_eval_compute

content: "Implement script 40_eval_selective.py: Run inference with active policy only (no mixing). Compute risk-coverage curves with bootstrap CIs (1000 samples, 95% CI). Compute AUGRC with bootstrap CI. CRITICAL: Compute and report FNR_on_exited and coverage for active policy with bootstrap CIs. Verify FNR_on_exited ≤ target_fnr_exit (2%) using CI upper bound. Compute proper scoring rules (NLL, Brier) with bootstrap CIs. Generate plots: risk-coverage curve with CI bands, calibration curve, FNR/coverage distributions. Save metrics.csv with all metrics + uncertainty estimates (mean, std, CI_lower, CI_upper)."status: pendingdependencies:

    - script_eval_load_all
- id: script_export_package

content: "Implement script 50_export_bundle.py: collect all artifacts (model, exactly ONE policy file based on exit_policy, splits, metrics, config), create bundle.json manifest with active_exit_policy and file pointers, validate all JSON files against schemas, enforce mutual exclusivity (only one policy file per bundle), create deployment bundle directory, copy all files"status: pendingdependencies:

    - script_export
- id: script_export_readme

content: "Implement script 50_export_bundle.py: generate deployment README with inference instructions, artifact descriptions, usage examples"status: pendingdependencies:

    - script_export_package
- id: artifacts_run_id

content: "Implement run_id generation: use timestamp format YYYYMMDD_HHMMSS, create artifacts/stage1_pro_modular_training_system/runs/<run_id>/ directory structure"status: pendingdependencies:

    - artifacts_structure
- id: artifacts_paths

content: "Ensure all scripts use correct artifact paths: config.json, splits.json, model_best.pth, bundle.json, exactly ONE policy file (thresholds.json OR gateparams.json OR scrcparams.json), calibrators/, metrics/, checkpoints/, logs/ all in run_id directory"status: pendingdependencies:

    - artifacts_run_id
- id: cli_all_args

content: "Implement CLI preserving ALL arguments from train_stage1_head.py: --mode, --model_path, --train_image_dir, --train_labels_file, --val_image_dir, --val_labels_file, --cached_features_dir, --use_extra_roadwork, --roadwork_iccv_dir, --roadwork_extra_dir, --use_kaggle_data, --kaggle_construction_dir, --kaggle_road_issues_dir, --max_batch_size, --fallback_batch_size, --grad_accum_steps, --epochs, --warmup_epochs, --lr_head, --lr_backbone, --weight_decay, --dropout, --label_smoothing, --max_grad_norm, --use_amp, --use_ema, --ema_decay, --early_stop_patience, --legacy_exit_threshold_for_logging (monitoring only), --resume_checkpoint, --output_dir, --log_file"status: pendingdependencies:

    - cli_interface
- id: cli_new_args

content: "Add NEW 2025 CLI arguments: --target_fnr_exit (single constraint), --exit_policy {softmax,gate,scrc}, --peft_type, --peft_r, --peft_blocks, --optimizer, --val_select_ratio, --lambda_cov, --aux_weight, --use_dirichlet, --calibration_iters, --gate_loss_weight, --gate_threshold. CRITICAL: NO --target_coverage argument. Validate exit_policy matches phase."status: pendingdependencies:

    - cli_interface
- id: cli_mode_routing

content: "Implement CLI mode routing: if mode == 'extract_features': call extract_features(), elif mode == 'train_cached': call train_cached(), elif mode == 'train': call train()"status: pendingdependencies:

    - cli_all_args
- id: tests_config_unit

content: "Create unit test for config: test config creation, test validation, test save/load, test default values"status: pendingdependencies:

    - tests_unit
- id: tests_datasets_unit

content: "Create unit test for datasets: test NATIXDataset loading, test MultiRoadworkDataset combining, test header skip, test path handling"status: pendingdependencies:

    - tests_unit
- id: tests_splits_unit

content: "Create unit test for splits: test deterministic splitting, test class balance preservation, test reproducibility with same seed"status: pendingdependencies:

    - tests_unit
- id: tests_model_unit

content: "Create unit test for model: test Stage1Head forward pass, test 3-head outputs, test PEFT application, test backbone freeze"status: pendingdependencies:

    - tests_unit
- id: tests_losses_unit

content: "Create unit test for losses: test SelectiveLoss computation, test ConformalRiskLoss batch splitting, test AuxiliaryLoss"status: pendingdependencies:

    - tests_unit
- id: tests_optimizers_unit

content: "Create unit test for optimizers: test F-SAM forward/backward, test AdamW fallback, test per-layer LR"status: pendingdependencies:

    - tests_unit
- id: tests_metrics_unit

content: "Create unit test for metrics: test ECE computation, test risk-coverage, test AUGRC, test NLL, test Brier"status: pendingdependencies:

    - tests_unit
- id: tests_calibration_unit

content: "Create unit test for calibration: test Dirichlet fitting, test Platt fitting, test SCRC threshold computation"status: pendingdependencies:

    - tests_unit
- id: tests_integration_pipeline

content: "Create integration test for full pipeline: test split creation → explora → train → calibrate → eval flow end-to-end"status: pendingdependencies:

    - tests_integration
- id: tests_integration_cache

content: "Create integration test for feature cache: test extract_features → train_cached flow, verify cached features match direct extraction"status: pendingdependencies:

    - tests_integration
- id: tests_integration_resume

content: "Create integration test for checkpoint resume: test save checkpoint → load checkpoint → resume training, verify state restoration"status: pendingdependencies:

    - tests_integration
- id: tests_reproducibility

content: "Create reproducibility test: same seed → same splits → same training results. Test with multiple runs"status: pendingdependencies:

    - tests_integration
- id: docs_readme_architecture

content: "Write README.md architecture section: explain modular structure, data flow, training pipeline, calibration pipeline"status: pendingdependencies:

    - docs_readme
- id: docs_readme_usage

content: "Write README.md usage section: examples for extract_features mode, train_cached mode, train mode, calibration, evaluation"status: pendingdependencies:

    - docs_readme
- id: docs_readme_migration

content: "Write README.md migration guide: how to migrate from train_stage1_head.py, what's new, what's preserved, comparison table"status: pendingdependencies:

    - docs_readme
- id: docs_api_classes

content: "Write API.md for all classes: Stage1ProConfig, NATIXDataset, MultiRoadworkDataset, Stage1Head, DINOv3Backbone, Stage1Trainer, etc. with full docstrings"status: pendingdependencies:

    - docs_api
- id: docs_api_functions

content: "Write API.md for all functions: compute_ece, compute_risk_coverage, fit_dirichlet_calibrator, etc. with parameter descriptions and return types"status: pendingdependencies:

    - docs_api
- id: validation_paths

content: "Implement path validation: check model_path exists, check train_image_dir exists, check train_labels_file exists, check val paths exist, provide helpful error messages"status: pendingdependencies:

    - validation_config
- id: validation_hyperparams

content: "Implement hyperparameter validation: dropout in [0,1], lr_head > 0, weight_decay >= 0, epochs > 0, warmup_epochs < epochs, etc."status: pendingdependencies:

    - validation_config
- id: validation_mode_compat

content: "Implement mode compatibility validation: extract_features mode doesn't need cached_features_dir to exist, train_cached needs cached features, train needs datasets"status: pendingdependencies:

    - validation_config
- id: error_handling_oom

content: "Implement OOM error handling: catch RuntimeError with 'out of memory', try smaller batch size, provide helpful message"status: pendingdependencies:

    - error_handling
- id: error_handling_missing_data

content: "Implement missing data error handling: check dataset paths exist, provide list of missing datasets, allow graceful degradation (skip missing datasets)"status: pendingdependencies:

    - error_handling
- id: error_handling_checkpoint

content: "Implement checkpoint error handling: validate checkpoint file exists, validate keys present, handle corrupted checkpoints gracefully, provide recovery options"status: pendingdependencies:

    - error_handling
- id: reproducibility_seeds

content: "Implement seed setting: set random.seed(), np.random.seed(), torch.manual_seed(), torch.cuda.manual_seed_all(), cudnn.deterministic=True, cudnn.benchmark=False"status: pendingdependencies:

    - reproducibility
- id: reproducibility_deterministic

content: "Ensure deterministic operations: use deterministic algorithms where possible, save all seeds in config.json, document non-deterministic operations"status: pendingdependencies:

    - reproducibility_seeds
- id: performance_compile

content: "Preserve torch.compile optimization: compile Stage1Head with mode='default', document expected speedup (40%)"status: pendingdependencies:

    - performance
- id: performance_tf32

content: "Preserve TF32 precision: torch.set_float32_matmul_precision('high'), torch.backends.cuda.matmul.allow_tf32=True, torch.backends.cudnn.allow_tf32=True"status: pendingdependencies:

    - performance
- id: performance_dataloader

content: "Optimize DataLoader: use num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True (if supported)"status: pendingdependencies:

    - performance
- id: monitoring_progress

content: "Implement progress bars: use tqdm for training/validation loops, show loss, accuracy, LR in progress bar, update every batch"status: pendingdependencies:

    - monitoring
- id: monitoring_gpu

content: "Implement GPU monitoring: track VRAM usage with torch.cuda.memory_allocated(), log peak memory, warn if approaching limit"status: pendingdependencies:

    - monitoring
- id: monitoring_metrics_viz

content: "Implement metrics visualization: optional real-time plots for loss/accuracy curves, risk-coverage curves, calibration curves (use matplotlib)"status: pendingdependencies:

    - monitoring
- id: phase1_threshold_sweep

content: "PHASE 1 CRITICAL: Implement threshold_sweep.py matching cascade exit logic. Sweep softmax thresholds, compute exit coverage/accuracy/FNR, find threshold satisfying FNR≤2% and maximizing coverage. Save thresholds.json with exit_threshold, fnr_on_exits, coverage, exit_accuracy, sweep_results"status: pendingdependencies:

    - metrics_exit_softmax
- id: phase1_deploy_bundle

content: "PHASE 1 CRITICAL: Implement deploy bundle export with minimum files: model_best.pth, thresholds.json, splits.json, metrics.csv, config.json. Validate bundle structure, create bundle README. Ensure inference loads from bundle (no hard-coded thresholds)"status: pendingdependencies:

    - script_export_package
    - phase1_threshold_sweep
- id: phase1_acceptance_tests

content: "PHASE 1 GATE: Create acceptance tests - Test 1: Training completes successfully, Test 2: Model accuracy matches baseline (within 1%), Test 3: Threshold sweep produces valid thresholds.json, Test 4: Deploy bundle exports successfully, Test 5: Inference loads thresholds from bundle. STOP POINT: Do not proceed to Phase 2 until all tests pass"status: pendingdependencies:

    - phase1_deploy_bundle
    - trainer_train_full
- id: phase2_acceptance_tests

content: "PHASE 2 GATE: Create acceptance tests - Test 1: Selective metrics computed correctly, Test 2: Checkpoint selection uses selective KPIs, Test 3: Metrics match expected ranges, Test 4: No regression in Phase 1 functionality. STOP POINT: Do not proceed to Phase 3 until all tests pass"status: pendingdependencies:

    - metrics_selective_suite
    - script_eval_compute
- id: phase3_acceptance_tests

content: "PHASE 3 GATE: Create acceptance tests - Test 1: Gate head trains successfully, Test 2: Gate-based exit improves over softmax-threshold, Test 3: No regression in classification accuracy, Test 4: Gate calibration works correctly. STOP POINT: Do not proceed to Phase 4 until all tests pass"status: pendingdependencies:

    - calib_gate_platt
    - head_forward
- id: phase4_acceptance_tests

content: "PHASE 4 GATE: Create acceptance tests - Test 1: ExPLoRA pretraining completes successfully, Test 2: PEFT applies correctly, Test 3: Performance improves or maintains baseline, Test 4: No memory issues. STOP POINT: Do not proceed to Phase 5 until all tests pass"status: pendingdependencies:

    - explora_pretrain_loop
    - peft_apply
- id: phase5_acceptance_tests

content: "PHASE 5 GATE (OPTIONAL): Create acceptance tests - Test 1: F-SAM training completes successfully, Test 2: Performance improves over AdamW, Test 3: No memory issues, Test 4: Training stability maintained. STOP POINT: Do not proceed to Phase 6 until all tests pass (if Phase 5 attempted)"status: pendingdependencies:

    - optimizer_fsam_backward
- id: phase6_acceptance_tests

content: "PHASE 6 GATE: Create acceptance tests - Test 1: SCRC calibration completes successfully, Test 2: End-to-end conformal risk training works, Test 3: FNR≤2% constraint satisfied, Test 4: Coverage maximized subject to FNR constraint, Test 5: Prediction sets work correctly. FINAL PHASE"status: pendingdependencies:

    - calib_scrc_inference
    - risk_trainer_step
- id: inference_bundle_loader

content: "CRITICAL: Create inference bundle loader that loads thresholds from thresholds.json (Phase 1), gate calibrator (Phase 3+), SCRC params (Phase 6+). NO hard-coded exit_threshold values in inference code. All thresholds loaded from bundle files"status: pendingdependencies:

    - phase1_deploy_bundle
    - calib_scrc_inference
- id: schemas_create_all

content: "Create stage1_pro_modular_training_system/schemas/ directory with JSON schemas: config.schema.json (training settings, target_fnr_exit, exit_policy, legacy_exit_threshold_for_logging), splits.schema.json (indices, seed, metadata), thresholds.schema.json (exit_threshold, metrics, sweep_results), gateparams.schema.json (gate_threshold, calibrator_type, calibrator_params, metrics), scrcparams.schema.json (lambda1, lambda2, scrc_variant, bounds, version info), bundle.schema.json (active_exit_policy, file pointers, version). Define strict structure for each artifact type."status: pending

- id: schemas_validation

content: "Implement schema validation on all artifact writes and loads using jsonschema library. Validate config.json, splits.json, thresholds.json, gateparams.json, scrcparams.json, bundle.json against their schemas. Fail fast with clear error messages. Add validation to config.save(), splits.save(), threshold_sweep.py, calibrate_gate.py, calibrate_scrc.py, export_bundle.py."status: pendingdependencies:

    - schemas_create_all
- id: bundle_manifest_create

content: "Create bundle.json manifest in 50_export_bundle.py: Contains active_exit_policy, policy_file pointer (thresholds.json OR gateparams.json OR scrcparams.json), model_file, splits_file, config_file, metrics_file, version, created_at. Validates exactly ONE policy file exists. Validates all files against schemas before export."status: pendingdependencies:

    - schemas_validation
    - script_export

---

# Stage-1 Pro Modular Training System (Phased with Gates - Dec 2025)

## Overview

Create a comprehensive modular `stage1_pro_modular_training_system/` package implementing 2025 best practices in **RISK-ORDERED PHASES**.**CRITICAL: Phase gates with hard acceptance criteria - DO NOT proceed to next phase until current phase passes all criteria.Baseline Reference:** `train_stage1_head.py` (preserve EXACT logic from this file, not train_stage1_head.py)**Deploy Bundle is Central:** All inference loads thresholds/params from bundle files; NO hard-coded exit_threshold in inference code.

## Phase Gate Strategy

Each phase has:

- **Acceptance Criteria:** Hard requirements that MUST pass before proceeding
- **Stop Point:** Do not start next phase until current phase is complete
- **Deliverables:** Specific artifacts that must be produced
- **Risk Level:** Low (Phase 1) to High (Phase 6)

## Phase Overview

- **Phase 1 (LOW RISK):** Baseline training + threshold sweep + deploy bundle
- **Phase 2 (LOW-MEDIUM RISK):** Selective metrics + checkpoint selection
- **Phase 3 (MEDIUM RISK):** Learned gate head (if needed)
- **Phase 4 (MEDIUM-HIGH RISK):** PEFT/domain adaptation (ExPLoRA)
- **Phase 5 (HIGH RISK):** F-SAM optimizer experiments (optional)
- **Phase 6 (HIGHEST RISK):** SCRC + end-to-end conformal risk training

## Complete Directory Structure

```javascript
streetvision_cascade/
├── train_stage1_head.py            # Baseline reference (preserve EXACT logic from this)
├── train_stage1_head.py              # Alternative baseline (if exists, prove equivalence)
├── stage1_pro_modular_training_system/                     # NEW: Phased professional training system
│   ├── __init__.py
│   ├── README.md                   # Comprehensive architecture docs
│   ├── config.py                   # Complete config system with validation
│   ├── cli.py                      # CLI interface (all modes + args)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py             # NATIXDataset, MultiRoadworkDataset (ALL datasets)
│   │   ├── splits.py               # Deterministic 4-way split (val_select/val_calib/val_test)
│   │   ├── transforms.py           # TimmStyleAugmentation (aggressive + moderate)
│   │   └── loaders.py              # DataLoader creation + batch auto-detection
│   ├── model/
│   │   ├── __init__.py
│   │   ├── backbone.py             # DINOv3Backbone wrapper (freeze + PEFT hooks)
│   │   ├── head.py                 # Stage1Head (3-head: cls + gate + aux)
│   │   └── peft.py                 # DoRAN/DoRA adapters
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                # SelectiveLoss + ConformalRiskLoss + AuxiliaryLoss
│   │   ├── optimizers.py            # F-SAM + AdamW (per-layer LR)
│   │   ├── schedulers.py            # Cosine + warmup (preserve exact logic)
│   │   ├── trainer.py              # Stage1Trainer (ALL modes: extract/train_cached/train)
│   │   ├── risk_training.py        # End-to-end conformal risk training
│   │   └── ema.py                  # EMA (preserve exact implementation)
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── dirichlet.py            # DirichletCalibrator + ODIR
│   │   ├── gate_calib.py           # PlattCalibrator + IsotonicCalibrator
│   │   └── scrc.py                 # SCRCCalibrator (prediction sets)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── selective.py            # Risk-coverage, AUGRC, selective metrics
│   │   ├── calibration.py          # ECE, NLL, Brier (preserve + extend)
│   │   └── exit.py                 # Exit metrics (FNR_on_exited, coverage) with bootstrap CIs
│   ├── domain_adaptation/
│   │   ├── __init__.py
│   │   ├── explora.py              # ExPLoRATrainer (PEFT extended pretraining)
│   │   └── data.py                 # Unlabeled data loading
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── feature_cache.py        # extract_features mode (preserve logic)
│   │   ├── checkpointing.py        # Save/load checkpoints (preserve format)
│   │   ├── logging.py              # CSV logging (preserve + extend)
│   │   └── monitoring.py          # Progress bars, GPU tracking
│   └── scripts/
│       ├── 00_make_splits.py      # Create 4-way splits (val_select/val_calib/val_test)
│       ├── 10_explora_pretrain.py # ExPLoRA domain adaptation
│       ├── 20_train.py             # Unified training (ALL phases, ALL modes)
│       ├── 25_threshold_sweep.py  # Phase 1: Threshold sweep → thresholds.json
│       ├── 33_calibrate_gate.py   # Phase 3: Gate calibration → gateparams.json
│       ├── 36_calibrate_scrc.py   # Phase 6: SCRC calibration → scrcparams.json
│       ├── 40_eval_selective.py   # Per-policy evaluation on val_test with bootstrap CIs
│       └── 50_export_bundle.py    # Export deployment bundle
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_datasets.py
│   │   ├── test_splits.py
│   │   ├── test_model.py
│   │   ├── test_losses.py
│   │   ├── test_optimizers.py
│   │   ├── test_metrics.py
│   │   └── test_calibration.py
│   └── integration/
│       ├── test_pipeline.py
│       ├── test_feature_cache.py
│       └── test_checkpoint_resume.py
└── artifacts/
    └── stage1_pro_modular_training_system/
        └── runs/
            └── <run_id>/
                ├── config.json
                ├── splits.json
                ├── model_best.pth
                ├── backbone_explora.pth (optional)
                ├── bundle.json              # Manifest: active_exit_policy, file pointers
                ├── thresholds.json           # Phase 1 policy (OR gateparams.json OR scrcparams.json)
                ├── gateparams.json           # Phase 3 policy (optional)
                ├── scrcparams.json          # Phase 6 policy (production default)
                ├── calibrators/
                │   ├── class_dirichlet.pth
                │   └── gate_platt.pth
                ├── metrics/
                │   ├── metrics.csv
                │   ├── risk_coverage_curve.png
                │   └── calibration_curve.png
                ├── checkpoints/
                │   ├── checkpoint_epoch1.pth
                │   ├── checkpoint_epoch2.pth
                │   └── ...
                └── logs/
                    ├── training.log
                    └── calibration.log
```



## Complete Implementation Plan

### Phase 1: Core Infrastructure (Preserve + Extend)

**1.1 Config System** (`stage1_pro_modular_training_system/config.py`)

- `Stage1ProConfig` dataclass with ALL existing fields from `TrainingConfig` in `train_stage1_head.py`:
- Model paths (model_path, train_image_dir, train_labels_file, val_image_dir, val_labels_file)
- Training mode (mode: extract_features/train_cached/train)
- Cached features directory
- Multi-dataset flags (use_extra_roadwork, roadwork_iccv_dir, roadwork_extra_dir)
- Batch sizes (max_batch_size, fallback_batch_size, grad_accum_steps)
- Training schedule (epochs, warmup_epochs)
- Optimizer (lr_head, lr_backbone, weight_decay)
- Regularization (dropout, label_smoothing, max_grad_norm)
- Advanced features (use_amp, use_ema, ema_decay, early_stop_patience)
- Cascade exit (legacy_exit_threshold_for_logging) - **Monitoring only, NEVER used for inference. Inference loads from thresholds.json/gateparams.json/scrcparams.json based on exit_policy**
- Checkpointing (resume_checkpoint)
- Output (output_dir, log_file)
- **Phase 1 ONLY fields:**
- Validation splits: `val_select_ratio: float = 0.33`, `val_calib_ratio: float = 0.33`, `val_test_ratio: float = 0.34` (for 4-way split creation)
- Risk constraint: `target_fnr_exit: float = 0.02` (2% FNR on exited samples) - **ONLY constraint, maximize coverage**
- Exit policy: `exit_policy: str = "softmax"` (Phase 1: only "softmax", Phase 3+: "softmax" or "gate", Phase 6: "scrc" as production default)
- Bootstrap settings: `bootstrap_samples: int = 1000`, `bootstrap_confidence: float = 0.95` (for CI computation)
- **NO target_coverage** (maximize coverage subject to FNR constraint)
- **Phase 2+ fields (disabled in Phase 1):**
- PEFT settings: `peft_type: str = "none"` (disabled until Phase 4)
- Optimizer choice: `optimizer: str = "adamw"` (F-SAM disabled until Phase 5)
- Gate: `gate_loss_weight: float = 0.0` (disabled until Phase 3)
- Selective loss: `lambda_cov: float = 0.0` (disabled until Phase 3)
- Calibration: `use_dirichlet: bool = False` (disabled until Phase 2)
- Config validation: Check paths exist, hyperparameters in valid ranges, mode compatibility
- Save/load with full reproducibility (seeds, timestamps)

**1.2 Data Module** (`stage1_pro_modular_training_system/data/`)**datasets.py** - Preserve EXACT logic from `train_stage1_head.py`:

- `NATIXDataset`: Image loading, header skip logic, label parsing, augmentation toggle
- **Preserve EXACT implementation from train_stage1_head.py**
- `MultiRoadworkDataset`: Multi-source combining with dataset source tracking
- Support datasets: NATIX, ROADWork ICCV, Roboflow (as in baseline)
- Preserve path handling (absolute/relative), header detection, CSV parsing
- Dataset statistics printing
- **Preserve EXACT implementation from train_stage1_head.py**
- Both classes preserve exact `__getitem__` logic with ImageNet normalization

**splits.py** - NEW deterministic 4-way splitting (CRITICAL: Strict separation):

- `create_val_splits()`: Hash-based deterministic 4-way split
- Input: NATIX val dataset
- Output: val_select (33%) + val_calib (33%) + val_test (34%)
- **CRITICAL USAGE RULES:**
- **val_select**: Model selection/early stopping ONLY (checkpoint selection, hyperparameter tuning)
- **val_calib**: Fitting calibrators/policies ONLY (threshold sweep, gate calibration, SCRC fitting)
- **val_test**: Final evaluation ONLY (no model selection, no calibration fitting - unbiased evaluation)
- Stratification: Preserve class balance across all splits
- Reproducibility: Use hash(image_path + seed) for deterministic assignment
- `save_splits()`: Save splits.json with:
- Indices for train/val_select/val_calib/val_test
- Seed used
- Class distribution per split
- Dataset source metadata
- Usage rules documented in metadata
- `load_splits()`: Load and validate splits.json

**transforms.py** - Preserve augmentation:

- `TimmStyleAugmentation`: Preserve exact implementation
- Support both modes: aggressive (scale 0.7-1.0, ColorJitter, p=0.4 RandomErasing) and moderate (scale 0.8-1.0, p=0.25 RandomErasing)
- Configurable via config flag
- Validation transforms: Resize(256) + CenterCrop(224) + ToTensor

**loaders.py** - Batch handling:

- `create_dataloaders()`: Create train/val DataLoaders
- Preserve num_workers=4, pin_memory=True, drop_last logic
- Support gradient accumulation
- `pick_batch_size()`: Preserve exact auto-detection logic from `train_stage1_head.py`
- Try max_batch_size, fallback to fallback_batch_size on OOM
- GPU memory testing with dummy tensors

**1.3 Threshold Sweep** (`stage1_pro_modular_training_system/utils/threshold_sweep.py`)

- **CRITICAL for Phase 1:** Implement threshold sweep matching cascade exit logic
- **MUST use val_calib ONLY** (not val_select, not val_test) for threshold selection
- Sweep softmax thresholds: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]
- For each threshold:
- Compute exit coverage (fraction of samples with max_prob >= threshold)
- Compute exit accuracy (accuracy on exited samples)
- Compute FNR on exited samples
- Find threshold that satisfies FNR ≤ 2% and maximizes coverage
- Save to `thresholds.json` (Phase 1 policy artifact, validate against thresholds.schema.json):
  ```json
                            {
                              "exit_threshold": 0.88,
                              "fnr_on_exits": 0.019,
                              "coverage": 0.75,
                              "exit_accuracy": 0.981,
                              "sweep_results": [...],
                              "val_calib_metrics": {...}
                            }
  ```




- **Separate policy artifacts:**
- `gateparams.json` (Phase 3): gate_threshold, calibrator_type, calibrator_params, metrics
- `scrcparams.json` (Phase 6): lambda1, lambda2, scrc_variant, fnr_bound, coverage, version info
- **Inference contract:** All inference code loads threshold from `thresholds.json`, NO hard-coded values

**1.4 Deploy Bundle Export** (`stage1_pro_modular_training_system/scripts/50_export_bundle.py`)

- **CRITICAL for Phase 1:** Create deploy bundle with minimum required files
- Bundle structure:
  ```javascript
                            deploy_bundle/
                            ├── bundle.json             # Manifest: active_exit_policy, file pointers
                            ├── model_best.pth          # Trained classifier head
                            ├── thresholds.json         # Phase 1: Softmax policy (OR gateparams.json OR scrcparams.json)
                            ├── splits.json             # Train/val splits (for reference)
                            ├── metrics.csv             # Training metrics
                            └── config.json             # Full training configuration
  ```




- **Exactly ONE policy file:** thresholds.json (Phase 1) OR gateparams.json (Phase 3) OR scrcparams.json (Phase 6)
- **bundle.json declares:** active_exit_policy and which policy_file to use
- Bundle validation: Verify all files exist, validate JSON structure
- Bundle README: Document inference usage, file descriptions
- **Inference contract:** Stage-1 inference MUST load from bundle files

**1.5 Training Script** (`stage1_pro_modular_training_system/scripts/20_train.py` - UNIFIED)

- **Single unified script** with `--phase {1..6}` and `--exit_policy {softmax,gate,scrc}` arguments
- **Phase 1:** Baseline training matching `train_stage1_head.py` exactly
- Single-head model (no gate, no aux)
- Standard CrossEntropyLoss with class weights + label smoothing
- AdamW optimizer (exact config from baseline)
- Cosine scheduler + warmup (exact lr_lambda from baseline)
- EMA (exact implementation from baseline)
- Save validation logits for threshold sweep
- **NO selective loss, NO conformal risk, NO gate head**
- **Phase 3:** Train with gate head (3-head, SelectiveLoss)
- **Phase 6:** Train with conformal risk (3-head, ConformalRiskLoss)
- Validate phase supports exit_policy

**1.6 Threshold Sweep Script** (`stage1_pro_modular_training_system/scripts/25_threshold_sweep.py`)

- Load best model checkpoint
- Load validation logits
- Run threshold sweep
- Save `thresholds.json`
- Generate sweep plots (coverage vs threshold, FNR vs threshold)

**1.7 Phase 1 Acceptance Test** (`tests/phase1_acceptance.py`)

- Test 1: Training completes successfully
- Test 2: Model accuracy matches baseline (within 1%)
- Test 3: Threshold sweep produces valid thresholds.json
- Test 4: Deploy bundle exports successfully
- Test 5: Inference loads thresholds from bundle (no hard-coded values)
- **STOP POINT:** Do not proceed to Phase 2 until all tests pass

---

### PHASE 2: Selective Metrics + Checkpoint Selection (LOW-MEDIUM RISK)

**STOP POINT:** Do NOT start Phase 3 until Phase 2 passes all acceptance criteria.**Acceptance Criteria:**

1. ✅ Selective metrics computed correctly (risk-coverage curves, AUGRC)
2. ✅ Checkpoint selection uses selective KPIs (not just accuracy)
3. ✅ Metrics match expected ranges
4. ✅ No regression in Phase 1 functionality

**Deliverables:**

- Selective metrics implementation
- Checkpoint selection by selective KPIs
- Extended metrics.csv with selective metrics

**Exit Strategy:** Still uses softmax-threshold exit from Phase 1. Metrics are for evaluation only, not production exit control.

#### Phase 2 Tasks

**2.1 Selective Metrics** (`stage1_pro_modular_training_system/metrics/selective.py`)

- `compute_risk_coverage()`: Risk-coverage curves
- `compute_augrc()`: AUGRC (NeurIPS 2024)
- `compute_selective_metrics()`: Full suite
- **Evaluation only, not used for exit control**

**2.2 Checkpoint Selection** (`stage1_pro_modular_training_system/training/checkpoint_selection.py`)

- Extend checkpoint selection to use selective KPIs
- Select best checkpoint by AUGRC or risk@coverage
- Still save accuracy-based best checkpoint for compatibility

---

### PHASE 3: Learned Gate Head (MEDIUM RISK)

**STOP POINT:** Do NOT start Phase 4 until Phase 3 passes all acceptance criteria.**Acceptance Criteria:**

1. ✅ Gate head trains successfully
2. ✅ Gate-based exit improves over softmax-threshold
3. ✅ No regression in classification accuracy
4. ✅ Gate calibration works correctly

**Deliverables:**

- 3-head model (cls + gate + aux)
- Gate calibrator
- Gate-based exit implementation

**Exit Strategy:** Gate head becomes primary exit controller. Softmax-threshold remains as logged baseline metric only.

#### Phase 3 Tasks

**3.1 Gate Head Architecture** (`stage1_pro_modular_training_system/model/head.py`)

- Extend Stage1Head to 3-head architecture
- `gate_head`: `Linear(768, 1)` - Selection score
- `aux_head`: `Linear(768, 2)` - Auxiliary classifier
- `forward()` returns: `(logits, gate_logit, aux_logits)`

**3.2 Selective Loss** (`stage1_pro_modular_training_system/training/losses.py`)

- `SelectiveLoss`: Coverage-constrained selective loss
- `AuxiliaryLoss`: Full-coverage classifier loss
- Train gate head with selective loss

**3.3 Gate Calibration** (`stage1_pro_modular_training_system/calibration/gate_calib.py`)

- `PlattCalibrator`: Logistic calibration for gate
- Fit on val_calib gate logits
- Save as `gate_platt.pth`

**3.4 Gate-Based Exit** (`stage1_pro_modular_training_system/inference/gate_exit.py`)

- Implement gate-based exit logic
- Load gate calibrator from gateparams.json (NOT thresholds.json)
- Use gate score for exit decision
- Save gateparams.json with gate_threshold, calibrator_params, metrics
- **Separate artifact:** gateparams.json is Phase 3 policy, thresholds.json is Phase 1 policy

---

### PHASE 4: PEFT/Domain Adaptation (MEDIUM-HIGH RISK)

**STOP POINT:** Do NOT start Phase 5 until Phase 4 passes all acceptance criteria.**Acceptance Criteria:**

1. ✅ ExPLoRA pretraining completes successfully
2. ✅ PEFT (DoRAN/DoRA) applies correctly
3. ✅ Performance improves or maintains baseline
4. ✅ No memory issues

**Deliverables:**

- ExPLoRA adapted backbone
- PEFT implementation
- PEFT training script

#### Phase 4 Tasks

**4.1 ExPLoRA** (`stage1_pro_modular_training_system/domain_adaptation/explora.py`)

- `ExPLoRATrainer`: Parameter-efficient extended pretraining
- Unfreeze last 1-2 blocks, PEFT on rest
- MAE-style masked modeling
- Save `backbone_explora.pth`

**4.2 PEFT** (`stage1_pro_modular_training_system/model/peft.py`)

- `apply_doran()`: DoRAN adapter implementation
- `apply_dora()`: DoRA fallback
- Apply to last N blocks

---

### PHASE 5: F-SAM Optimizer (HIGH RISK - OPTIONAL)

**STOP POINT:** Do NOT start Phase 6 until Phase 5 passes all acceptance criteria (if attempted).**Acceptance Criteria:**

1. ✅ F-SAM training completes successfully
2. ✅ Performance improves over AdamW
3. ✅ No memory issues
4. ✅ Training stability maintained

**Deliverables:**

- F-SAM optimizer implementation
- F-SAM training results
- Comparison with AdamW

#### Phase 5 Tasks

**5.1 F-SAM Optimizer** (`stage1_pro_modular_training_system/training/optimizers.py`)

- F-SAM implementation (CVPR 2024)
- Two-step optimizer
- Gradient checkpointing support
- Per-layer LR adaptation

---

### PHASE 6: SCRC + Conformal Risk Training (HIGHEST RISK)

**STOP POINT:** Final phase - only proceed after all previous phases pass.**Acceptance Criteria:**

1. ✅ SCRC calibration completes successfully
2. ✅ End-to-end conformal risk training works
3. ✅ FNR ≤ 2% constraint satisfied
4. ✅ Coverage maximized subject to FNR constraint
5. ✅ Prediction sets work correctly

**Deliverables:**

- SCRC calibrator
- Conformal risk training implementation
- Prediction set inference
- Updated deploy bundle with SCRC params

**Exit Strategy:** SCRC becomes the ONLY exit controller (Phase 6 production default). Gate and softmax-threshold remain as logged metrics only. Bundle contains scrcparams.json, NOT thresholds.json or gateparams.json.

#### Phase 6 Tasks

**6.1 SCRC Calibration** (`stage1_pro_modular_training_system/calibration/scrc.py`)

- `SCRCCalibrator`: Selective Conformal Risk Control (arXiv 2512.12844)
- SCRC-I (calibration-only, default) and SCRC-T (transductive, optional)
- Two thresholds: λ1 (selection), λ2 (set size)
- Output prediction sets: {0}, {1}, {0,1}
- Save `scrcparams.json` (NOT scrc_params.json, validate against scrcparams.schema.json)

**6.2 Conformal Risk Training** (`stage1_pro_modular_training_system/training/risk_training.py`)

- `ConformalRiskTrainer`: End-to-end conformal risk training
- Batch splitting (pseudo-calib/pseudo-pred)
- FNR ≤ 2% control
- Gradient through CRC objective

**6.3 Dirichlet Calibration** (`stage1_pro_modular_training_system/calibration/dirichlet.py`)

- `DirichletCalibrator`: Matrix scaling on logits
- `ODIRRegularizer`: Off-diagonal + intercept regularization
- Fit on val_calib
- Save `class_dirichlet.pth`

---

## Key Principles

1. **Baseline Reference:** `train_stage1_head.py` is the EXACT preservation baseline. All logic must match this file.
2. **Deploy Bundle Central:** All inference loads from bundle files. NO hard-coded thresholds.
3. **Single Constraint:** Choose FNR ≤ 2% (risk constraint). Maximize coverage subject to this constraint. Do NOT set both target_fnr and target_coverage.
4. **Exit Strategy Evolution:**

- Phase 1: Softmax-threshold exit only (from thresholds.json)
- Phase 3+: Gate-based exit (gate head + calibration)
- Phase 6: SCRC exit (prediction sets)
- Previous methods remain as logged metrics only

5. **Phase Gates:** Hard stop points. Do not proceed until acceptance criteria pass.

## Updated Implementation Details

**2.1 Backbone** (`stage1_pro_modular_training_system/model/backbone.py`)

- `DINOv3Backbone` class:
- Load from pretrained path (preserve AutoModel.from_pretrained logic)
- Freeze all parameters by default (preserve exact freeze logic)
- Feature extraction: `extract_features()` returns CLS token [B, hidden_size]
- PEFT hooks: Expose methods for applying PEFT adapters
- Processor: Preserve AutoImageProcessor handling
- Device management: Preserve .to(device) logic

**2.2 Head** (`stage1_pro_modular_training_system/model/head.py`)

- **Phase 1:** Single-head architecture (matches baseline):
- `Stage1Head` class: `Linear(hidden_size, 768) → ReLU → Dropout(dropout) → Linear(768, 2)`
- `forward()` returns: `logits` only
- **Preserve EXACT architecture from train_stage1_head.py**
- **Phase 3+:** Extend to 3-head architecture:
- `gate_head`: `Linear(768, 1)` - Selection score (sigmoid → exit prob)
- `aux_head`: `Linear(768, 2)` - Auxiliary classifier (training only)
- `forward()` returns: `(logits, gate_logit, aux_logits)`
- Support `torch.compile()` (preserve compilation logic from baseline)
- Preserve initialization (Xavier/Kaiming if needed)

**2.3 PEFT** (`stage1_pro_modular_training_system/model/peft.py`)

- `apply_doran()`: DoRAN adapter implementation
- Apply to last N transformer blocks (configurable, default 6)
- Target modules: QKV projections, attention output, MLP layers
- DoRA fallback: If DoRAN not available, use DoRA
- `apply_lora()`: Basic LoRA fallback
- Integration with backbone: Modify backbone parameters in-place
- Preserve frozen backbone for non-PEFT blocks

### Phase 3: Training Components (Preserve + Extend)

**3.1 Losses** (`stage1_pro_modular_training_system/training/losses.py`)

- `SelectiveLoss`: FNR-constrained selective loss (NO target_coverage)
- Input: logits, gate_logit, labels, target_fnr_exit
- Compute selective risk (error on accepted samples)
- Objective: Minimize selective risk subject to FNR ≤ target_fnr_exit
- Coverage maximized implicitly by minimizing selective risk
- NO coverage penalty - coverage is not targeted explicitly
- `ConformalRiskLoss`: End-to-end conformal risk term
- Split batch: pseudo-calib (50%) + pseudo-pred (50%)
- Compute conformal threshold on calib set
- Differentiate through CRC objective on pred set
- Control FNR ≤ target_fnr on exited samples
- `AuxiliaryLoss`: Full-coverage classifier loss
- Standard CrossEntropyLoss on aux_logits
- Weight: `aux_weight` (default 0.5)
- Prevents collapse during selective training
- Preserve existing: `CrossEntropyLoss` with class weights + label smoothing

**3.2 Optimizers** (`stage1_pro_modular_training_system/training/optimizers.py`)

- `create_optimizer()`: Factory function
- F-SAM: Friendly SAM (CVPR 2024) implementation
    - Two-step optimizer (forward + backward)
    - Adversarial perturbation formation
    - Gradient checkpointing support for memory efficiency
- AdamW: Fallback (preserve exact config: betas=(0.9,0.999), eps=1e-8)
- Per-layer LR adaptation:
- Lower LR for transformer (if PEFT enabled)
- Higher LR for head
- Configurable via param_groups

**3.3 Schedulers** (`stage1_pro_modular_training_system/training/schedulers.py`)

- `create_scheduler()`: Factory function
- Cosine annealing + warmup: Preserve EXACT `lr_lambda` logic from train_stage1_head.py
  ```python
                              def lr_lambda(step):
                                  if step < warmup_steps:
                                      return step / warmup_steps
                                  progress = (step - warmup_steps) / (total_steps - warmup_steps)
                                  return 0.5 * (1 + math.cos(math.pi * progress))
  ```




- Support warmup_epochs configuration

**3.4 Risk Training** (`stage1_pro_modular_training_system/training/risk_training.py`)

- `ConformalRiskTrainer`: End-to-end conformal risk training
- Implements NeurIPS 2025 "End-to-End Optimization of Conformal Risk Control"
- Batch splitting: Each batch → calib (50%) + pred (50%)
- Compute conformal threshold on calib (for FNR control)
- Backprop through risk-control objective on pred
- Target: FNR ≤ 2% on exited samples
- Integration with main training loop

**3.5 EMA** (`stage1_pro_modular_training_system/training/ema.py`)

- Preserve EXACT implementation from train_stage1_head.py:
- `register()`: Initialize shadow parameters
- `update()`: Update shadow with decay
- `apply_shadow()`: Replace params with shadow
- `restore()`: Restore original params
- Decay: 0.9999 (preserve)

**3.6 Main Trainer** (`stage1_pro_modular_training_system/training/trainer.py`)

- `Stage1Trainer` class supporting ALL modes:

**Mode: extract_features**

- Load backbone (frozen)
- Extract CLS features for train/val sets
- Save to cached_features_dir as .pt files
- Preserve exact logic from train_stage1_head.py

**Mode: train_cached**

- Load cached features from disk
- Create TensorDataset from cached features
- Train head only (10x faster)
- Preserve exact training loop logic
- Support checkpoint resuming

**Mode: train**

- Full end-to-end training:
- Load backbone (frozen or with PEFT)
- Load adapted backbone if ExPLoRA checkpoint exists
- Create 3-head model
- Multi-dataset support (preserve MultiRoadworkDataset logic)
- Training loop with:
    - Selective loss + conformal risk loss + auxiliary loss
    - F-SAM or AdamW optimizer
    - Cosine scheduler with warmup
    - EMA updates
    - Gradient accumulation
    - Mixed precision (AMP)
    - Early stopping on val_select
- Validation on val_select (checkpoint selection)
- Save validation logits for calibration
- Comprehensive logging (preserve CSV format + extend)
- Checkpoint saving (preserve format + extend)

### Phase 4: Calibration (New)

**4.1 Dirichlet Calibration** (`stage1_pro_modular_training_system/calibration/dirichlet.py`)

- `DirichletCalibrator`: Matrix scaling on logits
- Architecture: `Linear(num_classes, num_classes, bias=True)`
- Initialize to identity (safe start)
- Forward: `log_probs = log_softmax(logits)` → `cal_logits = linear(log_probs)`
- `ODIRRegularizer`: Off-diagonal + intercept regularization
- Penalty: `lambda_odir * (||W - diag(W)||_F^2 + ||b||_2^2)`
- Prevents overfitting of calibration mapping
- `fit_dirichlet_calibrator()`: Fit on val_calib logits
- Use LBFGS optimizer (300 iterations)
- CrossEntropyLoss objective
- Save as `class_dirichlet.pth`

**4.2 Gate Calibration** (`stage1_pro_modular_training_system/calibration/gate_calib.py`)

- `PlattCalibrator`: Logistic calibration for gate
- Fit sigmoid(scale * gate_logit + bias) to correctness labels
- Use sklearn LogisticRegression or custom PyTorch implementation
- `IsotonicCalibrator`: Alternative (fallback)
- Non-parametric calibration
- Use sklearn IsotonicRegression
- Fit on val_calib gate logits + correctness labels
- Save as `gate_platt.pth`

**4.3 SCRC** (`stage1_pro_modular_training_system/calibration/scrc.py`)

- `SCRCCalibrator`: Selective Conformal Risk Control
- Two-stage procedure:

    1. Selection control: Compute λ1 threshold for gate acceptance
    2. Risk control: Compute λ2 threshold for set size

- Output prediction sets: {0}, {1}, {0,1}
- {0,1} means "reject → Stage-2"
- `fit_scrc()`: Fit on val_calib
- Target: FNR ≤ 2% on exited samples
- Maximize coverage subject to risk constraint
- Save `scrcparams.json` (NOT scrc_params.json) with:
    - λ1, λ2 thresholds
    - Achieved risk/coverage bounds
    - scrc_variant (SCRC-I default, SCRC-T optional)
    - Calibration alpha
    - Model version, splits version (for reproducibility)
    - Validate against scrcparams.schema.json

### Phase 5: Metrics & Evaluation (Preserve + Extend)

**5.1 Selective Metrics** (`stage1_pro_modular_training_system/metrics/selective.py`)

- `compute_risk_coverage()`: Risk-coverage curves
- Input: gate scores, predictions, labels, thresholds
- Output: (coverage, risk) pairs for each threshold
- Risk = error rate on accepted samples
- Coverage = fraction of samples accepted
- `compute_risk_coverage_with_bootstrap()`: Risk-coverage curves with bootstrap CIs
- Bootstrap sampling: 1000 resamples with replacement
- Compute 95% confidence intervals (percentile method: [2.5th, 97.5th] percentiles)
- Output: (coverage_mean, coverage_ci_lower, coverage_ci_upper, risk_mean, risk_ci_lower, risk_ci_upper)
- `compute_augrc()`: AUGRC (NeurIPS 2024)
- Area Under Generalized Risk Curve
- Addresses flaws in AURC
- Interpretable as average risk of undetected failures
- `compute_augrc_with_bootstrap()`: AUGRC with bootstrap confidence intervals
- Bootstrap sampling: 1000 resamples
- Output: (augrc_mean, augrc_std, augrc_ci_lower, augrc_ci_upper)
- `compute_selective_metrics()`: Full suite
- Risk@Coverage(c) for various c values
- Coverage@Risk(r) for various r values
- Average set size
- Rejection rate
- `compute_selective_metrics_with_bootstrap()`: Full suite with bootstrap CIs
- All metrics reported with mean ± std and [CI_lower, CI_upper]

**5.2 Calibration Metrics** (`stage1_pro_modular_training_system/metrics/calibration.py`)

- `compute_ece()`: Preserve EXACT implementation from train_stage1_head.py
- 10 bins, confidence-accuracy matching
- `compute_nll()`: Negative Log-Likelihood
- Proper scoring rule for probabilistic predictions
- `compute_brier()`: Brier score
- Mean squared error between probs and one-hot labels
- `compute_classwise_ece()`: Per-class ECE (extend)

**5.3 Exit Metrics** (`stage1_pro_modular_training_system/metrics/exit.py`)

- Preserve exit coverage/accuracy computation from train_stage1_head.py
- Support all exit policies:
- Phase 1: Softmax threshold (from thresholds.json)
- Phase 3: Gate threshold (from gateparams.json)
- Phase 6: SCRC prediction sets (from scrcparams.json)
- `compute_exit_metrics()`: Compute exit accuracy, exit coverage, FNR_on_exited
- **CRITICAL: Always report FNR_on_exited and coverage for active policy**
- FNR_on_exited = False Negative Rate on samples that exited (missed positives)
- Coverage = Fraction of samples that exited (accepted by Stage-1)
- `compute_exit_metrics_with_bootstrap()`: Exit metrics with bootstrap CIs
- Bootstrap sampling: 1000 resamples
- Output: (fnr_mean, fnr_std, fnr_ci_lower, fnr_ci_upper, coverage_mean, coverage_std, coverage_ci_lower, coverage_ci_upper, exit_accuracy_mean, exit_accuracy_std, exit_accuracy_ci_lower, exit_accuracy_ci_upper)
- Verify FNR_on_exited ≤ target_fnr_exit (2%) with CI upper bound

### Phase 6: Domain Adaptation (New)

**6.1 ExPLoRA** (`stage1_pro_modular_training_system/domain_adaptation/explora.py`)

- `ExPLoRATrainer`: Parameter-efficient extended pretraining
- Load DINOv3 backbone
- Unfreeze last 1-2 transformer blocks
- Apply PEFT (DoRAN/LoRA) to remaining blocks
- Self-supervised objective: MAE-style masked image modeling
    - Mask random patches (75% masking ratio)
    - Predict masked patches from visible patches
    - Reconstruction loss (MSE)
- Train on unlabeled road images (5-10 epochs)
- Save adapted backbone: `backbone_explora.pth`

**6.2 Unlabeled Data** (`stage1_pro_modular_training_system/domain_adaptation/data.py`)

- `UnlabeledRoadDataset`: Load unlabeled road images
- Sources: NATIX extras (if available), SDXL synthetics
- No labels needed (self-supervised)
- Same transforms as training (augmentation)

### Phase 7: Utilities (Preserve)

**7.1 Feature Cache** (`stage1_pro_modular_training_system/utils/feature_cache.py`)

- Preserve EXACT `extract_features()` logic from train_stage1_head.py
- Save train_features.pt, train_labels.pt, val_features.pt, val_labels.pt
- Support loading for train_cached mode

**7.2 Checkpointing** (`stage1_pro_modular_training_system/utils/checkpointing.py`)

- `save_checkpoint()`: Preserve EXACT format from train_stage1_head.py
- Dictionary with: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_acc, patience_counter, ema_state_dict
- `load_checkpoint()`: Load and validate checkpoint
- Support resume from checkpoint

**7.3 Logging** (`stage1_pro_modular_training_system/utils/logging.py`)

- Preserve CSV logging format from train_stage1_head.py:
- Header: "Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,ECE,Exit_Coverage,Exit_Acc,Best_Val_Acc,LR"
- Extend with selective metrics:
- Add: Risk@Coverage, Coverage@Risk, AUGRC, GateCoverage, GateAcc
- **CRITICAL: Add FNR_on_exited and coverage columns for active policy** (with CI columns: FNR_mean, FNR_ci_lower, FNR_ci_upper, Coverage_mean, Coverage_ci_lower, Coverage_ci_upper)
- Support both file and console logging

**7.4 Monitoring** (`stage1_pro_modular_training_system/utils/monitoring.py`)

- Progress bars: Preserve tqdm usage
- GPU memory tracking: Monitor VRAM usage
- Training metrics visualization: Real-time plots (optional)
- Checkpoint size monitoring

### Phase 8: Pipeline Scripts (Complete)

**8.1 Split Creation** (`stage1_pro_modular_training_system/scripts/00_make_splits.py`)

- Load NATIX val dataset
- Create val_select/val_calib splits deterministically
- Save splits.json with full metadata
- Support all dataset sources (if multi-dataset val needed)

**8.2 Domain Adaptation** (`stage1_pro_modular_training_system/scripts/10_explora_pretrain.py`)

- Load unlabeled road images
- Run ExPLoRA pretraining
- Save backbone_explora.pth
- Optional: Can skip if checkpoint exists

**8.3 Unified Training** (`stage1_pro_modular_training_system/scripts/20_train.py`)

- Single unified training script supporting ALL modes:
- `--phase {1..6}`: Phase number (determines training mode)
- `--exit_policy {softmax,gate,scrc}`: Exit policy (validated against phase)
- `--mode extract_features`: Extract and cache features
- `--mode train_cached`: Train on cached features
- `--mode train`: Full training (routes to phase-specific training)
- Phase 1: Baseline training (single-head, CrossEntropyLoss)
- Phase 3: Gate training (3-head, SelectiveLoss)
- Phase 6: Conformal risk training (3-head, ConformalRiskLoss)
- Load adapted backbone if exists (from ExPLoRA)
- Save model + validation logits (for calibration)
- Comprehensive logging

**8.4 Calibration Scripts** (Separate policy artifacts)

- `25_threshold_sweep.py` (Phase 1): Load validation logits, sweep thresholds, save thresholds.json
- `33_calibrate_gate.py` (Phase 3): Load gate logits, fit Platt calibrator, save gateparams.json (NOT thresholds.json)
- `36_calibrate_scrc.py` (Phase 6): Load logits, fit Dirichlet calibrator, fit SCRC (compute λ1, λ2), save scrcparams.json
- Each script validates output against corresponding JSON schema
- Policy artifacts are mutually exclusive (one per bundle)

**8.5 Selective Evaluation** (`stage1_pro_modular_training_system/scripts/40_eval_selective.py`)

- **CRITICAL: Per-policy evaluation (no mixing)**
- Load `bundle.json` first to determine `active_exit_policy`
- Load exactly ONE policy file according to `bundle.json.active_exit_policy`:
- If `active_exit_policy == "softmax"`: Load `thresholds.json` ONLY
- If `active_exit_policy == "gate"`: Load `gateparams.json` ONLY
- If `active_exit_policy == "scrc"`: Load `scrcparams.json` ONLY
- Load model + corresponding calibrators (if needed)
- **MUST evaluate on val_test** (NOT val_calib, NOT val_select) to avoid optimistic results
- Compute risk-coverage curves with bootstrap confidence intervals (1000 bootstrap samples, 95% CI)
- Compute AUGRC with bootstrap CI (mean ± std across bootstrap samples)
- **CRITICAL: Report real constraint metrics for active policy:**
- `FNR_on_exited`: False Negative Rate on exited samples (with bootstrap CI)
- `coverage`: Fraction of samples exited (with bootstrap CI)
- Verify FNR_on_exited ≤ target_fnr_exit (2%)
- Compute proper scoring rules (NLL, Brier) with bootstrap CIs
- Generate plots: risk-coverage curve with CI bands, calibration curve, FNR/coverage distributions
- Save metrics.csv with all metrics + uncertainty estimates (mean, std, CI_lower, CI_upper)

**8.6 Export Bundle** (`stage1_pro_modular_training_system/scripts/50_export_bundle.py`)

- Package all artifacts for deployment:
- `bundle.json` (manifest: active_exit_policy, file pointers, version)
- `model_best.pth` (model weights)
- Exactly ONE policy file: `thresholds.json` (Phase 1) OR `gateparams.json` (Phase 3) OR `scrcparams.json` (Phase 6)
- `splits.json` (for reproducibility)
- `metrics.csv` (evaluation results)
- `config.json` (full training config)
- `backbone_explora.pth` (if exists, Phase 4+)
- `calibrators/class_dirichlet.pth` (if exists, Phase 6)
- `calibrators/gate_platt.pth` (if exists, Phase 3+)
- Validate all JSON files against schemas
- Enforce mutual exclusivity: only one policy file per bundle
- Create deployment README with inference instructions

### Phase 9: CLI Interface (Complete)

**9.1 CLI** (`stage1_pro_modular_training_system/cli.py`)

- Comprehensive argparse preserving ALL arguments from train_stage1_head.py:
- Mode: --mode (extract_features/train_cached/train)
- Paths: --model_path, --train_image_dir, --train_labels_file, etc.
- Multi-dataset: --use_extra_roadwork, --use_kaggle_data, etc.
- Training: --epochs, --warmup_epochs, --lr_head, --weight_decay, etc.
- Regularization: --dropout, --label_smoothing, --max_grad_norm
- Advanced: --use_amp, --use_ema, --early_stop_patience
- Checkpointing: --resume_checkpoint
- Output: --output_dir, --log_file
- Phase-specific arguments:
- Phase 1: --target_fnr_exit (default 0.02) - ONLY constraint, maximize coverage, --exit_policy softmax
- Phase 2: --selective_metrics (flag)
- Phase 3: --gate_loss_weight, --aux_weight, --gate_threshold (default 0.90), --exit_policy {softmax,gate}
- Phase 4: --peft_type (doran/dora/lora/none), --peft_r, --peft_blocks
- Phase 5: --optimizer (fsam/adamw)
- Phase 6: --use_dirichlet (flag), --calibration_iters, --exit_policy scrc (production default)
- **CRITICAL: NO --target_coverage argument** (use single FNR constraint, maximize coverage)
- **CRITICAL: --exit_policy validated against phase** (Phase 1: only softmax, Phase 3+: softmax or gate, Phase 6: all three)
- Mode routing: Call appropriate trainer method
- Help text with examples

### Phase 10: Testing & Validation

**10.1 Unit Tests** (`tests/unit/`)

- `test_config.py`: Config validation, save/load
- `test_datasets.py`: Dataset loading, multi-dataset combining, header skip
- `test_splits.py`: Deterministic splitting, reproducibility
- `test_model.py`: Head forward pass, PEFT application
- `test_losses.py`: Selective loss, conformal risk loss, auxiliary loss
- `test_optimizers.py`: F-SAM, AdamW, per-layer LR
- `test_metrics.py`: ECE, NLL, Brier, risk-coverage, AUGRC
- `test_calibration.py`: Dirichlet, Platt, SCRC fitting

**10.2 Integration Tests** (`tests/integration/`)

- `test_pipeline.py`: Full pipeline (split → explora → train → calibrate → eval)
- `test_feature_cache.py`: extract_features → train_cached flow
- `test_checkpoint_resume.py`: Save → load → resume training
- `test_reproducibility.py`: Same seed → same results

### Phase 11: Documentation (Complete)

**11.1 README** (`stage1_pro_modular_training_system/README.md`)

- Architecture overview with diagram
- Complete feature list (existing + new)
- Usage examples for all modes
- Migration guide from train_stage1_head.py
- Performance benchmarks
- Troubleshooting guide

**11.2 API Docs** (`stage1_pro_modular_training_system/docs/API.md`)

- Detailed docstrings for all classes/functions
- Parameter descriptions with types
- Return value documentation
- Usage examples for each component
- Cross-references between modules

## Key Features Summary

### Preserved from train_stage1_head.py:

1. ✅ Three training modes (extract_features, train_cached, train)
2. ✅ Multi-dataset support (NATIX, ROADWork, Roboflow, Kaggle, Open Images, GTSRB)
3. ✅ Feature caching for fast iteration
4. ✅ Batch size auto-detection
5. ✅ Gradient accumulation
6. ✅ Checkpoint resuming
7. ✅ Comprehensive CSV logging
8. ✅ Class weights for imbalanced data
9. ✅ Exit threshold monitoring
10. ✅ ECE computation
11. ✅ EMA, AMP, torch.compile
12. ✅ LR scheduling (cosine + warmup)
13. ✅ Early stopping
14. ✅ Exact augmentation logic
15. ✅ Header skip logic for CSV files
16. ✅ Path handling (absolute/relative)

### New 2025 Pro Features (Phased by Risk):

**Phase 1 (LOW RISK - SHIP FIRST):**

1. ✅ Deploy bundle with thresholds.json (softmax-threshold exit)
2. ✅ Threshold sweep matching cascade exit logic
3. ✅ 4-way split (train/val_select/val_calib/val_test) with strict usage rules (leakage-free evaluation)

**Phase 2 (LOW-MEDIUM RISK):**

4. ✅ Selective metrics (AUGRC, risk-coverage curves)
5. ✅ Checkpoint selection by selective KPIs
6. ✅ Proper scoring rules (NLL, Brier)

**Phase 3 (MEDIUM RISK):**

7. ✅ 3-head architecture (class + gate + aux)
8. ✅ Gate calibration (Platt/isotonic)
9. ✅ Gate-based exit (replaces softmax-threshold)

**Phase 4 (MEDIUM-HIGH RISK):**

10. ✅ ExPLoRA domain adaptation (PEFT extended pretraining)
11. ✅ DoRAN/DoRA PEFT (efficient backbone fine-tuning)

**Phase 5 (HIGH RISK - OPTIONAL):**

12. ✅ F-SAM optimizer (better generalization)

**Phase 6 (HIGHEST RISK - LAST):**

13. ✅ SCRC (Selective Conformal Risk Control) - final exit controller
14. ✅ End-to-end conformal risk training (FNR ≤ 2% control)
15. ✅ Dirichlet calibration with ODIR

## Migration Path

- Keep `train_stage1_head.py` completely unchanged (baseline reference)
- Phase 1: Ship deploy bundle with baseline training + threshold sweep
- Phase 2+: Add selective metrics and checkpoint selection
- Phase 3+: Add gate head (if needed for improvement)
- Phase 4+: Add PEFT/domain adaptation (if needed)
- Phase 5+: Add F-SAM optimizer (optional experiments)
- Phase 6: Add SCRC + conformal risk training (final upgrade)
- Can compare results: baseline vs Phase 1 vs Phase 6
- Deployment uses Phase 1+ bundle artifacts
- Phase gates ensure each phase is production-ready before proceeding
- Backward compatibility: Can load baseline checkpoints if needed

## Testing Strategy

- Unit tests: Each module tested independently
- Integration tests: Full pipeline validation
- Regression tests: Compare pro vs baseline on same data
- Reproducibility tests: Same seed → same results
- Performance tests: Training speed, memory usage

## Documentation

- `stage1_pro_modular_training_system/README.md`: Complete architecture + usage guide
- `stage1_pro_modular_training_system/docs/API.md`: Detailed API reference
- Docstrings: All classes/functions fully documented
- Examples: Usage examples for all modes
- Migration guide: How to migrate from baseline

## Success Criteria

1. ✅ All existing features preserved exactly
2. ✅ All 2025 pro features implemented
3. ✅ All 2025 pro features implemented
4. ✅ Modular, maintainable structure
5. ✅ Full test coverage

## Critical Fixes Applied (Dec 2025)

### 1. Strict Dataset Separation (4-Way Split)

- **val_select**: Model selection/early stopping ONLY (checkpoint selection, hyperparameter tuning)
- **val_calib**: Fitting calibrators/policies ONLY (threshold sweep, gate calibration, SCRC fitting)
- **val_test**: Final evaluation ONLY (no model selection, no calibration fitting - unbiased evaluation)
- **train**: Training data only
- **Usage rules enforced**: No leakage between splits, documented in splits.json metadata

### 2. Per-Policy Evaluation (No Mixing)

- Evaluation script loads `bundle.json` first to determine `active_exit_policy`
- Loads exactly ONE policy file according to `bundle.json.active_exit_policy`:
- `"softmax"` → `thresholds.json` ONLY
- `"gate"` → `gateparams.json` ONLY
- `"scrc"` → `scrcparams.json` ONLY
- No mixing of policies during evaluation
- All evaluation on `val_test` (NOT `val_calib`, NOT `val_select`)

### 3. Real Constraint Metrics Reporting

- **Always report FNR_on_exited** (False Negative Rate on exited samples) for active policy
- **Always report coverage** (fraction of samples exited) for active policy