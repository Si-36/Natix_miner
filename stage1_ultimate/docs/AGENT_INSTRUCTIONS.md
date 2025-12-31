# 🤖 IMPLEMENTATION AGENT INSTRUCTIONS

**Date**: 2025-12-31  
**Project**: Stage1 Ultimate Upgrade for Bittensor Subnet 72 Mining  
**Goal**: Top 10% competitive performance (not production deployment)

---

## 📋 WHAT YOU NEED TO DO

Implement the complete upgrade plan from `docs/TODO_CHECKLIST.md` (203 tasks).

**This is a COMPETITION system** - performance matters more than production polish. Focus on:
- ✅ Maximum MCC/accuracy/FNR optimization
- ✅ Fast training (BF16 + compile)
- ✅ Advanced features (ExPLoRA, CVFM, multi-objective calibration)
- ❌ Skip: CI/CD, Docker, monitoring (not needed for competition)

---

## 🎯 CRITICAL: PHASE ORDER CLARIFICATION

**There are TWO different orders - don't confuse them!**

### 1. **Implementation Order** (What to BUILD first)
This is the order tasks appear in `TODO_CHECKLIST.md`:
```
Phase 2 (Tasks #9-35) → Phase 1 (Tasks #36-71) → Phase 4a (Tasks #72-105) → Phase 4c (Tasks #106-140) → Phase 5 (Tasks #141-160) → Phase 6 (Tasks #161-175)
```

**Why this order?**
- Phase 2 is easiest (fast win, validates schema)
- Phase 1 validates training stack
- Phase 4a adds domain adaptation
- Phase 4c adds CVFM fusion
- Phase 5 adds calibration
- Phase 6 exports bundle

### 2. **Runtime Order** (What to RUN at training time)
This is the order phases execute when training:
```
phase4a_explora → phase1 → phase2 → phase4c_cvfm → phase5 → phase6
```

**Why this order?**
- Phase 4a adapts DINOv3 backbone from ImageNet → NATIX domain (unsupervised SimCLR)
- Phase 1 trains classification head on domain-adapted backbone (supervised)
- Phase 2 optimizes threshold for MCC on Phase 1 logits
- Phase 4c trains CVFM fusion (optional, improves multi-view)
- Phase 5 calibrates probabilities for better confidence
- Phase 6 exports deployment bundle

**IMPORTANT**: 
- ✅ BUILD phases in implementation order (follow TODO_CHECKLIST.md)
- ✅ RUN phases in runtime order (`phase4a_explora → phase1 → phase2 → phase4c_cvfm → phase5 → phase6`)
- ✅ Phase 4a adapts backbone, Phase 1 trains on adapted backbone, Phase 2 optimizes threshold

---

## 📚 DOCUMENTATION STRUCTURE

1. **`docs/TODO_CHECKLIST.md`** - **START HERE!** 203 atomic tasks
   - Day-by-day breakdown
   - File paths, exact code changes, verification steps
   - Agent-friendly format
   - Progress tracking

2. **`docs/MASTER_PLAN.md`** - Complete technical plan (8,067 lines)
   - Architecture & contracts
   - Artifact schema
   - File-by-file implementation guide
   - Full code templates
   - 2025/2026 upgrades

3. **`docs/START_HERE.md`** - Quick start guide

---

## 🚀 EXECUTION STRATEGY

### Tier 1 (Must Do First): Days 1-2
- **Tasks #1-71**: Setup + Training Stack + MCC Sweep
- **Result**: Competitive baseline (top 30-50%)
- **Risk**: Low (proven techniques)
- **Time**: 2 days
- **✅ DO THIS FIRST, VERIFY IT WORKS**

### Tier 2 (High Value): Days 3-4
- **Tasks #72-105**: ExPLoRA SimCLR domain adaptation
- **Result**: +6-8% MCC (top 20%)
- **Risk**: Medium (complex, but code templates exist)
- **Time**: 2 days
- **✅ DO THIS if Tier 1 works and you have time**

### Tier 3 (High Value): Days 5-6
- **Tasks #106-140**: CVFM learned fusion
- **Result**: +8-12% MCC (top 10%)
- **Risk**: Medium (trainable fusion can overfit)
- **Time**: 2 days
- **✅ DO THIS if Tier 1+2 work and you have time**

### Tier 4 (Polish): Days 7-9
- **Tasks #141-195**: Calibration + Export + Evaluation
- **Result**: Production-ready
- **Risk**: Low (mostly infrastructure)
- **Time**: 3 days
- **✅ DO THIS to finish the system**

---

## ✅ KEY PRINCIPLES

1. **Follow Checklist Exactly**
   - Each task has file path, code snippet, verification step
   - Don't skip verification steps
   - Don't move forward until verification passes

2. **Test After Each Day**
   - Run smoke tests after completing each day's tasks
   - Verify no regressions before continuing
   - Commit after each day's completion

3. **Use Code Templates**
   - All code templates are in `MASTER_PLAN.md`
   - Copy/paste templates, adjust names to match existing modules
   - No sys.path hacks - use proper imports (`pip install -e .`)

4. **2025/2026 Best Practices**
   - PyTorch 2.6+ APIs (`.to(device)` not `.cuda()`)
   - `torchvision.transforms.v2` not deprecated v1
   - Type hints on all functions
   - Atomic writes + manifest tracking

5. **No Data Leakage**
   - TRAIN: training only (Phase 4, Phase 1, CVFM training)
   - VAL_SELECT: early stopping/model selection, CVFM validation
   - VAL_CALIB: Phase 2 threshold fit, Phase 5 calibration fit
   - VAL_TEST: final report only (never tune on this)

---

## 🎯 EXPECTED RESULTS

After completing all 203 tasks:

- **MCC**: 0.94-1.03 (+29-38% improvement)
- **Training Speed**: 3× faster (BF16 + compile)
- **Inference Speed**: 2× faster (compile + CVFM)
- **ECE**: <0.02 (multi-objective calibration)
- **Zero Data Leakage**: Strict split enforcement
- **Full Validator Compliance**: All artifacts pass validation

---

## ⚠️ OPTIONAL FEATURES (Can Skip for v1.0)

- **Task #202**: BYOL/SwAV hybrid (only if SimCLR fails or bottlenecked by batch size)
- **Task #203**: FlexAttention CVFM (only if num_views >5 or learned spatial weighting needed)

**Recommendation**: Skip these for initial implementation. Add only if needed.

---

## 📝 PROMPT TEMPLATE FOR IMPLEMENTATION

Use this exact prompt to start:

```
I want to implement the TODO_CHECKLIST.md (203 tasks) to build a competitive mining system for Bittensor Subnet 72.

CONTEXT:
- This is for COMPETITION mining (need top 10% performance)
- "Average" features = average ranking = low rewards
- Performance matters more than production polish

EXECUTION STRATEGY:
1. Start with Days 1-2 (Tasks #1-71): Baseline + optimizations
2. TEST this baseline works before continuing
3. Then add Days 3-4 (Tasks #72-105): ExPLoRA SimCLR domain adaptation
4. Then add Days 5-6 (Tasks #106-140): CVFM learned fusion
5. Then add Days 7-9 (Tasks #141-195): Calibration + export + evaluation

PHASE ORDER CLARIFICATION:
- Implementation Order (what to build): Phase 2 → Phase 1 → Phase 4a → Phase 4c → Phase 5 → Phase 6
- Runtime Order (what to run): phase4a_explora → phase1 → phase2 → phase4c_cvfm → phase5 → phase6
- Build in implementation order, run in runtime order

Follow the checklist exactly, test after each day, commit regularly.

Start with Task #1: Create git branch for upgrade.
```

---

## ✅ VERIFICATION CHECKLIST

After each day, verify:

- [ ] All tasks for that day completed
- [ ] All verification steps passed
- [ ] Code follows 2025/2026 best practices (no deprecated APIs)
- [ ] No sys.path hacks (proper imports)
- [ ] Tests pass (smoke tests, unit tests)
- [ ] Git commit created with descriptive message
- [ ] Progress tracking updated in TODO_CHECKLIST.md

---

## 🚨 COMMON PITFALLS TO AVOID

1. **Don't confuse phase orders**
   - Build in implementation order (Phase 2 → Phase 1 → Phase 4a)
   - Run in runtime order (phase4a → phase1 → phase2)

2. **Don't skip verification steps**
   - Each task has a verification step - don't skip it
   - If verification fails, fix before moving forward

3. **Don't use deprecated APIs**
   - `.cuda()` → `.to(device)`
   - `transforms.ToTensor()` → built into v2
   - `torch.save(model)` → `torch.save(model.state_dict())`

4. **Don't hardcode paths**
   - Use `ArtifactSchema` for all file paths
   - No hardcoded `"outputs/phase1/..."` paths

5. **Don't leak data**
   - Never train on VAL_CALIB or VAL_TEST
   - Never tune thresholds/calibration on TRAIN or VAL_SELECT

---

## 📊 PROGRESS TRACKING

Update the progress tracking section in `TODO_CHECKLIST.md` after each day:

```markdown
## 📝 PROGRESS TRACKING
- Total Tasks: 203
- Completed: 35   ← Update this
- In Progress: 6
- Blocked: 0
- Skipped: 0

**Completion Rate**: 17.2%
```

---

## 🎯 SUCCESS CRITERIA

You'll know you're done when:

1. ✅ All 203 tasks completed
2. ✅ Full pipeline runs end-to-end: `python scripts/train_cli_v2.py pipeline.phases=[phase4a_explora,phase1,phase2,phase4c_cvfm,phase5,phase6]`
3. ✅ Final MCC >0.90 on VAL_TEST
4. ✅ All artifacts pass validator checks
5. ✅ No data leakage (strict split enforcement)
6. ✅ All code follows 2025/2026 best practices

---

**Status**: ✅ Ready for Implementation  
**Estimated Time**: 7-10 days of focused work  
**Expected Outcome**: Top 10% competitive mining system

