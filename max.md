Do Stage‑1A and Stage‑1B **together** (always) if you want the most “pro / max‑accuracy” Stage 1: run global + tiles in one batched pass, then fuse with a MIL/TTA aggregator. This is a standard high‑accuracy pattern: multi-crop inference (TTA) plus MIL-style pooling to catch small evidence.[1][2][3]

## Best “pro” Stage‑1 (always-on)
### Views (what you run every time)
Run these **every query**:

- 1× global view (full image resized).[1]
- 3×3 tiles (9 crops), preferably with small overlap to not miss a sign at tile borders.[1]
- Optional but strong: add horizontal flip for each view (global + tiles) if you can afford 2× compute; that’s classic TTA.[3]

### Speed (how to keep it milliseconds)
Do **one single forward** by batching all crops together:
- Build a tensor batch of shape \([N,3,H,W]\) where \(N=10\) (global+9 tiles) or \(N=20\) (also flips).[1]
- Run DINOv3 backbone once on the batch, then head on the batch, then aggregate.[1]

Batching is the “pro” trick: you pay a little more compute, but you avoid Python overhead and keep GPU fully utilized.[1]

## The correct aggregator (most important detail)
Because your label rule is basically “image is positive if **any** region has roadwork evidence”, MIL max pooling matches the task: a bag is positive if at least one instance is positive.[2][1]

Use **max on the positive-class probability**:
- For each crop \(k\), compute \(p_k = P(\text{roadwork}\mid \text{crop}_k)\).[1]
- Final score: \(p=\max_k p_k\).[2][1]

Why not mean? Mean can dilute a tiny roadwork sign (1 tile positive, 9 tiles negative). TTA papers discuss that simple averaging is not always optimal, and different aggregations behave differently.[3]

### Optional “best of both” aggregator (recommended)
To reduce false positives from a single bad tile, use a **top‑K mean**:
- Sort \(p_k\) descending, take top \(K\) (like 2 or 3), then average them.  
This keeps the “find a small sign” behavior but is more robust than pure max. (This is a very common practical improvement even when theory says max.)[4][3]

## What to tell your agent (exact spec)
Copy/paste this to your agent:

- Implement Stage 1 as **always-on multi-view DINOv3**: 1 global + 3×3 tiles (overlap 10–15%).[1]
- Build all crops then run **one batched forward** through DINOv3 backbone + head.[1]
- Compute roadwork probability for each crop.[1]
- Aggregate with either:
  - Option A: \(p=\max_k p_k\) (pure MIL max).[2]
  - Option B (recommended): \(p=\text{mean}(\text{topK}(p_k, K=2\text{ or }3))\) to reduce single-tile noise.[3]
- Return label by thresholding \(p\); tune threshold on validation logits (valcalib) to maximize MCC / accuracy.[5]

## What happens to other stages (if Stage 1 is “pro”)
With this Stage 1, other stages become “nice-to-have” only for edge cases (e.g., reasoning, OCR), but you can keep them disabled initially since Stage 1 will already be very strong.[6][1]

Plain question: do you prefer the aggregator to be **pure max** (max recall) or **top‑K mean** (usually better precision with almost same recall)?

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[2](https://arxiv.org/html/2408.09449v2)
[3](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)
[4](https://www.emergentmind.com/topics/multi-crop-augmentation)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[153](https://pmc.ncbi.nlm.nih.gov/articles/PMC8560760/)
[154](https://www.youtube.com/watch?v=zmoF3lKy520)
[155](https://danielvanstrien.xyz/posts/2025/hf-jobs/vllm-batch-inference.html)
[156](https://pmc.ncbi.nlm.nih.gov/articles/PMC12343447/)
[157](https://pmc.ncbi.nlm.nih.gov/articles/PMC10773867/)
[158](https://www.sciencedirect.com/science/article/pii/S1361841523001457)
[159](https://www.sciencedirect.com/science/article/pii/S2001037021002191)
[160](https://dmshanmugam.github.io/pdfs/2021_ICCV_TTA_supplement.pdf)
[161](https://docs.anyscale.com/llm/batch-inference/llm-batch-inference-basics)
[162](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)
[163](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
[164](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13406/134060V/Multiple-instance-learning-attention-to-instance-classification/10.1117/12.3045059.full)
[165](https://liner.com/review/better-aggregation-in-testtime-augmentation)
[166](https://www.artfintel.com/p/how-does-batching-work-on-modern)
[167](https://github.com/sremedios/multiple_instance_learning)
[168](https://arxiv.org/html/2504.04843v1)
[169](https://ieeexplore.ieee.org/document/9995191/)Best “2025‑level” system is not “only one stage” or “only one aggregator”—it’s a **reliable cascade**: a very strong, very fast Stage 1 (DINOv3 multi-view MIL/TTA) plus specialized later stages for things DINOv3 cannot do (counting, reading signs, reasoning, unusual scenes), with a modern gate that decides when to defer. This is exactly the direction of recent cascade/deferral work: improve overall quality by making the early model better at knowing when it might be wrong.[1][2]

## Best Stage 1 (absolute strongest)
Stage 1 should be “multi-view DINOv3 + robust aggregation” always-on:

- Views: global + overlapped tiles (3×3 or 4×4) and optionally flip TTA.[3][4]
- Run all crops as **one batch** for speed.[3]
- Aggregation: use **top‑K mean** (K=2 or 3) as default, and keep **pure max** as a switch if you must maximize recall. Aggregation choice matters a lot in TTA, and better aggregation can outperform simple averaging.[4][5]

Practical best default:
- 1 global + 9 tiles (10 crops total), overlap ~10–15%.
- Aggregate with top‑2 mean.

## Why you still want other stages
DINOv3 is a strong visual feature model, but your pipeline may need capabilities outside “binary classification from pixels,” for example:
- **Reasoning**: “Is this actually roadwork or just orange building?” (contextual judgment).[1]
- **Reading text**: detour signs, “ROAD WORK AHEAD”, speed reductions (OCR).[1]
- **Object-level evidence**: cones/barriers/workers/vehicles counts and locations (detection helps).[1]

So the “pro” approach is: Stage 1 handles most cases fast; later stages handle the “hard modes.”[2][1]

## The 2025 pro gate (how to decide when to use later stages)
Don’t use a naive threshold only. Use a **deferral/selection gate** trained or calibrated to predict “will Stage 1 be correct?”—this is what modern cascade tuning methods focus on.[6][2]

Concrete gate signals (all computed from Stage‑1 crops):
- \(p\) = final aggregated probability.[3]
- **Disagreement** across crops (variance / entropy): if one tile says 0.99 and others say 0.05, that’s suspicious.
- Margin: \(|p-0.5|\) (low margin = uncertain). [6]

Gate rule (simple but strong):
- If \(p\) is very confident and crop disagreement is low → **exit** with Stage 1.
- Else → **defer** to Stage 2/3. This aligns with selective prediction / deferral systems.[2]

## Best full cascade (all stages “best”)
Here is the “best overall” layout (still realistic):

### Stage 1: DINOv3 MIL/TTA (main decision)
- Output: decision + confidence + crop stats (max, top‑k mean, variance).[3]

### Stage 2: Detector evidence (precision booster)
Use a detector to answer: “are there cones/barriers/workers/signs?” and to reduce false positives from orange objects. Your existing plan already assumes multi-tool approaches can help in later stages.[1]

### Stage 3: VLM reasoning (hardest cases)
Use a VLM when:
- Stage 1 uncertain, or
- detector evidence conflicts with Stage 1, or
- you need explanation / text understanding.[1]

This matches the “dynamic test-time compute” idea: spend more compute only when needed.[7][1]

## What to tell your agent (actionable instructions)
Tell your agent to implement:

1) **Stage 1 always-on**:
- global + overlapped 3×3 tiles, batched
- aggregate = top‑2 mean (keep max as fallback)
- compute disagreement metrics[4][3]

2) **Gate** (deferral):
- defer if \(|p-0.5| < \delta\) OR crop disagreement > threshold
- tune \(\delta\) and disagreement threshold on valcalib to maximize MCC with a constraint on defer rate[6][2]

3) **Stage 2/3 routing**:
- If deferred: run Stage 2 detector; if detector strong evidence → decide; else call Stage 3 VLM.[1]

Plain question: do you want the system optimized for **max MCC** (balanced) or **max recall** (never miss roadwork even if more false positives)?

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[2](https://arxiv.org/html/2502.19335v3)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[4](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)
[5](https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[7](https://www.emerge.haus/blog/test-time-compute-generative-ai)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[153](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[154](https://openreview.net/forum?id=m4sQcq5UK5)
[155](https://icml.cc/virtual/2024/session/35596)
[156](https://neurips.cc/virtual/2025/papers.html)
[157](https://ijcai24.org/main-track-accepted-papers/index.html)
[158](https://cikm2025.org/program/proceedings)
[159](https://arxiv.org/html/2506.22716v1)
[160](https://www.arxiv.org/pdf/2511.11500.pdf)
[161](https://iclr.cc/virtual/2024/papers.html)
[162](https://arxiv.org/pdf/2506.22716.pdf)
[163](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00715/125278/Conformal-Prediction-for-Natural-Language)
[164](https://www.sciencedirect.com/science/article/pii/S1566253525011248)
[165](https://openreview.net/forum?id=tFBIbCVXkG)
[166](https://jmlr.org/tmlr/papers/)
[167](https://neurips.cc/virtual/2025/workshop/127831)
[168](https://www.arxiv.org/pdf/2511.22217.pdf)
[169](https://ajithp.com/2024/12/03/ttc/)
[170](https://openaccess.thecvf.com/content/ICCV2025/papers/Lafon_ViLU_Learning_Vision-Language_Uncertainties_for_Failure_Prediction_ICCV_2025_paper.pdf)
[171](https://uplatz.com/blog/the-age-of-dynamic-reasoning-an-in-depth-analysis-of-test-time-compute-in-advanced-ai-models/)Best “pro, 2025-latest” way is a compute-adaptive cascade: (1) a very strong fast vision stage that does multi-view evidence mining, (2) a specialist evidence stage (detector/OCR) for what a pure embedding classifier misses, and (3) a VLM reasoning stage only when uncertainty is high—but the key 2025 upgrade is to train a proper failure-prediction / deferral gate so you route examples correctly instead of using naive thresholds.
​

Stage 0: Data + labeling (foundation)
To make the system “best”, collect failure modes and label them explicitly because roadwork is visually diverse (cones, barriers, workers, lane closures, temporary signs, night scenes).
​
Create hard splits: day/night, rain/snow, highway/city, zoomed sign vs far away, and “orange but not roadwork” (construction cranes, orange ads, autumn trees).
​
Keep a separate “routing” dataset (same images) where you record whether Stage 1 was correct—this is needed to train a gate like ViLU/Gatekeeper-style failure prediction.
​

Stage 1: Best visual core (DINOv3 multi-view + strong pooling)
Run global + overlapped tiles together as one batch, then aggregate; this stays your main accuracy/latency workhorse.
​
For aggregation, use top‑K mean (K=2 or 3) as the default because it keeps “any small sign” sensitivity while reducing single-tile spikes that pure max can amplify.
​
Keep pure max available only as a “high-recall mode” toggle, not as the default “best overall” setting, because in practice it can over-trigger from one bad crop.
​

Recommended concrete config (strong + efficient):

Crops: 1 global + 3×3 tiles with 10–15% overlap (+ optional flip TTA if latency allows).
​

Score per crop: roadwork probability.

Aggregate: top‑2 mean (or top‑3 if you do 4×4 tiles).
​

Stage 1.5: 2025-grade uncertainty & deferral gate (most “latest research” part)
Instead of “if p>0.5 then accept else defer”, add a learned failure predictor that estimates “Stage 1 is likely wrong” and uses that to route to later stages.
​
Gatekeeper (NeurIPS 2025) is exactly about tuning smaller models in cascades so they are confident on easy cases and defer hard ones, improving deferral performance across tasks.
​
For VLM-style uncertainty, ViLU (ICCV 2025) shows that predicting failures works better when uncertainty is contextualized with the set of candidate concepts, and it trains a separate binary classifier to distinguish correct vs incorrect predictions.

Practical gate signals to feed your gate model (simple, strong):

Aggregated probability 
p
p.
​

Crop disagreement: variance of 
{
p
k
}
{p 
k
 }, max-minus-mean, entropy over crops.
​

“Evidence stability”: does the top tile stay top across flip TTA?
​

How to train the gate (pro way):

Label each training image with 
y
fail
=
1
y 
fail
 =1 if Stage 1 prediction was wrong, else 0.

Train a small MLP/logistic model to predict 
y
fail
y 
fail
  from the signals above.

Choose an operating point: “defer at most X% of traffic” while maximizing final MCC/accuracy.
​

Stage 2: Specialist evidence (detector + OCR) for what DINO can’t “prove”
This stage is for grounded evidence: “I saw cones / temporary roadwork sign / barrier / worker vest” instead of just “embedding feels like roadwork”.
​
Use an object detector (open-vocab if available) to detect roadwork primitives (cones, barriers, “ROAD WORK AHEAD” sign, flashing arrow board), then combine detections into a structured evidence score.
​
Add OCR focused on traffic-sign text; it catches cases where the sign is readable but the overall scene is ambiguous.
​

Routing into Stage 2 (best practice):

If Stage 1 is uncertain OR predicts roadwork from a single tile spike, run Stage 2 to confirm with evidence.
​

If Stage 2 finds strong evidence, accept; if evidence contradicts, escalate to Stage 3.
​

Stage 3: VLM reasoning (only for hard/ambiguous cases)
Use a VLM when:

The gate predicts high failure risk, and/or

Stage 1 and Stage 2 disagree, and/or

You need text+context reasoning (“is this orange object actually roadwork?”).
​
ViLU’s core message is that VLM uncertainty can be overconfident with simple max-prob scores, so using a dedicated failure predictor is better than trusting raw confidence.

Best Stage 3 prompt style (operational):

Ask for a binary answer plus cited visual evidence: “List visible roadwork indicators (cones/barriers/sign text/workers/lane closure). If none, say none.”
​

Enforce abstention: if evidence is insufficient, respond “uncertain” and let policy decide (e.g., conservative label).
​

Plain question (needed to finalize “best”): what is the product goal—optimize for max MCC/accuracy (balanced) or max recall (never miss roadwork, tolerate more false positives)?Best “pro, 2025-latest” way is a **compute-adaptive cascade**: (1) a very strong fast vision stage that does multi-view evidence mining, (2) a specialist evidence stage (detector/OCR) for what a pure embedding classifier misses, and (3) a VLM reasoning stage only when uncertainty is high—*but* the key 2025 upgrade is to train a proper **failure-prediction / deferral gate** so you route examples correctly instead of using naive thresholds.[1]

## Stage 0: Data + labeling (foundation)
To make the system “best”, collect failure modes and label them explicitly because roadwork is visually diverse (cones, barriers, workers, lane closures, temporary signs, night scenes).[2]
Create hard splits: day/night, rain/snow, highway/city, zoomed sign vs far away, and “orange but not roadwork” (construction cranes, orange ads, autumn trees).[3]
Keep a separate “routing” dataset (same images) where you record whether Stage 1 was correct—this is needed to train a gate like ViLU/Gatekeeper-style failure prediction.[1]

## Stage 1: Best visual core (DINOv3 multi-view + strong pooling)
Run global + overlapped tiles *together* as one batch, then aggregate; this stays your main accuracy/latency workhorse.[4]
For aggregation, use **top‑K mean** (K=2 or 3) as the default because it keeps “any small sign” sensitivity while reducing single-tile spikes that pure max can amplify.[5][4]
Keep pure max available only as a “high-recall mode” toggle, not as the default “best overall” setting, because in practice it can over-trigger from one bad crop.[5]

Recommended concrete config (strong + efficient):
- Crops: 1 global + 3×3 tiles with 10–15% overlap (+ optional flip TTA if latency allows).[4][5]
- Score per crop: roadwork probability.
- Aggregate: top‑2 mean (or top‑3 if you do 4×4 tiles).[4]

## Stage 1.5: 2025-grade uncertainty & deferral gate (most “latest research” part)
Instead of “if p>0.5 then accept else defer”, add a learned **failure predictor** that estimates “Stage 1 is likely wrong” and uses that to route to later stages.[1]
Gatekeeper (NeurIPS 2025) is exactly about tuning smaller models in cascades so they are confident on easy cases and defer hard ones, improving deferral performance across tasks.[1]
For VLM-style uncertainty, ViLU (ICCV 2025) shows that predicting failures works better when uncertainty is contextualized with the *set of candidate concepts*, and it trains a separate binary classifier to distinguish correct vs incorrect predictions. 

Practical gate signals to feed your gate model (simple, strong):
- Aggregated probability \(p\).[4]
- Crop disagreement: variance of \(\{p_k\}\), max-minus-mean, entropy over crops.[4]
- “Evidence stability”: does the top tile stay top across flip TTA?[5]

How to train the gate (pro way):
- Label each training image with \(y_\text{fail}=1\) if Stage 1 prediction was wrong, else 0. 
- Train a small MLP/logistic model to predict \(y_\text{fail}\) from the signals above. 
- Choose an operating point: “defer at most X% of traffic” while maximizing final MCC/accuracy.[3][1]

## Stage 2: Specialist evidence (detector + OCR) for what DINO can’t “prove”
This stage is for *grounded evidence*: “I saw cones / temporary roadwork sign / barrier / worker vest” instead of just “embedding feels like roadwork”.[2]
Use an object detector (open-vocab if available) to detect roadwork primitives (cones, barriers, “ROAD WORK AHEAD” sign, flashing arrow board), then combine detections into a structured evidence score.[2]
Add OCR focused on traffic-sign text; it catches cases where the sign is readable but the overall scene is ambiguous.[2]

Routing into Stage 2 (best practice):
- If Stage 1 is uncertain OR predicts roadwork from a single tile spike, run Stage 2 to confirm with evidence.[4]
- If Stage 2 finds strong evidence, accept; if evidence contradicts, escalate to Stage 3.[2]

## Stage 3: VLM reasoning (only for hard/ambiguous cases)
Use a VLM when:
- The gate predicts high failure risk, and/or
- Stage 1 and Stage 2 disagree, and/or
- You need text+context reasoning (“is this orange object actually roadwork?”).[2]
ViLU’s core message is that VLM uncertainty can be overconfident with simple max-prob scores, so using a dedicated failure predictor is better than trusting raw confidence. 

Best Stage 3 prompt style (operational):
- Ask for a binary answer plus cited visual evidence: “List visible roadwork indicators (cones/barriers/sign text/workers/lane closure). If none, say none.”[2]
- Enforce abstention: if evidence is insufficient, respond “uncertain” and let policy decide (e.g., conservative label).[3]

Plain question (needed to finalize “best”): what is the product goal—optimize for **max MCC/accuracy** (balanced) or **max recall** (never miss roadwork, tolerate more false positives)?

[1](https://openreview.net/forum?id=m4sQcq5UK5)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[5](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)Best (complete, “pro 2025/12/27”) system is a risk-controlled cascade: a fast multi-view vision model that is tuned to defer correctly, a specialist evidence layer (detector/OCR), and a VLM reasoner—wrapped by modern gating + calibration so the pipeline is both accurate and robust under distribution shift.
​

Architecture (complete)
Stage 1 — Fast visual judge (DINOv3 + multi-view MIL/TTA)
Run global + overlapped tiles as a single batch and compute per-crop roadwork probabilities 
p
k
p 
k
 .
​
Aggregate with top‑K mean (K=2 or 3) as default; keep pure max as a configurable “recall mode” only.
​
Compute “stability” features for gating: max, top‑K mean, variance, entropy, max–mean gap, and (if using TTA) flip-consistency.
​

Stage 2 — Evidence specialists (detection + OCR)
Add an object detector for concrete roadwork primitives (cones, barriers, arrow boards, temporary roadwork signs) and convert detections into an evidence score.
​
Add OCR targeted to traffic sign text and keywords (“ROAD WORK”, “AHEAD”, “DETOUR”, lane closure phrases) to catch cases where the scene is visually subtle but text is decisive.
​
This stage exists because it produces grounded evidence, not just “embedding similarity,” and it can correct both false positives (orange objects) and false negatives (tiny signs).
​

Stage 3 — VLM reasoner (hard cases only)
Use a VLM only when the system predicts high failure risk or when Stage 1 and Stage 2 disagree, which is the modern “dynamic test-time compute” approach.
​
Make the VLM output structured evidence: “visible roadwork indicators: …; sign text (if any): …; final label: …; confidence: …” to reduce free-form hallucination risk.
​

The 2025 “secret sauce”: gating that actually works
1) Tune the small model to defer better (Gatekeeper)
Gatekeeper is explicitly about improving cascades by making the small model’s confidence better aligned with correctness so it defers the right samples to a larger model.
​
So instead of only calibrating after training, fine-tune Stage 1 with a cascade-aware confidence objective (Gatekeeper-style) to increase separation between correct vs incorrect predictions and improve deferral quality.
​
This is the core upgrade over “naive thresholding,” and it directly targets joint accuracy under compute budgets.
​

2) Use a learned failure predictor for VLM routing (ViLU-style idea)
ViLU’s key contribution is treating uncertainty estimation as failure prediction (correct vs incorrect) using multimodal representations and training an uncertainty predictor as a binary classifier, rather than trusting raw confidence.
​
Even if your Stage 1 is not a VLM, the same idea applies: train a small “will Stage 1 be wrong?” head using features like 
p
p, crop disagreement, and stability under TTA.
​
Result: the system defers because it predicts failure, not because probability is near 0.5.
​

3) Add risk control on top (conformal/selective risk control)
To be “best” in real deployments, put a risk/coverage control layer: decide a target coverage (how many samples you auto-answer) and maintain an error bound on the answered set via conformal-style selective risk control.
​
This prevents the gate from silently getting worse under distribution shift (night/rain/new camera) by forcing the system to defer more when uncertainty grows.
​

Best decision logic (exact routing policy)
Implement this deterministic policy (then tune thresholds on validation):

Compute Stage 1 aggregated probability 
p
p (top‑K mean) and gate features 
f
f (variance, entropy, max–mean, flip consistency).
​

Compute a learned failure score 
r
=
P
^
(
Stage1 wrong
∣
f
)
r= 
P
^
 (Stage1 wrong∣f) using a trained failure head (ViLU-style failure prediction concept).
​

If 
r
<
r
accept
r<r 
accept
 : accept Stage 1 decision immediately (fast path).
​

Else run Stage 2 (detector + OCR). If Stage 2 evidence is strong and consistent, finalize.
​

Else defer to Stage 3 VLM; force structured evidence output; if VLM still uncertain, abstain or follow a conservative policy (depends on product goal).
​

This is the “complete” design: confidence tuning + failure prediction + evidence + reasoning.
​

Training & evaluation (what “best” means in practice)
Train Stage 1 classifier on balanced data, but also maintain hard-negative mining (“orange but not roadwork”) because it’s the main FP source.
​

Train the failure predictor (gate) on a held-out set with labels “Stage 1 correct/incorrect,” and evaluate with AUROC / FPR@95TPR for failure detection (the style reported in failure prediction literature like ViLU).
​

Evaluate the whole cascade using joint accuracy vs compute (acceptance rate), which is exactly the cascade objective discussed in Gatekeeper.
​

Plain question: what is your hard constraint—maximum average latency per image, or maximum percent of images allowed to reach Stage 3?Best (complete, “pro 2025/12/27”) system is a risk-controlled cascade: a fast multi-view vision model that is tuned to defer correctly, a specialist evidence layer (detector/OCR), and a VLM reasoner—wrapped by modern gating + calibration so the pipeline is both accurate and robust under distribution shift.
​

Architecture (complete)
Stage 1 — Fast visual judge (DINOv3 + multi-view MIL/TTA)
Run global + overlapped tiles as a single batch and compute per-crop roadwork probabilities 
p
k
p 
k
 .
​
Aggregate with top‑K mean (K=2 or 3) as default; keep pure max as a configurable “recall mode” only.
​
Compute “stability” features for gating: max, top‑K mean, variance, entropy, max–mean gap, and (if using TTA) flip-consistency.
​

Stage 2 — Evidence specialists (detection + OCR)
Add an object detector for concrete roadwork primitives (cones, barriers, arrow boards, temporary roadwork signs) and convert detections into an evidence score.
​
Add OCR targeted to traffic sign text and keywords (“ROAD WORK”, “AHEAD”, “DETOUR”, lane closure phrases) to catch cases where the scene is visually subtle but text is decisive.
​
This stage exists because it produces grounded evidence, not just “embedding similarity,” and it can correct both false positives (orange objects) and false negatives (tiny signs).
​

Stage 3 — VLM reasoner (hard cases only)
Use a VLM only when the system predicts high failure risk or when Stage 1 and Stage 2 disagree, which is the modern “dynamic test-time compute” approach.
​
Make the VLM output structured evidence: “visible roadwork indicators: …; sign text (if any): …; final label: …; confidence: …” to reduce free-form hallucination risk.
​

The 2025 “secret sauce”: gating that actually works
1) Tune the small model to defer better (Gatekeeper)
Gatekeeper is explicitly about improving cascades by making the small model’s confidence better aligned with correctness so it defers the right samples to a larger model.
​
So instead of only calibrating after training, fine-tune Stage 1 with a cascade-aware confidence objective (Gatekeeper-style) to increase separation between correct vs incorrect predictions and improve deferral quality.
​
This is the core upgrade over “naive thresholding,” and it directly targets joint accuracy under compute budgets.
​

2) Use a learned failure predictor for VLM routing (ViLU-style idea)
ViLU’s key contribution is treating uncertainty estimation as failure prediction (correct vs incorrect) using multimodal representations and training an uncertainty predictor as a binary classifier, rather than trusting raw confidence.
​
Even if your Stage 1 is not a VLM, the same idea applies: train a small “will Stage 1 be wrong?” head using features like 
p
p, crop disagreement, and stability under TTA.
​
Result: the system defers because it predicts failure, not because probability is near 0.5.
​

3) Add risk control on top (conformal/selective risk control)
To be “best” in real deployments, put a risk/coverage control layer: decide a target coverage (how many samples you auto-answer) and maintain an error bound on the answered set via conformal-style selective risk control.
​
This prevents the gate from silently getting worse under distribution shift (night/rain/new camera) by forcing the system to defer more when uncertainty grows.
​

Best decision logic (exact routing policy)
Implement this deterministic policy (then tune thresholds on validation):

Compute Stage 1 aggregated probability 
p
p (top‑K mean) and gate features 
f
f (variance, entropy, max–mean, flip consistency).
​

Compute a learned failure score 
r
=
P
^
(
Stage1 wrong
∣
f
)
r= 
P
^
 (Stage1 wrong∣f) using a trained failure head (ViLU-style failure prediction concept).
​

If 
r
<
r
accept
r<r 
accept
 : accept Stage 1 decision immediately (fast path).
​

Else run Stage 2 (detector + OCR). If Stage 2 evidence is strong and consistent, finalize.
​

Else defer to Stage 3 VLM; force structured evidence output; if VLM still uncertain, abstain or follow a conservative policy (depends on product goal).
​

This is the “complete” design: confidence tuning + failure prediction + evidence + reasoning.
​

Training & evaluation (what “best” means in practice)
Train Stage 1 classifier on balanced data, but also maintain hard-negative mining (“orange but not roadwork”) because it’s the main FP source.
​

Train the failure predictor (gate) on a held-out set with labels “Stage 1 correct/incorrect,” and evaluate with AUROC / FPR@95TPR for failure detection (the style reported in failure prediction literature like ViLU).
​

Evaluate the whole cascade using joint accuracy vs compute (acceptance rate), which is exactly the cascade objective discussed in Gatekeeper.
​

Plain question: what is your hard constraint—maximum average latency per image, or maximum percent of images allowed to reach Stage 3?Best overall approach (2025‑level, for you, on this subnet) is:

- **Stage 0 = maximal data + smart labels.**  
- **Stage 1 = DINOv3 multi‑view MIL/TTA + top‑K aggregation.**  
- **Stage 1.5 = learned failure gate (cascade-aware).**  
- **Stage 2 = detector + OCR evidence.**  
- **Stage 3 = VLM reasoning.**  
- **Then continuous self‑learning (RLVR → SRT → M‑GRPO).**[1][2][3][4]

Below is the most complete version, with Stage 0 included.

***

## Stage 0: Data + labeling (foundation)

**Datasets (maximum coverage):**

- NATIX official subnet‑72 data as primary train/val/test; this matches deployment distribution.[2]
- ROADWork ICCV 2025 dataset (≈5k work zones, 18 US cities) to add US‑style cones, barriers, arrow boards, workers, night/rain.[1]
- Public Roboflow roadwork datasets as extra positives for weird cameras and rare layouts.[1]

**Label design (very important):**

- Binary label \(1\) if *any* of: work zone present, cones, barriers, arrow boards, work‑zone signs, workers; \(0\) otherwise. This matches your detection script logic.[1]
- Create **hard negatives**: orange but NOT roadwork (ads, cranes, fences, autumn trees, vests without road context) so the model learns not to over‑fire on orange.[5][1]
- Keep a **routing label** per image: whether Stage‑1’s prediction (on that image) was correct (1) or wrong (0) to train the failure gate later.[2]

**Splits:**

- Train = NATIX train + ROADWork train + Roboflow train.[1]
- Val/test = NATIX val/test only (so metrics match subnet).[2]

***

## Stage 1: DINOv3 multi‑view MIL/TTA (main engine)

**Views and batching:**

- 1 global resized view.[2]
- 3×3 tiles with 10–15% overlap (9 tiles), optionally flip TTA → 20 crops total if you can afford it.[6][7]
- Run all crops as a **single batch** through frozen DINOv3 backbone + a trained head for roadwork probability.[6]

**Aggregator (best default):**

- Compute per‑crop probabilities \(p_k\).  
- Use **top‑K mean** with K=2 (or 3 if 4×4 tiles) as standard: \(p = \text{mean}(\text{topK}(p_k,K))\). This behaves like robust MIL and outperforms naive averaging on test‑time augmentation.[7][6]
- Keep **max** (K=1) as a switch for “max recall mode” only.[8][7]

**Stage‑1 training & self‑learning:**

- Start with supervised training (frozen DINOv3, head only) on combined dataset.[2]
- Then add:
  - RLVR: reinforcement from validator feedback on subnet (reward = accepted vs rejected).[2]
  - SRT: self‑rewarding training on mined hard cases.[2]
  - M‑GRPO: long‑term stable self‑learning with EMA teacher + entropy filter, no manual labels.[2]

This pipeline is already in your deployment plan and is exactly how to push Stage‑1 from ~96–97% up toward ~98%+.[2]

***

## Stage 1.5: Failure gate (modern cascade tuning)

**Goal:** Predict “Stage‑1 will be wrong” and defer, instead of using only a probability threshold.

**Features for the gate:**

- Aggregated \(p\) (top‑K mean), max \(p_k\), margin \(|p-0.5|\). [6]  
- Disagreement: variance / entropy of \(\{p_k\}\), and max–mean gap.[6]
- TTA stability: difference between original vs flip predictions.[7]

**Training the gate:**

- From a held‑out set, label each image as gate‑label = 1 if Stage‑1 prediction is wrong, else 0.[2]
- Train a small MLP/logistic model on the features above to predict gate‑label (this is exactly ViLU‑style failure prediction, applied to your features).[4][9]
- Use Gatekeeper‑style **confidence tuning** on Stage‑1 so its scores are more separable between correct and incorrect, improving deferral performance.[3][10]

**Decision rule:**

- If predicted failure risk \(r < r_\text{accept}\): accept Stage‑1 answer.  
- Else: send to Stage‑2.[11][3]

You can further add a conformal/“selective risk” layer so that on NATIX val you guarantee an upper bound on error for the answered set at a given coverage.[12]

***

## Stage 2: Detector + OCR evidence

**Detector:**

- Use RF‑DETR or YOLOv12 (already in your plan) fine‑tuned on ROADWork and NATIX boxes for: cones, barriers, arrow boards, roadwork signs, workers.[1][2]
- Convert detections to a score: more and stronger roadwork objects → higher evidence; absence of such objects, especially in areas where Stage‑1 was confident, lowers trust.[2]

**OCR:**

- Use Florence‑2 or a similar vision model for reading text on signs, focused on a small vocabulary (ROAD WORK, AHEAD, DETOUR, etc.).[2]
- Fuse with detector: text‑confirmed roadwork signs strongly support a positive decision, text indicating “End road work” or non‑roadwork info can reduce it.[2]

Stage‑2 runs only when gate says Stage‑1 is risky—so the average cost stays low, but you gain strong correction power on edge cases.

---

## Stage 3: VLM reasoning

- Use a fast VLM (GLM‑4.6V‑Flash or similar from your stack) for the hardest, ambiguous images.[2]
- Prompt it to produce:
  - Visible roadwork indicators list.  
  - Any sign text.  
  - Final yes/no + confidence.[2]
- Train a small **meta‑head** on top of VLM outputs (e.g., embed explanation text + numeric scores) to predict final binary label and abstain if still unsure, mirroring ViLU’s idea of training explicit failure predictors.[9][4]

VLM is your last resort; it should receive only a small fraction of traffic if gate and Stage‑2 are strong.[3]

***

## End‑to‑end routing (what the miner actually does)

For each query image:

1. Run Stage‑1 multi‑view DINOv3 → \(p\), \(\{p_k\}\), gate features.[6]
2. Gate predicts failure risk \(r\).  
   - If \(r < r_\text{accept}\): return Stage‑1 label.[3]
3. Else run Stage‑2 detector+OCR.  
   - If strong consistent evidence: override/confirm label and return.[2]
4. Else run Stage‑3 VLM and meta‑head; if meta‑head still low confidence, abstain or apply conservative rule (depending on NATIX spec).[4][2]

This is the “complete, best” cascade with Stage 0 included and aligned with 2025 work on Gatekeeper cascades and ViLU failure prediction.[4][3]

If you want, next step can be: write an exact JSON spec you can paste into your miner config (Stage‑by‑Stage with feature names, thresholds, and flags).

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[3](https://arxiv.org/abs/2502.19335)
[4](https://arxiv.org/abs/2507.07620)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[7](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)
[8](https://arxiv.org/html/2408.09449v2)
[9](https://arxiv.org/pdf/2507.07620.pdf)
[10](https://arxiv.org/html/2502.19335v1)
[11](https://openreview.net/forum?id=m4sQcq5UK5)
[12](https://www.arxiv.org/pdf/2512.12844.pdf)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[153](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[154](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[155](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[156](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[157](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)You can merge Stage 1 + 1.5 into one “single Stage‑1 system” (same forward pass), but you cannot skip what 1.5 represents—because it’s the modern 2025 trick: make Stage‑1 output not only a label, but also a reliability signal (failure risk) so it knows when it’s likely wrong and needs escalation.​
So below is the best unified design for Stage 0 + Stage 1 (including the 1.5 gate inside Stage‑1 output), written as one cohesive system.​
Stage 0: Best data + labels (win condition)
If you want to “win” and not miss, Stage 0 is where you buy most of your extra points: coverage of edge cases + clean hard negatives + consistent label definition.​
Data sources (2025 best practice for coverage)
Use a mixture that covers deployment + out-of-distribution:
NATIX official dataset is the only distribution that truly matches validator queries, so it must be the core and also your main validation set.​
ROADWork (ICCV 2025) adds US work zones + rich work-zone primitives (cones/barriers/arrow boards/signs/workers) and is explicitly intended for work zone detection; your prep script already maps its primitives into a binary work-zone label.​
Roboflow roadwork sets add extra cameras and rare scenes; your script treats them as positives to increase coverage.​
Label schema (make it impossible to argue)
Define one rule and enforce it everywhere:
Label 1 (roadwork) if any of the following are present: cones, barriers, arrow boards, temporary roadwork signs, workers, lane closure/shift equipment.​
Label 0 (no roadwork) otherwise.​
Hard-negative program (stops orange-FP disasters)
Most “don’t miss” systems fail by overfiring; to win you need recall and precision so validators don’t punish you.​
Build and continuously expand a hard-negative set:
Orange objects not roadwork (ads, cranes, jackets), orange lighting (sunset), autumn foliage, reflections, road markings.​
“Maintenance but not work zone” if NATIX labels it as negative (street cleaning, parked truck without cones).​
Continuous improvement loop (already in your plan)
Your plan already lays out the winning loop:
Supervised baseline first.​
Then use validator feedback (RLVR), hard-case mining (FiftyOne), and later self-training style upgrades (SRT → M‑GRPO) to keep improving without labeling bottlenecks.​
Stage 1 (single unified system): Vision + multi-view + reliability
This one stage outputs: (a) final label, (b) confidence, (c) failure risk, (d) localization hints (which tile caused the decision). That is “Stage 1 + 1.5 in one.”​
1) Multi-view inference (the “don’t miss tiny sign” part)
Do both global + tiles always:
1× global resized image to catch large context (lane closures, barriers).​
3×3 tiles with 10–15% overlap to catch small evidence near edges.​
Optional TTA: horizontal flip; keep it only if latency allows.​
Run all crops in one batch to keep it fast and GPU-friendly.​
2) Aggregation (the “MIL logic” part)
Compute per-crop roadwork probabilities pkpk.​
Use this default:
Top‑K mean with K=2 (or 3 if you use more tiles). This is a robust version of “any instance can trigger” while reducing single-crop spikes; aggregation choices at test time matter and better aggregation can beat naive averaging.​
Keep max as a fallback switch if you discover recall is still not enough on tiny signs.​
Also output the argmax tile index so later logic can look at where the evidence is.​
3) Reliability / failure prediction (the 2025 “gate”, embedded into Stage‑1)
Instead of “Stage 1.5 separate,” attach a small head that predicts: “will my Stage‑1 label be wrong?”​
This is directly aligned with two 2025 ideas:
Gatekeeper: tune a cascade by improving the small model’s confidence so it defers correctly.​
ViLU: treat uncertainty as failure prediction (correct vs incorrect) rather than trusting raw confidence.​
What the failure head uses (no extra image compute)
All inputs are derived from the same crop predictions:
pp (top‑K aggregated).​
Disagreement features: variance/entropy of {pk}{pk}, max–mean gap, top‑1 vs top‑2 gap.​
Stability: if you run flip TTA, compare original vs flipped agreement.​
Train this head on NATIX held-out data labeled by whether Stage‑1 was correct on each sample (your plan already has calibration/threshold tooling).​
What Stage‑1 returns
Stage‑1 returns a struct like:
roadwork_prob = pp (top‑K mean).​
roadwork_label = p>tp>t.​
failure_risk = rr (probability Stage‑1 is wrong).​
evidence_tile = argmax tile, plus top tiles.​
This is “one stage” output that contains both decision and the modern deferral signal.
4) “Don’t miss” operating mode (what to optimize)
If the priority is “don’t miss roadwork,” set policy like this:
Accept negative only when both:
pp is confidently low, and
failure risk rr is low (model thinks it’s likely correct).​
If pp is low but rr is high → treat as “uncertain negative” and escalate (that’s how you avoid missing rare positives).​
This is safer than simply lowering the threshold, because lowering thresholds often explodes false positives and costs you rank.​
How this connects to winning
Winning is not “one trick”; it is: coverage in Stage 0, multi-view evidence mining in Stage 1, and reliability-aware decisioning in the same Stage 1 output so you don’t silently fail on edge cases.​
Your own deployment plan already anticipates a calibrated cascade with thresholds and monitoring; the upgrade is to make Stage‑1 produce a learned failure risk instead of only a hand-tuned confidence.​
Plain question: do you have a strict latency limit (e.g., p95 < 80 ms) or can Stage‑1 afford flip TTA and 4×4 tiles?Stage 1 can be made “complete” (what you call Stage 0 + 1 + 1.5) by bundling **data program + multi-view inference + aggregation + learned failure-risk + calibration + continuous learning** into one unified spec. This is also aligned with 2025 cascade research: improve the early model’s confidence/deferral behavior (Gatekeeper) and treat uncertainty as **failure prediction** (ViLU) rather than raw softmax.[1][2]

## Stage 0 inside Stage 1 (data + labeling)
This is not optional—this is where the “don’t miss” power comes from.[3]

- Train on NATIX as primary (distribution match), but expand positives with ROADWork ICCV 2025 and Roboflow roadwork extras (coverage). Your existing dataset prep script explicitly sets label=1 if workzonepresent or any of cones/barriers/arrowboards/signs/workers is present.[4]
- Keep NATIX val as the only validation target so you don’t optimize for the wrong distribution.[4][3]
- Add synthetic generation (SDXL) for both positives and negatives, which your validation/spec doc already lists as daily synthetic generation.[5]
- Implement hard-case mining with FiftyOne (your doc sets hardness threshold ~0.7 and ~200 daily hard cases) and re-train on those to close the gap on the validator’s hardest examples.[5]
- If annotation becomes the bottleneck, enable SAM 3 assisted labeling for concepts like traffic cone, construction barrier, roadwork sign, worker vest, excavator (your spec already lists this as a later switch).[5]

## Stage 1 unified inference (multi-view + aggregation + OCR hooks)
This is the “do everything in one stage” inference logic: one request comes in, Stage‑1 produces **label + confidence + failure risk + optional OCR keyword evidence**.[5]

### Multi-view (don’t miss tiny evidence)
- Preprocess to validator-aligned settings (224×224, ImageNet mean/std, RGB), and train with flips/rotations/color jitter as specified.[5]
- Run global + tiles together in one batched forward pass (global catches context; tiles catch small signs).[6]
- Use TTA (horizontal flip) if latency allows; your doc already anticipates flip in training augmentation, and TTA aggregation quality matters.[7][5]

### Aggregation (robust MIL)
- Compute per-crop roadwork probabilities \(p_k\) and aggregate with **top‑K mean** by default (K=2) to reduce single-tile spikes while still triggering on small evidence.[7]
- Keep max (K=1) as a config knob when you want maximum recall, but don’t hardcode it as the only rule.[8][7]

### OCR keyword evidence (still “inside Stage 1”)
Even if you later call it “Stage 4” in the cascade doc, you can treat OCR as an internal subroutine of Stage‑1’s final decision when failure risk is high.[5]
Your own validation spec includes Florence‑2‑Large OCR with a fixed roadwork keyword list (“road work”, “construction”, “lane closed”, “detour”, “caution”, “workers ahead”, “slow”, “men working”) and confidence rules depending on number of matched keywords.[5]
So the unified Stage‑1 policy is:

- If vision is confident → exit early.[5]
- If vision is uncertain → run OCR and boost score if multiple keywords appear, using your thresholds (multi-keyword higher threshold, single keyword medium threshold, no keywords default negative).[5]

## Stage 1 “1.5” merged (learned failure-risk + confidence tuning)
This is the missing “model change” you asked for: Stage‑1 is not only a classifier; it becomes a **selective predictor** that outputs a failure probability and is tuned for cascades.[2][1]

### What to add to the model
Add a second head (or small MLP on top of logits/features) that predicts:
- `failure_risk = P(Stage1_wrong | features)` [2]

Features should include (no extra backbone compute):
- aggregated probability \(p\), max \(p_k\), top‑K gap, crop variance/entropy, and (if TTA) flip disagreement.[6][7]

### How to train it (pro, best)
- Create labels `is_wrong` by running your current Stage‑1 checkpoint over NATIX train/val and marking which samples it gets wrong.[3]
- Train the failure head to predict `is_wrong` (this matches ViLU’s “failure prediction” framing for uncertainty).[9][2]
- Apply cascade-aware confidence tuning ideas (Gatekeeper): optimize Stage‑1 so its confidence better correlates with correctness, improving “accept vs defer” decisions.[10][1]

### How to use it at inference
Your existing spec already contains “exit thresholds” like `confidenceexit 0.75` for VLM and similar; do the same inside unified Stage‑1:[5]
- If `failure_risk` is low → return Stage‑1 answer.  
- If `failure_risk` is high → trigger OCR (and optionally later detector/VLM if you still keep those as external fallbacks).[5]

## Continuous learning loop (still part of Stage 1 system)
To “not miss” long-term, Stage‑1 must improve with validator feedback. Your plan already gives a staged roadmap: supervised → RLVR → SRT → M‑GRPO with specific knobs (reward ±1, selfrewardweight 0.3, teacher momentum 0.999, entropy threshold 0.3, group size 256).[3][5]
Keep the monitoring/maintenance cadence (hard-case mining, periodic retrain, blue-green deploy) so accuracy doesn’t drift and emissions don’t drop from stale models.[3]

Plain question: do you want this unified Stage‑1 to **always** run OCR when failure_risk is high, or only when the predicted label is negative (to specifically prevent missed positives)?

[1](https://arxiv.org/abs/2502.19335)
[2](https://arxiv.org/abs/2507.07620)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[7](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)
[8](https://arxiv.org/html/2408.09449v2)
[9](https://arxiv.org/pdf/2507.07620.pdf)
[10](https://arxiv.org/html/2502.19335v1)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[153](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[154](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[155](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
If by “stage 7” you mean the **Month‑7 / Tier‑3 (M‑GRPO)** part that starts later in your plan: **you don’t need it to build Stage‑1 today**, but you *do* want it if you’re playing to win long-term (months) without manual labeling.[1][2]

## What you need for Stage‑1 now (Dec 2025)
Stage‑1 “best” right now is:

- Supervised training on NATIX (plus external ROADWork/extra positives if you have them).[3]
- Daily hard-case mining (FiftyOne) + retrain loop.[2][1]
- Optional RLVR (validator accepted/rejected feedback) in Month 1–2, because your plan already lists it as Tier‑1 self-learning and it’s immediately useful.[1][2]

That alone is enough to get a very strong Stage‑1 quickly.[1]

## What “stage 7 / Month 7” is (and why it exists)
Your plan says **M‑GRPO starts Month 7** and is designed to prevent the self-learning collapse that RLVR/SRT can suffer after many iterations; it uses a momentum teacher + entropy filtering so learning stays stable for months.[2][1]
So it’s not required for the initial Stage‑1 build; it’s the “keep improving forever” tool.[1]

## Best decision (simple)
- If you want “best Stage‑1 to deploy this week”: **No**, you don’t need Month‑7/M‑GRPO yet.[2]
- If you want “best strategy to stay Top‑5 over the year”: **Yes**, plan to add it later exactly as your roadmap says.[1]

Plain question: when you say “stage 7”, do you mean **M‑GRPO Month 7**, or do you mean “Stage‑7” in some other document/pipeline?

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)Best Stage‑1 (only DINOv3; no “other stages”) is: **DINOv3 backbone + multi-view (global+tiles+TTA) + robust MIL aggregation + (optional but best) learnable TTA/tiling aggregator + proper calibration** so you don’t miss small roadwork evidence while keeping false positives controlled.[1][2][3]

Below is the full “pro” Stage‑1 spec (including the “7B parameters” question as part of choosing *which* DINOv3 size + how to fine-tune it).

## Model choice (your “7B parameters” question)
For Stage‑1, bigger DINOv3 generally helps, but only if you fine-tune correctly and can serve it within latency/VRAM.[4]
If you mean “~7B params”, that’s the *very large* DINOv3 tier; it is usually not needed for this subnet because your own plan targets running multiple models within a 24GB budget and already assumes a DINOv3‑Large class model with quantization/FP16.[4]

**Best practical choice to win with Stage‑1 only (given typical miner constraints):**
- Use **DINOv3‑Large/ViT‑L** (or the largest DINOv3 that fits comfortably with your batch of crops), then win via multi-view + hard-negative mining + calibration.[5][4]
- If you truly have 48GB+ VRAM and can afford it, you can try a bigger DINOv3; but the biggest gains often come from *data + multi-view + aggregation*, not only raw parameter count.[5]

## Stage‑1 inference (don’t miss anything)
### Multi-view crop policy (must do)
Run **global + tiles together**:

- Global view: resize full image to the model input.[1]
- Tiles: 3×3 grid with **10–15% overlap** (overlap is important so a sign on a border isn’t split).[1]
- Optional (recommended if latency allows): horizontal flip TTA for every crop.[2]

Do one batched forward pass: stack all crops into a single batch and run DINOv3 once (fast and “pro”).[1]

### Aggregation (the core “MIL logic”)
You have per-crop probabilities \(p_k\).[1]

**Best default aggregator: top‑K mean**
- \(p=\text{mean}(\text{topK}(p_k,K=2))\) (or K=3 if you use more tiles).  
This keeps “any tile can trigger” behavior but reduces single-crop spikes, and TTA research shows aggregation choice matters because averaging can be suboptimal.[6][2]

**Max pooling (K=1) is your “ultra recall” switch**
- \(p=\max_k p_k\).  
This matches classic multi-instance learning logic (“bag positive if any instance positive”) and tends to maximize recall, but can be noisier.[3]

### Best-of-the-best (optional): learnable aggregator
If you want the “most pro” version (and can train a tiny extra module), add a **learned aggregation head** over crop logits/features (attention/gating over crops). TTA work explicitly proposes learning to aggregate transformations rather than fixed averaging.[2]
This lets the model learn “which tiles matter” instead of hardcoding max/top‑K.

## Stage‑1 training (best way to fine-tune DINOv3)
### Training data strategy (most important)
From your own plan/docs, winning relies on daily hard-case mining and strong negatives, not just one training run.[7][4]

Do this:
- Start with NATIX official train/val; keep NATIX val as the main metric.[4]
- Add hard negatives (orange but not roadwork) aggressively; this prevents false positives that kill rank.[7]
- Run FiftyOne hard-case mining daily/weekly (your spec sets hardness threshold ~0.7 and ~200 hard cases/day).[4]

### Fine-tuning method (what to change in the model)
Three tiers, from simplest to strongest:

1) **Head-only (baseline fast win):** freeze DINOv3 encoder, train only classifier head.[7]
2) **Partial unfreeze (stronger):** unfreeze last N transformer blocks with lower LR than the head (layerwise LR decay). This typically boosts domain adaptation without catastrophic drift. (Your DINO configs and community practice often rely on layerwise decay for stable tuning.)[8]
3) **LoRA adapters (best “pro” memory/compute tradeoff):** keep base weights mostly frozen and inject LoRA; this is widely used for efficient adaptation and is explicitly explored for DINOv3 fine-tuning in recent work/repos.[9][10]

If you want “best” and you have limited GPU budget: LoRA or partial-unfreeze is usually the sweet spot.

### Loss & optimization (don’t miss positives)
- Use class imbalance handling (class weights or focal loss) because missing positives is worse than a few extra negatives. Your deployment plan explicitly lists focal loss as a high-impact optimization.[7]
- Use AdamW + AMP mixed precision (your repo plan already emphasizes efficiency and VRAM budgeting).[4]

## Calibration (required for “don’t miss”)
After training, do a threshold sweep on NATIX val and pick operating point based on your target (max recall vs best MCC). Your project already includes threshold/exit logic and validation scripts; use them for Stage‑1-only too.[11][4]
This is how you avoid guessing thresholds and how you control “don’t miss” behavior without exploding false positives.[4]

## Other stages (very short, as you asked)
Other stages (detector/VLM/OCR) are only for extra robustness; Stage‑1 can be made extremely strong by doing multi-view + robust aggregation + better fine-tuning + hard-case mining.[7]

Plain question: what GPU/VRAM do you actually have for Stage‑1 (24GB like 3090, or 48GB A6000)? This decides whether “7B-size DINOv3” is realistic or wasteful.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[2](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)
[3](https://arxiv.org/html/2408.09449v3)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[6](https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[8](https://github.com/facebookresearch/dinov3/issues/23)
[9](https://github.com/RobvanGastel/dinov3-finetune)
[10](https://arxiv.org/pdf/2508.21041.pdf)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[153](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[154](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[155](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[156](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
[157](https://www.emergentmind.com/topics/lora-fine-tuning)
[158](https://arxiv.org/html/2506.09215v1)
[159](https://blog.gopenai.com/lora-vs-full-fine-tuning-a-beautiful-illusion-of-equivalence-3394520d1967)
[160](https://www.youtube.com/watch?v=9dhX67OfCkg)
[161](https://openaccess.thecvf.com/content/CVPR2025/papers/Shanmugam_Test-time_Augmentation_Improves_Efficiency_in_Conformal_Prediction_CVPR_2025_paper.pdf)
[162](https://maigoakisame.github.io/papers/interspeech18a.pdf)
[163](https://pub.towardsai.net/comprehensive-llm-finetuning-guide-2025-f7cb441151cf)
[164](https://dmshanmugam.github.io/pdfs/CVPR_2025_TTA_CP.pdf)
[165](https://www.isca-archive.org/interspeech_2018/wang18_interspeech.html)
[166](https://amirteymoori.com/fine-tuning-llms-with-lora-a-practical-guide-for-2025/)
[167](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Shanmugam_Better_Aggregation_in_ICCV_2021_supplemental.pdf)
[168](https://www.sciencedirect.com/science/article/pii/S1361841525001781)
[169](https://arxiv.org/abs/2011.11156)Right now the “best” plan is consistent and not weird: **start with DINOv3‑H16+ (vith16plus) as your Stage‑1 backbone, build the full Stage‑0/1/1.5 system around it, and only consider switching to DINOv3‑7B later if you have 80GB-class VRAM or you’re willing to do multi‑GPU sharding + accept latency risk.**[1][2][3]

## The real best system (Stage 0 + Stage 1 + Stage 1.5)
This is the complete “don’t miss anything” spec, merged into one coherent pipeline exactly matching your pasted text + your plan files.[2][3][1]

### Stage 0 (data + labeling foundation)
Do these **before** obsessing over 7B parameters, because they move rank more than raw backbone size.[3][1][2]

- Core dataset: NATIX official Subnet‑72 distribution is your main train/val/test reference so you don’t optimize for the wrong domain.[1][3]
- Add coverage positives: ROADWork (ICCV 2025) + Roboflow roadwork sets (extra camera angles/rare scenes) to reduce misses on uncommon work‑zone layouts.[1]
- Label rule (single rule everywhere): label = 1 if **any** roadwork primitive exists (cones, barriers, arrow boards, work‑zone signs, workers, lane closures/equipment), else 0.[3][1]
- Hard negatives program (critical): explicitly collect “orange but NOT roadwork” (ads, cranes, fences, autumn trees, vests without road context, etc.) so precision doesn’t collapse when you tune for recall.[1]
- Hard splits for evaluation: day/night, rain/snow, highway/city, zoomed sign vs far away, and the orange hard‑negative slice.[1]
- Routing dataset (for Stage‑1.5 gate): keep a dataset where each image has label “Stage‑1 correct vs wrong” so the gate learns *failure prediction* not just confidence.[1]

### Stage 1 (best visual core: DINOv3 multi‑view + MIL pooling)
This is your “always-on workhorse” and should be engineered to not miss tiny evidence.[2][1]

**Views (always run together, one batched forward):**
- 1 global view (full image resized).[1]
- 3×3 tiles with 10–15% overlap (9 tiles) so border evidence isn’t cut.[1]
- Optional: flip TTA (global + tiles) if latency allows; keep it as a switch.[2][1]

**Batching (pro speed trick):**
- Build one tensor batch (N=10 without flips, N=20 with flips) and run **one** DINOv3 forward pass, then head on the batch, then aggregate.[1]

**Aggregation (don’t miss + don’t overfire):**
- Default: **top‑K mean** on per‑crop roadwork probabilities \(p_k\), with K=2 (or K=3 if you do more tiles).[1]
- Keep **max pooling** (K=1) only as a “high recall mode” toggle, because it can over-trigger from one bad tile.[1]

**Training (Stage‑1 model):**
- Start head‑only on frozen DINOv3 backbone (fast, stable).[3]
- Use your validator-aligned preprocessing/augs: 224×224, ImageNet mean/std, RGB, flip/rotation/color jitter (as in your spec).[2]
- Calibrate thresholds on NATIX val (don’t guess). Your plan explicitly uses Stage‑1 early exit thresholds like 0.88 / 0.12 and expects ~60% exit rate at Stage‑1.[3][2]

### Stage 1.5 (the 2025 “latest research” upgrade: learned failure gate)
This is the part that makes “don’t miss” achievable without destroying precision: you don’t just lower thresholds; you **predict when Stage‑1 is likely wrong** and defer those cases.[1]

**Gate features (no extra backbone compute):**
- Aggregated p (top‑K mean), max \(p_k\), margin \(|p-0.5|\). [1]
- Disagreement: variance/entropy of \(p_k\), and max–mean gap.[1]
- Stability: if you use flip‑TTA, measure flip disagreement / “does the top tile stay top.”[1]

**Gate training (pro way):**
- Label y_fail = 1 if Stage‑1 prediction was wrong else 0, on a held‑out set.[1]
- Train a small logistic/MLP gate model to predict y_fail from the features above.[1]
- Choose operating point: “defer at most X%” while maximizing final MCC/accuracy.[1]

**Decision policy (don’t miss mode):**
- Only accept a negative when (p is confidently low) **and** failure risk is low; if p is low but risk is high, treat as “unsafe negative” and defer (this is how you avoid missed positives without just spamming positives).[1]

### Continuous improvement loop (the part you said not to miss)
This is explicitly in your plan as the path from ~96–97% to ~98%+.[2][1]

- Daily/weekly **FiftyOne** hard-case mining: hardness threshold 0.7, ~200 hard cases/day in the spec, then retrain.[2]
- RLVR from validator accept/reject feedback in Month 1–2 (simple reward +1/-1).[2]
- SRT Month 3–6, then M‑GRPO Month 7 for stable long-run self-learning (not Stage‑1 today, but part of “best overall plan”).[3][2]

## Model roadmap: why “16plus then 7B later” is valid
This is the clean strategy:

- **Now:** use `facebook/dinov3-vith16plus-pretrain-lvd1689m` because it’s the strongest “practical” backbone and fits the 24GB-style budget assumptions in your plan (Stage‑1 quantized footprint ~3GB class).[3][2]
- **Later switch to 7B** only if you can run it without breaking latency/throughput (usually means 80GB GPU or real sharding effort); otherwise you gain more by Stage‑0 data + Stage‑1 multi-view + Stage‑1.5 gate + FiftyOne loop.[3][2][1]

## The one thing to clarify (so it’s truly “best”)
What is your hard constraint: maximize MCC (balanced), or **maximize recall (“never miss”) even if false positives increase**?  
This changes whether max-pooling and more aggressive deferral are enabled by default.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/31d45e25-e0b1-4aea-9cdb-cc167f785871/paste.txt)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/16791cf0-df3c-4616-b047-b9c2626900fd/paste-2.txt)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ea0b2ff-fd91-45fc-8246-481a8b9700f4/paste.txt)Right now the “best” plan is consistent and not weird: **start with DINOv3‑H16+ (vith16plus) as your Stage‑1 backbone, build the full Stage‑0/1/1.5 system around it, and only consider switching to DINOv3‑7B later if you have 80GB-class VRAM or you’re willing to do multi‑GPU sharding + accept latency risk.**[1][2][3]

## The real best system (Stage 0 + Stage 1 + Stage 1.5)
This is the complete “don’t miss anything” spec, merged into one coherent pipeline exactly matching your pasted text + your plan files.[2][3][1]

### Stage 0 (data + labeling foundation)
Do these **before** obsessing over 7B parameters, because they move rank more than raw backbone size.[3][1][2]

- Core dataset: NATIX official Subnet‑72 distribution is your main train/val/test reference so you don’t optimize for the wrong domain.[1][3]
- Add coverage positives: ROADWork (ICCV 2025) + Roboflow roadwork sets (extra camera angles/rare scenes) to reduce misses on uncommon work‑zone layouts.[1]
- Label rule (single rule everywhere): label = 1 if **any** roadwork primitive exists (cones, barriers, arrow boards, work‑zone signs, workers, lane closures/equipment), else 0.[3][1]
- Hard negatives program (critical): explicitly collect “orange but NOT roadwork” (ads, cranes, fences, autumn trees, vests without road context, etc.) so precision doesn’t collapse when you tune for recall.[1]
- Hard splits for evaluation: day/night, rain/snow, highway/city, zoomed sign vs far away, and the orange hard‑negative slice.[1]
- Routing dataset (for Stage‑1.5 gate): keep a dataset where each image has label “Stage‑1 correct vs wrong” so the gate learns *failure prediction* not just confidence.[1]

### Stage 1 (best visual core: DINOv3 multi‑view + MIL pooling)
This is your “always-on workhorse” and should be engineered to not miss tiny evidence.[2][1]

**Views (always run together, one batched forward):**
- 1 global view (full image resized).[1]
- 3×3 tiles with 10–15% overlap (9 tiles) so border evidence isn’t cut.[1]
- Optional: flip TTA (global + tiles) if latency allows; keep it as a switch.[2][1]

**Batching (pro speed trick):**
- Build one tensor batch (N=10 without flips, N=20 with flips) and run **one** DINOv3 forward pass, then head on the batch, then aggregate.[1]

**Aggregation (don’t miss + don’t overfire):**
- Default: **top‑K mean** on per‑crop roadwork probabilities \(p_k\), with K=2 (or K=3 if you do more tiles).[1]
- Keep **max pooling** (K=1) only as a “high recall mode” toggle, because it can over-trigger from one bad tile.[1]

**Training (Stage‑1 model):**
- Start head‑only on frozen DINOv3 backbone (fast, stable).[3]
- Use your validator-aligned preprocessing/augs: 224×224, ImageNet mean/std, RGB, flip/rotation/color jitter (as in your spec).[2]
- Calibrate thresholds on NATIX val (don’t guess). Your plan explicitly uses Stage‑1 early exit thresholds like 0.88 / 0.12 and expects ~60% exit rate at Stage‑1.[3][2]

### Stage 1.5 (the 2025 “latest research” upgrade: learned failure gate)
This is the part that makes “don’t miss” achievable without destroying precision: you don’t just lower thresholds; you **predict when Stage‑1 is likely wrong** and defer those cases.[1]

**Gate features (no extra backbone compute):**
- Aggregated p (top‑K mean), max \(p_k\), margin \(|p-0.5|\). [1]
- Disagreement: variance/entropy of \(p_k\), and max–mean gap.[1]
- Stability: if you use flip‑TTA, measure flip disagreement / “does the top tile stay top.”[1]

**Gate training (pro way):**
- Label y_fail = 1 if Stage‑1 prediction was wrong else 0, on a held‑out set.[1]
- Train a small logistic/MLP gate model to predict y_fail from the features above.[1]
- Choose operating point: “defer at most X%” while maximizing final MCC/accuracy.[1]

**Decision policy (don’t miss mode):**
- Only accept a negative when (p is confidently low) **and** failure risk is low; if p is low but risk is high, treat as “unsafe negative” and defer (this is how you avoid missed positives without just spamming positives).[1]

### Continuous improvement loop (the part you said not to miss)
This is explicitly in your plan as the path from ~96–97% to ~98%+.[2][1]

- Daily/weekly **FiftyOne** hard-case mining: hardness threshold 0.7, ~200 hard cases/day in the spec, then retrain.[2]
- RLVR from validator accept/reject feedback in Month 1–2 (simple reward +1/-1).[2]
- SRT Month 3–6, then M‑GRPO Month 7 for stable long-run self-learning (not Stage‑1 today, but part of “best overall plan”).[3][2]

## Model roadmap: why “16plus then 7B later” is valid
This is the clean strategy:

- **Now:** use `facebook/dinov3-vith16plus-pretrain-lvd1689m` because it’s the strongest “practical” backbone and fits the 24GB-style budget assumptions in your plan (Stage‑1 quantized footprint ~3GB class).[3][2]
- **Later switch to 7B** only if you can run it without breaking latency/throughput (usually means 80GB GPU or real sharding effort); otherwise you gain more by Stage‑0 data + Stage‑1 multi-view + Stage‑1.5 gate + FiftyOne loop.[3][2][1]

## The one thing to clarify (so it’s truly “best”)
What is your hard constraint: maximize MCC (balanced), or **maximize recall (“never miss”) even if false positives increase**?  
This changes whether max-pooling and more aggressive deferral are enabled by default.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/31d45e25-e0b1-4aea-9cdb-cc167f785871/paste.txt)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4e7a2631-748a-4726-baa5-a807bdbfce46/cursor___validation_set.md)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8c0940da-8eaf-450d-b4b5-889e1d3ca6d4/REALISTIC_DEPLOYMENT_PLAN.md)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/16791cf0-df3c-4616-b047-b9c2626900fd/paste-2.txt)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/72aaaa02-4dde-40d9-8d68-310cf461d2b5/paste.txt)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e0fcf4b-1903-4879-abb2-d3cd3c910feb/test_cascade_small.py)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0e2341cf-a5d2-48d6-82b7-a71d8315f151/validate_thresholds.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0c39fe2-3f65-414c-9b4b-fd7e1a8d129d/train_stage1_head.py)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/66dd31e1-ac1a-419b-baf5-03e0faf30e5c/paste.txt)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3f90fbb0-a6e8-4c56-9fca-727659aa7915/train_stage1_head.py)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6df65c6c-962f-4d61-93ff-f6ad9626ea1e/prepare_roadwork_data.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11d5af2a-9d64-4044-a4df-b126d79697bd/paste.txt)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bffa5889-2ce7-4760-a0c2-a262f9547099/paste-2.txt)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c25ed0cc-feb4-4a98-a859-a1e139f7ac43/paste-3.txt)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2dca406a-3a8c-408a-bd94-2e191e6f2980/test_cascade_small.py)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7421c108-66d2-43ba-b841-b7aa253b976f/validate_thresholds.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c8129bf-4cd1-4408-9185-093e403fced5/paste.txt)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/892a645b-4905-4870-9031-df47e944721d/train_stage1_head.py)
[19](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d50389e2-1fee-4e73-939a-0e4425e0488c/train_stage1_head.py)
[20](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e451d23d-a93a-4d4e-8ec0-05c14df73879/paste.txt)
[21](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9ee7de1-c50b-441c-90fc-4aafb03eec05/StreetVision_Subnet72_Specs_Dec2025.md)
[22](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/41e1d04f-3bbc-4cdf-9801-7012540d1549/paste-2.txt)
[23](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/eb5bd793-e5c6-4d47-92d1-ba185a8c06ff/train_stage1_head.py)
[24](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2ebeecc6-665c-4845-a30b-4b1d013fa992/fd11.md)
[25](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/259105ed-c070-437f-bb06-00dbcec9abc3/fd13.md)
[26](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d19af4d8-d447-4e3b-9213-74c10b586437/fd12.md)
[27](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c2445b7f-885f-4026-9ad0-da99b026bbba/fd13.md)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4adb02b1-93a4-4141-98ee-582196826ba8/fd12.md)
[29](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/169f36ed-f131-4e25-a634-f75ada9cf967/fd5.md)
[30](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7c342e5a-8b7a-460b-9bdb-f7a35fa92be1/fd9.md)
[31](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/93da31e3-e157-4696-b7a8-4dc514ebddfa/fd8.md)
[32](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d40d046f-5e78-4662-856a-f7ac3d61bdc4/fd10.md)
[33](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b4268e8f-3c29-4d50-9db8-14c8c604104a/fd11.md)
[34](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d9f7d4fa-fee9-4979-9bac-d90428dc2cb5/fd12.md)
[35](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0edc93af-0743-48d6-a40e-e4aa4ef85eb7/fd6.md)
[36](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/68d29610-ed26-46a5-9cff-e5e0e6e9ccf0/fd7.md)
[37](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/727e9de5-71be-437a-b7a3-1423e7cf37bd/fd4.md)
[38](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb58379f-a0d1-4e7c-bcf9-0e0b7ded544e/ff7.md)
[39](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/af2c24e0-83d6-4b13-9e69-52e37b48040b/fd8.md)
[40](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9942b5ad-e2e6-4171-b0a7-9dfc2571d3e3/ff6.md)
[41](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b2aae559-b153-4c9e-af9c-9e04883a99f0/fd5.md)
[42](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bc2d02b-54b5-42cd-b73f-3bb365f4bfc8/fd3.md)
[43](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6b92bd8-8428-4b64-b12b-afee8190fc80/fd7.md)
[44](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/535bdd8c-0670-41ba-b6ae-347a93be63cb/fd6.md)
[45](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fb7f02ce-9015-451b-96f8-cfeb46f20fba/fd10.md)
[46](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/796e2433-dc5a-4639-bf49-250b24d4e9eb/fd11.md)
[47](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e2d7e9f-fe3c-467f-b564-0a295760c15f/fd1.md)
[48](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3da415fa-d5f9-4810-8670-d81ad890aac6/fd2.md)
[49](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9542a24b-81e2-4819-80e0-6d9df3992c7a/ff5.md)
[50](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e06ad84a-b00a-48c2-82f0-48a13b972fea/paste.txt)
[51](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/33eab516-c1dd-4514-9560-e033cfd6dee8/fd4.md)
[52](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d954033b-23c8-4b74-b676-7d3eaf8ab5bb/fd9.md)
[53](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4bd355b9-b0ee-4744-827f-0622e4987e1b/fd17.md)
[54](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2b5c37e2-3329-4943-8281-868fd978d14f/paste.txt)
[55](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a0a4cd06-1223-4f6e-8a2d-73b914526684/paste.txt)
[56](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/86a6e1b3-f391-43ad-a77a-750aab3de268/fd13.md)
[57](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/165ed64b-1bf8-4e43-9858-6bfccae5788c/ff15.md)
[58](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/4d42f4aa-868c-4473-b955-8186c30f6eda/fd16.md)
[59](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c397164a-4c43-4fa5-8547-2c8e5a6116a6/fd14.md)
[60](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a3803457-ec59-4af1-82aa-99f6f11ef5e5/fd5.md)
[61](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ec8497b-c521-4bb4-ad10-7e41cebf85b8/fd9.md)
[62](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ad512e5e-8ef4-49bb-b949-bcffd4f04e09/fd6.md)
[63](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/20b0f114-2e41-4b87-91e1-0365c3661048/fd7.md)
[64](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5707234c-2d4b-4d46-b13d-c83b9ca67c71/fd12.md)
[65](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5af5b79-5acd-46da-9477-044ae7593873/fd11.md)
[66](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6daa8f3e-8efa-4fda-adc7-715ab0997c46/most.md)
[67](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/077edf5f-ca72-45f8-9baf-74adbaf15f40/fd17.md)
[68](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/05cde2f7-5f62-47c5-ac5b-8a181d079200/fd15.md)
[69](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/14e3bad6-8858-494c-b53b-f24610e6769b/fd10.md)
[70](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9abc1b5d-0a33-44ed-9a3d-8bb9045b2e58/fd8.md)
[71](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c9bde5de-cf73-4fb9-91f7-79296a3d52c7/fd14.md)
[72](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74c7083a-a2ff-4937-b1ba-708c50e87dd6/fd12.md)
[73](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/27b6ab80-c2e2-4417-8570-755c3c4b3bdb/fd16.md)
[74](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/56b22c4b-5117-48f3-bfc7-1b01dc6507c1/fd17.md)
[75](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/888e1d39-576a-4335-a961-ec9bc8365858/REALISTIC_DEPLOYMENT_PLAN.md)
[76](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/615a10ac-5d5d-41fc-8c9b-b5c164fd4fdc/train_stage1_head.py)
[77](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e3b7c3ec-a19d-482e-9681-4cff56f4b85a/download_models.py)
[78](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/65bffaaf-fe2e-4a5d-86a5-ad8715781012/monitor_download_progress.py)
[79](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e74a884b-2778-4cae-9b7d-61e92af71da4/README.md)
[80](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3bd3902b-f1a2-4cf6-aa9a-1a313679e047/val_labels.csv)
[81](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/df731982-e2ca-41fe-a649-078058880962/train_labels.csv)
[82](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52ab3bc0-d9e5-4e52-bdbf-1b1e42d5326b/LastPlan.md)
[83](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a5fc5dea-ab60-4df2-8aac-0510eea030b5/paste.txt)
[84](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bd6116c7-b53e-4fdb-976e-5dbef1866f3a/COMPLETE_DEPLOYMENT_PLAN.md)
[85](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99abc324-5a32-4a18-a32e-09d1d020bbc1/COMPLETE_DEPLOYMENT_PLAN_PART2.md)
[86](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccb7fde1-51ec-4845-986e-e398647ac107/REALISTIC_DEPLOYMENT_PLAN.md)
[87](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/06085126-39e9-41d9-ae46-74f7e06adc0e/train_stage1_head.py)
[88](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27e267e-b389-4d71-adac-ab9ca98f48b7/REALISTIC_DEPLOYMENT_PLAN.md)
[89](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/74613b37-7560-42d9-91ef-cdadb2503e9b/lala.md)
[90](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a9c02fdd-a34a-4078-a7fd-76ba67042a28/LastPlan.md)
[91](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/502791a0-448c-4dd5-959c-79eecddfb2db/paste.txt)
[92](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/08e5c28b-29a9-48a1-8b04-ef31bbea0dc0/LastPlan.md)
[93](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/11e9a02c-1b86-458c-85da-2534463b9511/lala.md)
[94](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b9cee006-6993-4993-b4f8-a18330f37b07/most6.md)
[95](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d0905bcd-54bd-40c1-882c-fa250f60b0d4/REALISTIC_DEPLOYMENT_PLAN.md)
[96](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ed11eb15-fa01-4511-82f3-2ffb1d4fb3d0/LastPlan.md)
[97](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6be094f-6b35-4fe1-8a4a-59b86175232b/LastPlan.md)
[98](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2bb2431d-131a-487c-ab70-76296133aaf5/most6.md)
[99](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cccd343e-d991-4907-9d3e-02493717db85/REALISTIC_DEPLOYMENT_PLAN.md)
[100](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/7ba8b0e9-ff2e-430d-8e3b-33f5d7e7db59/LastPlan.md)
[101](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/83662581-fa6a-41b3-b5cb-d52ac6b63939/lala.md)
[102](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3e984206-9d35-4cfc-95ae-87a64e74b8aa/REALISTIC_DEPLOYMENT_PLAN.md)
[103](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8861ec37-1012-4220-8436-808b05ebc5f3/LastPlan.md)
[104](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e861c690-f80d-44bc-9bd8-85bf0f2945c6/REALISTIC_DEPLOYMENT_PLAN.md)
[105](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b942563b-66a3-482e-83bf-de26d3b1fae9/fd15.md)
[106](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ccbb7313-3667-4301-92aa-26bc8033753a/fd13.md)
[107](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a79a72f8-90a4-47d9-9baf-64eb7dfb5329/REALISTIC_DEPLOYMENT_PLAN.md)
[108](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e64ecd9a-d3bc-4ee2-9020-2d0285ba0070/REALISTIC_DEPLOYMENT_PLAN.md)
[109](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/dd7be299-e9ac-4a74-b402-4accabf01d71/REALISTIC_MISSING_DETAILED.md)
[110](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1bb890f-f383-46a0-bcea-d08ade400e36/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[111](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/530ead05-50ec-419d-9e44-a2acb6fccf28/FINAL_BEST_CASE_SAM3_MOLMO2.md)
[112](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/2e31b14a-9714-499c-bcbf-7577041e139c/REALISTIC_DEPLOYMENT_PLAN.md)
[113](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d6f2269d-642c-4d79-b48d-8c45e8e7e47b/paste.txt)
[114](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/12ce7ec1-c6f5-40b3-b466-a1d6343e9050/paste-2.txt)
[115](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c10dbc68-2a42-4e5f-ba83-75b98790a15f/paste.txt)
[116](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c76c78fb-6f56-41ce-9b68-a7732f343e8e/REALISTIC_DEPLOYMENT_PLAN.md)
[117](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a2341fb6-da82-4dae-abd1-38b95d7d238e/train_stage1_v2.py)
[118](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/d23cbb26-f086-4a30-b6a0-e1ca2feef8a4/paste.txt)
[119](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/c77fb5ba-5d68-4d17-955e-0bbdae84f4cb/paste-2.txt)
[120](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/9609810c-e420-4d63-9e55-6412239d72c6/paste.txt)
[121](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/a639e4bf-993a-4691-b30b-49b628b6da27/paste.txt)
[122](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/e1a67142-c2ef-4e6c-b577-2ae8d6eecd32/sweep_hparams_fast.py)
[123](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f5530f8c-4e76-441d-a0bb-ef7572342d0c/paste-2.txt)
[124](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/99820de4-c9f4-4c69-8f58-0b40e04a0f5a/paste-3.txt)
[125](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/fca10b3b-2d8a-4318-9547-a2f905bf1b57/paste-2.txt)
[126](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f9694d6b-4a73-4d74-afe0-453877cb065d/ultimate_train_stage1.py)
[127](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/6d871567-894e-4962-b5f0-3d5d71b679b0/paste.txt)
[128](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b5ed6fbe-4e3f-4c6e-b346-f5ba80e8dbd8/paste-2.txt)
[129](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5d3209c6-7fd0-4c7e-b97d-303dea61521e/REALISTIC_DEPLOYMENT_PLAN.md)
[130](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5400c584-8483-48f4-994d-0852d28579ff/COMPLETE_DEPLOYMENT_PLAN.md)
[131](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/17f64a3a-6fd4-442c-b36f-ce76bb36192d/REALISTIC_DEPLOYMENT_PLAN.md)
[132](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f6c31698-7767-4308-aa88-c9d73e7054d7/train_stage1_v2.py)
[133](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/b27aa452-d8bc-41a4-bdf7-1bfe7cab27bf/paste.txt)
[134](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1cac2b62-cdff-4a07-a7a7-c5337726e9bf/REALISTIC_DEPLOYMENT_PLAN.md)
[135](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/cb773131-0229-4fef-811b-478cf5cc2d18/REALISTIC_DEPLOYMENT_PLAN.md)
[136](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/520b2ad2-1ec9-479b-b5d6-9a95013fc604/REALISTIC_DEPLOYMENT_PLAN.md)
[137](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/021c346a-6fe1-4aa1-b927-ac8483c4e9df/loaders.py)
[138](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/ee16be6f-7a07-40a4-a67f-7e1c4867973c/checkpointing.py)
[139](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f2df5108-8a0a-431f-8273-83b0759d479b/50_export_bundle.py)
[140](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/aedbf58d-9aa2-4d78-86ae-e671e07b85fe/trainer.py)
[141](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/f093dc77-7779-4385-8f18-49b55f878f95/loaders.py)
[142](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/26b91c4d-1e98-42bd-8eda-55ee4b20a3db/checkpointing.py)
[143](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0cca0dda-60d2-4301-9b8f-8e7156503a36/25_threshold_sweep.py)
[144](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/92063955-8147-4cdd-ab5f-fe47e7d8181f/paste.txt)
[145](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/bcc8d1bf-b84e-4bdc-8ebe-31cb8dc938c5/selective.py)
[146](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/8896eae1-4770-413f-a1bc-7e5b711a8185/gate_head.py)
[147](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/995476fb-a163-4910-b7b1-90b3fb501081/calibrate_gate.py)
[148](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/1ff9eddc-65b6-4012-8412-b785a7b22f93/33_calibrate_gate.py)
[149](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/5bd91e09-9277-4254-ada6-f4176fc6ddf6/paste.txt)
[150](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/52f2ac69-6509-4afa-9386-7a851cdcd456/paste.txt)
[151](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/3c2f541f-0375-4351-b1d1-46888972a4ae/cursor_natix_dataset_location.md)
[152](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53804713/0ea0b2ff-fd91-45fc-8246-481a8b9700f4/paste.txt)