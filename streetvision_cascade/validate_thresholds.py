import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torch.utils.data import DataLoader
from train_stage1_head import NATIXDataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
import numpy as np

def validate_thresholds():
    """Test different confidence thresholds to find optimal exit rate"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load trained model
    print("Loading trained DINOv3 + classifier head...")
    model_path = "models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    hidden_size = backbone.config.hidden_size
    classifier_head = nn.Sequential(
        nn.Linear(hidden_size, 768),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(768, 2)
    ).to(device)

    # Load trained weights
    classifier_head.load_state_dict(
        torch.load(f"{model_path}/classifier_head.pth", map_location=device)
    )

    backbone.eval()
    classifier_head.eval()

    # Load validation set
    val_dataset = NATIXDataset(
        image_dir="data/natix_official/val",
        labels_file="data/natix_official/val_labels.csv",
        processor=processor
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Collect predictions
    print("Running inference on validation set...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = backbone(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]
            logits = classifier_head(features)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Test different thresholds
    print("\n" + "="*80)
    print("THRESHOLD VALIDATION - Target: ~60% exit rate, ≥99% accuracy on exits")
    print("="*80)

    thresholds = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]

    for thresh in thresholds:
        # High confidence mask (exit at Stage 1)
        # Exit if p(roadwork) >= thresh OR p(roadwork) <= (1-thresh)
        high_conf_mask = (all_probs[:, 1] >= thresh) | (all_probs[:, 1] <= (1-thresh))

        # Metrics on exited samples
        exited_probs = all_probs[high_conf_mask]
        exited_labels = all_labels[high_conf_mask]
        exited_preds = (exited_probs[:, 1] > 0.5).astype(int)

        if len(exited_labels) == 0:
            print(f"\nThreshold {thresh:.2f}: No samples exit (threshold too high)")
            continue

        accuracy = accuracy_score(exited_labels, exited_preds)
        f1 = f1_score(exited_labels, exited_preds)
        mcc = matthews_corrcoef(exited_labels, exited_preds)
        coverage = high_conf_mask.mean()

        print(f"\nThreshold {thresh:.2f}:")
        print(f"  Exit Rate (Coverage): {coverage*100:.1f}%")
        print(f"  Accuracy on Exits:    {accuracy*100:.2f}%")
        print(f"  F1 Score:             {f1*100:.2f}%")
        print(f"  MCC:                  {mcc:.4f}")

        # Check if meets target from REALISTIC_DEPLOYMENT_PLAN
        # Target: ~60% exit rate, ≥99.2% accuracy
        if 55 <= coverage*100 <= 65 and accuracy*100 >= 99:
            print(f"  ✅ MEETS TARGET! Recommended threshold: {thresh:.2f}")
        elif accuracy*100 >= 99:
            print(f"  ⚠️ High accuracy but exit rate is {coverage*100:.1f}% (target: 60%)")
        elif 55 <= coverage*100 <= 65:
            print(f"  ⚠️ Good exit rate but accuracy is {accuracy*100:.2f}% (target: ≥99%)")

    print("\n" + "="*80)

if __name__ == "__main__":
    validate_thresholds()
