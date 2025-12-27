import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss with label smoothing and class weights.

    Phase 1: Only this loss is used (baseline exact match).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            n_classes = logits.size(-1)
            targets_one_hot = F.one_hot(targets, n_classes).float()
            targets_one_hot = (
                1 - self.label_smoothing
            ) * targets_one_hot + self.label_smoothing / n_classes
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(targets_one_hot * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, targets, weight=self.weight)
        return loss


# Phase 2+ losses (disabled in Phase 1)


class SelectiveLoss(nn.Module):
    """
    Selective loss for Phase 2.
    Combines CE + coverage penalty.
    """

    def __init__(self, lambda_cov: float = 1.0):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.ce_loss = CrossEntropyLoss()

    def forward(self, cls_logits, gate_logits, targets):
        ce = self.ce_loss(cls_logits, targets)

        gate_probs = torch.sigmoid(gate_logits)
        coverage_penalty = (1 - gate_probs.mean()) * self.lambda_cov

        return ce + coverage_penalty


class RiskLoss(nn.Module):
    """
    Risk-aware loss for Phase 2.
    Combines CE + risk penalty.
    """

    def __init__(self, lambda_risk: float = 1.0):
        super().__init__()
        self.lambda_risk = lambda_risk
        self.ce_loss = CrossEntropyLoss()

    def forward(self, cls_logits, gate_logits, targets):
        ce = self.ce_loss(cls_logits, targets)

        cls_probs = F.softmax(cls_logits, dim=-1)
        gate_probs = torch.sigmoid(gate_logits)

        preds = cls_probs.argmax(dim=-1)
        errors = (preds != targets).float()

        risk = (gate_probs * errors).mean()

        return ce + self.lambda_risk * risk


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for Phase 2.
    Combines main CE + auxiliary CE.
    """

    def __init__(self, alpha_aux: float = 0.5):
        super().__init__()
        self.alpha_aux = alpha_aux
        self.ce_loss = CrossEntropyLoss()

    def forward(self, cls_logits, aux_logits, targets):
        ce_main = self.ce_loss(cls_logits, targets)
        ce_aux = self.ce_loss(aux_logits, targets)
        return ce_main + self.alpha_aux * ce_aux
