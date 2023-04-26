import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target, alpha):
        if alpha is None:
            alpha = self.smoothing

        # Log softmax is used for numerical stability
        pred = pred.log_softmax(dim=self.dim)
        if len(pred.shape) < 2:
            pred = pred.unsqueeze(0)

        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            if torch.is_tensor(alpha):
                true_dist[torch.arange(len(pred)), target.argmax(-1)] = torch.ones_like(alpha) - alpha
                alpha = alpha.unsqueeze(-1)
            else:
                if len(target.shape) < 2:
                    target = target.unsqueeze(0)

                true_dist.scatter_(-1, target.argmax(-1, keepdim=True), 1. - alpha)
            true_dist += alpha / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, logits_provided=True, one_hot_encode_trgts=False, num_classes=-1):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        self.dim = dim

        # Greater zero threshold
        self.gz_threshold = 0.1

        self.logits_provided = logits_provided
        self.one_hot_encode_trgts = one_hot_encode_trgts

        self.num_classes = num_classes

    def forward(self, pred, target, alpha):
        if alpha is None:
            alpha = self.alpha

        if self.logits_provided:
            pred = pred.softmax(dim=self.dim)

        # with torch.no_grad():
        # Apply one-hot encoding to targets
        if self.one_hot_encode_trgts:
            target = F.one_hot(target, num_classes=self.num_classes)

        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.unsqueeze(-1)
            # pred_hat = alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            pred_hat = alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - alpha, pred_hat)
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)

        pred = torch.sum(pred * target, dim=-1)

        result = torch.where(torch.gt(pred, 1. - alpha), torch.zeros_like(divergence), divergence)

        return torch.mean(result)
