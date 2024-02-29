import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, ignore_index=1024, reduction='mean'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        target = target.unsqueeze(-1)
        nll_loss = -log_probs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Apply label smoothing
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            pad_mask = pad_mask.squeeze(-1)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
            count = (~pad_mask).sum()
        else:
            count = target.numel()

        loss = (1.0 - self.alpha) * nll_loss + self.alpha * smooth_loss

        if self.reduction == 'mean':
            loss = loss.sum() / count
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
