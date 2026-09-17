# Portions of this file are ported from SG-SCL (https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), which does not include a license file.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
"""
Gradient Reversal Layer — exact port of the official code of the SG-SCL/DAT family of papers.

Source: kaen2891/stethoscope-guided_supervised_contrastive_learning models/functions.py
(ICASSP 2024, https://arxiv.org/abs/2312.09603)
"""
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
