"""
Gradient Reversal Layer — SG-SCL/DAT 계열 논문의 공식 코드 충실 이식.

출처: kaen2891/stethoscope-guided_supervised_contrastive_learning models/functions.py
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
