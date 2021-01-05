import torch
import numpy as np

class GradientReverseLayer(torch.autograd.Function):
    iter_num = 0

    @staticmethod
    def forward(ctx, input):
        GradientReverseLayer.iter_num += 1
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        coeff = np.float(
            2.0 * (GradientReverseLayer.high_value - GradientReverseLayer.low_value) /
            (1.0 + np.exp(-GradientReverseLayer.alpha * GradientReverseLayer.iter_num / GradientReverseLayer.max_iter))
            - (GradientReverseLayer.high_value - GradientReverseLayer.low_value) + GradientReverseLayer.low_value
        )
        return - coeff * grad_output