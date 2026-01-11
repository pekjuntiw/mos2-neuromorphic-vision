import math
import torch
from spikingjelly.clock_driven import neuron


class arctan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mag):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.mag = mag
        return neuron.surrogate.heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0].abs()
            grad_x = grad_output * (1. / ctx.mag / (1. + (math.pi * ctx.saved_tensors[0]).pow_(2)))
        return grad_x, None, None


class Arctan(neuron.surrogate.MultiArgsSurrogateFunctionBase):
    def __init__(self, mag=math.pi, spiking=True):
        super().__init__(spiking)
        self.mag = mag
        self.spiking = spiking
        if spiking:
            self.f = self.spiking_function
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.mag)

    @staticmethod
    def spiking_function(x, mag):
        return arctan.apply(x, mag)

    @staticmethod
    def primitive_function(x: torch.Tensor, mag):
        return (math.pi * x).atan_() / mag / math.pi + 0.5
