import torch
import os

os.environ["LINE_PROFILE"] = "0"
from line_profiler import profile
profile.disable()

class CustomAdam:
    def __init__(self, size, device, alpha = 1e-2, beta1 = 0.9, beta2 = 0.999) -> None:
        self.device = device
        self.m = torch.zeros(size, requires_grad=False)
        self.v = torch.zeros(size, requires_grad=False)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0 # step num
        self.eps = 1e-8
        self.set_device(device)
    
    def set_device(self, device):
        self.device = device
        self.m = self.m.to(device)
        self.v = self.m.to(device)
    
    @profile
    def get_grads(self, grads):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads # calculate first moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2) # calculate second moment
        mhat = self.m / (1 - (self.beta1 ** (self.t + 1)))
        vhat: torch.Tensor = self.v / (1 - (self.beta2 ** (self.t + 1)))
        self.t += 1
        return self.alpha * mhat / (vhat.sqrt() + self.eps)