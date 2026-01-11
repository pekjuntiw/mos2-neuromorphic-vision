import math
import torch
import torch.nn as nn
from typing import Callable
from spikingjelly.clock_driven import neuron, base
from spikingjelly.clock_driven.model import sew_resnet
from surrogate import Arctan


class IdealLIF(neuron.BaseNode):
    """
    dV/dt = -V/tau + R·(Σw·spike_in)/tau
    V(t) = exp(-Δt/tau)·V(t-1) + (1-exp(-Δt/tau))·R·Σ(w·spike_in)
    """
    def __init__(self, tau, v_threshold, v_reset, dt,
                 surrogate_function: Callable = Arctan(mag=math.pi),
                 detach_reset: bool = True):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.dt = dt
        self.factor = torch.exp(torch.tensor(-dt / tau))

        self.v = 0.
        self.register_memory('spike', 0)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.factor * self.v + (1 - self.factor) * x

    def neuronal_fire(self):
        spike = self.surrogate_function((self.v - self.v_threshold) / self.v_threshold)
        self.spike = spike
        return spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = (1 - spike_d) * self.v + spike_d * self.v_reset

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x.shape, device=x.device)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike


class LPFilter(base.MemoryModule):
    """
        Low-pass filter, think of it as a leaky spike counter
        Modeled using an RC circuit
    """
    def __init__(self, tau: float = 20e-3, dt: float = 1e-3):
        super(LPFilter, self).__init__()
        self.tau = tau
        self.dt = dt
        self.factor = torch.exp(torch.tensor(-dt / tau))
        self.register_memory("out", 0)

    def forward(self, x: torch.Tensor):
        self.out = self.factor * self.out + (1 - self.factor) * x

        return self.out


class RateNet(nn.Module):
    def __init__(self, num_in, num_out, tau, dt, device,):
        super(RateNet, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.tau = tau
        self.dt = dt
        self.device = device

        self.fc1 = nn.Linear(num_in, num_out, bias=True)
        self.out = IdealLIF(
            tau=tau, v_threshold=1., v_reset=0., dt=dt, surrogate_function=Arctan(mag=math.pi), detach_reset=True
        )

        fc1_bound = math.sqrt(3. / num_in) * 100
        nn.init.uniform_(self.fc1.weight, -fc1_bound, fc1_bound)

        self.fc1.weight.register_hook(lambda grad: print(f'FC1 Grad\n{grad}\nFC1 Weight\n{self.fc1.weight.data}') if torch.isnan(grad).any() else None)

        self.v_1 = None
        self.spike_1 = None
        self.neuron_in_1 = None
        self.p_1 = None
        self.spike_for_reg_1 = None
        self.v_for_reg_1 = None
        self.x_for_reg_1 = None
        self.p_for_stat_1 = None

    def forward(self, x: torch.Tensor):
        # x.shape = [batch, times, features]
        self.spike_for_reg_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.v_for_reg_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.x_for_reg_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)
        self.p_for_stat_1 = torch.empty([x.shape[0], x.shape[1], self.num_out], device=self.device)

        if x.shape[0] == 1:
            self.v_1 = torch.empty([x.shape[1], self.num_out])
            self.p_1 = torch.empty([x.shape[1], self.num_out])
            self.spike_1 = torch.empty([x.shape[1], self.num_out])
            self.neuron_in_1 = torch.empty([x.shape[1], self.num_out])

        y_seq = []

        x = x.permute(1, 0, 2)  # [times, batch, features]

        for t in range(x.shape[0]):
            y = self.fc1(x[t])
            neuron_in_1 = y
            y = self.out(y)
            self.p_for_stat_1[:, t] = self.out.v * neuron_in_1
            self.spike_for_reg_1[:, t] = y

            y_seq.append(y.unsqueeze(0))

            if x.shape[1] == 1:
                self.v_1[t] = self.out.v
                self.spike_1[t] = self.out.spike
                self.neuron_in_1[t] = neuron_in_1
                self.p_1[t] = self.out.v * neuron_in_1

        return torch.cat(y_seq, 0)


class SEWResnet18(nn.Module):
    def __init__(self, num_classes=5, tau=4, dt=5e-2):
        super().__init__()
        self.resnet = sew_resnet.sew_resnet18(
            pretrained=False, progress=True,
            cnf='ADD',
            single_step_neuron=neuron.IFNode,
            v_threshold=1., surrogate_function=Arctan(mag=math.pi), detach_reset=True
        )

        # replace the first convolutional layer to accept the desired input size
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        self.lpfilter = LPFilter(tau=tau, dt=dt)
        self.dropout = nn.Dropout(p=0.5)

        # replace the final fully connected layer to classify 5 categories
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        nn.init.normal_(self.resnet.fc.weight, 0.03, 1.)
        self.num_classes = num_classes
        
        self.l4_out = None
        self.fc_in = None

    def forward(self, x):
        
        if x.shape[0] == 1:
            self.l4_out = torch.empty([x.shape[1], 20])
            self.fc_in = torch.empty([x.shape[1], 512])
    
        x = x.permute(1, 0, 2, 3, 4)  # [times, batch, 1, width, height]
        y_seq = torch.zeros([x.shape[0], x.shape[1], self.num_classes], device=x.device)

        for t in (range(x.shape[0])):
            y = self.resnet.conv1(x[t])
            y = self.resnet.bn1(y)
            y = self.resnet.sn1(y)
            y = self.resnet.maxpool(y)
            y = self.resnet.layer1(y)
            y = self.resnet.layer2(y)
            y = self.resnet.layer3(y)
            y = self.resnet.layer4(y)
            if x.shape[1] == 1:
                self.l4_out[t] = torch.flatten(y)[:20]
            y = self.resnet.avgpool(y)
            y = torch.flatten(y, 1)
            y = self.lpfilter(y)
            if x.shape[1] == 1:
                self.fc_in[t] = y
            y = self.dropout(y)
            y = self.resnet.fc(y)
            y_seq[t] = y

        return y_seq
