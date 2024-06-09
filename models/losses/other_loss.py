import torch
import torch.nn as nn

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, output, target):
        abs_error = torch.abs(output - target)
        C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        loss = torch.where(
            abs_error < self.omega,
            self.omega * torch.log(1 + abs_error / self.epsilon),
            abs_error - C
        )
        return loss.mean()


class JointsHuberLoss(nn.Module):
    def __init__(self, use_target_weight=False, delta=1.0):
        super(JointsHuberLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.delta = delta

    def forward(self, output, target, target_weight=None):
        error = output - target
        if self.use_target_weight and target_weight is not None:
            error = error * target_weight
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()
