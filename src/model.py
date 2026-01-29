"""Model definition for FastChessNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparams as hp

class ResBlock(nn.Module):
    """Stacked residual block."""
    def __init__(self):
        # Two convolutional layers with a skip connection
        super().__init__()
        self.conv1 = nn.Conv2d(hp.N_CHANNELS, hp.N_CHANNELS, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hp.N_CHANNELS)
        self.conv2 = nn.Conv2d(hp.N_CHANNELS, hp.N_CHANNELS, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hp.N_CHANNELS)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class FastChessNet(nn.Module):
    """CNN for evaluating chess positions and suggesting moves."""
    def __init__(self):
        super().__init__()

        # Initial convolution
        self.conv0 = nn.Conv2d(18, hp.N_CHANNELS, 3, padding=1, bias=False)
        self.bn0   = nn.BatchNorm2d(hp.N_CHANNELS)

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock() for _ in range(hp.N_RES_BLOCKS)])

        # Policy head
        self.policy_conv = nn.Conv2d(hp.N_CHANNELS, 32, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, hp.POLICY_DIM)

        # Value head
        self.value_conv = nn.Conv2d(hp.N_CHANNELS, 3, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(3)
        self.value_fc1  = nn.Linear(hp.V_FC1_DIM, hp.V_FC2_DIM)
        self.value_fc2  = nn.Linear(hp.V_FC2_DIM, 1)

    def forward(self, x, legal_mask):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.view(p.size(0), -1))
        p = p.masked_fill(~legal_mask, -1e9)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.value_fc2(v))

        return p, v
