import torch.nn as nn
from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self, input_dim=256, genes=100, side=32):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, side * side),
            nn.Unflatten(1, (1, side, side)),
        )

        self.backbone = resnet18(
            weights=None,
            norm_layer=lambda channels: nn.GroupNorm(8, channels),
        )

        self.backbone.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, genes)
        self.activation = nn.ReLU()

    def forward(self, x):
        # [N,256] → [N,1,32,32]
        x = self.adapter(x)

        # [N,1,32,32] → [N,G]
        x = self.backbone(x)
        x = self.activation(x)

        return x