from pathlib import Path
import torch.nn as nn
from .gcn import GCN
from .uni import UNI
from .resnet import ResNet

class MyModel(nn.Module):
    def __init__(self, feature_dim=1024, genes=100):
        super().__init__()
        weights_path = Path(__file__).resolve().parents[1] / "ckpts" / "pytorch_model.bin"
        self.uni = UNI(weights_path=weights_path)
        self.gcn = GCN(nfeat=feature_dim, nhid=256, nclass=256, dropout=False)
        self.resnet = ResNet(input_dim=256, genes=genes)

    def forward(self, x, A_norm):
        print("Input")
        x = self.uni(x)
        print("UNI")
        x = self.gcn(x, A_norm)
        print("GCN")
        x = self.resnet(x)
        print("RESNET")
        return x
