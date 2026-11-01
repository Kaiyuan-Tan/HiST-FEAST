import torch
import torch.nn as nn
import timm


class UNI(nn.Module):
    def __init__(self, weights_path, frozen=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(weights, strict=True)
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, patches):
        if patches.shape[-1] == 3:
            patches = patches.permute(0, 3, 1, 2)
        patches = (patches.float() / 255.0 - self.mean) / self.std
        if self.frozen:
            with torch.no_grad():
                return self.backbone(patches)
        else:
            return self.backbone(patches)