import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder that produces dense spatial feature maps."""

    BACKBONES = {
        "dinov2_vitb14": {"hub": "dinov2_vitb14", "dim": 768, "patch": 14},
        "dinov2_vitl14": {"hub": "dinov2_vitl14", "dim": 1024, "patch": 14},
    }

    def __init__(self, backbone="dinov2_vitl14", freeze=True):
        super().__init__()
        spec = self.BACKBONES[backbone]
        self.feat_dim = spec["dim"]
        self.patch_size = spec["patch"]

        self.model = torch.hub.load("facebookresearch/dinov2", spec["hub"])

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

        self.freeze = freeze
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, images):
        """Normalize images from [0, 255] uint8 to ImageNet-normalized floats."""
        x = images.float() / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) tensor, pixel values in [0, 255].
                    H, W should be divisible by patch_size (14).
        Returns:
            features: (B, h, w, D) dense feature map where h=H/14, w=W/14.
        """
        if self.freeze:
            self.model.eval()

        x = self.preprocess(images)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        patch_tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        return patch_tokens.reshape(B, h, w, self.feat_dim)
