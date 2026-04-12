import torch
import torch.nn as nn
from torchvision import transforms

from cavr.models.policy import MLPPolicy


class R3MBaseline(nn.Module):
    """R3M (ResNet-50) baseline for comparison.

    Uses the pre-trained R3M visual encoder (frozen) with the same MLP
    policy head architecture as CAVR for fair comparison.
    """

    EMBED_DIM = 2048

    def __init__(self, pol_cfg, device="cpu"):
        super().__init__()
        self._device = device
        self._encoder = None
        self.policy = MLPPolicy(
            visual_dim=self.EMBED_DIM,
            proprio_dim=pol_cfg["proprio_dim"],
            hidden_dim=pol_cfg["hidden_dim"],
            action_dim=pol_cfg["action_dim"],
            num_layers=pol_cfg["num_layers"],
            dropout=pol_cfg.get("dropout", 0.0),
        )
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _lazy_load_encoder(self):
        if self._encoder is not None:
            return
        from r3m import load_r3m
        self._encoder = load_r3m("resnet50")
        self._encoder.eval()
        self._encoder.to(self._device)
        for p in self._encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images, task_description=None):
        self._lazy_load_encoder()
        x = images.float() / 255.0
        x = self.preprocess(x)
        return self._encoder(x * 255.0)

    def forward(self, images, proprio, task_description=None):
        visual_emb = self.encode(images, task_description)
        return self.policy(visual_emb, proprio)


class VC1Baseline(nn.Module):
    """VC-1 (ViT-L) baseline for comparison.

    Uses the pre-trained VC-1 visual encoder (frozen) with the same MLP
    policy head.
    """

    EMBED_DIM = 1024

    def __init__(self, pol_cfg, device="cpu"):
        super().__init__()
        self._device = device
        self._encoder = None
        self._transform = None
        self.policy = MLPPolicy(
            visual_dim=self.EMBED_DIM,
            proprio_dim=pol_cfg["proprio_dim"],
            hidden_dim=pol_cfg["hidden_dim"],
            action_dim=pol_cfg["action_dim"],
            num_layers=pol_cfg["num_layers"],
            dropout=pol_cfg.get("dropout", 0.0),
        )

    def _lazy_load_encoder(self):
        if self._encoder is not None:
            return
        try:
            from vc_models.models.vit import model_utils
            self._encoder, _, self._transform, _ = model_utils.load_model(
                model_utils.VC1_LARGE_NAME
            )
        except ImportError:
            import timm
            self._encoder = timm.create_model("vit_large_patch16_224", pretrained=True)
            self._encoder.head = nn.Identity()
            self._transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        self._encoder.eval()
        self._encoder.to(self._device)
        for p in self._encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images, task_description=None):
        self._lazy_load_encoder()
        x = images.float() / 255.0
        x = self._transform(x)
        return self._encoder(x)

    def forward(self, images, proprio, task_description=None):
        visual_emb = self.encode(images, task_description)
        return self.policy(visual_emb, proprio)
