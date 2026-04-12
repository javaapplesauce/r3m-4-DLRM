import torch
import torch.nn as nn

from cavr.models.encoder import DINOv2Encoder
from cavr.models.concept_mask import ConceptMasker
from cavr.models.policy import MLPPolicy


class CAVR(nn.Module):
    """Concept-Aware Visual Representation pipeline.

    Full forward pass:
        RGB image -> DINOv2 dense features -> SAM concept mask -> filtered features
        -> spatial average pool -> concat with proprioception -> MLP -> 6-DOF action
    """

    def __init__(self, cfg):
        super().__init__()
        enc_cfg = cfg["encoder"]
        mask_cfg = cfg["masking"]
        pol_cfg = cfg["policy"]

        self.encoder = DINOv2Encoder(
            backbone=enc_cfg["backbone"],
            freeze=enc_cfg["freeze"],
        )
        self.masking_enabled = mask_cfg["enabled"]
        if self.masking_enabled:
            self.masker = ConceptMasker(threshold=mask_cfg["threshold"])

        self.policy = MLPPolicy(
            visual_dim=self.encoder.feat_dim,
            proprio_dim=pol_cfg["proprio_dim"],
            hidden_dim=pol_cfg["hidden_dim"],
            action_dim=pol_cfg["action_dim"],
            num_layers=pol_cfg["num_layers"],
            dropout=pol_cfg.get("dropout", 0.0),
        )

    def encode(self, images, task_description=None):
        """Extract pooled visual embedding from images.

        Args:
            images: (B, 3, H, W) in [0, 255].
            task_description: str or None. If provided and masking is enabled,
                concept-aware filtering is applied.
        Returns:
            visual_emb: (B, feat_dim)
        """
        features = self.encoder(images)
        B, h, w, D = features.shape

        if self.masking_enabled and task_description is not None:
            mask = self.masker(images, task_description, h, w)
            features = features * mask

        return features.mean(dim=(1, 2))

    def forward(self, images, proprio, task_description=None):
        """
        Args:
            images: (B, 3, H, W) in [0, 255].
            proprio: (B, proprio_dim).
            task_description: str, e.g. "the red cube".
        Returns:
            action: (B, action_dim) predicted 6-DOF delta action.
        """
        visual_emb = self.encode(images, task_description)
        return self.policy(visual_emb, proprio)
