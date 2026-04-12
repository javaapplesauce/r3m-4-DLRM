import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """3-layer MLP policy head for behavioral cloning.

    Takes concatenated [visual_embedding, proprioceptive_state] and outputs
    a 6-DOF delta action (dx, dy, dz, droll, dpitch, dyaw).
    """

    def __init__(self, visual_dim, proprio_dim=14, hidden_dim=256, action_dim=6,
                 num_layers=3, dropout=0.0):
        super().__init__()
        input_dim = visual_dim + proprio_dim

        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, visual_emb, proprio):
        """
        Args:
            visual_emb: (B, visual_dim) pooled visual features.
            proprio: (B, proprio_dim) proprioceptive state.
        Returns:
            action: (B, action_dim) predicted delta action.
        """
        x = torch.cat([visual_emb, proprio], dim=-1)
        return self.net(x)
