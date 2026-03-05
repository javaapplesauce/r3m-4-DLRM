import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class DINO_CLIP_Fusion(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        
        # 1. Load the Vision Backbones
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # 2. Cross-Attention Setup
        self.hidden_dim = 768 # Default for ViT-Base models
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Expects [B, 3, 224, 224]
        """
        with torch.no_grad(): 
            # DINO processing: Get dense patch tokens [B, N, 768]
            dino_features = self.dino.forward_features(x)
            dino_patches = dino_features['x_norm_patchtokens'] 
            
            # CLIP processing: Get global CLS token and reshape to [B, 1, 768]
            clip_features = self.clip(x)
            clip_cls = clip_features.pooler_output.unsqueeze(1) 
            
        # Fuse: CLIP [CLS] queries the DINO patches
        attn_output, _ = self.cross_attn(query=clip_cls, key=dino_patches, value=dino_patches)
        
        # Squeeze out the sequence dimension: [B, 1, 768] -> [B, 768]
        fused_representation = attn_output.squeeze(1)
        
        # Project down to 512-dim for the Behavior Cloning MLP
        out = self.projector(fused_representation) 
        
        return out