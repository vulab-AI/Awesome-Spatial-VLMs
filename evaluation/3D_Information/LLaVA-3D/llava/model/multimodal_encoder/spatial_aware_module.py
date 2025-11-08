import torch
from torch import nn
from .position_encodings import PositionEmbeddingLearnedMLP
    

class SpatialAwareModule(nn.Module):

    def __init__(self, latent_dim=1024):
        super(SpatialAwareModule, self).__init__()
        self.positional_embedding = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.positional_embedding(xyz)
        
    def forward(
        self, feature_list=None, xyz_list=None,
        shape=None, multiview_data=None, voxelize=None,
        ) -> torch.Tensor:
        """
        Args:
            feature_list: list of tensor (B*V, C, H, W)
            xyz_list: list of tensor (B*V, H, W, 3)
            shape: (B, V)
        """
        out_features = []
        bs, v = shape
        for j, (feature, xyz) in enumerate(zip(feature_list, xyz_list)):
            # B*V, F, H, W -> B, V, F, H, W -> B, V, H, W, F
            bv, f, h, w = feature.shape
            feature = feature.reshape(bs, v, f, h, w).permute(0, 1, 3, 4, 2)
            xyz = xyz.reshape(bs, v, h, w, 3)
            pos_embed = self.encode_pe(xyz) # (B, V, H, W, F)
            feature = feature + pos_embed
            feature = feature.flatten(1, 3)  # (B, V*H*W, F)
            out_features.append(feature)
        return out_features