import torch
import torch.nn as nn

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize
from torch_scatter import scatter_mean
from .position_encodings import PositionEmbeddingLearnedMLP

class PromptEncoder(nn.Module):
    
    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)
    
    def forward(self, clicks):
        # (n, 3)
        pos_embed = self.encode_pe(clicks) #  (N, F)
        return pos_embed

class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = 'voxelize'
        self.voxel_size = 0.2
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return

        self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, features, depths, poses, intrinsics, lengths=None):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 336, 336), image token features
            - depths: (B, V, H, W), depth images
            - poses: (B, V, 4, 4) pose information
            - instrinsics: (B, V, 4, 4), intriniscs
            - lengths: (B,)  view number of each scene

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        B, V, C, H, W = features.shape
        assert intrinsics.dim() == 4
        # (B, V, 24, 24, 3)
        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], depths, poses, intrinsics)
        # (B, V*H*W, C)
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_xyz = feat_xyz.reshape(B, V*H*W, 3)
        if lengths is not None:
            lengths = lengths*H*W

        if self.pooling == 'voxelize':
            p2v = voxelize(feat_xyz, self.voxel_size)  # （B, N)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(len(video_features))]) # bn, F
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
        else:
            raise NotImplementedError
        
        return pooled_video_features, batch_offset  # (B, num_token, 1024) or (Bn, 1024)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
