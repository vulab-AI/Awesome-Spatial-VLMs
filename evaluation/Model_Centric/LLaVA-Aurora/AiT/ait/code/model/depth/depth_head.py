import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule, auto_fp16
from mmdet.models.builder import HEADS

from model import build_model
import math


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        pred = pred.clip(min=1e-3)
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


@HEADS.register_module()
class DepthHead(BaseModule):
    def __init__(self,
                 task_id,
                 transformer,
                 loss_weight=1.,
                 depth_token_offset=0,
                 vae_cfg=dict(
                     token_length=16,
                     mask_size=64,
                     embedding_dim=64,
                     num_embeddings=256,
                     pretrained='vqvae_patch16_256_64.pth',
                     freeze=True
                 ),
                 shift_window_test=True,
                 shift_size=2,
                 flip_test=True,
                 decoder_loss_weight=0.,
                 soft_vae=False,
                 ):
        super().__init__()
        self.fp16_enabled = False
        self.decoder_loss_weight = decoder_loss_weight
        self.dloss_fn = SiLogLoss()
        self.loss_weight = loss_weight
        self.transformer = transformer
        self.task_id = task_id
        self.depth_token_offset = depth_token_offset
        self.vae_cfg = vae_cfg
        self.vqvae = build_model(vae_cfg)
        self.loss_seq = nn.CrossEntropyLoss(reduction='sum')

        # test setting
        self.shift_window_test = shift_window_test
        self.shift_size = shift_size
        self.flip_test = flip_test
        self.soft_vae = soft_vae

    def simple_test(self, feats, **kwargs):
        x = feats[-1]  # using only last layer of x
        masks = x.new_zeros((x.shape[0], x.shape[2], x.shape[3]))
        pred_logits, pred_tokens = self.transformer(
            x, None, masks, self.task_id, pred_len=self.vae_cfg.token_length)
        if self.soft_vae:
            B, N, C = pred_logits.shape
            n = int(math.sqrt(N))
            pred_depth = self.vqvae.decode_soft(
                pred_logits.transpose(1, 2).view(B, C, n, n))
        else:
            pred_depth = self.vqvae.decode(pred_tokens)
        return pred_depth

    def get_seq(self, depth, **kwargs):
        vae_gt = self.vqvae.encode(depth.squeeze(1)) + self.depth_token_offset
        self.depth_map = depth
        return vae_gt[:, :-1], vae_gt

    @force_fp32(apply_to='out_seq')
    def loss2(self, out_seq, target_seq):
        loss = self.loss_seq(out_seq.transpose(1, 2).contiguous(), target_seq)
        if self.decoder_loss_weight > 0:
            B, nL, C = out_seq.shape
            soft_out_mask = self.vqvae.decode_soft(out_seq.view(B, int(math.sqrt(nL)), int(math.sqrt(nL)), C)[
                                                   :, :, :, self.depth_token_offset: self.depth_token_offset+self.vae_cfg.num_embeddings].permute(0, 3, 1, 2))
            decoder_aux_ce = self.dloss_fn(
                soft_out_mask.float(), self.depth_map.squeeze(1).float())
        else:
            decoder_aux_ce = torch.tensor(
                0., device=f'cuda:{torch.cuda.current_device()}')
        return {'loss': (loss + decoder_aux_ce * self.decoder_loss_weight * (target_seq > -1).sum()) * self.loss_weight, 'ce': loss/(target_seq > -1).sum(), 'decoder_aux_ce': decoder_aux_ce}
