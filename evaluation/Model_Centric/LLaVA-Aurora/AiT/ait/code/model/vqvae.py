import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from math import log2, sqrt
from .utils import MODELS


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 /
                                             self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs, return_indices=False):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if return_indices:
            return encoding_indices.squeeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer('_embedding', torch.empty(
            self._num_embeddings, self._embedding_dim))
        self._embedding.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs, return_indices=False):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if return_indices:
            return encoding_indices.squeeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input.detach())
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            self._embedding = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices.view(input_shape[0:3])


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class ResBlock_v1(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class ResBlock_v2(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


@MODELS.register_module()
class VQVAE(nn.Module):
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if 'codebook.weight' in state_dict:
            del state_dict['codebook.weight']

    def __init__(self, token_length=16, mask_size=64, embedding_dim=512, num_embeddings=1024, pretrained='vae.pt', freeze=True, num_resnet_blocks=2, hidden_dim=256, use_norm=True, use_sigmoid=False, tau=1.0):
        # adapt param
        self.tau = tau
        self.token_length = token_length
        image_size = mask_size
        num_tokens = num_embeddings
        codebook_dim = embedding_dim
        channels = 1
        loss_type = 'mse'
        temperature = 0.9
        straight_through = False
        kl_div_loss_weight = 0.
        simplify = False
        max_value = 1.0
        use_softmax = False
        determistic = False
        downsample_ratio = mask_size // int(sqrt(token_length))
        train_objective = "regression"
        residul_type = 'v1'
        ema_decay = 0.99
        commitment_cost = 0.25
        super().__init__()
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        hidden_dim = hidden_dim // (downsample_ratio // 2)
        self.num_resnet_blocks = num_resnet_blocks
        self.simplify = simplify
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.straight_through = straight_through
        self.use_softmax = use_softmax
        self.determistic = determistic
        self.downsample_ratio = downsample_ratio
        self.layer_num = int(log2(downsample_ratio) - 1)
        self.use_norm = use_norm
        self.use_sigmoid = use_sigmoid
        self.train_objective = train_objective
        self.max_value = max_value
        self.residul_type = residul_type
        if self.residul_type == 'v1':
            ResBlock = ResBlock_v1
        elif self.residul_type == 'v2':
            ResBlock = ResBlock_v2

        dim = hidden_dim
        enc_layers = [
            nn.Conv2d(channels, dim, 4, stride=2, padding=1), nn.ReLU()]

        for i in range(self.layer_num):
            enc_layers.append(nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            dim = dim * 2

        for i in range(num_resnet_blocks):
            enc_layers.append(ResBlock(dim))
        enc_layers.append(nn.Conv2d(dim, codebook_dim, 1))

        dim = hidden_dim * self.downsample_ratio // 2

        dec_layers = [nn.Conv2d(codebook_dim, dim, 1), nn.ReLU()]

        for i in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dim))

        dec_layers.append(nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1))
        dec_layers.append(nn.ReLU())
        for i in range(self.layer_num):
            dec_layers.append(nn.ConvTranspose2d(
                dim, dim // 2, 4, stride=2, padding=1))
            dec_layers.append(nn.ReLU())
            dim = dim // 2
        dec_layers.append(nn.Conv2d(dim, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        if loss_type == 'smooth_l1':
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_type == 'cross_entropy':
            assert self.train_objective == 'classification'
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            assert "loss_type {0} is not implemented".format(loss_type)

        self.kl_div_loss_weight = kl_div_loss_weight

        if ema_decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_tokens, codebook_dim,
                                              commitment_cost, ema_decay)
        else:
            self._vq_vae = VectorQuantizer(num_tokens, codebook_dim,
                                           commitment_cost)

        assert pretrained is not None
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'weights' in state_dict:
            state_dict = state_dict['weights']
        if 'module' in state_dict:
            state_dict = state_dict['module']
        self.load_state_dict(state_dict, strict=True)
        if freeze:
            self.freeze_layer()

    def freeze_layer(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self, pretrained=None):
        # assert pretrained == Nones
        if pretrained is not None:
            pretrained_model = torch.load(pretrained)
            self.load_state_dict(pretrained_model['weights'])

    def norm(self, images):
        images = 2. * (images / self.max_value - 0.5)
        return images

    def denorm(self, images):
        images = images * 0.5 + 0.5
        return images * self.max_value

    @torch.no_grad()
    @eval_decorator
    def encode(
        self,
        img,
        use_norm=None
    ):
        B = img.shape[0]
        image_size = self.image_size
        assert img.shape[-1] == image_size and img.shape[
            -2] == image_size, f'input must have the correct image size {image_size}'
        img = img.float()

        use_norm = self.use_norm if use_norm is None else use_norm
        if use_norm:
            img = self.norm(img)

        logits = self.encoder(img.unsqueeze(1))
        return self._vq_vae(logits, return_indices=True).view(B, self.token_length)

    def decode(
        self,
        img_seq
    ):
        image_embeds = self._vq_vae._embedding[img_seq]
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))
        image_embeds = image_embeds.view(b, h, w, d)

        image_embeds = rearrange(image_embeds, 'b h w d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)

        if self.train_objective == 'classification':
            images = images.argmax(dim=1, keepdim=True)
        if self.use_sigmoid:
            return images.sigmoid().squeeze(1)
        elif self.use_norm:
            return self.denorm(images.squeeze(1))
        else:
            return images.squeeze(1)

    def decode_soft(
        self,
        logits,
    ):
        # Actually, the effect of the selection of tau is minor
        soft_one_hot = F.softmax(logits*self.tau, dim=1)
        image_embeds = einsum('b n h w, n d -> b d h w',
                              soft_one_hot, self._vq_vae._embedding)
        images = self.decoder(image_embeds)

        if self.train_objective == 'classification':
            images = images.argmax(dim=1, keepdim=True)
        if self.use_sigmoid:
            return images.sigmoid().squeeze(1)
        elif self.use_norm:
            return self.denorm(images.squeeze(1))
        else:
            return images.squeeze(1)

    def forward(
        self,
        img,
        use_norm=None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[
            -2] == image_size, f'input must have the correct image size {image_size}'
        label_img = img.clone()
        img = img.float()

        use_norm = self.use_norm if use_norm is None else use_norm
        if use_norm:
            img = self.norm(img)

        logits = self.encoder(img)
        vq_loss, quantized, perplexity, _, code_indices = self._vq_vae(logits)
        out = self.decoder(quantized)

        # reconstruction loss
        if self.train_objective == 'classification':
            recon_loss = self.loss_fn(out, label_img[:, 0, :, :].long())
        else:
            recon_loss = self.loss_fn(out, img)

        total_loss = recon_loss + vq_loss

        if self.train_objective == 'classification':
            out = out.argmax(dim=1, keepdim=True)

        if self.use_sigmoid:
            out = out.sigmoid()
        elif use_norm:
            out = self.denorm(out)
        return total_loss, recon_loss, vq_loss, out
