from typing import Optional

import torch

# from bailingmm.common.logger import logger


def unwrap_feats(feats: torch.Tensor, feats_lengths: torch.Tensor):
    """
    The input feats are in the "wrapped" format, which means that features from (at most) N audios are concatenated
    as a single sample feats[i]. In this case, each row of feats_lengths contains the lengths of the concatenated
    feature. This function unwraps the features.
    For samples with less than N segments, one should pad feats_lengths with 0. The result will contain valid
    segments only.

    feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    feats_lengths: torch.LongTensor, size = [B, N]

    Example ('X' for padding):
    Inputs:
        feats = [[A, A, A, A, X],
                 [B, B, C, C, C]]
        feats_lengths = [[4, 0],
                         [2, 3]]
    Outputs:
        feat_segs = [[A, A, A, A],
                     [B, B, X, X],
                     [C, C, C, X]]
        feat_seg_lengths = [4, 2, 3]
    """
    feat_segs = []
    feat_seg_lengths = []
    for i in range(feats_lengths.shape[0]):
        feat_index = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0: break
            feat_segs.append(feats[i, feat_index:feat_index + feat_len])
            feat_seg_lengths.append(feat_len)
            feat_index += feat_len
    feat_segs_batch = torch.nn.utils.rnn.pad_sequence(feat_segs, True).to(feats.device)
    feat_seg_lengths = torch.tensor(feat_seg_lengths, dtype=torch.long, device=feats.device)
    return feat_segs_batch, feat_seg_lengths


def unwrap_feats_n1(feats: torch.Tensor, feats_lengths: torch.Tensor):
    """
    Equivalent version of unwrap_feats when N=1.
    """
    return feats, feats_lengths.squeeze(-1)


def wrap_feats(feat_segs: torch.Tensor, feats_lengths: torch.Tensor, feats_seg_lengths: Optional[torch.Tensor] = None):
    """
    Wrap segmented features back to the wrapped format.
    This function is the inverse operation of unwrap_feats(). See its documentation for details.
    Note that the feats_lengths value does not matter a lot. We only check the location of the first 0 to determine the
    number of feature segments.
    """
    feat_idx = 0
    feats_buffer = []
    feats_locs_buffer = []
    feats_lengths_buffer = []
    for i in range(feats_lengths.shape[0]):
        feat_buffer = []
        feat_locs_buffer = []
        feat_lengths_buffer = []
        feat_total_len = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0:
                break
            if feats_seg_lengths is not None:
                feat_len = feats_seg_lengths[feat_idx].item()
            feat_buffer.append(feat_segs[feat_idx, :feat_len])
            feat_locs_buffer.append(feat_total_len)
            feat_lengths_buffer.append(feat_len)
            feat_idx += 1
            feat_total_len += feat_len
        feats_buffer.append(torch.cat(feat_buffer))
        feats_locs_buffer.append(torch.tensor(feat_locs_buffer, dtype=torch.long))
        feats_lengths_buffer.append(torch.tensor(feat_lengths_buffer, dtype=torch.long))
    feats = torch.nn.utils.rnn.pad_sequence(feats_buffer, True).to(feat_segs.device)
    feats_locs = torch.nn.utils.rnn.pad_sequence(feats_locs_buffer, True).to(feats_lengths.device)
    feats_new_lengths = torch.nn.utils.rnn.pad_sequence(feats_lengths_buffer, True).to(feats_lengths.device)
    return feats, feats_locs, feats_new_lengths


def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
):
    """
    Patch continuous features into input embeddings, while keeping a valid gradient flow.

    input_embeddings: torch.Tensor, size = [B, C?, T, D]
    placeholder_loc_lens: torch.LongTensor, size = [B, N, 2]
        Each 2-tuple represents (start, length) of a placeholder.
    encoded_feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    encoded_feat_lens: torch.LongTensor, size = [B, N]

    Example ('X' for patch placeholder tokens):
    Inputs:
        input_embeddings = [[1, 2, 3, X, X, X, 4, 5, 6, X, X, X, 7, 8]]
        placeholder_loc_lens = [[3, 3], [9, 3]]
        encoded_feats = [[A, A, A, B, B]]
        encoded_feat_lens = [[3, 2]]
    Outputs:
        embeddings = [[1, 2, 3, A, A, A, 4, 5, 6, B, B, X, 7, 8]]
    """
    batch_size = input_embeddings.size(0)
    audio_feats_mask = torch.zeros_like(input_embeddings, dtype=torch.bool)
    audio_feats_buffer = []
    for i in range(batch_size):
        sample_len = 0
        audio_feat_start = 0
        audio_feat_buffer = []
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                print(
                    f"Feature length ({feat_len}) > placeholder length ({placeholder_len}). This is not expected. Please "
                    "check the implementation of estimate_audio_feature_length(). We truncate the feature to avoid errors."
                )
                feat_len = placeholder_len
            if placeholder_start > sample_len:
                audio_feat_buffer.append(input_embeddings.new_zeros((placeholder_start - sample_len, input_embeddings.shape[2])))
                sample_len = placeholder_start
            audio_feat_buffer.append(encoded_feats[i, audio_feat_start:audio_feat_start + feat_len])
            if feat_len < placeholder_len:
                audio_feat_buffer.append(encoded_feats.new_zeros(placeholder_len - feat_len))
            audio_feats_mask[i, sample_len:sample_len + feat_len] = 1
            audio_feat_start += real_feat_len
            sample_len += placeholder_len
        if sample_len < input_embeddings.shape[1]:
            audio_feat_buffer.append(
                input_embeddings.new_zeros((input_embeddings.shape[1] - sample_len, input_embeddings.shape[2]))
            )
        audio_feats_buffer.append(torch.cat(audio_feat_buffer))
    audio_feats_buffer = torch.stack(audio_feats_buffer, dim=0)
    embeddings = audio_feats_buffer * audio_feats_mask + input_embeddings * ~audio_feats_mask
    return embeddings

