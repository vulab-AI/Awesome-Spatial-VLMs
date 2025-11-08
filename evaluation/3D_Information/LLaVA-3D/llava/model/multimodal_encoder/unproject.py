import torch
from torch.nn import functional as F


def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3
    """
    # (B, V, 336, 336)
    B, V, H, W = depths.shape 
    # (B, V, 1)
    fx, fy, px, py = intrinsics[..., 0, 0][..., None], intrinsics[..., 1, 1][..., None], intrinsics[..., 0, 2][..., None], intrinsics[..., 1, 2][..., None]

    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    z = depths.flatten(2)  # (B, V, H*W)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    return world_coords

def backproject_depth(depths, poses, intrinsics=None):
    B, V, H, W = depths.shape
    xyz = unproject(intrinsics, poses, depths)
    return xyz  # (B X V X H X W X 3)

def interpolate_depth(xyz, multi_scale_features, method="nearest"):
    multi_scale_xyz = []
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode=method).permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_)    
    return multi_scale_xyz

def backprojector_dataloader(
    multi_scale_features, depths, poses,
    intrinsics=None, method='nearest', 
    padding=None):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 1024, 24, 24], [B*V, 1024, 48, 48], [B*V, 1024, 96, 96]
        depths: tensor [B, 5, 336, 336]
        poses: tensor [B, 5, 4, 4]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    # (B, V, H, W, 3)
    new_xyz = backproject_depth(
        depths, poses, intrinsics)

    if padding is not None:
        new_xyz = F.pad(new_xyz.permute(0, 1, 4, 2, 3), (0, padding[1], 0, padding[0]), mode='constant', value=0).permute(0, 1, 3, 4, 2)

    multi_scale_xyz = interpolate_depth(
        new_xyz, multi_scale_features,
        method=method)

    if len(multi_scale_xyz) == 1:
        multi_scale_xyz =  multi_scale_xyz[0]
    
    return multi_scale_xyz, new_xyz

def voxelize(xyz, voxel_size=0.28):
    """
    Inputs: 
        xyz: list of tensors [B, V, H, W, 3]
        voxel_size: voxel size

    Outputs:
        N=V*H*W
        p2v: tensors [B, N]
    """
    B, V, H, W, _ = xyz.shape
    xyz = xyz.reshape(B, V*H*W, 3)
    p2v = voxelization(xyz, voxel_size)
    return p2v


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def voxelization(xyz, voxel_size):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
    Outputs: 
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0)
    return point_to_voxel


def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, C)
        point2voxel (B, N)
    Output:
        src_new (B, N, C)
    """
    bs, n, c = voxel_map.shape
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new