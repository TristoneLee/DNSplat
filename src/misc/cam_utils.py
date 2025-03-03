import cv2
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]

    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32, device=pivotal_pose.device)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


####### Pose update from delta

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(cam_trans_delta: Float[Tensor, "batch 3"],
                cam_rot_delta: Float[Tensor, "batch 3"],
                extrinsics: Float[Tensor, "batch 4 4"],
                # original_rot: Float[Tensor, "batch 3 3"],
                # original_trans: Float[Tensor, "batch 3"],
                # converged_threshold: float = 1e-4
                ):
    # extrinsics is c2w, here we need w2c as input, so we need to invert it
    view_input = len(extrinsics.shape) == 4
    if view_input:
        B,V = extrinsics.shape[0], extrinsics.shape[1]
        extrinsics = extrinsics.reshape(B*V, 4, 4)
        cam_trans_delta = cam_trans_delta.reshape(B*V, 3)
        cam_rot_delta = cam_rot_delta.reshape(B*V, 3)
    bs = cam_trans_delta.shape[0]

    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=-1)
    T_w2c = extrinsics.inverse()

    new_w2c_list = []
    for i in range(bs):
        new_w2c = SE3_exp(tau[i]) @ T_w2c[i]
        new_w2c_list.append(new_w2c)

    new_w2c = torch.stack(new_w2c_list, dim=0)
    return new_w2c.inverse() if not view_input else new_w2c.inverse().reshape(B, V, 4, 4)

    # converged = tau.norm() < converged_threshold
    # camera.update_RT(new_R, new_T)
    #
    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)
    # return converged


#######  Pose estimation
def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def get_pnp_pose(pts3d, opacity, K, H, W, opacity_threshold=0.3):
    pixels = np.mgrid[:W, :H].T.astype(np.float32)
    pts3d = pts3d.cpu().numpy()
    opacity = opacity.cpu().numpy()
    K = K.cpu().numpy()

    K[0, :] = K[0, :] * W
    K[1, :] = K[1, :] * H

    mask = opacity > opacity_threshold

    res = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
    success, R, T, inliers = res

    assert success

    R = cv2.Rodrigues(R)[0]  # world to cam
    pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world

    return torch.from_numpy(pose.astype(np.float32))


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs

def quaternion_from_matrix(matrix: torch.Tensor):
    """Convert rotation matrices to quaternions.
    Args:
        matrix: [B, 4, 4] or [4, 4] transformation matrix
    Returns:
        position: [B, 3] or [3] translation vector
        quaternion: [B, 4] or [4] rotation quaternion
    """
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    
    B = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype
    
    rotation = matrix[:, :3, :3]
    trace = rotation.diagonal(dim1=1, dim2=2).sum(-1)  # [B]
    
    q = torch.zeros(B, 4, device=device, dtype=dtype)
    eps = 1e-7  # numerical stability
    
    # Handle positive trace
    pos_mask = trace > 0
    if pos_mask.any():
        S = torch.sqrt(trace[pos_mask] + 1.0) * 2
        q[pos_mask, 0] = 0.25 * S
        q[pos_mask, 1] = (rotation[pos_mask, 2, 1] - rotation[pos_mask, 1, 2]) / (S + eps)
        q[pos_mask, 2] = (rotation[pos_mask, 0, 2] - rotation[pos_mask, 2, 0]) / (S + eps)
        q[pos_mask, 3] = (rotation[pos_mask, 1, 0] - rotation[pos_mask, 0, 1]) / (S + eps)
    
    # Handle negative trace
    neg_mask = ~pos_mask
    if neg_mask.any():
        # Find largest diagonal element
        max_diag = torch.argmax(rotation[neg_mask].diagonal(dim1=1, dim2=2), dim=1)  # [B']
        
        for i in range(3):
            i_mask = neg_mask.clone()
            i_mask[neg_mask] = max_diag == i
            if not i_mask.any():
                continue
                
            if i == 0:  # max value at rotation[0,0]
                S = torch.sqrt(1.0 + rotation[i_mask, 0, 0] - rotation[i_mask, 1, 1] - rotation[i_mask, 2, 2]) * 2
                q[i_mask, 0] = (rotation[i_mask, 2, 1] - rotation[i_mask, 1, 2]) / (S + eps)
                q[i_mask, 1] = 0.25 * S
                q[i_mask, 2] = (rotation[i_mask, 0, 1] + rotation[i_mask, 1, 0]) / (S + eps)
                q[i_mask, 3] = (rotation[i_mask, 0, 2] + rotation[i_mask, 2, 0]) / (S + eps)
            elif i == 1:  # max value at rotation[1,1]
                S = torch.sqrt(1.0 + rotation[i_mask, 1, 1] - rotation[i_mask, 0, 0] - rotation[i_mask, 2, 2]) * 2
                q[i_mask, 0] = (rotation[i_mask, 0, 2] - rotation[i_mask, 2, 0]) / (S + eps)
                q[i_mask, 1] = (rotation[i_mask, 0, 1] + rotation[i_mask, 1, 0]) / (S + eps)
                q[i_mask, 2] = 0.25 * S
                q[i_mask, 3] = (rotation[i_mask, 1, 2] + rotation[i_mask, 2, 1]) / (S + eps)
            else:  # max value at rotation[2,2]
                S = torch.sqrt(1.0 + rotation[i_mask, 2, 2] - rotation[i_mask, 0, 0] - rotation[i_mask, 1, 1]) * 2
                q[i_mask, 0] = (rotation[i_mask, 1, 0] - rotation[i_mask, 0, 1]) / (S + eps)
                q[i_mask, 1] = (rotation[i_mask, 0, 2] + rotation[i_mask, 2, 0]) / (S + eps)
                q[i_mask, 2] = (rotation[i_mask, 1, 2] + rotation[i_mask, 2, 1]) / (S + eps)
                q[i_mask, 3] = 0.25 * S
    
    # Normalize quaternions
    q = q / (torch.norm(q, dim=1, keepdim=True) + eps)
    
    # Get positions
    position = matrix[:, :3, 3]
    
    # Remove batch dimension if input was unbatched
    if matrix.shape[0] == 1:
        position = position.squeeze(0)
        q = q.squeeze(0)
        
    return position, q