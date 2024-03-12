import math
from functools import wraps

import cv2
import numpy as np
import torch

try:
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
except:
    print("Pytorch3D is not installed.")


def unif_numpy_torch(func):
    """
    Summary:
        Decorator function for unifying numpy/torch input and output.
        Function to be decorated should be written in PyTorch.
        The decorated function accepts both numpy and torch.
        If the input is ndarray, it returns ndarray,
        and if in torch, output is also torch.
        Doesn't support for dictionary returns.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list([*args])
        contain_numpy = False

        for arg_i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[arg_i] = torch.from_numpy(arg)
                contain_numpy = True

        for key, arg in kwargs.items():
            if isinstance(arg, np.ndarray):
                kwargs[key] = torch.from_numpy(arg)
                contain_numpy = True

        rets = func(*args, **kwargs)

        if contain_numpy:
            if isinstance(rets, tuple):
                numpy_rets = []
                for ret in rets:
                    if isinstance(ret, torch.Tensor):
                        ret = ret.numpy()
                    numpy_rets.append(ret)
                return tuple(numpy_rets)
            else:
                return rets.numpy()
        return rets

    return wrapper


@unif_numpy_torch
def vector_to_pitchyaw(vec):
    unit_vec = vec / torch.norm(vec, dim=-1, keepdim=True)
    theta = torch.arcsin(unit_vec[..., 1])
    phi = torch.atan2(unit_vec[..., 0], unit_vec[..., 2])
    pitchyaw = torch.stack([theta, phi], dim=-1)
    return pitchyaw


@unif_numpy_torch
def pitchyaw_to_vector(pitchyaws):
    """
    Args:
        pitchyaw:   [..., 2]
    Returns:
        unit_vec3d: [..., 3]
    """
    if torch.any(torch.isinf(pitchyaws)):
        print("INPUT PITCHYAW IS INF")
        print("pitchyaw = ", pitchyaws)
    if torch.any(torch.isnan(pitchyaws)):
        print("INPUT PITCHYAW IS NAN")
        print("pitchyaw = ", pitchyaws)
    pitch = pitchyaws[..., :1].clone()
    yaw = pitchyaws[..., 1:].clone()
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    unit_vec3d = torch.cat(
        [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw], dim=-1
    )
    # print('cos_pitch = ', cos_pitch)
    # print('cos-yaw = ', cos_yaw)
    if torch.any(torch.isnan(unit_vec3d)):
        print("during PITCHYAW_TO_VECTOR")
        print("unit-vec3d = ", unit_vec3d)
    return unit_vec3d


@unif_numpy_torch
def rotate_pitchyaw(rot, pitchyaw):
    """
    Summary:
        returns `rot @ gaze` in pitchyaw format
    Args:
        rot:      [batch_size, ..., 3, 3]
        pitchyaw: [batch_size, ..., 2]
    Returns:
        rot_gaze: [batch_size, ..., 2]
    """
    vec = pitchyaw_to_vector(pitchyaw)
    vec = rot @ vec.unsqueeze(-1)
    rot_gaze = vector_to_pitchyaw(vec.squeeze(-1))
    return rot_gaze


@unif_numpy_torch
def pitchyaw_to_psi_theta(pitchyaws):
    # TODO declare the definition of psi, theta. what's the coords?
    pitch = pitchyaws[..., :1]
    yaw = pitchyaws[..., 1:]

    psi = torch.atan2(-yaw.sin(), pitch.tan())
    theta = torch.atan2(
        -pitch.cos() * (-yaw).sin() / psi.sin(), (-pitch).cos() * (-yaw).cos()
    )
    return psi, theta


def feat_to_mat(feat):
    """
    Args:
        feat: [..., 6]
    Returns:
        mat: [..., 3, 3]
    """
    x_raw = feat[..., 0:3]
    y_raw = feat[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x)

    mat = torch.stack([x, y, z], dim=-1)
    return mat


def rotate_along_x(angle_x):
    batch_size = angle_x.shape[0]
    device = angle_x.device
    rots = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    rots[:, [1, 2], [1, 2]] = torch.cos(angle_x).reshape(-1, 1)
    rots[:, [1, 2], [2, 1]] = torch.sin(angle_x).reshape(-1, 1)
    rots[:, 1, 2] *= -1
    return rots


def rotate_along_y(angle_y):
    batch_size = angle_y.shape[0]
    device = angle_y.device
    rots = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    rots[:, [0, 2], [0, 2]] = torch.cos(angle_y).reshape(-1, 1)
    rots[:, [0, 2], [2, 0]] = torch.sin(angle_y).reshape(-1, 1)
    rots[:, 2, 0] *= -1
    return rots


def to_rvec(rotmat):
    """
    Args:
        rotmat: Rotation matrix \in 3x3
    Returns:
        rvec  : Rotation in Rodrigues formula.
                Since the head pose is normalized, the last element is always 0.
                We ommit the last element, and thus the shape is 2
    """
    M = cv2.Rodrigues(rotmat)
    Zv = M[:, 2]
    rotvec = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])
    return rotvec


@unif_numpy_torch
def angular_error(a, b, is_degree=True):
    a = pitchyaw_to_vector(a) if a.shape[-1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[-1] == 2 else b

    a_norm = torch.norm(a, dim=-1)
    b_norm = torch.norm(b, dim=-1)
    similarity = torch.sum(a * b, dim=-1) / (a_norm * b_norm + 1e-6)

    error = torch.arccos(similarity.clip(max=1.0 - 1e-6))
    if is_degree:
        error *= 180.0 / np.pi
    return error


def l1_distance_in_pitchyaw(a, b):
    return torch.abs(a - b)


def gaze_2d_error(a, b):
    a = pitchyaw_to_vector(a)
    b = pitchyaw_to_vector(b)
    return torch.norm(a[..., :2] - b[..., :2], dim=-1)


def pitchyaw_to_rotmat(pitchyaw):
    is_torch = isinstance(pitchyaw, torch.Tensor)
    # print('pitchyaw = ', pitchyaw)
    # print('fliplr', torch.flip(pitchyaw, dims=[0]))
    rotmat = rotate_from_euler("xy", pitchyaw)
    # rotmat = SciRotation.from_euler("yx", pitchyaw, degrees=False).as_matrix()
    # rotmat = SciRotation.from_euler("yx", torch.flip(pitchyaw, dims=[0]), degrees=False).as_matrix()

    return rotmat


@unif_numpy_torch
def rotate_from_euler(order: str, degrees):
    """
    order: str, sequenece of [x, y, z]
    degrees: torch.tensor, [..., 3]
    """
    device = degrees.device
    ones = torch.ones(*degrees.shape[:-1], 1, device=device)
    zeros = torch.zeros(*degrees.shape[:-1], 1, device=device)
    rotations = {}
    if "x" in order:
        dim = order.index("x")
        degrees_x = degrees[..., dim : dim + 1]
        rotations["rx"] = torch.stack(  # NOQA
            [
                torch.cat([ones, zeros, zeros], dim=-1),
                torch.cat([zeros, degrees_x.cos(), -degrees_x.sin()], dim=-1),
                torch.cat([zeros, degrees_x.sin(), degrees_x.cos()], dim=-1),
            ],
            dim=-1,
        )
    if "y" in order:
        dim = order.index("y")
        degrees_y = degrees[..., dim : dim + 1]
        rotations["ry"] = torch.stack(  # NOQA
            [
                torch.cat([degrees_y.cos(), zeros, degrees_y.sin()], dim=-1),
                torch.cat([zeros, ones, zeros], dim=-1),
                torch.cat([-degrees_y.sin(), zeros, degrees_y.cos()], dim=-1),
            ],
            dim=-1,
        )
    if "z" in order:
        dim = order.index("z")
        degrees_z = degrees[..., dim : dim + 1]
        rotations["rz"] = torch.stack(  # NOQA
            [
                torch.cat([degrees_z.cos(), -degrees_z.sin(), zeros], dim=-1),
                torch.cat([degrees_z.sin(), degrees_z.cos(), zeros], dim=-1),
                torch.cat([zeros, zeros, ones], dim=-1),
            ],
            dim=-1,
        )
    r = rotations[f"r{order[0]}"]
    for axis in order[1:]:
        r = r @ rotations[f"r{axis}"]
    return r.transpose(-1, -2).to(torch.float)


def rotation_matrix_slerp(
    rot_a: torch.Tensor, rot_b: torch.Tensor, ratio: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        rot_*: [batch_size, 3, 3]
        ratio: [batch_size, 1]. Float value ranges 0 ~ 1.
    """
    EPS = 1e-5
    quat_a = matrix_to_quaternion(rot_a)
    quat_b = matrix_to_quaternion(rot_b)
    theta = torch.arccos((quat_a * quat_b).sum(dim=-1, keepdim=True))
    is_near_pi = torch.isnan(theta)

    w_a = torch.sin(theta * (1 - ratio)) / (torch.sin(theta) + EPS)
    w_b = torch.sin(theta * ratio) / (torch.sin(theta) + EPS)

    w_a[is_near_pi] = (1 - ratio.to(w_a.dtype))[is_near_pi]
    w_b[is_near_pi] = ratio.to(w_b.dtype)[is_near_pi]
    quat_slerp = quat_a * w_a + quat_b * w_b

    return quaternion_to_matrix(quat_slerp)
