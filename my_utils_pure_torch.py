"""
Pure-PyTorch utilities for fitting GaussianImage to an image -- no gsplat,
no compiled CUDA extension.

Two covariance decompositions (paper Sec. 3.1), one shared renderer:
  * GaussianImage_Cholesky   Sigma = L L^T          (Eq. 1)
  * GaussianImage_RS         Sigma = (R S)(R S)^T   (Eqs. 2-3)

Adapted from https://github.com/Xinjie-Q/GaussianImage.
torch.optim.Adam replaces Adan.

Renderer: each Gaussian is splatted into a local window around its center
(sized from its own 3-sigma radius), weights are evaluated in PyTorch, and
`weight * color` is scatter-added into the image. Autograd handles backward.
Cost is ~N * (window size)^2. Sizing the window needs one `.item()` sync per
forward call. Covariance entries are in pixel units for both models.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Loss
# =============================================================================

def loss_fn(pred, target):
    """L2 (MSE) reconstruction loss."""
    return F.mse_loss(pred.float(), target.detach().float())


# =============================================================================
# Covariance from each decomposition
# =============================================================================

def cholesky_to_cov(cholesky_vector):
    """(N, 3) (l1, l2, l3), L = [[l1, 0], [l2, l3]] -> (N, 3) (S_xx, S_xy, S_yy) of L L^T."""
    l1, l2, l3 = cholesky_vector[:, 0], cholesky_vector[:, 1], cholesky_vector[:, 2]
    return torch.stack([l1 * l1, l1 * l2, l2 * l2 + l3 * l3], dim=-1)


def scale_rot_to_cov(scales, rotation):
    """(N, 2) scales (s1, s2) in px + (N, 1) angle in rad -> (N, 3) (S_xx, S_xy, S_yy).

    Matches gsplat's project_gaussians_2d_scale_rot kernel: M = R S with
    R = [[cos, sin], [-sin, cos]] (the mirror of paper Eq. 3; theta just turns
    the other way, fitting is unaffected) and Sigma = M M^T.
    """
    theta = rotation[:, 0]
    cos, sin = torch.cos(theta), torch.sin(theta)
    s1sq, s2sq = scales[:, 0] ** 2, scales[:, 1] ** 2
    return torch.stack([
        cos * cos * s1sq + sin * sin * s2sq,
        cos * sin * (s2sq - s1sq),
        sin * sin * s1sq + cos * cos * s2sq,
    ], dim=-1)


# =============================================================================
# Renderer (pure PyTorch)
# =============================================================================

def render_gaussians_cov(gaussian_centers, cov, weighted_color, H, W,
                         k_sigma=3.0, min_radius=2, max_radius=None, background=None):
    """Additive splat of 2D Gaussians into an (H, W, 3) image (not clamped).

    Args:
        gaussian_centers: (N, 2) centers in pixels (x, y).
        cov: (N, 3) covariance entries (S_xx, S_xy, S_yy) in pixel^2.
        weighted_color: (N, 3) color * opacity (opacity is fixed at 1).
        H, W: image size.
        k_sigma: radius = k_sigma * sqrt(largest eigenvalue).
        min_radius / max_radius: radius floor / cap in pixels (cap defaults to max(H, W)).
        background: unused by additive blending, kept for API compatibility.
    """
    device, dtype = gaussian_centers.device, gaussian_centers.dtype
    N = gaussian_centers.shape[0]

    if background is None:
        background = torch.ones(3, device=device, dtype=dtype)
    if max_radius is None:
        max_radius = max(H, W)

    a, b, c = cov[:, 0], cov[:, 1], cov[:, 2]
    det = (a * c - b * b).clamp_min(1e-8)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det  # conic = Sigma^-1

    # Per-Gaussian radius from the largest eigenvalue (like gsplat's radii).
    mid = 0.5 * (a + c)
    half_diff = 0.5 * (a - c)
    eig_max = mid + torch.sqrt((half_diff * half_diff + b * b).clamp_min(0.0))
    radius_f = (k_sigma * torch.sqrt(eig_max.clamp_min(1e-12))).clamp(min_radius, max_radius)
    radius_i = torch.ceil(radius_f).detach()  # discrete pixel selection, not differentiated

    # Shared window sized to the largest radius this call (needs a GPU->CPU sync).
    patch_radius = int(radius_i.max().item())
    P = 2 * patch_radius + 1

    cx, cy = gaussian_centers[:, 0], gaussian_centers[:, 1]
    cx0 = torch.round(cx).detach().long()
    cy0 = torch.round(cy).detach().long()

    offsets = torch.arange(-patch_radius, patch_radius + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")  # (P, P)

    px = cx0.view(N, 1, 1) + ox.view(1, P, P)  # (N, P, P) pixel x
    py = cy0.view(N, 1, 1) + oy.view(1, P, P)  # (N, P, P) pixel y
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    dx = px.to(dtype) - cx.view(N, 1, 1)  # subpixel-accurate offsets
    dy = py.to(dtype) - cy.view(N, 1, 1)

    # Cut off at each Gaussian's own radius, not the shared window size.
    within_own_radius = (dx * dx + dy * dy) <= (radius_i.view(N, 1, 1) ** 2)

    power = -0.5 * (
        inv_a.view(N, 1, 1) * dx * dx
        + 2 * inv_b.view(N, 1, 1) * dx * dy
        + inv_c.view(N, 1, 1) * dy * dy
    )
    weight = torch.exp(power.clamp(max=0.0)) * in_bounds * within_own_radius  # (N, P, P)
    contrib = weight.unsqueeze(-1) * weighted_color.view(N, 1, 1, 3)  # (N, P, P, 3)

    flat_idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)  # masked entries are 0
    out = torch.zeros(H * W, 3, device=device, dtype=dtype)
    out = out.index_add(0, flat_idx, contrib.reshape(-1, 3))

    return out.view(H, W, 3)


def render_gaussians(gaussian_centers, cholesky_vector, weighted_color, H, W, **kwargs):
    """Cholesky front-end to `render_gaussians_cov` (original signature)."""
    return render_gaussians_cov(gaussian_centers, cholesky_to_cov(cholesky_vector),
                                weighted_color, H, W, **kwargs)


# =============================================================================
# Model: GaussianImage_Cholesky
# =============================================================================

class GaussianImage_Cholesky(nn.Module):
    """2D Gaussians with position, Cholesky covariance factor and color."""

    def __init__(self, **kwargs):
        super().__init__()
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.device = kwargs["device"]
        # Render window: k-sigma cutoff plus radius floor / cap.
        self.k_sigma = kwargs.get("k_sigma", 3.0)
        self.min_radius = kwargs.get("min_radius", 2)
        self.max_radius = kwargs.get("max_radius", None)

        # Position, mapped to (-1, 1) via tanh.
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        # Cholesky vector (l1, l2, l3); `cholesky_bound` adds 0.5 to l1, l3.
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer("_opacity", torch.ones((self.init_num_points, 1)))
        # Weighted color c' (Eq. 7): color merged with opacity.
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer("background", torch.ones(3))
        self.register_buffer("cholesky_bound", torch.tensor([0.5, 0, 0.5]).view(1, 3))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    @property
    def get_xyz(self):
        """Centers in (-1, 1)."""
        return torch.tanh(self._xyz)

    @property
    def get_xyz_pixels(self):
        """Centers in pixel coordinates."""
        xy01 = (self.get_xyz + 1) * 0.5
        scale = torch.tensor([self.W - 1, self.H - 1], device=xy01.device, dtype=xy01.dtype)
        return xy01 * scale

    @property
    def get_features(self):
        """Weighted color coefficients."""
        return self._features_dc

    @property
    def get_opacity(self):
        """Opacity, fixed at 1."""
        return self._opacity

    @property
    def get_cholesky_elements(self):
        """Cholesky vector offset by `cholesky_bound`."""
        return self._cholesky + self.cholesky_bound

    def forward(self):
        """Render to a (1, 3, H, W) image in [0, 1]."""
        out_img = render_gaussians(
            self.get_xyz_pixels, self.get_cholesky_elements, self.get_features,
            self.H, self.W, k_sigma=self.k_sigma, min_radius=self.min_radius,
            max_radius=self.max_radius, background=self.background,
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.permute(2, 0, 1).unsqueeze(0).contiguous()
        return {"render": out_img}


# =============================================================================
# Model: GaussianImage_RS
# =============================================================================

class GaussianImage_RS(nn.Module):
    """2D Gaussians with position, rotation + scaling covariance and color.
    Identical to GaussianImage_Cholesky except for the covariance."""

    def __init__(self, **kwargs):
        super().__init__()
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.device = kwargs["device"]
        # Render window: k-sigma cutoff plus radius floor / cap.
        self.k_sigma = kwargs.get("k_sigma", 3.0)
        self.min_radius = kwargs.get("min_radius", 2)
        self.max_radius = kwargs.get("max_radius", None)

        # Position, mapped to (-1, 1) via tanh.
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        # Scales (s1, s2) and angle theta (Eqs. 2-3); `scaling_bound` adds 0.5 to s1, s2.
        self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self.register_buffer("_opacity", torch.ones((self.init_num_points, 1)))
        # Weighted color c' (Eq. 7): color merged with opacity.
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer("background", torch.ones(3))
        self.register_buffer("scaling_bound", torch.tensor([0.5, 0.5]).view(1, 2))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    @property
    def get_xyz(self):
        """Centers in (-1, 1)."""
        return torch.tanh(self._xyz)

    @property
    def get_xyz_pixels(self):
        """Centers in pixel coordinates."""
        xy01 = (self.get_xyz + 1) * 0.5
        scale = torch.tensor([self.W - 1, self.H - 1], device=xy01.device, dtype=xy01.dtype)
        return xy01 * scale

    @property
    def get_features(self):
        """Weighted color coefficients."""
        return self._features_dc

    @property
    def get_opacity(self):
        """Opacity, fixed at 1."""
        return self._opacity

    @property
    def get_scaling(self):
        """Scales (s1, s2) in pixels: |raw + 0.5|."""
        return torch.abs(self._scaling + self.scaling_bound)

    @property
    def get_rotation(self):
        """Angle in radians, in (0, 2*pi) via sigmoid."""
        return torch.sigmoid(self._rotation) * 2 * math.pi

    def forward(self):
        """Render to a (1, 3, H, W) image in [0, 1]."""
        out_img = render_gaussians_cov(
            self.get_xyz_pixels, scale_rot_to_cov(self.get_scaling, self.get_rotation),
            self.get_features, self.H, self.W, k_sigma=self.k_sigma,
            min_radius=self.min_radius, max_radius=self.max_radius, background=self.background,
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.permute(2, 0, 1).unsqueeze(0).contiguous()
        return {"render": out_img}


# =============================================================================
# Helpers
# =============================================================================

def image_to_tensor(image, device="cuda"):
    """Path, PIL.Image or numpy array -> [1, 3, H, W] float tensor in [0, 1]."""
    from PIL import Image as PILImage

    if isinstance(image, (str, Path)):
        image = PILImage.open(image)
    if isinstance(image, PILImage.Image):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image)
    if image.ndim == 2:  # grayscale -> 3 channels
        image = np.stack([image, image, image], axis=-1)
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)