"""
Self-contained GaussianImage fitting with a C++/CUDA rasterizer.

Everything needed is in this file plus gi_render.cu (same folder). No gsplat.

Models: 
  * GaussianImage_Cholesky   Sigma = L L^T          
  * GaussianImage_RS         Sigma = (R S)(R S)^T   

Each takes backend="cuda" (default when a GPU is present; the custom
forward/backward kernels in gi_render.cu, JIT-compiled with
torch.utils.cpp_extension) or backend="torch" (the pure-PyTorch renderer,
kept as the reference for checking and benchmarking).

init_scale=None follows the paper (Gaussians start at ~1 px); a float f starts them at ~f * spacing.

Helpers: loss_fn, image_to_tensor, load_extension, check_against_torch.

Reference: Zhang, Ge et al., "GaussianImage: 1000 FPS Image Representation and
Compression by 2D Gaussian Splatting", ECCV 2024. https://arxiv.org/abs/2403.08551
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
    """(N, 2) scales in px + (N, 1) angle in rad -> (N, 3) (S_xx, S_xy, S_yy).
    Matches gsplat: M = R S with R = [[cos, sin], [-sin, cos]], Sigma = M M^T."""
    theta = rotation[:, 0]
    cos, sin = torch.cos(theta), torch.sin(theta)
    s1sq, s2sq = scales[:, 0] ** 2, scales[:, 1] ** 2
    return torch.stack([
        cos * cos * s1sq + sin * sin * s2sq,
        cos * sin * (s2sq - s1sq),
        sin * sin * s1sq + cos * cos * s2sq,
    ], dim=-1)


def _conic_and_radius(cov, k_sigma, min_radius, max_radius):
    """Sigma^-1 as (A, B, C) and the integer per-Gaussian radius (shared by both backends)."""
    a, b, c = cov[:, 0], cov[:, 1], cov[:, 2]
    det = (a * c - b * b).clamp_min(1e-8)
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)
    mid = 0.5 * (a + c)
    half_diff = 0.5 * (a - c)
    eig_max = mid + torch.sqrt((half_diff * half_diff + b * b).clamp_min(0.0))
    radius_f = (k_sigma * torch.sqrt(eig_max.clamp_min(1e-12))).clamp(min_radius, max_radius)
    return conics, torch.ceil(radius_f).detach()


# =============================================================================
# Renderer 1: pure PyTorch (reference)
# =============================================================================

def render_torch(centers, cov, colors, H, W, k_sigma=3.0, min_radius=2, max_radius=None):
    """Additive splat into an (H, W, 3) image (not clamped). One shared window
    sized to the largest radius; needs one .item() sync per call."""
    if max_radius is None:
        max_radius = max(H, W)
    device, dtype = centers.device, centers.dtype
    N = centers.shape[0]
    conics, radius_i = _conic_and_radius(cov, k_sigma, min_radius, max_radius)
    inv_a, inv_b, inv_c = conics[:, 0], conics[:, 1], conics[:, 2]

    R = int(radius_i.max().item())
    P = 2 * R + 1
    cx, cy = centers[:, 0], centers[:, 1]
    cx0 = torch.round(cx).detach().long()
    cy0 = torch.round(cy).detach().long()
    offsets = torch.arange(-R, R + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")

    px = cx0.view(N, 1, 1) + ox.view(1, P, P)
    py = cy0.view(N, 1, 1) + oy.view(1, P, P)
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    dx = px.to(dtype) - cx.view(N, 1, 1)
    dy = py.to(dtype) - cy.view(N, 1, 1)
    within = (dx * dx + dy * dy) <= (radius_i.view(N, 1, 1) ** 2)

    power = -0.5 * (inv_a.view(N, 1, 1) * dx * dx
                    + 2 * inv_b.view(N, 1, 1) * dx * dy
                    + inv_c.view(N, 1, 1) * dy * dy)
    weight = torch.exp(power.clamp(max=0.0)) * in_bounds * within
    contrib = weight.unsqueeze(-1) * colors.view(N, 1, 1, 3)

    flat_idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
    out = torch.zeros(H * W, 3, device=device, dtype=dtype)
    return out.index_add(0, flat_idx, contrib.reshape(-1, 3)).view(H, W, 3)


# =============================================================================
# Renderer 2: C++/CUDA (gi_render.cu)
# =============================================================================

_EXT = None


def load_extension(verbose=False):
    """Compile gi_render.cu on first call (~1 min), then reuse the cached build."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(
            name="gi_render_cuda",
            sources=[str(Path(__file__).with_name("gi_render.cu"))],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    return _EXT


class _RasterizeSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, conics, colors, radii, H, W):
        centers = centers.contiguous().float()
        conics = conics.contiguous().float()
        colors = colors.contiguous().float()
        radii = radii.contiguous().int()
        out = load_extension().forward(centers, conics, colors, radii, H, W)
        ctx.save_for_backward(centers, conics, colors, radii)
        ctx.H, ctx.W = H, W
        return out

    @staticmethod
    def backward(ctx, grad_out):
        centers, conics, colors, radii = ctx.saved_tensors
        d_centers, d_conics, d_colors = load_extension().backward(
            centers, conics, colors, radii, grad_out.contiguous().float(), ctx.H, ctx.W)
        return d_centers, d_conics, d_colors, None, None, None


def render_cuda(centers, cov, colors, H, W, k_sigma=3.0, min_radius=2, max_radius=None):
    """Same inputs/output as render_torch, rasterized by the CUDA kernels (no sync)."""
    if max_radius is None:
        max_radius = max(H, W)
    conics, radii = _conic_and_radius(cov, k_sigma, min_radius, max_radius)
    return _RasterizeSum.apply(centers, conics, colors, radii.int(), H, W)


RENDERERS = {"torch": render_torch, "cuda": render_cuda}


# =============================================================================
# Models
# =============================================================================

class _GaussianImageBase(nn.Module):
    """Shared parameters: position (tanh), weighted color, opacity fixed at 1.
    Subclasses add the covariance parameters and `get_cov`."""

    def __init__(self, **kwargs):
        super().__init__()
        n = self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.device = kwargs["device"]
        self.backend = kwargs.get("backend", "cuda" if torch.cuda.is_available() else "torch")  # default for model(); override per call
        if self.backend not in RENDERERS:
            raise ValueError("backend must be 'cuda' or 'torch'")
        # Render window: k-sigma cutoff plus radius floor / cap. BLOCK_* kwargs are ignored.
        self.k_sigma = kwargs.get("k_sigma", 3.0)
        self.min_radius = kwargs.get("min_radius", 2)
        self.max_radius = kwargs.get("max_radius", None)

        # None = paper (Gaussians start at ~1 px). A float f starts them at ~f * spacing,
        # spacing = sqrt(H*W/N), so they overlap from step 1 at any N.
        init_scale = kwargs.get("init_scale", None)
        self.init_sigma = None if init_scale is None else init_scale * math.sqrt(self.H * self.W / n)

        # Overlapping Gaussians add up: scale initial colors down so the first render isn't clamped.
        coverage = 1.0 if init_scale is None else max(1.0, 2 * math.pi * init_scale ** 2)
        
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(n, 2) - 0.5)))
        self._init_covariance(n)
        self.register_buffer("_opacity", torch.ones((n, 1)))
        self._features_dc = nn.Parameter(torch.rand(n, 3) / coverage)
        self.register_buffer("background", torch.ones(3))

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

    def forward(self, backend=None):
        """Render to a (1, 3, H, W) image in [0, 1]. `backend` overrides self.backend for this call."""
        backend = backend or self.backend
        if backend not in RENDERERS:
            raise ValueError("backend must be 'cuda' or 'torch'")
        out = RENDERERS[backend](
            self.get_xyz_pixels, self.get_cov(), self.get_features, self.H, self.W,
            k_sigma=self.k_sigma, min_radius=self.min_radius, max_radius=self.max_radius)
        out = torch.clamp(out, 0, 1)
        return {"render": out.permute(2, 0, 1).unsqueeze(0).contiguous()}


class GaussianImage_Cholesky(_GaussianImageBase):
    """2D Gaussians with a Cholesky covariance factor."""

    def _init_covariance(self, n):
        # (l1, l2, l3); `cholesky_bound` adds 0.5 to l1, l3.
        self.register_buffer("cholesky_bound", torch.tensor([0.5, 0, 0.5]).view(1, 3))
        raw = torch.rand(n, 3)
        if self.init_sigma is not None:   # l = (rand + bound) * sigma instead of rand + bound
            raw = (raw + self.cholesky_bound) * self.init_sigma - self.cholesky_bound
        self._cholesky = nn.Parameter(raw)

    @property
    def get_cholesky_elements(self):
        """(l1, l2, l3) in pixels: raw + cholesky_bound."""
        return self._cholesky + self.cholesky_bound

    def get_cov(self):
        return cholesky_to_cov(self.get_cholesky_elements)


class GaussianImage_RS(_GaussianImageBase):
    """2D Gaussians with a rotation + scaling covariance."""

    def _init_covariance(self, n):
        # Scales (s1, s2) and angle theta; `scaling_bound` adds 0.5 to s1, s2.
        self.register_buffer("scaling_bound", torch.tensor([0.5, 0.5]).view(1, 2))
        raw = torch.rand(n, 2)
        if self.init_sigma is not None:
            raw = (raw + self.scaling_bound) * self.init_sigma - self.scaling_bound
        self._scaling = nn.Parameter(raw)
        self._rotation = nn.Parameter(torch.rand(n, 1))

    @property
    def get_scaling(self):
        """Scales in pixels: |raw + 0.5|."""
        return torch.abs(self._scaling + self.scaling_bound)

    @property
    def get_rotation(self):
        """Angle in radians, in (0, 2*pi) via sigmoid."""
        return torch.sigmoid(self._rotation) * 2 * math.pi

    def get_cov(self):
        return scale_rot_to_cov(self.get_scaling, self.get_rotation)


MODELS = {"Cholesky": GaussianImage_Cholesky, "RS": GaussianImage_RS}


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
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def check_against_torch(N=2000, H=256, W=256, seed=0, device="cuda"):
    """Random Gaussians of mixed sizes: compare CUDA vs torch image and gradients."""
    g = torch.Generator().manual_seed(seed)
    centers0 = torch.rand(N, 2, generator=g) * torch.tensor([W - 1.0, H - 1.0])
    l = torch.rand(N, 3, generator=g) * torch.tensor([6.0, 4.0, 6.0]) + torch.tensor([0.3, -2.0, 0.3])
    cov0 = cholesky_to_cov(l)
    colors0 = torch.rand(N, 3, generator=g)
    grad_img = torch.randn(H, W, 3, generator=g).to(device)

    res = {}
    for name, fn in RENDERERS.items():
        centers = centers0.clone().to(device).requires_grad_()
        cov = cov0.clone().to(device).requires_grad_()
        colors = colors0.clone().to(device).requires_grad_()
        img = fn(centers, cov, colors, H, W)
        (img * grad_img).sum().backward()
        res[name] = (img.detach(), centers.grad, cov.grad, colors.grad)

    ok = True
    for name, t, c in zip(("image", "d_centers", "d_cov", "d_colors"), res["torch"], res["cuda"]):
        err = (t - c).abs().max().item()
        rel = err / (t.abs().max().item() + 1e-12)
        ok &= rel < 1e-3
        print(f"{name:10s} max abs err {err:.3e}   relative {rel:.3e}")
    print("MATCH" if ok else "MISMATCH -- check the build")
    return ok