"""
Self-contained GaussianImage fitting with a C++/CUDA rasterizer.

Everything needed is in this file plus gi_render.cu (same folder). No gsplat.

Models (covariance form):
  * GaussianImage_Cholesky   Sigma = L L^T
  * GaussianImage_RS         Sigma = (R S)(R S)^T
Kernels (any model, via kernel=...), with q = d^T Sigma^-1 d:
  * "gaussian"   exp(-q/2)
  * "beta"       (1 - q)^beta for q <= 1, beta = 4 e^b     (+1 parameter)
  * "gabor"      exp(-q/2) cos(2 pi f.d + phi)             (+3 parameters)

Each takes backend="cuda" (default when a GPU is present; the custom
forward/backward kernels in gi_render.cu, JIT-compiled with
torch.utils.cpp_extension) or backend="torch" (the pure-PyTorch renderer,
kept as the reference for checking and benchmarking).

init_scale=None follows the paper (Gaussians start at ~1 px); a float f starts them at ~f * spacing.

Helpers: loss_fn, image_to_tensor, load_extension, check_against_torch, make_combined_video, Densifier.

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

def _kernel_torch(kernel, power, dx, dy, extra):
    """Kernel value per pixel; power = -0.5 d^T Sigma^-1 d, shape (N, P, P)."""
    if kernel == "gaussian":
        return torch.exp(power.clamp(max=0.0))
    e = extra.view(extra.shape[0], extra.shape[1], 1, 1)
    if kernel == "beta":     # (1 - q)^beta, zero for q >= 1 (Beta Splatting, Eq. 3)
        q = (-2 * power).clamp_min(0.0)
        return (1 - q).clamp_min(1e-6) ** e[:, 0] * (q < 1)
    if kernel == "gabor":    # Gaussian envelope x cos(2 pi f.d + phi)
        phase = 2 * math.pi * (e[:, 0] * dx + e[:, 1] * dy) + e[:, 2]
        return torch.exp(power.clamp(max=0.0)) * torch.cos(phase)
    raise ValueError(f"unknown kernel {kernel!r}")


def render_torch(centers, cov, colors, H, W, k_sigma=3.0, min_radius=2, max_radius=None,
                 kernel="gaussian", extra=None):
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
    weight = _kernel_torch(kernel, power, dx, dy, extra) * in_bounds * within
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


KERNEL_IDS = {"gaussian": 0, "beta": 1, "gabor": 2}
KERNELS = list(KERNEL_IDS)   # ["gaussian", "beta", "gabor"]


class _RasterizeSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, conics, colors, extra, radii, H, W, kernel_id):
        centers = centers.contiguous().float()
        conics = conics.contiguous().float()
        colors = colors.contiguous().float()
        extra = extra.contiguous().float()
        radii = radii.contiguous().int()
        out = load_extension().forward(centers, conics, colors, extra, radii, H, W, kernel_id)
        ctx.save_for_backward(centers, conics, colors, extra, radii)
        ctx.H, ctx.W, ctx.kernel_id = H, W, kernel_id
        return out

    @staticmethod
    def backward(ctx, grad_out):
        centers, conics, colors, extra, radii = ctx.saved_tensors
        d_centers, d_conics, d_colors, d_extra = load_extension().backward(
            centers, conics, colors, extra, radii, grad_out.contiguous().float(),
            ctx.H, ctx.W, ctx.kernel_id)
        return d_centers, d_conics, d_colors, d_extra, None, None, None, None


def render_cuda(centers, cov, colors, H, W, k_sigma=3.0, min_radius=2, max_radius=None,
                kernel="gaussian", extra=None):
    """Same inputs/output as render_torch, rasterized by the CUDA kernels (no sync)."""
    if max_radius is None:
        max_radius = max(H, W)
    if extra is None:
        extra = centers.new_zeros(centers.shape[0], 0)
    conics, radii = _conic_and_radius(cov, k_sigma, min_radius, max_radius)
    return _RasterizeSum.apply(centers, conics, colors, extra, radii.int(), H, W, KERNEL_IDS[kernel])


RENDERERS = {"torch": render_torch, "cuda": render_cuda}


# =============================================================================
# Models
# =============================================================================

BETA_SUPPORT = 2 * math.sqrt(2)   # (1 - r^2)^4 ~ Gaussian with sigma = support / 2.83


class _GaussianImageBase(nn.Module):
    """Shared parameters: position (tanh), weighted color, opacity fixed at 1, plus the
    kernel's own parameters. Subclasses add the covariance parameters and `get_cov`."""

    def __init__(self, **kwargs):
        super().__init__()
        n = self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.device = kwargs["device"]
        self.backend = kwargs.get("backend", "cuda" if torch.cuda.is_available() else "torch")  # default for model(); override per call
        if self.backend not in RENDERERS:
            raise ValueError("backend must be 'cuda' or 'torch'")
        self.kernel = kwargs.get("kernel", "gaussian")
        if self.kernel not in KERNEL_IDS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        # Render window: k-sigma cutoff plus radius floor / cap. Beta has support q <= 1, so 1 sigma.
        self.k_sigma = kwargs.get("k_sigma", 1.0 if self.kernel == "beta" else 3.0)
        self.min_radius = kwargs.get("min_radius", 2)
        self.max_radius = kwargs.get("max_radius", None)

        # None = paper (Gaussians start at ~1 px). A float f starts them at ~f * spacing,
        # spacing = sqrt(H*W/N), so they overlap from step 1 at any N.
        # Beta: Sigma marks the support edge, so start BETA_SUPPORT x larger to match the Gaussian footprint.
        init_scale = kwargs.get("init_scale", None)
        size_mult = BETA_SUPPORT if self.kernel == "beta" else 1.0
        base_sigma = 1.0 if init_scale is None else init_scale * math.sqrt(self.H * self.W / n)
        self.init_sigma = None if (init_scale is None and size_mult == 1.0) else base_sigma * size_mult

        # Overlapping Gaussians add up: scale initial colors down so the first render isn't clamped.
        coverage = 1.0 if init_scale is None else max(1.0, 2 * math.pi * init_scale ** 2)

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(n, 2) - 0.5)))
        self._init_covariance(n)
        self.register_buffer("_opacity", torch.ones((n, 1)))
        self._features_dc = nn.Parameter(torch.rand(n, 3) / coverage)
        self.register_buffer("background", torch.ones(3))
        self._init_kernel(n)   # after colors: same seed -> same centers/shapes/colors for every kernel

        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_kernel(self, n):
        """Kernel-specific per-primitive parameters (none for Gaussian)."""
        if self.kernel == "beta":
            self._beta_b = nn.Parameter(torch.zeros(n, 1))   # beta = 4 e^b; b = 0 is Gaussian-like
        elif self.kernel == "gabor":
            # f = 0.5 tanh(raw) cycles/px (Nyquist cap). Nonzero start: at f = 0, f and phi get no gradient.
            self._freq = nn.Parameter(0.05 * torch.randn(n, 2))
            self._phase = nn.Parameter(torch.zeros(n, 1))

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

    def get_extra(self):
        """Activated kernel parameters: None (gaussian), beta (N, 1), or (fx, fy, phi) (N, 3)."""
        if self.kernel == "beta":
            return 4 * torch.exp(self._beta_b)
        if self.kernel == "gabor":
            return torch.cat([0.5 * torch.tanh(self._freq), self._phase], dim=1)
        return None

    def forward(self, backend=None):
        """Render to a (1, 3, H, W) image in [0, 1]. `backend` overrides self.backend for this call."""
        backend = backend or self.backend
        if backend not in RENDERERS:
            raise ValueError("backend must be 'cuda' or 'torch'")
        out = RENDERERS[backend](
            self.get_xyz_pixels, self.get_cov(), self.get_features, self.H, self.W,
            k_sigma=self.k_sigma, min_radius=self.min_radius, max_radius=self.max_radius,
            kernel=self.kernel, extra=self.get_extra())
        out = torch.clamp(out, 0, 1)
        return {"render": out.permute(2, 0, 1).unsqueeze(0).contiguous()}


class GaussianImage_Cholesky(_GaussianImageBase):
    """2D primitives with a Cholesky covariance factor."""

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
    """2D primitives with a rotation + scaling covariance."""

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
# Densification (GaussianImage++, Li et al., AAAI 2026)
# =============================================================================

def _per_primitive_tensors(model):
    """Names of the parameters and buffers that have one row per primitive."""
    n = model._xyz.shape[0]
    params = [k for k, p in model.named_parameters() if p.dim() >= 1 and p.shape[0] == n]
    buffers = [k for k, b in model.named_buffers() if b.dim() >= 2 and b.shape[0] == n]
    return params, buffers


def _replace_rows(model, keep, new_rows):
    """Keep the rows selected by the bool mask `keep` and append `new_rows[name]` for every
    per-primitive parameter and buffer. Adam moments: kept rows keep theirs, new rows start at 0.
    model.optimizer and model.scheduler stay valid."""
    opt = model.optimizer
    params, buffers = _per_primitive_tensors(model)
    for name in params:
        old = getattr(model, name)
        add = new_rows[name].to(old.device, old.dtype)
        new = nn.Parameter(torch.cat([old.detach()[keep], add], dim=0))
        state = opt.state.pop(old, None)
        if state is not None:
            for key, v in state.items():
                if torch.is_tensor(v) and v.shape == old.shape:   # exp_avg, exp_avg_sq
                    state[key] = torch.cat([v[keep], torch.zeros_like(add)], dim=0)
            opt.state[new] = state
        for group in opt.param_groups:
            group["params"] = [new if p is old else p for p in group["params"]]
        setattr(model, name, new)
    for name in buffers:
        old = getattr(model, name)
        setattr(model, name, torch.cat([old[keep], new_rows[name].to(old.device, old.dtype)], dim=0))


class Densifier:
    """GaussianImage++-style densification for any model (Cholesky / RS) and kernel.

    Start the model with N0 primitives (GaussianImage++ uses N0 = M / 2); it grows to the budget M:
      * grow:  every `grow_every` steps (t < iterations) add k = ceil((M - N) / 2) primitives at the
               k pixels with the largest L1 error; the last growth step adds all that is left.
               Centers sit on those pixels; colors start at 0 (official code) or at the target
               color (color_init="target", as in the paper); covariance and kernel parameters are
               drawn like a fresh model's (paper initialization).
      * prune: every `prune_every` steps, drop primitives with non-finite values or a
               degenerate covariance.
    Call step() once per iteration, after optimizer.step() / zero_grad().
    """

    def __init__(self, model, max_points, iterations, grow_every=5000, prune_every=100,
                 color_init="zero"):
        self.model, self.M, self.T = model, max_points, iterations
        self.grow_every, self.prune_every = grow_every, prune_every
        self.color_init = color_init
        self.last_grow = (iterations - 1) // grow_every * grow_every   # last growth step < iterations
        self.history = [(0, self.num_points)]                          # (iteration, N) after each change

    @property
    def num_points(self):
        return self.model._xyz.shape[0]

    @torch.no_grad()
    def step(self, it, image, gt_image):
        changed = False
        if self.prune_every and it % self.prune_every == 0:
            changed |= self._prune()
        if self.grow_every and it % self.grow_every == 0 and it < self.T:
            changed |= self._grow(it, image, gt_image)
        if changed:
            self.history.append((it, self.num_points))

    def _grow(self, it, image, gt_image):
        m, n = self.model, self.num_points
        k = self.M - n if it == self.last_grow else math.ceil((self.M - n) / 2)
        if k <= 0:
            return False
        H, W = m.H, m.W

        # Top-k pixels of the per-pixel L1 error (Eq. 3).
        err = (image - gt_image).abs().sum(dim=1).view(-1)
        idx = torch.topk(err, k).indices
        ys, xs = idx // W, idx % W

        # A fresh model of the same class and kernel supplies covariance and kernel parameters.
        fresh = type(m)(num_points=k, H=H, W=W, device="cpu", lr=1e-3, backend="torch", kernel=m.kernel)
        rows = dict(fresh.named_parameters())
        rows.update(dict(fresh.named_buffers()))
        rows = {name: t.detach() for name, t in rows.items()}

        # Centers on the chosen pixels (inverse of get_xyz_pixels).
        px = torch.stack([xs, ys], dim=1).float()
        scale = torch.tensor([W - 1, H - 1], dtype=px.dtype, device=px.device)
        rows["_xyz"] = torch.atanh((2 * px / scale - 1).clamp(-1 + 1e-4, 1 - 1e-4))
        if self.color_init == "target":                     # paper, Eq. 4
            rows["_features_dc"] = gt_image[0, :, ys, xs].T
        else:                                               # official code: zeros
            rows["_features_dc"] = torch.zeros(k, 3)

        _replace_rows(m, torch.ones(n, dtype=torch.bool, device=m._xyz.device), rows)
        return True

    def _prune(self):
        m = self.model
        cov = m.get_cov()
        valid = (cov[:, 0] * cov[:, 2] - cov[:, 1] ** 2 > 0) & (cov[:, 0] > 0) & (cov[:, 2] > 0)
        params, buffers = _per_primitive_tensors(m)
        for name in params:
            valid &= torch.isfinite(getattr(m, name)).flatten(1).all(dim=1)
        if valid.all() or not valid.any():
            return False
        _replace_rows(m, valid, {name: getattr(m, name)[:0] for name in params + buffers})
        return True

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


def check_against_torch(N=2000, H=256, W=256, seed=0, device="cuda", kernel="gaussian"):
    """Random primitives of mixed sizes: compare CUDA vs torch image and gradients for one kernel."""
    g = torch.Generator().manual_seed(seed)
    centers0 = torch.rand(N, 2, generator=g) * torch.tensor([W - 1.0, H - 1.0])
    l = torch.rand(N, 3, generator=g) * torch.tensor([6.0, 4.0, 6.0]) + torch.tensor([0.3, -2.0, 0.3])
    cov0 = cholesky_to_cov(l)
    colors0 = torch.rand(N, 3, generator=g)
    grad_img = torch.randn(H, W, 3, generator=g).to(device)

    # Kernel-specific parameters, in the form the renderers take (beta; or fx, fy, phi).
    k_sigma = 1.0 if kernel == "beta" else 3.0
    if kernel == "beta":
        extra0 = 4 * torch.exp(0.3 * torch.randn(N, 1, generator=g))
    elif kernel == "gabor":
        extra0 = torch.cat([0.4 * torch.rand(N, 2, generator=g) - 0.2,
                            2 * math.pi * torch.rand(N, 1, generator=g)], dim=1)
    else:
        extra0 = torch.zeros(N, 0)

    res = {}
    for name, fn in RENDERERS.items():
        centers = centers0.clone().to(device).requires_grad_()
        cov = cov0.clone().to(device).requires_grad_()
        colors = colors0.clone().to(device).requires_grad_()
        extra = extra0.clone().to(device).requires_grad_()
        img = fn(centers, cov, colors, H, W, k_sigma=k_sigma, kernel=kernel, extra=extra)
        (img * grad_img).sum().backward()
        d_extra = extra.grad if extra.grad is not None else torch.zeros_like(extra)
        res[name] = (img.detach(), centers.grad, cov.grad, colors.grad, d_extra)

    print(f"[{kernel}]")
    ok = True
    for name, t, c in zip(("image", "d_centers", "d_cov", "d_colors", "d_extra"), res["torch"], res["cuda"]):
        if t.numel() == 0:   # Gaussian has no extra parameters
            continue
        err = (t - c).abs().max().item()
        rel = err / (t.abs().max().item() + 1e-12)
        ok &= rel < 1e-3
        print(f"{name:10s} max abs err {err:.3e}   relative {rel:.3e}")
    print("MATCH" if ok else "MISMATCH -- check the build")
    return ok


def make_combined_video(histories, num_points_list, gt_image,
                        out_path="training_combined.mp4", fps=50, dpi=120, label="{} Gaussians"):
    """Side-by-side training video, one panel per run plus the ground truth.
    histories[key] needs "snapshots", "snapshot_iters" and "psnr"; all runs must share
    the snapshot schedule. `label` formats each key for its panel title."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    snap_iters = histories[num_points_list[0]]["snapshot_iters"]
    for n in num_points_list:
        assert histories[n]["snapshot_iters"] == snap_iters, \
            f"Run {label.format(n)} has a different snapshot schedule."

    gt_np = gt_image[0].permute(1, 2, 0).cpu().numpy()
    fig, axes = plt.subplots(1, len(num_points_list) + 1,
                             figsize=(3.2 * (len(num_points_list) + 1), 3.6))
    ims = [ax.imshow(histories[n]["snapshots"][0]) for ax, n in zip(axes, num_points_list)]
    titles = [ax.set_title("", fontsize=8) for ax in axes[:-1]]
    axes[-1].imshow(gt_np)
    axes[-1].set_title("Ground truth", fontsize=8)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    writer = FFMpegWriter(fps=fps, metadata={"title": "training_combined"}, bitrate=6000)
    with writer.saving(fig, out_path, dpi=dpi):
        for i, it in enumerate(snap_iters):
            for im, title, n in zip(ims, titles, num_points_list):
                im.set_data(histories[n]["snapshots"][i])
                title.set_text(f"{label.format(n)} | it {it} | {histories[n]['psnr'][it - 1]:.2f} dB")
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved combined video to {out_path}")