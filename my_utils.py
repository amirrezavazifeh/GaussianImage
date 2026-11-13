"""
Self-contained utilities for fitting GaussianImage (Cholesky variant) to an
arbitrary image, without cloning the full GaussianImage repository.

Adapted from https://github.com/Xinjie-Q/GaussianImage (utils.py and
gaussianimage_cholesky.py), with the VQ / bitstream-compression codec path
removed (only fitting a representation to an image is needed here, not
compression). Uses plain torch.optim.Adam instead of the paper's Adan
optimizer.

Still requires the compiled `gsplat` CUDA package to be importable.

Reference: Zhang, Ge et al., "GaussianImage: 1000 FPS Image Representation and
Compression by 2D Gaussian Splatting", ECCV 2024. https://arxiv.org/abs/2403.08551
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum


# =============================================================================
# Loss
# =============================================================================

def loss_fn(pred, target):
    """L2 (MSE) reconstruction loss between the rendered and target images."""
    return F.mse_loss(pred.float(), target.detach().float())


# =============================================================================
# Model: GaussianImage_Cholesky
# =============================================================================

class GaussianImage_Cholesky(nn.Module):
    """
    Represents an image as a fixed set of 2D Gaussians, each parameterized by
    a position, a Cholesky factor of its covariance, and a color. Rendering
    splats and sums these Gaussians onto a pixel grid (GaussianImage, ECCV 2024).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        # Position (Sec 3.1): unconstrained, mapped to (-1, 1) via tanh.
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        
        # Cholesky vector l = (l1, l2, l3); `cholesky_bound` keeps l1, l3 away from 0.
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer("_opacity", torch.ones((self.init_num_points, 1)))
        
        # Weighted color coefficients c' (Eq. 7): merges color + opacity, unconstrained.
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer("background", torch.ones(3))
        self.register_buffer("cholesky_bound", torch.tensor([0.5, 0, 0.5]).view(1, 3))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    @property
    def get_xyz(self):
        """Gaussian centers, mapped into (-1, 1) via tanh."""
        return torch.tanh(self._xyz)

    @property
    def get_features(self):
        """Weighted color coefficients (color merged with opacity)."""
        return self._features_dc

    @property
    def get_opacity(self):
        """Per-Gaussian opacity (fixed at 1, kept as a buffer for API compatibility)."""
        return self._opacity

    @property
    def get_cholesky_elements(self):
        """Cholesky covariance parameters, offset by `cholesky_bound`."""
        return self._cholesky + self.cholesky_bound
        
    def forward(self):
        """Projects all Gaussians and rasterizes them into an RGB image."""
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False,
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}


# =============================================================================
# Convenience helpers
# =============================================================================

def image_to_tensor(image, device="cuda"):
    """Converts a file path, PIL.Image, or numpy array into a [1, 3, H, W]
    float tensor in [0, 1], ready to use as the training target for
    GaussianImage_Cholesky."""
    from PIL import Image as PILImage

    if isinstance(image, (str, Path)):
        image = PILImage.open(image)
    if isinstance(image, PILImage.Image):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image)
    if image.ndim == 2:  # grayscale -> replicate to 3 channels
        image = np.stack([image, image, image], axis=-1)
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)



