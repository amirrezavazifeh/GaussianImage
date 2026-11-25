import math

import torch
import torch.nn.functional as F

from gaussianimage import RENDERERS


# =============================================================================
# Pseudo-3D surfels: depth -> lift 2D primitives to front-facing 3D surfels ->
# re-project to new cameras with the same additive renderer.
# Camera convention: OpenCV (x right, y down, z forward); original camera at the origin.
# =============================================================================

def estimate_disparity(gt_image, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
    """(1, 3, H, W) image in [0, 1] -> (H, W) relative disparity in [0, 1], 1 = near.
    Same model and normalization as depth_estimator.ipynb."""
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    H, W = gt_image.shape[-2:]
    device = gt_image.device
    img = (gt_image[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
    processor = AutoImageProcessor.from_pretrained(model_id)
    net = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    inputs = processor(images=PILImage.fromarray(img), return_tensors="pt").to(device)
    with torch.no_grad():
        d = net(**inputs).predicted_depth.unsqueeze(1)
    d = F.interpolate(d, size=(H, W), mode="bicubic", align_corners=False)[0, 0]
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def disparity_to_depth(disp, z_near=1.0, z_far=3.0):
    """Relative disparity in [0, 1] -> metric-like depth in [z_near, z_far] (1/Z is linear in disp)."""
    return 1.0 / (disp / z_near + (1.0 - disp) / z_far)


def _sample_map(img, xy, H, W):
    """Bilinear sample of a (C, H, W) map at (N, 2) pixel coordinates -> (N, C)."""
    grid = torch.stack([xy[:, 0] / (W - 1), xy[:, 1] / (H - 1)], -1) * 2 - 1
    out = F.grid_sample(img[None], grid.view(1, -1, 1, 2), mode="bilinear", align_corners=True)
    return out[0, :, :, 0].T


@torch.no_grad()
def lift_to_surfels(model, depth, fov_deg=60.0, gt_image=None):
    """Lift a fitted 2D model to front-facing 3D surfels (2DGS parameterization):
    center p, orthonormal tangents t_u, t_v, scales s_u, s_v; seen from the original
    camera, each surfel projects back exactly to its 2D primitive."""
    H, W = model.H, model.W
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
    cx, cy = (W - 1) / 2, (H - 1) / 2

    mu = model.get_xyz_pixels                                     # (N, 2) pixels
    Z = _sample_map(depth[None].to(mu), mu, H, W)[:, 0]           # (N,)
    p = torch.stack([(mu[:, 0] - cx) * Z / f, (mu[:, 1] - cy) * Z / f, Z], -1)

    a, b, c = model.get_cov().unbind(-1)                          # Sigma = [[a, b], [b, c]]
    phi = 0.5 * torch.atan2(2 * b, a - c)                         # angle of the major axis
    mid, rad = 0.5 * (a + c), torch.sqrt((0.5 * (a - c)) ** 2 + b ** 2)
    sigma = torch.stack([mid + rad, (mid - rad).clamp_min(1e-8)], -1).sqrt()   # (N, 2) pixels
    cos, sin, zero = torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)
    t_u = torch.stack([cos, sin, zero], -1)                       # (e1, 0)
    t_v = torch.stack([-sin, cos, zero], -1)                      # (e2, 0); normal t_u x t_v = +z

    surf = dict(p=p, t_u=t_u, t_v=t_v, s=sigma * (Z / f)[:, None],
                colors=model.get_features.detach(), extra=model.get_extra(),
                kernel=model.kernel, k_sigma=model.k_sigma, min_radius=model.min_radius,
                max_radius=model.max_radius, f=f, cx=cx, cy=cy, H=H, W=W, fov_deg=fov_deg)
    surf["vis_colors"] = (_sample_map(gt_image[0], mu, H, W).clamp(0, 1) if gt_image is not None
                          else surf["colors"].clamp(0, 1))
    return surf


def look_at(eye, target, device="cuda"):
    """Camera-to-world rotation (columns = camera x, y, z axes) for a camera at `eye`
    looking at `target`, image y pointing down (OpenCV)."""
    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
    z = torch.as_tensor(target, dtype=torch.float32, device=device) - eye
    z = z / z.norm()
    x = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0], device=device), z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z], 1), eye


@torch.no_grad()
def project_surfels(surf, R_c2w, C):
    """Surfels -> 2D centers (N, 2), covariances (N, 3), view depth (N,), visibility mask,
    and kernel parameters for this view. Local affine (Jacobian) projection as in EWA/3DGS.
    `order` sorts the visible surfels near -> far (unused by the additive renderer)."""
    f, cx, cy = surf["f"], surf["cx"], surf["cy"]
    R_w2c = R_c2w.T
    X = (surf["p"] - C) @ R_w2c.T                                 # (N, 3) camera coordinates
    T = R_w2c @ torch.stack([surf["t_u"] * surf["s"][:, :1],      # (N, 3, 2) scaled tangents
                             surf["t_v"] * surf["s"][:, 1:]], -1)
    x, y, z = X.unbind(-1)
    visible = z > 1e-3
    z = z.clamp_min(1e-3)
    J = torch.zeros(X.shape[0], 2, 3, device=X.device, dtype=X.dtype)
    J[:, 0, 0], J[:, 0, 2] = f / z, -f * x / z ** 2
    J[:, 1, 1], J[:, 1, 2] = f / z, -f * y / z ** 2
    A = J @ T                                                     # (N, 2, 2) surfel uv -> pixels
    S = A @ A.transpose(1, 2)
    cov = torch.stack([S[:, 0, 0] + 1e-6, S[:, 0, 1], S[:, 1, 1] + 1e-6], -1)
    centers = torch.stack([f * x / z + cx, f * y / z + cy], -1)

    extra = surf["extra"]
    if surf["kernel"] == "gabor":   # carry the pixel-space frequency through the change of view
        A0 = _jacobian(surf["p"], surf["f"]) @ torch.stack([surf["t_u"] * surf["s"][:, :1],
                                                            surf["t_v"] * surf["s"][:, 1:]], -1)
        rhs = (A0.transpose(1, 2) @ extra[:, :2, None])[..., 0]          # solve A^T freq = rhs (2x2)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
        freq = torch.stack([A[:, 1, 1] * rhs[:, 0] - A[:, 1, 0] * rhs[:, 1],
                            -A[:, 0, 1] * rhs[:, 0] + A[:, 0, 0] * rhs[:, 1]], -1) / det[:, None]
        extra = torch.cat([freq, extra[:, 2:]], 1)
    order = torch.argsort(z[visible])
    return centers, cov, z, visible, extra, order


@torch.no_grad()
def render_surfels(surf, R_c2w, C, backend="cuda"):
    """Render surfels from a camera with the existing additive renderer.
    Returns RGB (H, W, 3) in [0, 1], depth (H, W) and coverage (H, W)."""
    centers, cov, z, vis, extra, _ = project_surfels(surf, R_c2w, C)
    H, W = surf["H"], surf["W"]
    kw = dict(k_sigma=surf["k_sigma"], min_radius=surf["min_radius"], max_radius=surf["max_radius"])
    render = RENDERERS[backend]
    ex = None if extra is None else extra[vis].contiguous()
    rgb = render(centers[vis].contiguous(), cov[vis].contiguous(), surf["colors"][vis].contiguous(),
                 H, W, kernel=surf["kernel"], extra=ex, **kw).clamp(0, 1)
    # Depth = sum Z_n w_n / sum w_n, via a 3-channel "color" (Z, 1, 0); Gaussian weights for Gabor.
    kd, exd = ("gaussian", None) if surf["kernel"] == "gabor" else (surf["kernel"], ex)
    zc = torch.stack([z, torch.ones_like(z), torch.zeros_like(z)], -1)[vis].contiguous()
    acc = render(centers[vis].contiguous(), cov[vis].contiguous(), zc, H, W, kernel=kd, extra=exd, **kw)
    coverage = acc[..., 1]
    depth = acc[..., 0] / coverage.clamp_min(1e-6)
    return rgb, depth, coverage


def orbit_cameras(pivot_depth, radius=0.1, n_frames=90, device="cuda"):
    """Cameras on a small circle around the original camera, all looking at a pivot
    at `pivot_depth` on the optical axis (pivot stays fixed; nearer/farther parallax)."""
    cams = []
    for t in torch.linspace(0, 2 * math.pi, n_frames + 1)[:-1].tolist():
        eye = (radius * math.cos(t), 0.5 * radius * math.sin(t), 0.0)
        cams.append(look_at(eye, (0.0, 0.0, pivot_depth), device=device))
    return cams


def make_novel_view_video(surf, cams, out_path="novel_views.mp4", backend="cuda",
                          fps=30, dpi=120, cmap="inferno"):
    """Video: novel-view RGB | rendered inverse depth (bright = near), one frame per camera."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    def frame(R, C):
        rgb, depth, cov = render_surfels(surf, R, C, backend)
        inv = torch.where(cov > 1e-3, 1.0 / depth.clamp_min(1e-6), torch.zeros_like(depth))
        return rgb.cpu().numpy(), inv.cpu().numpy()

    z = surf["p"][:, 2]
    vmin, vmax = 1.0 / z.max().item(), 1.0 / z.min().item()
    rgb, inv = frame(*cams[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    im_rgb = axes[0].imshow(rgb)
    im_dep = axes[1].imshow(inv, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Novel view (RGB)", fontsize=8)
    axes[1].set_title("Rendered inverse depth", fontsize=8)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    writer = FFMpegWriter(fps=fps, bitrate=6000)
    with writer.saving(fig, out_path, dpi=dpi):
        for R, C in cams:
            rgb, inv = frame(R, C)
            im_rgb.set_data(rgb)
            im_dep.set_data(inv)
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved novel-view video to {out_path}")


def plot_surfels_3d(surf, max_points=20000, size=2, seed=0):
    """Interactive plotly 3D scatter of surfel centers, colored by the image.
    Axes: X right, Z (depth) into the screen, -Y up."""
    import plotly.graph_objects as go

    p, col = surf["p"].cpu(), (surf["vis_colors"].cpu() * 255).round().int()
    idx = torch.randperm(p.shape[0], generator=torch.Generator().manual_seed(seed))[:max_points]
    p, col = p[idx], col[idx]
    fig = go.Figure(go.Scatter3d(
        x=p[:, 0], y=p[:, 2], z=-p[:, 1], mode="markers",
        marker=dict(size=size, color=[f"rgb({r},{g},{b})" for r, g, b in col.tolist()])))
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Z (depth)", zaxis_title="-Y",
                                 aspectmode="data",
                                 camera=dict(eye=dict(x=0.0, y=-2.0, z=0.4))),
                      margin=dict(l=0, r=0, t=0, b=0), height=600)
    return fig


def _jacobian(X, f):
    """(N, 3) camera-space points -> (N, 2, 3) pinhole projection Jacobian."""
    x, y, z = X.unbind(-1)
    J = torch.zeros(X.shape[0], 2, 3, device=X.device, dtype=X.dtype)
    J[:, 0, 0], J[:, 0, 2] = f / z, -f * x / z ** 2
    J[:, 1, 1], J[:, 1, 2] = f / z, -f * y / z ** 2
    return J


@torch.no_grad()
def tilt_surfels(surf, depth, smooth=7, max_tilt_deg=75.0):
    """Rotate each surfel into the local surface plane of the depth map.
    Normals come from the back-projected depth (point map); tangents are chosen so the
    original view is unchanged. Returns a new dict; `surf` is not modified."""
    H, W, f, cx, cy = surf["H"], surf["W"], surf["f"], surf["cx"], surf["cy"]
    dev = surf["p"].device

    # 1) Point map P(u, v) and normals n = dP/du x dP/dv (depth smoothed against noise).
    Z = depth.to(dev)[None, None]
    if smooth > 1:
        Z = F.avg_pool2d(F.pad(Z, [smooth // 2] * 4, mode="replicate"), smooth, stride=1)
    Z = Z[0, 0]
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=Z.dtype),
                          torch.arange(W, device=dev, dtype=Z.dtype), indexing="ij")
    P = torch.stack([(u - cx) * Z / f, (v - cy) * Z / f, Z], 0)          # (3, H, W)
    dPv, dPu = torch.gradient(P, dim=(1, 2))
    n_map = torch.linalg.cross(dPu, dPv, dim=0)
    n_map = n_map / n_map.norm(dim=0, keepdim=True).clamp_min(1e-12)

    # 2) Normal at each surfel center, facing the camera, tilt limited to max_tilt_deg.
    p = surf["p"]
    mu = torch.stack([f * p[:, 0] / p[:, 2] + cx, f * p[:, 1] / p[:, 2] + cy], -1)
    n = _sample_map(n_map, mu, H, W)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = p / p.norm(dim=-1, keepdim=True)                                 # unit view ray
    n = torch.where((n * r).sum(-1, keepdim=True) > 0, -n, n)
    cos_t = -(n * r).sum(-1, keepdim=True)
    cos_m = math.cos(math.radians(max_tilt_deg))
    n_perp = n + cos_t * r
    n_perp = n_perp / n_perp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n = torch.where(cos_t < cos_m, -r * cos_m + n_perp * math.sqrt(1 - cos_m ** 2), n)

    # 3) B = d(ray-plane intersection)/d(pixel): maps pixel offsets onto the tilted plane.
    ray = p / p[:, 2:]                                                   # (x/z, y/z, 1)
    n_r = (n * ray).sum(-1, keepdim=True)
    e_u = torch.tensor([1.0, 0.0, 0.0], device=dev) / f
    e_v = torch.tensor([0.0, 1.0, 0.0], device=dev) / f
    B = torch.stack([e_u - ray * (n[:, :1] / n_r) / f,
                     e_v - ray * (n[:, 1:2] / n_r) / f], -1) * p[:, 2:, None]   # (N, 3, 2)

    # 4) Pixel-space axes of each primitive (original view) -> 3D on the plane -> orthonormal.
    T0 = torch.stack([surf["t_u"] * surf["s"][:, :1], surf["t_v"] * surf["s"][:, 1:]], -1)
    A0 = _jacobian(p, f) @ T0                                            # (N, 2, 2) pixels
    T = B @ A0                                                           # (N, 3, 2) on the plane
    G = T.transpose(1, 2) @ T                                            # 2x2: closed-form SVD
    phi = 0.5 * torch.atan2(2 * G[:, 0, 1], G[:, 0, 0] - G[:, 1, 1])
    v1 = torch.stack([torch.cos(phi), torch.sin(phi)], -1)[..., None]
    v2 = torch.stack([-torch.sin(phi), torch.cos(phi)], -1)[..., None]
    a1, a2 = (T @ v1)[..., 0], (T @ v2)[..., 0]                          # orthogonal in-plane axes
    s1, s2 = a1.norm(dim=-1), a2.norm(dim=-1)
    t_u = a1 / s1[:, None].clamp_min(1e-12)
    t_v = torch.linalg.cross(n, t_u, dim=-1)
    out = dict(surf)
    out.update(t_u=t_u, t_v=t_v, s=torch.stack([s1, s2], -1), normal=n)
    return out