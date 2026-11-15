// CUDA forward + backward for GaussianImage's accumulated-sum rasterizer.
//
// Same math as my_utils2.render_gaussians_cov:
//   out[p] = sum_n  w_n(p) * color_n,   w_n(p) = exp(min(power, 0)),
//   power  = -0.5 * (A dx^2 + 2 B dx dy + C dy^2),  dx = px - cx, dy = py - cy,
// over pixels p inside the image with dx^2 + dy^2 <= r_n^2.
// conic (A, B, C) = Sigma^-1 and integer radius r_n are computed in PyTorch.
//
// One CUDA block per Gaussian; its threads stride over the Gaussian's own
// bounding box. Forward scatters with atomicAdd. Backward reduces each
// Gaussian's gradient inside its block, so no atomics are needed there.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

constexpr int THREADS = 128;

#define CHECK_INPUT(x, dt)                                                   \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                 \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");              \
    TORCH_CHECK((x).scalar_type() == (dt), #x " has the wrong dtype")

// Clipped bounding box of Gaussian n. Returns false if it is empty or invalid.
__device__ __forceinline__ bool gaussian_bbox(
    float cx, float cy, int r, int H, int W,
    int& x0, int& y0, int& bw, int& bh)
{
    if (!isfinite(cx) || !isfinite(cy) || r < 0) return false;
    x0 = max(0, (int)floorf(cx - r));
    int x1 = min(W - 1, (int)ceilf(cx + r));
    y0 = max(0, (int)floorf(cy - r));
    int y1 = min(H - 1, (int)ceilf(cy + r));
    if (x0 > x1 || y0 > y1) return false;
    bw = x1 - x0 + 1;
    bh = y1 - y0 + 1;
    return true;
}

__global__ void rasterize_sum_forward_kernel(
    int N, int H, int W,
    const float* __restrict__ centers,   // (N, 2)
    const float* __restrict__ conics,    // (N, 3)
    const float* __restrict__ colors,    // (N, 3)
    const int*   __restrict__ radii,     // (N,)
    float*       __restrict__ out)       // (H, W, 3)
{
    const int n = blockIdx.x;
    if (n >= N) return;

    const float cx = centers[2 * n], cy = centers[2 * n + 1];
    const int r = radii[n];
    int x0, y0, bw, bh;
    if (!gaussian_bbox(cx, cy, r, H, W, x0, y0, bw, bh)) return;

    const float A = conics[3 * n], B = conics[3 * n + 1], C = conics[3 * n + 2];
    const float c0 = colors[3 * n], c1 = colors[3 * n + 1], c2 = colors[3 * n + 2];
    const float r2 = (float)r * (float)r;
    const int total = bw * bh;

    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int px = x0 + t % bw;
        const int py = y0 + t / bw;
        const float dx = (float)px - cx;
        const float dy = (float)py - cy;
        if (dx * dx + dy * dy > r2) continue;

        const float power = -0.5f * (A * dx * dx + 2.f * B * dx * dy + C * dy * dy);
        const float w = power > 0.f ? 1.f : expf(power);

        float* o = out + 3 * ((long long)py * W + px);
        atomicAdd(o + 0, w * c0);
        atomicAdd(o + 1, w * c1);
        atomicAdd(o + 2, w * c2);
    }
}

__global__ void rasterize_sum_backward_kernel(
    int N, int H, int W,
    const float* __restrict__ centers,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const int*   __restrict__ radii,
    const float* __restrict__ grad_out,  // (H, W, 3)
    float*       __restrict__ d_centers, // (N, 2)
    float*       __restrict__ d_conics,  // (N, 3)
    float*       __restrict__ d_colors)  // (N, 3)
{
    const int n = blockIdx.x;
    if (n >= N) return;  // uniform per block

    const float cx = centers[2 * n], cy = centers[2 * n + 1];
    const int r = radii[n];
    int x0, y0, bw, bh;
    if (!gaussian_bbox(cx, cy, r, H, W, x0, y0, bw, bh)) return;  // grads stay 0

    const float A = conics[3 * n], B = conics[3 * n + 1], C = conics[3 * n + 2];
    const float c0 = colors[3 * n], c1 = colors[3 * n + 1], c2 = colors[3 * n + 2];
    const float r2 = (float)r * (float)r;
    const int total = bw * bh;

    // acc: d_cx, d_cy, d_A, d_B, d_C, d_c0, d_c1, d_c2
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int px = x0 + t % bw;
        const int py = y0 + t / bw;
        const float dx = (float)px - cx;
        const float dy = (float)py - cy;
        if (dx * dx + dy * dy > r2) continue;

        const float* g = grad_out + 3 * ((long long)py * W + px);
        const float g0 = g[0], g1 = g[1], g2 = g[2];

        const float power = -0.5f * (A * dx * dx + 2.f * B * dx * dy + C * dy * dy);
        const float w = power > 0.f ? 1.f : expf(power);

        acc[5] += w * g0;
        acc[6] += w * g1;
        acc[7] += w * g2;

        if (power <= 0.f) {  // clamp(max=0) passes gradient only here
            const float dpow = (g0 * c0 + g1 * c1 + g2 * c2) * w;
            acc[0] += dpow * (A * dx + B * dy);   // d power / d cx
            acc[1] += dpow * (B * dx + C * dy);   // d power / d cy
            acc[2] += dpow * (-0.5f * dx * dx);
            acc[3] += dpow * (-dx * dy);
            acc[4] += dpow * (-0.5f * dy * dy);
        }
    }

    // Block reduction of the 8 accumulators.
    __shared__ float sh[8][THREADS];
    const int tid = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < 8; ++k) sh[k][tid] = acc[k];
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) sh[k][tid] += sh[k][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_centers[2 * n]     = sh[0][0];
        d_centers[2 * n + 1] = sh[1][0];
        d_conics[3 * n]      = sh[2][0];
        d_conics[3 * n + 1]  = sh[3][0];
        d_conics[3 * n + 2]  = sh[4][0];
        d_colors[3 * n]      = sh[5][0];
        d_colors[3 * n + 1]  = sh[6][0];
        d_colors[3 * n + 2]  = sh[7][0];
    }
}

torch::Tensor rasterize_sum_forward(
    torch::Tensor centers, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor radii, int64_t H, int64_t W)
{
    CHECK_INPUT(centers, torch::kFloat32);
    CHECK_INPUT(conics, torch::kFloat32);
    CHECK_INPUT(colors, torch::kFloat32);
    CHECK_INPUT(radii, torch::kInt32);
    const at::cuda::OptionalCUDAGuard guard(device_of(centers));

    const int N = (int)centers.size(0);
    auto out = torch::zeros({H, W, 3}, centers.options());
    if (N == 0) return out;

    rasterize_sum_forward_kernel<<<N, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        N, (int)H, (int)W,
        centers.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), radii.data_ptr<int>(),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> rasterize_sum_backward(
    torch::Tensor centers, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor radii, torch::Tensor grad_out, int64_t H, int64_t W)
{
    CHECK_INPUT(centers, torch::kFloat32);
    CHECK_INPUT(conics, torch::kFloat32);
    CHECK_INPUT(colors, torch::kFloat32);
    CHECK_INPUT(radii, torch::kInt32);
    CHECK_INPUT(grad_out, torch::kFloat32);
    const at::cuda::OptionalCUDAGuard guard(device_of(centers));

    const int N = (int)centers.size(0);
    auto d_centers = torch::zeros_like(centers);
    auto d_conics = torch::zeros_like(conics);
    auto d_colors = torch::zeros_like(colors);
    if (N == 0) return {d_centers, d_conics, d_colors};

    rasterize_sum_backward_kernel<<<N, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        N, (int)H, (int)W,
        centers.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), radii.data_ptr<int>(),
        grad_out.data_ptr<float>(),
        d_centers.data_ptr<float>(), d_conics.data_ptr<float>(),
        d_colors.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_centers, d_conics, d_colors};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterize_sum_forward, "GaussianImage sum rasterizer (forward, CUDA)");
    m.def("backward", &rasterize_sum_backward, "GaussianImage sum rasterizer (backward, CUDA)");
}
