// CUDA forward + backward for GaussianImage's accumulated-sum rasterizer,
// with three per-primitive kernels: Gaussian, Beta and Gabor.
//
// Same math as render_torch:
//   out[p] = sum_n  w_n(p) * color_n,   power = -0.5 * (A dx^2 + 2 B dx dy + C dy^2),
//   dx = px - cx, dy = py - cy, q = -2 * power = d^T Sigma^-1 d,
// over pixels p inside the image with dx^2 + dy^2 <= r_n^2, where w_n(p) is
//   Gaussian (0): exp(min(power, 0))
//   Beta     (1): (1 - q)^beta for q < 1, else 0              extra = (beta)
//   Gabor    (2): exp(min(power, 0)) * cos(2 pi (fx dx + fy dy) + phi)   extra = (fx, fy, phi)
// conic (A, B, C) = Sigma^-1, integer radius r_n and the activated extra
// parameters are computed in PyTorch.
//
// One CUDA block per primitive; its threads stride over the primitive's own
// bounding box. Forward scatters with atomicAdd. Backward reduces each
// primitive's gradient inside its block, so no atomics are needed there.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

constexpr int THREADS = 128;
constexpr int MAX_EXTRA = 3;                 // Gabor: fx, fy, phi
constexpr float TWO_PI = 6.283185307179586f;

#define CHECK_INPUT(x, dt)                                                   \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                 \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");              \
    TORCH_CHECK((x).scalar_type() == (dt), #x " has the wrong dtype")

// Clipped bounding box of primitive n. Returns false if it is empty or invalid.
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

// Kernel value w at offset (dx, dy) and the partials the backward pass needs:
//   dw_dpow        dw / d power
//   dw_ddx, dw_ddy direct dependence on dx, dy (Gabor phase only)
//   dw_de[k]       dw / d extra[k]
// KERNEL: 0 = Gaussian, 1 = Beta (e = beta), 2 = Gabor (e = fx, fy, phi).
template <int KERNEL>
__device__ __forceinline__ float eval_kernel(
    float power, float dx, float dy, const float* e,
    float& dw_dpow, float& dw_ddx, float& dw_ddy, float* dw_de)
{
    dw_ddx = dw_ddy = 0.f;
    #pragma unroll
    for (int k = 0; k < MAX_EXTRA; ++k) dw_de[k] = 0.f;

    if (KERNEL == 1) {                       // Beta: (1 - q)^beta, zero for q >= 1
        const float q = fmaxf(-2.f * power, 0.f);
        if (q >= 1.f) { dw_dpow = 0.f; return 0.f; }
        const float base = 1.f - q, beta = e[0];
        const float w = powf(base, beta);
        dw_dpow = power > 0.f ? 0.f : 2.f * beta * powf(base, beta - 1.f);
        dw_de[0] = w * logf(base);
        return w;
    }

    const float g = power > 0.f ? 1.f : expf(power);   // Gaussian (envelope for Gabor)
    if (KERNEL == 2) {                       // Gabor: g * cos(2 pi (fx dx + fy dy) + phi)
        float s, c;
        sincosf(TWO_PI * (e[0] * dx + e[1] * dy) + e[2], &s, &c);
        const float dw_dph = -g * s;
        dw_dpow = power > 0.f ? 0.f : g * c;
        dw_ddx = dw_dph * TWO_PI * e[0];
        dw_ddy = dw_dph * TWO_PI * e[1];
        dw_de[0] = dw_dph * TWO_PI * dx;
        dw_de[1] = dw_dph * TWO_PI * dy;
        dw_de[2] = dw_dph;
        return g * c;
    }

    dw_dpow = power > 0.f ? 0.f : g;         // Gaussian: clamp(max=0) passes gradient only here
    return g;
}

template <int KERNEL>
__global__ void rasterize_sum_forward_kernel(
    int N, int H, int W, int E,
    const float* __restrict__ centers,   // (N, 2)
    const float* __restrict__ conics,    // (N, 3)
    const float* __restrict__ colors,    // (N, 3)
    const float* __restrict__ extra,     // (N, E)
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
    float e[MAX_EXTRA] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < MAX_EXTRA; ++k) if (k < E) e[k] = extra[E * n + k];
    float dwp, dwx, dwy, dwe[MAX_EXTRA];  // partials, unused in forward

    const float r2 = (float)r * (float)r;
    const int total = bw * bh;

    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int px = x0 + t % bw;
        const int py = y0 + t / bw;
        const float dx = (float)px - cx;
        const float dy = (float)py - cy;
        if (dx * dx + dy * dy > r2) continue;

        const float power = -0.5f * (A * dx * dx + 2.f * B * dx * dy + C * dy * dy);
        const float w = eval_kernel<KERNEL>(power, dx, dy, e, dwp, dwx, dwy, dwe);

        float* o = out + 3 * ((long long)py * W + px);
        atomicAdd(o + 0, w * c0);
        atomicAdd(o + 1, w * c1);
        atomicAdd(o + 2, w * c2);
    }
}

template <int KERNEL>
__global__ void rasterize_sum_backward_kernel(
    int N, int H, int W, int E,
    const float* __restrict__ centers,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ extra,     // (N, E)
    const int*   __restrict__ radii,
    const float* __restrict__ grad_out,  // (H, W, 3)
    float*       __restrict__ d_centers, // (N, 2)
    float*       __restrict__ d_conics,  // (N, 3)
    float*       __restrict__ d_colors,  // (N, 3)
    float*       __restrict__ d_extra)   // (N, E)
{
    const int n = blockIdx.x;
    if (n >= N) return;  // uniform per block

    const float cx = centers[2 * n], cy = centers[2 * n + 1];
    const int r = radii[n];
    int x0, y0, bw, bh;
    if (!gaussian_bbox(cx, cy, r, H, W, x0, y0, bw, bh)) return;  // grads stay 0

    const float A = conics[3 * n], B = conics[3 * n + 1], C = conics[3 * n + 2];
    const float c0 = colors[3 * n], c1 = colors[3 * n + 1], c2 = colors[3 * n + 2];
    float e[MAX_EXTRA] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < MAX_EXTRA; ++k) if (k < E) e[k] = extra[E * n + k];

    const float r2 = (float)r * (float)r;
    const int total = bw * bh;

    // acc: d_cx, d_cy, d_A, d_B, d_C, d_c0, d_c1, d_c2, d_e0, d_e1, d_e2
    constexpr int NACC = 8 + MAX_EXTRA;
    float acc[NACC];
    #pragma unroll
    for (int k = 0; k < NACC; ++k) acc[k] = 0.f;

    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int px = x0 + t % bw;
        const int py = y0 + t / bw;
        const float dx = (float)px - cx;
        const float dy = (float)py - cy;
        if (dx * dx + dy * dy > r2) continue;

        const float* g = grad_out + 3 * ((long long)py * W + px);
        const float g0 = g[0], g1 = g[1], g2 = g[2];

        const float power = -0.5f * (A * dx * dx + 2.f * B * dx * dy + C * dy * dy);
        float dw_dpow, dw_ddx, dw_ddy, dw_de[MAX_EXTRA];
        const float w = eval_kernel<KERNEL>(power, dx, dy, e, dw_dpow, dw_ddx, dw_ddy, dw_de);

        acc[5] += w * g0;
        acc[6] += w * g1;
        acc[7] += w * g2;

        const float gc = g0 * c0 + g1 * c1 + g2 * c2;   // dL/dw
        const float dpow = gc * dw_dpow;                // dL/dpower
        acc[0] += dpow * (A * dx + B * dy) - gc * dw_ddx;   // d/dcx (dx = px - cx)
        acc[1] += dpow * (B * dx + C * dy) - gc * dw_ddy;   // d/dcy
        acc[2] += dpow * (-0.5f * dx * dx);
        acc[3] += dpow * (-dx * dy);
        acc[4] += dpow * (-0.5f * dy * dy);
        #pragma unroll
        for (int k = 0; k < MAX_EXTRA; ++k) acc[8 + k] += gc * dw_de[k];
    }

    // Block reduction of the accumulators.
    __shared__ float sh[NACC][THREADS];
    const int tid = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < NACC; ++k) sh[k][tid] = acc[k];
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int k = 0; k < NACC; ++k) sh[k][tid] += sh[k][tid + s];
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
        for (int k = 0; k < E; ++k) d_extra[E * n + k] = sh[8 + k][0];
    }
}

// extra must be (N, E) with E = 0 / 1 / 3 for Gaussian / Beta / Gabor (any E allowed for Gaussian).
static void check_extra(const torch::Tensor& extra, const torch::Tensor& centers, int64_t kernel)
{
    CHECK_INPUT(extra, torch::kFloat32);
    const int64_t need[3] = {0, 1, 3};
    TORCH_CHECK(kernel >= 0 && kernel < 3, "kernel must be 0 (gaussian), 1 (beta) or 2 (gabor)");
    TORCH_CHECK(extra.dim() == 2 && extra.size(0) == centers.size(0)
                && extra.size(1) <= MAX_EXTRA
                && (kernel == 0 || extra.size(1) == need[kernel]),
                "extra has the wrong shape for this kernel");
}

torch::Tensor rasterize_sum_forward(
    torch::Tensor centers, torch::Tensor conics, torch::Tensor colors, torch::Tensor extra,
    torch::Tensor radii, int64_t H, int64_t W, int64_t kernel)
{
    CHECK_INPUT(centers, torch::kFloat32);
    CHECK_INPUT(conics, torch::kFloat32);
    CHECK_INPUT(colors, torch::kFloat32);
    CHECK_INPUT(radii, torch::kInt32);
    check_extra(extra, centers, kernel);
    const at::cuda::OptionalCUDAGuard guard(device_of(centers));

    const int N = (int)centers.size(0);
    const int E = (int)extra.size(1);
    auto out = torch::zeros({H, W, 3}, centers.options());
    if (N == 0) return out;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_FWD(K)                                                              \
    rasterize_sum_forward_kernel<K><<<N, THREADS, 0, stream>>>(                    \
        N, (int)H, (int)W, E,                                                      \
        centers.data_ptr<float>(), conics.data_ptr<float>(),                       \
        colors.data_ptr<float>(), extra.data_ptr<float>(), radii.data_ptr<int>(),  \
        out.data_ptr<float>())

    if (kernel == 0)      LAUNCH_FWD(0);
    else if (kernel == 1) LAUNCH_FWD(1);
    else                  LAUNCH_FWD(2);
#undef LAUNCH_FWD

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> rasterize_sum_backward(
    torch::Tensor centers, torch::Tensor conics, torch::Tensor colors, torch::Tensor extra,
    torch::Tensor radii, torch::Tensor grad_out, int64_t H, int64_t W, int64_t kernel)
{
    CHECK_INPUT(centers, torch::kFloat32);
    CHECK_INPUT(conics, torch::kFloat32);
    CHECK_INPUT(colors, torch::kFloat32);
    CHECK_INPUT(radii, torch::kInt32);
    CHECK_INPUT(grad_out, torch::kFloat32);
    check_extra(extra, centers, kernel);
    const at::cuda::OptionalCUDAGuard guard(device_of(centers));

    const int N = (int)centers.size(0);
    const int E = (int)extra.size(1);
    auto d_centers = torch::zeros_like(centers);
    auto d_conics = torch::zeros_like(conics);
    auto d_colors = torch::zeros_like(colors);
    auto d_extra = torch::zeros_like(extra);
    if (N == 0) return {d_centers, d_conics, d_colors, d_extra};
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_BWD(K)                                                              \
    rasterize_sum_backward_kernel<K><<<N, THREADS, 0, stream>>>(                   \
        N, (int)H, (int)W, E,                                                      \
        centers.data_ptr<float>(), conics.data_ptr<float>(),                       \
        colors.data_ptr<float>(), extra.data_ptr<float>(), radii.data_ptr<int>(),  \
        grad_out.data_ptr<float>(),                                                \
        d_centers.data_ptr<float>(), d_conics.data_ptr<float>(),                   \
        d_colors.data_ptr<float>(), d_extra.data_ptr<float>())

    if (kernel == 0)      LAUNCH_BWD(0);
    else if (kernel == 1) LAUNCH_BWD(1);
    else                  LAUNCH_BWD(2);
#undef LAUNCH_BWD

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_centers, d_conics, d_colors, d_extra};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterize_sum_forward,
          "GaussianImage sum rasterizer, Gaussian/Beta/Gabor kernels (forward, CUDA)");
    m.def("backward", &rasterize_sum_backward,
          "GaussianImage sum rasterizer, Gaussian/Beta/Gabor kernels (backward, CUDA)");
}