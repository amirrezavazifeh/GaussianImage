# GaussianImage: Fitting any image with 2D Gaussians

A simplified, fitting-only reimplementation of
[GaussianImage](https://github.com/Xinjie-Q/GaussianImage) (ECCV 2024), used to
make visualizations of an image being represented by 2D Gaussians.

## Repository structure

```
GaussianImage/
├── Code/
│   ├── gaussianimage_gsplat/     # fitting with the authors' gsplat CUDA backend
│   └── gaussianimage_custom/     # no gsplat: pure-PyTorch and custom C++/CUDA renderers
│       ├── gaussianimage.py      # models (Cholesky, RS), both renderers, helpers
│       ├── gi_render.cu          # CUDA forward + backward kernels
│       └── fit_image.ipynb       # fit an image, plot results, make videos
├── Videos/                       # training-progress videos
└── README.md
```

## Quick start

1. Open `Code/gaussianimage_custom/fit_image.ipynb`.
2. Set `BACKEND = "cuda"` (needs a GPU and `nvcc`) or `"torch"` (runs anywhere).
3. Run all cells.

## Videos

Training-progress videos are in [`Videos/`](Videos/).

## Citation

```bibtex
@inproceedings{zhang2024gaussianimage,
  title     = {GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting},
  author    = {Zhang, Xinjie and Ge, Xingtong and Xu, Tongda and He, Dailan and Wang, Yan and Qin, Hongwei and Lu, Guo and Geng, Jing and Zhang, Jun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
