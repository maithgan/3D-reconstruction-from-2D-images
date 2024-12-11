# 3D-reconstruction-from-2D-images
We look into overviews of Gaussian Splatting implementation and also point cloud generation, depth estimation, intrinsic matrix calculation, depth anything and MiDASNet

# Implemetation #1 Gaussian Splatting 

## Overview
This project implements **3D Gaussian Splatting** based on the paper **"3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)** by Kerbl et al. The method projects **3D Gaussians** derived from point clouds for real-time rendering, blending speed and fidelity in applications like neural rendering.

## Features
- **Efficient Rendering**: Real-time rendering using a tile-based rasterizer for fast projection of 3D Gaussians onto 2D views.
- **Optimization**:
  - Supports iterative refinement of Gaussian parameters (position, covariance, SH coefficients).
  - Includes a differentiable rasterization process for gradient-based updates.
- **Metrics**:
  - Computes PSNR (Peak Signal-to-Noise Ratio) to evaluate image quality.
  - Visualizes rendered results compared to ground truth.
## PSNR Evaluation
The quality of the rendered images is evaluated using the **PSNR (Peak Signal-to-Noise Ratio)** metric, which compares the rendered output to the ground truth.

### PSNR Formula
\[
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)
\]
Where:
- **MAX**: The maximum possible pixel value (e.g., 255 for 8-bit images).
- **MSE**: The mean squared error between the rendered and ground truth images.

### Steps to Compute PSNR
1. Render the 2D image using Gaussian Splatting.
2. Compare the rendered image (`I_rendered`) with the ground truth (`I_ground_truth`).
3. Run the following script:
   ```bash
   python evaluate.py --rendered ./output/rendered_image.png --ground_truth ./data/ground_truth.png
