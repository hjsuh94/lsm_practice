import torch 
import matplotlib.pyplot as plt
import cv2
import skfmm
from conv import *
from shape_generator import *
from tqdm import tqdm

# 0. Hyperparameters.
num_iters = 200
sigma = 2.0
ksize_gaussian = 7
ksize_laplace = 31
h = 0.1
n_grid = 128

# 1. Initialize grid. 
grid = torch.ones((n_grid, n_grid))

# 2. Initialize signed distance function
for i in range(n_grid):
    for j in range(n_grid):
        if ((i - 64) / 32) ** 2 + ((j- 64) / 16) ** 2 <= 1:
            grid[i,j] = -1

# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid))

sigma = 2.0
kernel = log_kernel(sigma, 31)
laplace_image = convolve(grid, kernel)
curvature_image = convolve(phi_0, kernel)

#phi_smooth = cv2.GaussianBlur(phi.numpy(), (11,11), 0)
#curvature_image = cv2.Laplacian(phi_smooth, ddepth=cv2.CV_64F, ksize=9)

kernel = gaussian_kernel(sigma, 7)
gaussian_image = convolve(grid, kernel)

kernel_x = sobel_kernel(direction='x')
kernel_y = sobel_kernel(direction='y')
sobel_image_x = convolve(gaussian_image, kernel_x)
sobel_image_y = convolve(gaussian_image, kernel_y)
# MASK
sobel_image = torch.sqrt(sobel_image_x ** 2.0 + sobel_image_y ** 2.0)

smooth_phi = convolve(phi_0, gaussian_kernel(sigma, 7))
grad_phi_x = convolve(smooth_phi, kernel_x) 
grad_phi_y = convolve(smooth_phi, kernel_y) 
grad_phi_norm = torch.sqrt(grad_phi_x ** 2.0 + grad_phi_y ** 2.0)
curvature_image_masked = sobel_image * curvature_image

# Compute curvature-dependent velocity field.
grad_phi = torch.stack((grad_phi_x, grad_phi_y), dim=2)
velocity = curvature_image * grad_phi_norm

plt.figure()
plt.subplot(1,3,1)
plt.imshow(grad_phi_x.T, cmap='gray', origin='lower')
plt.subplot(1,3,2)
plt.imshow(grad_phi_y.T, cmap='gray', origin='lower')
plt.subplot(1,3,3)
plt.imshow(grad_phi_norm.T, cmap='gray', origin='lower')

"""
for i in range(n_grid):
    for j in range(n_grid):
        if sobel_image[i,j] >= 0.1:
            plt.arrow(i, j, 0.1 * grad_phi_y[i,j], 0.1 * grad_phi_x[i,j],
                head_width=1)
plt.show()
"""

plt.figure()
plt.subplot(2,3,1)
plt.imshow(grid.T, cmap='gray', origin='lower')

plt.subplot(2,3,2)
plt.imshow(phi_0.T, cmap='gray', origin='lower')

plt.subplot(2,3,3)
plt.imshow(gaussian_image.T, cmap='gray', origin='lower')

plt.subplot(2,3,4)
plt.imshow(sobel_image_x.T, cmap='gray', origin='lower')

plt.subplot(2,3,5)
plt.imshow(sobel_image_y.T, cmap='gray', origin='lower')
plt.subplot(2,3,6)
plt.imshow(curvature_image_masked.T, cmap='gray', origin='lower')
plt.show()
