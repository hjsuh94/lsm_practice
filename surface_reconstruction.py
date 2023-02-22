import os, time

import torch 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skfmm
from conv import *
from shape_generator import *
from tqdm import tqdm

# 1. Hyperparameters.
num_iters = 500
sigma = 1.0
ksize_gaussian = 7
ksize_laplace = 15
h = 1.0
data_noise = 0.5
cost_weight_0 = 20.0
cost_schedule = np.linspace(cost_weight_0, 1.0, num_iters)

# 2. Initialize grid. 
n_grid = 128
grid = cv2.imread("bunny_proc.png")[:,:,0]
grid = torch.Tensor(grid)
grid = grid.T
grid = torch.flip(grid, dims=(1,))
grid[grid == 255] = 1.0
grid[grid == 0] = -1.0

# 3. Apply 
kernel_log = log_kernel(sigma, ksize_laplace)
boundary = convolve(grid, kernel_log)

points = []
for i in range(n_grid):
    for j in range(n_grid):
        if torch.abs(boundary[i,j]) > 0.1:
            points.append(np.array([i,j]) + np.random.normal(0, data_noise, size=(2)))
points = np.array(points)
points = torch.tensor(points)

"""
plt.figure()
#plt.imshow(boundary.T, cmap='gray', origin='lower')
plt.plot(points[:,0], points[:,1], 'ro', markersize=1)
plt.show()
"""

grid = generate_ellipse(32, 32, n_grid)
grid = grid.to(torch.float32)
# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid))
phi_t_arr = torch.zeros((phi_0.shape[0], phi_0.shape[0], num_iters))
phi_t_arr[:,:,0] = phi_0

t0 = time.time()
for iter in tqdm(range(num_iters-1)):
    cost_weight = cost_schedule[iter]
    phi_t = phi_t_arr[:,:,iter]

    # 3.1. Gaussian smooth the signed distance field.
    kernel_g = gaussian_kernel(sigma, ksize_gaussian)    
    phi_t_smooth = convolve(phi_t, kernel_g)

    # 3.2. Compute the curvature field.
    kernel_log = log_kernel(sigma, ksize_laplace)
    phi_t_curvature = convolve(phi_t, kernel_log)

    # 3.3. Compute the gradient field.
    kernel_x = sobel_kernel(direction='x')
    kernel_y = sobel_kernel(direction='y')
    phi_t_grad_x = convolve(phi_t_smooth, kernel_x)
    phi_t_grad_y = convolve(phi_t_smooth, kernel_y)
    phi_t_grad = torch.stack((phi_t_grad_x, phi_t_grad_y), dim=2)
    phi_t_grad_norm = torch.norm(phi_t_grad, dim=2)

    # 3.4 Compute the cost at each grid.
    pos_x, pos_y = torch.meshgrid(
        torch.range(0, n_grid-1), torch.range(0, n_grid-1))
    pos = torch.stack((pos_x, pos_y), dim=2)
    diff = pos[:, :, None,:] - points[None, None, :, :]
    diff_norm = torch.norm(diff, dim=3)
    diff_norm_squared = torch.pow(diff_norm, 2.0)
    dist = torch.exp(-diff_norm_squared / (cost_weight ** 2.0))
    dist_mean = torch.mean(dist, dim=2)

    # 3.5 Compute the gradient of the cost on each grid.
    dist_grad = -2.0 * diff * dist[:,:,:,None] / (cost_weight ** 2.0)
    dist_grad_mean = torch.mean(dist_grad, dim=2)
    normals = phi_t_grad / phi_t_grad_norm[:,:,None]
    normals = normals.to(torch.float64)
    dist_grad_norm = torch.einsum('ijk,ijk->ij', dist_grad_mean, normals)
    
    # 3.6 Compute the shape derivative.
    dphi = dist_grad_norm + dist_mean * phi_t_curvature

    # 3.4 Update the signed distance field via advection
    phi_next = phi_t + h * dphi * phi_t_grad_norm
    phi_t_arr[:,:,iter+1] = phi_next

print("iteration took {:.5f} seconds".format(time.time() - t0))

# 4. Plot the array
for iter in tqdm(range(num_iters-1)):
    plt.figure()
    phi_t_thres = get_interior_phi(phi_t_arr[:,:,iter], 1.0)
    plt.imshow(phi_t_thres.T, cmap='gray', origin='lower')
    plt.plot(points[:,0], points[:,1], 'ro', markersize=1)
    plt.savefig("surface/{:03d}.png".format(iter))
    plt.close()
