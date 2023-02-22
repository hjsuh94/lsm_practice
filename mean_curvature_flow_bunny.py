import os 

import torch 
import matplotlib.pyplot as plt
import cv2
import skfmm
from conv import *
from shape_generator import *
from tqdm import tqdm

# 1. Hyperparameters.
num_iters = 500
sigma = 2.0
ksize_gaussian = 7
ksize_laplace = 31
h = 0.2
rho_al = 0.001

# 2. Initialize grid. 
n_grid = 128
grid = cv2.imread("bunny_proc.png")[:,:,0]
grid = torch.Tensor(grid)
grid = grid.T
grid = torch.flip(grid, dims=(1,))
plt.figure()
plt.imshow(grid.T, cmap='gray', origin='lower')
plt.show()

grid[grid == 255] = 1.0
grid[grid == 0] = -1.0
plt.figure()
plt.imshow(grid.T, cmap='gray', origin='lower')
plt.show()

# 3. Compute signed distance and initialize Lagrangian
phi_0 = torch.Tensor(skfmm.distance(grid))
lambda_0 = 0.0
phi_t_arr = torch.zeros((phi_0.shape[0], phi_0.shape[0], num_iters))
phi_t_arr[:,:,0] = phi_0
lambda_arr = torch.zeros(num_iters)
lambda_arr[0] = lambda_0
V_0 = compute_volume(phi_0)

# 4. Iterate and optimize
for iter in tqdm(range(num_iters-1)):
    phi_t = phi_t_arr[:,:,iter]
    lambda_t = lambda_arr[iter]

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

    # 3.4 Compute volume of current phi_t to compute Lagrangian.
    V_t = compute_volume(phi_t)
    dL = (phi_t_curvature + lambda_t + rho_al * (V_t - V_0))

    # 3.4 Update the signed distance field via advection
    phi_next = phi_t + h * dL * phi_t_grad_norm

    phi_t_arr[:,:,iter+1] = phi_next
    lambda_arr[iter+1] = lambda_t + rho_al * (compute_volume(phi_next) - V_0)

print(lambda_arr)

# 4. Plot the array
for iter in tqdm(range(num_iters-1)):
    plt.figure()
    phi_t_thres = get_interior_phi(phi_t_arr[:,:,iter], 1.0)
    plt.imshow(phi_t_thres.T, cmap='gray', origin='lower')
    plt.savefig("data_bunny/{:03d}.png".format(iter))
    plt.close()
