import os, time

import torch 
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
h = 0.1

# 2. Initialize grid. 
n_grid = 128
#grid = generate_ellipse(32, 16, n_grid)
grid = generate_star(32, 16, 8, n_grid)

# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid))
phi_t_arr = torch.zeros((phi_0.shape[0], phi_0.shape[0], num_iters))
phi_t_arr[:,:,0] = phi_0

t0 = time.time()
for iter in tqdm(range(num_iters-1)):
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

    # 3.4 Update the signed distance field via advection
    phi_next = phi_t + h * phi_t_curvature * phi_t_grad_norm
    phi_t_arr[:,:,iter+1] = phi_next

print("iteration took {:.5f} seconds".format(time.time() - t0))

# 4. Plot the array
for iter in tqdm(range(num_iters-1)):
    plt.figure()
    phi_t_thres = get_interior_phi(phi_t_arr[:,:,iter], 1.0)
    plt.imshow(phi_t_thres.T, cmap='gray', origin='lower')
    plt.savefig("data/{:03d}.png".format(iter))
    plt.close()
