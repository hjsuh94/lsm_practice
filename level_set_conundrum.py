import os, time

import torch 
import matplotlib.pyplot as plt
import cv2
import skfmm
from conv import *
from shape_generator import *
from tqdm import tqdm

num_iters = 100
n_grid = 128
sigma = 3.0
ksize_gaussian = 7
ksize_laplace = 15
h = 0.1

grid = generate_ellipse(30, 30, n_grid)
plot_image(grid)

phi_0 = torch.Tensor(skfmm.distance(grid)).cuda()
phi_t_arr = torch.zeros((phi_0.shape[0], phi_0.shape[0], num_iters)).cuda()
phi_t_arr[:,:,0] = phi_0


for iter in tqdm(range(num_iters-1)):
    phi_t = phi_t_arr[:,:,iter]

    # 3.1. Gaussian smooth the signed distance field.
    kernel_g = gaussian_kernel(sigma, ksize_gaussian).cuda()
    phi_t_smooth = convolve(phi_t, kernel_g)

    # 3.2. Compute the curvature field.
    kernel_log = log_kernel(sigma, ksize_laplace).cuda()
    phi_t_curvature = convolve(phi_t, kernel_log)

    # 3.3. Compute the boundary element
    kernel_log = log_kernel(2.0, ksize_laplace).cuda()
    boundary = convolve(get_interior_phi(phi_t, 0.0), kernel_log)

    # 3.4. Compute the gradient field.
    kernel_x = sobel_kernel(direction='x').cuda()
    kernel_y = sobel_kernel(direction='y').cuda()
    phi_t_grad_x = convolve(phi_t_smooth, kernel_x)
    phi_t_grad_y = convolve(phi_t_smooth, kernel_y)
    phi_t_grad = torch.stack((phi_t_grad_x, phi_t_grad_y), dim=2)
    phi_t_grad_norm = torch.norm(phi_t_grad, dim=2)
    phi_t_grad_normalized = torch.norm(phi_t_grad, dim=2)

    # 3.5 Describe the velocity field.
    V = torch.zeros((n_grid, n_grid, 2)).cuda()
    V[:,:,0] = 1.0
    V[:,:,1] = 0.0

    phi_next = phi_t + h * torch.einsum('ijk,ijk->ij', V, phi_t_grad)
    phi_t_arr[:,:,iter+1] = phi_next    

# 4. Plot the array
image_count = 0
for iter in tqdm(range(0, num_iters-1, 10)):
    plot_image(get_interior_phi(phi_t_arr[:,:,iter].cpu(), 0.0), show=False)
    plt.savefig("data_con/{:03d}.png".format(image_count))
    plt.close()

    image_count += 1
