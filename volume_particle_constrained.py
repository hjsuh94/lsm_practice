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
h = 0.01
rho_al_V = 0.1
rho_al_np = 0.1

# 2. Initialize grid. 
n_grid = 128
grid = generate_ellipse(32, 16, n_grid)
obstacle_grid = generate_ellipse_off_center(16, 16, [64, 85], n_grid)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(grid.T, cmap='gray', origin='lower')
plt.subplot(1,3,2)
plt.imshow(obstacle_grid.T, cmap='gray', origin='lower')
plt.subplot(1,3,3)
plt.imshow(grid.T + obstacle_grid.T, cmap='gray', origin='lower')
plt.show()

# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid))
phi_t_arr = torch.zeros((phi_0.shape[0], phi_0.shape[0], num_iters))
phi_t_arr[:,:,0] = phi_0

psi = torch.Tensor(skfmm.distance(obstacle_grid))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(phi_0.T, cmap='gray', origin='lower')
plt.subplot(1,2,2)
plt.imshow(psi.T, cmap='gray', origin='lower')
plt.show()

# 4. Initiate Dual field.
mu_0 = 0.0
mu_t_arr = torch.zeros(num_iters)
mu_t_arr[0] = mu_0
V_0 = compute_volume(phi_0)

lambda_0 = torch.zeros((n_grid, n_grid))
lambda_t_arr = torch.zeros((n_grid, n_grid, num_iters))
lambda_t_arr[:,:,0] = lambda_0

# Define zeta function
def get_zeta(psi, lambda_t, mu_al):
    zeta = torch.zeros((psi.shape[0], psi.shape[1]))
    zeta_neg = -lambda_t * psi + 0.5 * mu_al * torch.pow(psi, 2)
    zeta_pos = -1 / (2.0 * mu_al) * torch.pow(lambda_t, 2)
    zeta_neg_ind = (psi - (lambda_t / mu_al) <= 0)
    zeta_pos_ind = (psi - (lambda_t / mu_al) > 0)    
    zeta[zeta_neg_ind] = zeta_neg[zeta_neg_ind]
    zeta[zeta_pos_ind] = zeta_pos[zeta_pos_ind]
    return zeta

# 5. Iterate through augmented Lagrangain.
t0 = time.time()
for iter in tqdm(range(num_iters-1)):
    phi_t = phi_t_arr[:,:,iter]
    mu_t = mu_t_arr[iter]
    lambda_t = lambda_t_arr[:,:,iter]

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

    # 3.4 Compute the overlapping part of psi and phi
    penetration = torch.logical_and((psi < 0), (phi_t < 0))

    # 3.5 Compute the Lagrangian.
    zeta_t = get_zeta(psi, lambda_t, rho_al_np)
    V_t = compute_volume(phi_t)
    dL_t = phi_t_curvature + rho_al_V * (V_t - V_0) + zeta_t

    # 3.6 Update the signed distance field via advection
    phi_next = phi_t + h * dL_t * phi_t_grad_norm

    # 3.7 Update the storages
    phi_t_arr[:,:,iter+1] = phi_next    
    penetration = torch.logical_and((psi < 0), (phi_next < 0))    
    lambda_t_arr[:,:,iter+1][penetration] = (lambda_t - rho_al_np * psi)[penetration]
    lambda_t_arr[:,:,iter+1][~penetration] = 0.0
    mu_t_arr[iter+1] = mu_t + rho_al_V * (compute_volume(phi_next) - V_0)

    print("Max_phi: " + str(torch.max(phi_next)))
    print("Max_lambda: " + str(torch.max(lambda_t_arr[:,:,iter +1])))

    #if (torch.any(torch.isnan(phi_t))): 
    #    raise ValueError("NaN encountered. stopping iterations")

print("iteration took {:.5f} seconds".format(time.time() - t0))

# 4. Plot the array
for iter in tqdm(range(0, num_iters-1, 10)):
    plt.figure()
    plt.subplot(1,2,1)
    phi_t_thres = get_interior_phi(phi_t_arr[:,:,iter], 0.0)
    phi_t = phi_t_arr[:,:,iter]
    plt.imshow(phi_t_thres.T, cmap='gray', origin='lower')
    plt.subplot(1,2,2)    
    plt.imshow(lambda_t_arr[:,:,iter].T, cmap='gray', origin='lower')
    plt.savefig("data/{:03d}.png".format(iter))
    plt.close()
