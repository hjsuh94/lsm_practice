import os, time

import torch 
import matplotlib.pyplot as plt
import cv2
import skfmm
from conv import *
from shape_generator import *
from tqdm import tqdm

# 1. Hyperparameters.
num_iters = 1000
sigma = 3.0
ksize_gaussian = 7
ksize_laplace = 15
h = 0.005
rho_al_V = 1e-2
rho_al_np = 1e-2

# 2. Initialize grid. 
n_grid = 128
grid = generate_ellipse(36, 30, n_grid)
obstacle_grid_1 = generate_ellipse_off_center(16, 16, [64, 64 + 30], n_grid)
obstacle_grid_2 = generate_ellipse_off_center(16, 16, [64, 64 - 30], n_grid)
obstacle_grid = add_shape([obstacle_grid_1, obstacle_grid_2], n_grid)

plot_image(obstacle_grid)

# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid)).cuda()
phi_t_arr = torch.zeros((phi_0.shape[0], phi_0.shape[0], num_iters)).cuda()
phi_t_arr[:,:,0] = phi_0

psi = torch.Tensor(skfmm.distance(obstacle_grid)).cuda()

# 4. Initiate Dual field.
mu_0 = 0.0
mu_t_arr = torch.zeros(num_iters)
mu_t_arr[0] = mu_0
V_0 = compute_volume(phi_0)

lambda_0 = torch.zeros((n_grid, n_grid)).cuda()
lambda_t_arr = torch.zeros((n_grid, n_grid, num_iters)).cuda()
lambda_t_arr[:,:,0] = lambda_0

# Define zeta function
def get_zeta(psi, lambda_t, mu_al):
    zeta = torch.zeros((psi.shape[0], psi.shape[1])).cuda()
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
    kernel_g = gaussian_kernel(sigma, ksize_gaussian).cuda()
    phi_t_smooth = convolve(phi_t, kernel_g)

    # 3.2. Compute the curvature field.
    kernel_log = log_kernel(sigma, ksize_laplace).cuda()
    phi_t_curvature = convolve(phi_t, kernel_log)

    # 3.3. Compute the boundary element
    kernel_log = log_kernel(2.0, ksize_laplace).cuda()
    boundary = convolve(get_interior_phi(phi_t, 0.0), kernel_log)

    mask = torch.zeros((n_grid, n_grid)).cuda()
    mask[torch.abs(boundary) > 1e-4] = 1.0
    mask[torch.abs(boundary) < 1e-4] = 0.0
    mask = mask.to(torch.bool)

    # 3.3. Compute the gradient field.
    kernel_x = sobel_kernel(direction='x').cuda()
    kernel_y = sobel_kernel(direction='y').cuda()
    phi_t_grad_x = convolve(phi_t_smooth, kernel_x)
    phi_t_grad_y = convolve(phi_t_smooth, kernel_y)
    phi_t_grad = torch.stack((phi_t_grad_x, phi_t_grad_y), dim=2)
    phi_t_grad_norm = torch.norm(phi_t_grad, dim=2)

    # 3.4 Compute chamfer distance
    phi_t_chamfer = torch.clone(phi_t)
    phi_t_chamfer[phi_t_chamfer < 0.0] = 0.0
    phi_t_chamfer = torch.pow(phi_t_chamfer, 2)
    phi_t_chamfer[~mask] = 0.0

    # 3.5 Compute the Lagrangian.
    zeta_t = get_zeta(psi, lambda_t, rho_al_np)
    V_t = compute_volume(phi_t)
    volume_penalty = rho_al_V * (V_t - V_0)    
    dL_t = 1e-3 * phi_t_chamfer + phi_t_curvature + volume_penalty + zeta_t

    # 3.6 Update the signed distance field via advection
    phi_next = phi_t + h * dL_t * phi_t_grad_norm

    # 3.7 Update the storages
    phi_t_arr[:,:,iter+1] = phi_next    
    penetration = torch.logical_and((psi < 0), (phi_next < 0))

    lambda_t_arr[:,:,iter+1][penetration] = (lambda_t - rho_al_np * psi)[penetration]
    lambda_t_arr[:,:,iter+1][~penetration] = 0.0
    mu_t_arr[iter+1] = mu_t + rho_al_V * (compute_volume(phi_next) - V_0)

    """
    print("Max_phi: " + str(torch.max(phi_next)))
    print("Max_lambda: " + str(torch.max(lambda_t_arr[:,:,iter +1])))
    """

print("iteration took {:.5f} seconds".format(time.time() - t0))

# 4. Plot the array
image_count = 0
for iter in tqdm(range(0, num_iters-1, 10)):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Primal (phi)")
    plt.imshow(render_scene(phi_t_arr[:,:,iter].cpu(), psi, n_grid), 
        cmap='gray', origin='lower')
    plt.subplot(1,2,2)    
    plt.title("Dual (lambda)")
    plt.imshow(lambda_t_arr[:,:,iter].T.cpu(), cmap='gray', origin='lower')
    plt.savefig("data_pinch/{:03d}.png".format(image_count))
    plt.close()

    image_count += 1
