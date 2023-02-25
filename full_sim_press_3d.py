import os, time

import torch 
import matplotlib.pyplot as plt
import cv2
import skfmm
import meshcat
from meshcat.animation import Animation

from tqdm import tqdm

from conv_3d import *
from shape_generator_3d import *

vis = meshcat.Visualizer().open()
anim = Animation()

# 1. Hyperparameters.
num_iters = 1000
sigma = 2.0
ksize_gaussian = 5
ksize_laplace = 15
h = 1e-4
rho_al_V = 1e-2
rho_al_np = 1e-3

# 2. Initialize grid. 
n_grid = 64
grid = generate_ellipsoid(12, 8, 18, n_grid)
obstacle_grid_1 = generate_half_plane([0.0, 0.0, 1.0], [32, 64, 32 - 10], n_grid)
obstacle_grid_2 = generate_half_plane([0.0, 0.0, -1.0], [32, 32, 32 + 10], n_grid)
obstacle_grid = add_shape([obstacle_grid_1, obstacle_grid_2], n_grid)

obstacle_grid_1 = generate_ellipse_off_center(8, 8, 8, [32, 32, 32 - 15], n_grid)
#obstacle_grid_2 = generate_ellipse_off_center(8, 8, 8, [32, 32, 32 + 15], n_grid)
#obstacle_grid = add_shape([obstacle_grid_1, obstacle_grid_2], n_grid)

#plot_interior_meshcat(grid, "object", vis)
#plot_interior_meshcat(obstacle_grid, "obstacle", vis, color=np.array([1.0, 1.0, 1.0])) 

# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid)).cuda()
phi_t_arr = torch.zeros((n_grid, n_grid, n_grid, num_iters)).cuda()
phi_t_arr[:,:,:,0] = phi_0
psi = torch.Tensor(skfmm.distance(obstacle_grid)).cuda()

# 4. Initiate Dual field.
mu_0 = 0.0
mu_t_arr = torch.zeros(num_iters)
mu_t_arr[0] = mu_0
V_0 = compute_volume(phi_0)

lambda_0 = torch.zeros((n_grid, n_grid, n_grid)).cuda()
lambda_t_arr = torch.zeros((n_grid, n_grid, n_grid, num_iters)).cuda()
lambda_t_arr[:,:,:,0] = lambda_0

#plot_interior_meshcat(obstacle_grid, "z", vis, color=np.array([1.0, 1.0, 1.0])) 

# Define zeta function
def get_zeta(psi, lambda_t, mu_al):
    n_grid = psi.shape[0]
    zeta = torch.zeros((n_grid, n_grid, n_grid)).cuda()
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
    phi_t = phi_t_arr[:,:,:,iter]
    mu_t = mu_t_arr[iter]
    lambda_t = lambda_t_arr[:,:,:,iter]

    # 3.1. Gaussian smooth the signed distance field.
    kernel_g = gaussian_kernel(sigma, ksize_gaussian).cuda()
    phi_t_smooth = convolve(phi_t, kernel_g)

    # 3.2. Compute the curvature field.
    kernel_log = log_kernel(sigma, ksize_laplace).cuda()
    phi_t_curvature = convolve(phi_t, kernel_log)
    boundary = convolve(convolve(
        get_interior_phi(phi_t, 0.0), kernel_g), kernel_log)

    # 3.3. Compute the boundary mask
    boundary_mask = torch.clone(phi_t)
    boundary_mask[torch.abs(boundary_mask) > 1e-4] = 1.0
    boundary_mask[torch.abs(boundary_mask) < 1e-4] = 0.0
    boundary_mask = boundary_mask.to(torch.bool)

    # 3.3. Compute the gradient field.
    kernel_x = sobel_kernel(direction='x').cuda()
    kernel_y = sobel_kernel(direction='y').cuda()
    kernel_z = sobel_kernel(direction='z').cuda()
    phi_t_grad_x = convolve(phi_t_smooth, kernel_x)
    phi_t_grad_y = convolve(phi_t_smooth, kernel_y)
    phi_t_grad_z = convolve(phi_t_smooth, kernel_z)
    phi_t_grad = torch.stack(
        (phi_t_grad_x, phi_t_grad_y, phi_t_grad_z), dim=3)
    phi_t_grad_norm = torch.norm(phi_t_grad, dim=3)

    # 3.4 Compute chamfer distance
    phi_t_chamfer = torch.clone(phi_t)
    phi_t_chamfer[phi_t_chamfer < 0.0] = 0.0
    phi_t_chamfer = torch.pow(phi_t_chamfer, 2)
    phi_t_chamfer[~boundary_mask] = 0.0

    # 3.5 Compute the Lagrangian.
    zeta_t = get_zeta(psi, lambda_t, rho_al_np)
    V_t = compute_volume(phi_t)
    volume_penalty = mu_t + rho_al_V * (V_t - V_0)
    dL_t = 1e-3 * phi_t_chamfer + 1e-1 * phi_t_curvature + volume_penalty + zeta_t
    dL_t = 1e2 * phi_t_curvature + volume_penalty

    # CFL number.
    # Decompose dL_t * nabla phi_n to component vectors
    phi_t_grad_normalized = phi_t_grad / phi_t_grad_norm[:,:,:,None]
    dL_t_vec = phi_t_grad_normalized * dL_t[:,:,:,None] 
    v_max = torch.max(torch.abs(torch.sum(dL_t_vec, dim=3)))

    # 3.6 Update the signed distance field via advection
    phi_next = phi_t + h * dL_t * phi_t_grad_norm

    # 3.7 Update the storages
    phi_t_arr[:,:,:,iter+1] = phi_next
    penetration = torch.logical_and((psi < 0), (phi_next < 0))

    lambda_t_arr[:,:,:,iter+1][penetration] = (lambda_t - rho_al_np * psi)[penetration]
    lambda_t_arr[:,:,:,iter+1][~penetration] = 0.0
    mu_t_arr[iter+1] = mu_t + rho_al_V * (compute_volume(phi_next) - V_0)

    """
    print("Max_phi: " + str(torch.max(phi_next)))
    print("Max_lambda: " + str(torch.max(lambda_t_arr[:,:,iter +1])))
    """

    plot_interior_meshcat(phi_t, "object_{:03d}".format(iter), vis)
    #plot_dual_meshcat(lambda_t, "dual_{:03d}".format(iter), vis)


print("iteration took {:.5f} seconds".format(time.time() - t0))

#plot_image_plt(phi_t_arr[:,:,:,-1].cpu())


for iter in range(num_iters-1):
    with anim.at_frame(vis, iter) as frame:
        for i in range(num_iters):
            frame["object_{:03d}".format(i)].set_property(
                "visible", "boolean", False)        
        frame["object_{:03d}".format(iter)].set_property(
            "visible", "boolean", True)

vis.set_animation(anim)
#plot_dual_meshcat(lambda_t_arr[:,:,:,num_iters-1], "dual", vis)

input()