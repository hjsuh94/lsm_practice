import os, time, tempfile
from pyrsistent import b

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
num_iters = 300
sigma = 1.0
ksize_gaussian = 5
ksize_laplace = 7
h = 2e-2
rho_al_np = 100
rho_al_V = 1

# 2. Initialize grid. 
n_grid = 100
grid = generate_ellipsoid(24, 16, 36, n_grid)
obstacle_grid_1 = generate_half_plane([0.0, 0.0, 1.0], [50, 50, 50 - 15], n_grid)
obstacle_grid_2 = generate_half_plane([0.0, 0.0, -1.0], [50, 50, 50 + 15], n_grid)
obstacle_grid = add_shape([obstacle_grid_1, obstacle_grid_2], n_grid)

# plot_interior_meshcat(obstacle_grid, "z", vis)
plot_half_plane_meshcat([0.0, 0.0, 1.0], [50, 50, 50 - 15], 1e-2, "plane_1", vis)
plot_half_plane_meshcat([0.0, 0.0, -1.0], [50, 50, 50 + 15], 1e-2, "plane_2", vis)

# 3. Compute signed distance and initialize Lagrangian
phi_0 = torch.Tensor(skfmm.distance(grid)).cuda()
phi_t_arr = torch.zeros((n_grid, n_grid, n_grid, num_iters)).cuda()
phi_t_arr[:,:,:,0] = phi_0
psi = torch.Tensor(skfmm.distance(obstacle_grid)).cuda()

# 4. Initiate dual field and volume.
mu_0 = 0.0
mu_t_arr = torch.zeros(num_iters).cuda()
mu_t_arr[0] = mu_0
lambda_0 = torch.zeros((n_grid, n_grid, n_grid)).cuda()
lambda_t_arr = torch.zeros((n_grid, n_grid, n_grid, num_iters)).cuda()
lambda_t_arr[:,:,:,0] = lambda_0
V_0 = compute_volume(phi_0)

# 5. Define zeta function
def get_zeta(psi, lambda_t, rho_al):
    n_grid = psi.shape[0]
    zeta = torch.zeros((n_grid, n_grid, n_grid)).cuda()
    zeta_neg = -lambda_t * psi + 0.5 * rho_al * torch.pow(psi, 2)
    zeta_pos = -1 / (2.0 * rho_al) * torch.pow(lambda_t, 2)
    zeta_neg_ind = (psi - (lambda_t / rho_al) <= 0)
    zeta_pos_ind = (psi - (lambda_t / rho_al) > 0)
    zeta[zeta_neg_ind] = zeta_neg[zeta_neg_ind]
    zeta[zeta_pos_ind] = zeta_pos[zeta_pos_ind]
    return zeta

# 3. Compute chamfer distance.
phi_chamfer = torch.clone(phi_0)
phi_chamfer[phi_chamfer < 0.0] = 0.0
phi_chamfer = torch.pow(phi_chamfer, 2)

# 4. Iterate and optimize
for iter in tqdm(range(num_iters-1)):
    phi_t = phi_t_arr[:,:,:,iter]
    lambda_t = lambda_t_arr[:,:,:,iter]
    mu_t = mu_t_arr[iter]
    
    # 3.1. Gaussian smooth the signed distance field.
    kernel_g = gaussian_kernel(sigma, ksize_gaussian).cuda()
    phi_t_smooth = convolve(phi_t, kernel_g)

    # 3.2. Compute the boundary mask.
    kernel_log = log_kernel(4.0 * sigma, ksize_laplace).cuda()
    boundary = convolve(get_interior_phi(phi_t, 0.0), kernel_log)
    boundary_mask = torch.clone(phi_t)
    boundary_mask[torch.abs(boundary) > 1e-4] = 1.0
    boundary_mask[torch.abs(boundary) < 1e-4] = 0.0
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
    normals = phi_t_grad / phi_t_grad_norm[:,:,:,None]

    # 3.4 Compute curvature
    curvature_x = convolve(normals[:,:,:,0], kernel_x)
    curvature_y = convolve(normals[:,:,:,1], kernel_y)
    curvature_z = convolve(normals[:,:,:,2], kernel_z)
    curvature = (curvature_x + curvature_y + curvature_z)

    # 3.5 Compute chamfer distance.
    phi_t_chamfer = torch.clone(phi_chamfer)
    #phi_t_chamfer[~boundary_mask] = 0.0

    # 3.6 Get the particle-wise Lagrangain.
    zeta_t = get_zeta(psi, lambda_t, rho_al_np)
    V_t = compute_volume(phi_t)
    volume_penalty = mu_t + rho_al_V * (V_t - V_0)
    dL_t = 1e-2 * phi_t_chamfer + 1e-1 * curvature + volume_penalty + zeta_t

    # 3.5 Fill dL...
    dL_vec = normals * dL_t[:,:,:,None]
    v_max = torch.max(torch.abs(torch.sum(dL_vec, dim=3)))
    h_cfl = h / v_max

    # 3.4 Update the signed distance field via advection
    phi_next = phi_t + h_cfl * dL_t * phi_t_grad_norm

    # Update the storages
    phi_t_arr[:,:,:,iter+1] = phi_next
    penetration = torch.logical_and((psi < 0), (phi_next < 0))
    lambda_t_arr[:,:,:,iter+1][penetration] = (lambda_t - rho_al_np * psi)[penetration]
    lambda_t_arr[:,:,:,iter+1][~penetration] = 0.0
    lambda_t_arr[:,:,:,iter+1] = convolve(lambda_t_arr[:,:,:,iter+1], kernel_g)
    mu_t_arr[iter+1] = mu_t + rho_al_V * (compute_volume(phi_next) - V_0)

    if (iter % 20 == 0):
        plot_interior_meshcat(phi_t, "object_{:03d}".format(iter), vis, "position")

for iter in range(0, num_iters-1, 20):
    with anim.at_frame(vis, iter) as frame:
        for i in range(0, num_iters, 20):
            frame["object_{:03d}".format(i)].set_property(
                "visible", "boolean", False)        
        frame["object_{:03d}".format(iter)].set_property(
            "visible", "boolean", True)

plt.figure()
plt.imshow(phi_t_arr[50,:,:,num_iters-1].T.cpu(), cmap='gray', origin='lower')
plt.show()

vis.set_animation(anim)    
res = vis.static_html()
with open("press_3d.html", 'w') as f:
    f.write(res)
input()