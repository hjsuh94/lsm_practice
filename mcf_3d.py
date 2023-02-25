import os, time
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
num_iters = 1000
sigma = 2.0
ksize_gaussian = 11
ksize_laplace = 7
h = 1e-4
rho_al = 1e-3

# 2. Initialize grid. 
n_grid = 100
grid = generate_ellipsoid(24, 16, 36, n_grid)

# 3. Compute signed distance and initialize Lagrangian
# 3. Compute signed distance
phi_0 = torch.Tensor(skfmm.distance(grid)).cuda()
phi_t_arr = torch.zeros((n_grid, n_grid, n_grid, num_iters)).cuda()
phi_t_arr[:,:,:,0] = phi_0

lambda_0 = 0.0
lambda_arr = torch.zeros(num_iters).cuda()
lambda_arr[0] = lambda_0
V_0 = compute_volume(phi_0)

# 4. Iterate and optimize
for iter in tqdm(range(num_iters-1)):
    phi_t = phi_t_arr[:,:,:,iter]
    lambda_t = lambda_arr[iter]

    # 3.1. Gaussian smooth the signed distance field.
    kernel_g = gaussian_kernel(sigma, ksize_gaussian).cuda()
    phi_t_smooth = convolve(phi_t, kernel_g)

    # 3.2. Compute the curvature field.
    kernel_log = log_kernel(sigma, ksize_laplace).cuda()
    phi_t_curvature = convolve(phi_t, kernel_log)
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

    # Alternate way to compute the curvature by implementing divergence.
    curvature_x = convolve(normals[:,:,:,0], kernel_x)
    curvature_y = convolve(normals[:,:,:,1], kernel_y)
    curvature_z = convolve(normals[:,:,:,2], kernel_z)
    curvature = (curvature_x + curvature_y + curvature_z)

    # 3.4 Compute volume of current phi_t to compute Lagrangian.
    V_t = compute_volume(phi_t)
    dL = (curvature + lambda_t + rho_al * (V_t - V_0)) #* boundary_mask
    #dL = curvature #* boundary_mask

    dL_vec = normals * dL[:,:,:,None]
    v_max = torch.max(torch.abs(torch.sum(dL_vec, dim=3)))
    h_cfl = 2e-2 / v_max

    # 3.4 Update the signed distance field via advection
    phi_next = phi_t + h_cfl * dL * phi_t_grad_norm

    phi_t_arr[:,:,:,iter+1] = phi_next
    lambda_arr[iter+1] = lambda_t + rho_al * (compute_volume(phi_next) - V_0)

    plot_interior_meshcat(phi_t, "object_{:03d}".format(iter), vis)

for iter in range(num_iters-1):
    with anim.at_frame(vis, iter) as frame:
        for i in range(num_iters):
            frame["object_{:03d}".format(i)].set_property(
                "visible", "boolean", False)        
        frame["object_{:03d}".format(iter)].set_property(
            "visible", "boolean", True)

plt.figure()
plt.imshow(phi_t_arr[50,:,:,num_iters-1].T.cpu(), cmap='gray', origin='lower')
plt.show()

vis.set_animation(anim)    
input()