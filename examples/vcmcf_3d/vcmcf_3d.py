import time

import torch
import matplotlib.pyplot as plt

from shape_opt.shape_prog_3d import ShapeMathematicalProgram3D
from shape_opt.shape_prog_params import SMPParameters3D
from shape_opt.shape_generator import (
    generate_ellipsoid_3d, plot_image_2d, get_interior_phi)

# Set parameters for the solver.
params = SMPParameters3D()
params.sigma = 1.0
params.ksize_gaussian = 5
params.n_grid = 100
params.max_iters = 1000
params.gpu = True
params.h = 2e-2

# Generate initial shape.
grid_0 = generate_ellipsoid_3d(24, 16, 36, params.n_grid).cuda()

# Set up the program and solve.
prog = ShapeMathematicalProgram3D(grid_0, params)
prog.AddSurfaceElementCost(
    torch.ones(params.n_grid, params.n_grid, params.n_grid).cuda())
prog.AddVolumeEqualityConstraint(rho_al_v=1e-3)
start = time.time()
phi_T = prog.solve()
print("Max iterations reached at {:.3f}s.".format(time.time() - start))

grid_T = get_interior_phi(phi_T, 0.0, 1, -1)
plt.figure()
plt.subplot(2,3,1)
plt.title('Before x')
plt.imshow(grid_0[50,:,:].cpu(), origin='lower', cmap='gray')
plt.subplot(2,3,4)
plt.title('After x')
plt.imshow(grid_T[50,:,:].cpu(), origin='lower', cmap='gray')
plt.subplot(2,3,2)
plt.title('Before y')
plt.imshow(grid_0[:,50,:].cpu(), origin='lower', cmap='gray')
plt.subplot(2,3,5)
plt.title('After y')
plt.imshow(grid_T[:,50,:].cpu(), origin='lower', cmap='gray')
plt.subplot(2,3,3)
plt.title('Before z')
plt.imshow(grid_0[:,:,50].cpu(), origin='lower', cmap='gray')
plt.subplot(2,3,6)
plt.title('After z')
plt.imshow(grid_T[:,:,50].cpu(), origin='lower', cmap='gray')

plt.savefig('examples/vcmcf_3d/vcmcf_3d.png')
