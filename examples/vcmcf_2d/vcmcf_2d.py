import time

import torch
import matplotlib.pyplot as plt

from shape_opt.shape_prog_2d import ShapeMathematicalProgram2D
from shape_opt.shape_prog_params import SMPParameters2D
from shape_opt.shape_generator import (
    generate_ellipse_2d, plot_image_2d, get_interior_phi)

# Set parameters for the solver.
params = SMPParameters2D()
params.sigma = 1.0
params.ksize_gaussian = 5
params.n_grid = 128
params.max_iters = 5000
params.log_iterations = False
params.gpu = True
params.h = 2e-3

# Generate initial shape.
grid_0 = generate_ellipse_2d(48, 24, params.n_grid).cuda()

# Set up the program and solve.
prog = ShapeMathematicalProgram2D(grid_0, params)
prog.AddSurfaceElementCost(
    torch.ones(params.n_grid, params.n_grid).cuda())
prog.AddVolumeEqualityConstraint(rho_al_v=1e-3)
start = time.time()
phi_T = prog.solve()
print("Max iterations reached at {:.3f}s.".format(time.time() - start))

grid_T = get_interior_phi(phi_T, 0.0, 1, -1)
plt.figure()
plt.subplot(1,2,1)
plt.title('Before')
plt.imshow(grid_0.cpu(), origin='lower', cmap='gray')
plt.subplot(1,2,2)
plt.title('After')
plt.imshow(grid_T.cpu(), origin='lower', cmap='gray')
plt.savefig('examples/vcmcf_2d/vcmcf_2d.png')
