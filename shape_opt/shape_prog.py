import os, time

import numpy as np
import torch
import matplotlib.pyplot as plt
import skfmm

import shape_opt.shape_generator as sgen
from shape_opt.shape_prog_params import SMPParameters

class ShapeMathematicalProgram():
    def __init__(self, grid_0, params: SMPParameters):
        self.params = params
        self.check_valid_grid(grid_0)
        self.phi_0 = self.compute_sdf(grid_0)

        # Kernel variables.
        self.kernel_g = None
        self.kernel_log = None
        self.kernel_x = None
        self.kernel_y = None
        self.kernel_z = None

        # Use this to iterate over costs and constraints.
        self.fixed_surface_element_costs = [] # store cost_grids.
        self.surface_laplacian_cost = [] # store scalar weights.
        self.fixed_volume_element_costs = [] # store cost grids.
        self.volume_equality_constraint = False # store volume.
        self.particlewise_penetration_constraints = [] # store obstacle grids.

    def check_gpu(self, obj):
        if (self.params.gpu) and not(obj.is_cuda):
            raise ValueError(
                "parameters is gpu but the given object is not on the gpu.")

    def check_valid_grid(self, grid):
        # 1. Check for dimensions of the grids.
        if (grid.dim() != self.params.d):
            print(grid.dim())
            print(self.params.d)
            raise ValueError("dimension is inconsistent with grid and params.")
        for i in range(self.params.d):
             if (grid.shape[i] != self.params.n_grid):
                raise ValueError(
                    "dimension {} is inconsistent with n_grid.".format(i))

        # 2. GPU compatibility testing.
        self.check_gpu(grid)

    def transfer_to_gpu(self, obj_lst):
        """
        Given a list of objs, transfer them to cuda.
        """
        gpu_objects = []
        for obj in obj_lst:
            gpu_objects.append(obj.cuda())
        return gpu_objects

    def AddSurfaceElementCost(self, cost_grid):
        """
        Given a surface cost F(\Omega) = \int_{\partial\Omega} f(x) ds where
        f: R^d -> R, add a cost term that minimizes F(\Omega). 
        The computed shape gradient is DF(\Omega) = f(x)H(x). Note that for 
        implementation, it is sufficient to have a grid of f(x) in the level
        set method.

        Note that cost_grid has to be of shape (n_grid, n_grid).
        """
        self.check_valid_grid(cost_grid)
        self.fixed_surface_element_costs.append(cost_grid)

    def AddSurfaceLaplacianEnergyCost(self, weight):
        """
        Given a surface cost with a weighting term
        F(\Omega) = weight * \int_{\partial\Omega} H(x)^2ds where
        H is the additive curvature (d * mean curvature), add a term that
        models this objective. We model the shape derivative as 
        dH/dOmega = dH/dn + H^2 where dH/dn = nabla H \cdot n.
        nabla H is computed discretely using Sobel operations on the curvature
        grid. 
        
        NOTE(terry-suh): This is separate cost from FixedSurfaceElementCost
        as we need to recompute H(x) at every iteration.
        """
        self.surface_laplacian_cost.append(weight)

    def AddFixedVolumeElementCost(self, cost_grid):
        """
        Given a volume cost F(\Omega) = \int_\Omega f(x)dv where f:R^d->R,
        add a cost term taht minimizes F(\Omega). 
        The computed shape gradient is dF/dOmega = f(x).
        """
        self.check_valid_grid(cost_grid)
        self.fixed_volume_element_costs.append(cost_grid)

    def AddVolumeEqualityConstraint(self, rho_al_v = 1.0):
        """
        Given a volume G(\Omega) = \int_\Omega 1 dv, constrain this volume
        to be given by volume_sum. This constraint can only be added once.
        rho_al_v is the coefficient of the augmented Lagrangian, where the
        Lagrange multiplier will be initialized to 0.0.

        NOTE(terry-suh): We should be able to support unilateral constraints.        
        """
        # Initialize multipliers for augmented Lagrangian
        self.mu_0 = 0.0
        self.rho_al_v = rho_al_v
        self.volume_equality_constraint = True

    def compute_sdf(self, grid):
        """
        Given a grid where interior is -1 and exterior is 1, compute the 
        signed distance.
        """
        if grid.is_cuda:
            return torch.Tensor(skfmm.distance(grid.cpu())).cuda()
        else:
            return torch.Tensor(skfmm.distance(grid))

    def AddParticlePenetrationConstraint(self, obs_grid, rho_al_np = 10.0):
        """
        Enforce a particle penetration constraint on a grid imposed by
        obs_grid. The grid values need to be 1 outside the obstacle and -1
        inside the obstacle.
        """
        self.check_valid_grid(obs_grid)
        if len(self.particlewise_penetration_constraints) == 0:
            self.obs_grid = obs_grid
            self.psi = self.compute_sdf(self.obs_grid) # SDF for obstacles.
        else:
            self.obs_grid = sgen.add_shape(self.obs_grid, obs_grid,
                self.params.d)
            self.psi = self.compute_sdf(self.obs_grid)

        self.lambda_0 = self.initialize_scalar_field(0.0)
        self.rho_al_np = rho_al_np

    def get_zeta(self, psi, lambda_t, rho_al_np):
        """
        Obtain the zeta function for Lagrange multipliers with unilateral 
        constraints. Psi corresponds to the sdf of the obstacle grid, and
        lambda_t is the current field of lagrange multipliers for each
        particle. 
        """        
        zeta = self.initialize_scalar_field(0.0)
        zeta_neg = -lambda_t * psi + 0.5 * rho_al_np * torch.pow(psi, 2)
        zeta_pos = -1 / (2.0 * rho_al_np) * torch.pow(lambda_t, 2)
        zeta_neg_ind = (psi - (lambda_t / rho_al_np) <= 0)
        zeta_pos_ind = (psi - (lambda_t / rho_al_np) > 0)
        zeta[zeta_neg_ind] = zeta_neg[zeta_neg_ind]
        zeta[zeta_pos_ind] = zeta_pos[zeta_pos_ind]
        return zeta

    def initialize_scalar_field(self, value):
        """
        Initialize scalar field of equal values.
        """
        raise NotImplementedError("Virtual function.")

    def compute_kernels(self):
        """
        Compute kernels at the beginning of the program to save computation.
        TODO(terry-suh): technically, these don't have to be constructed at
        every mathematical program since they are fixed. Might be more 
        efficient to transfer these to the simulation class.        
        Returns a tuple of:
            (gaussian, log, sobel_x, sobel_y, sobel_z)
        """
        raise NotImplementedError("Virtual function.")

    def get_derivatives(self, phi):
        """
        Given a level set function phi (not necessarily sdf), get derivatives.
        Returns a conv.Derivatives class struct
        """
        raise NotImplementedError("Virtual function.")
    
    def smooth_field(self, phi):
        """
        Smooth a field with a gaussian kernel.
        """
        raise NotImplementedError("Virtual function.")

    def initialize_prog(self):
        raise NotImplementedError("Virtual function.")
    
    def find_cfl_timestep(self):
        raise NotImplementedError("Virtual function.")
    
    def solve(self):
        self.initialize_prog()

        V_0 = sgen.compute_volume(self.phi_0)
        self.phi_t = torch.clone(self.phi_0)
        
        if self.volume_equality_constraint:
            self.mu_t = self.mu_0
        if len(self.particlewise_penetration_constraints) > 0:
            self.lambda_t = torch.clone(self.lambda_0)

        for t in range(self.params.max_iters):
            # First, update the storages.
            if (self.params.log_iterations):
                self.log_field_storage(self.phi_t_arr,
                                       t, self.phi_t)

            dphi = self.get_derivatives(self.phi_t)
            dL_t = self.initialize_field(0.0)

            # Add costs to compute the Lagrangian.
            for cost_grid in self.fixed_volume_element_costs:
                dL_t += cost_grid

            for cost_grid in self.fixed_surface_element_costs:
                dL_t += cost_grid * dphi.H
                
            for weight in self.surface_laplacian_cost:
                dL_t += weight * (torch.pow(dphi.H, 2))

            if self.volume_equality_constraint:
                V_t = sgen.compute_volume(self.phi_t)
                dL_t += self.mu_t + self.rho_al_v * (V_t - V_0)
                if (self.params.log_iterations):
                    self.mu_t_arr[t] = self.mu_t

            if len(self.particlewise_penetration_constraints) > 0:
                dL_t += self.get_zeta(self.psi, self.lambda_t, self.rho_al_np)
                if (self.params.log_iterations):
                    self.log_field_storage(self.lambda_t_arr,
                                           t, self.lambda_t)
                    
            # Advect the Lagrangian along the normal with CFL condition.
            if (self.params.cfl_integration):
                h = self.find_cfl_timestep(dL_t, dphi)
            else:
                h = self.params.h

            # Level set equation.
            self.phi_t = self.phi_t + h * dL_t * dphi.dx_norm

            # Update the duals.
            if self.volume_equality_constraint:
                self.mu_t += self.rho_al_v * (
                    sgen.compute_volume(self.phi_t) - V_0)
                
            if len(self.particlewise_penetration_constraints) > 0:
                penetration = torch.logical_and(
                    (self.psi < 0), (self.phi_t < 0))

                self.lambda_t[penetration] = (
                    self.lambda_t - self.rho_al_np * self.psi)[penetration]
                self.lambda_t[~penetration] = 0.0
                self.lambda_t = self.smooth_field(self.lambda_t)

        return self.phi_t
