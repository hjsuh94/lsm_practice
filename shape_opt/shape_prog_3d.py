import os, time

import numpy as np
import torch
import matplotlib.pyplot as plt
import skfmm

import shape_opt.conv as conv
import shape_opt.shape_generator as sgen
from shape_opt.shape_prog_params import SMPParameters2D
from shape_opt.shape_prog import ShapeMathematicalProgram

class ShapeMathematicalProgram3D(ShapeMathematicalProgram):
    def __init__(self, grid_0, params: SMPParameters2D):
        super().__init__(grid_0, params)

    def compute_kernels(self):
        self.kernel_g = conv.gaussian_kernel_3d(
            self.params.sigma, self.params.ksize_gaussian)
        self.kernel_log = conv.log_kernel_3d(
            self.params.sigma, self.params.ksize_gaussian)
        self.kernel_x = conv.sobel_kernel_3d(direction='x')
        self.kernel_y = conv.sobel_kernel_3d(direction='y')
        self.kernel_z = conv.sobel_kernel_3d(direction='z')

        if (self.params.gpu):
            (self.kernel_g, self.kernel_log, self.kernel_x, self.kernel_y, 
                self.kernel_z) = self.transfer_to_gpu([self.kernel_g, 
                self.kernel_log, self.kernel_x, self.kernel_y, self.kernel_z])

        return (
            self.kernel_g, self.kernel_log,
            self.kernel_x, self.kernel_y, self.kernel_z)

    def initialize_field(self, value):
        if (self.params.gpu):
            return value * torch.ones(
                self.params.n_grid, self.params.n_grid,
                self.params.n_grid).cuda()
        else:
            return value * torch.ones(
                self.params.n_grid, self.params.n_grid,
                self.params.n_grid)

    def initialize_prog(self):
        # 1. Initialize storage arrays for logging.
        if (self.params.log_iterations):
            
            self.phi_t_arr = torch.zeros(
                (self.params.n_grid, self.params.n_grid,
                 self.params.n_grid, self.params.max_iters))
            if (self.params.gpu): self.phi_t_arr = self.phi_t_arr.cuda()
            self.phi_t_arr[:,:,:,0] = self.phi_0
            
            if self.volume_equality_constraint:
                self.mu_t_arr = torch.zeros(self.params.max_iters)
                if (self.params.gpu):
                    self.mu_t_arr = self.mu_t_arr.cuda()
                self.mu_t_arr[0] = self.mu_0
                    
            if len(self.particlewise_penetration_constraints) > 0:
                self.lambda_t_arr = torch.zeros(
                    (self.params.n_grid, self.params.n_grid,
                     self.params.n_grid, self.params.max_iters))
                if (self.params.gpu):
                    self.lambda_t_arr = self.lambda_t_arr.cuda()
                self.lambda_t_arr[:,:,:,0] = self.lambda_0

        # 2. Compute kernels.
        self.compute_kernels()

    def get_derivatives(self, phi):
        dphi = conv.Derivatives()

        # 1. First derivatives of phi.
        dphi.smooth = conv.convolve_3d(phi, self.kernel_g)
        phi_x = conv.convolve_3d(dphi.smooth, self.kernel_x)
        phi_y = conv.convolve_3d(dphi.smooth, self.kernel_y)
        phi_z = conv.convolve_3d(dphi.smooth, self.kernel_z)
        dphi.dx = torch.stack((phi_x, phi_y, phi_z), dim=3)

        dphi.dx_norm = torch.norm(dphi.dx, dim=3)
        dphi.n = dphi.dx / dphi.dx_norm[:,:,:,None]

        # 2. Second derivatives of phi.
        H_x = conv.convolve_3d(dphi.n[:,:,:,0], self.kernel_x)
        H_y = conv.convolve_3d(dphi.n[:,:,:,1], self.kernel_y)
        H_z = conv.convolve_3d(dphi.n[:,:,:,2], self.kernel_z)
        dphi.H = H_x + H_y + H_z

        # 3. Third derivatives of phi.
        dHdx = conv.convolve_3d(dphi.H, self.kernel_x)
        dHdy = conv.convolve_3d(dphi.H, self.kernel_y)
        dHdz = conv.convolve_3d(dphi.H, self.kernel_z)
        
        grad_H = torch.stack((dHdx, dHdy, dHdz), dim=3)
        dphi.dHdn = torch.einsum('ijkl,ijkl->ijk', grad_H, dphi.n)

        return dphi
    
    def log_field_storage(self, storage, idx, object):
        """
        Given a storage object, and index, and an object, store the
        object into the storage object.
        """
        storage[:,:,:,idx] = object
        
    def smooth_field(self, phi):
        return conv.convolve_3d(phi, self.kernel_g)
    
    def find_cfl_timestep(self, dL_t, dphi):
        dL_t_vec = dphi.n * dL_t[:,:,:,None]
        dL_max = torch.max(torch.abs(
            torch.sum(dL_t_vec, dim=3)))
        return self.params.h / dL_max
