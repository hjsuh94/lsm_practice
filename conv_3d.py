import torch
import matplotlib.pyplot as plt
import numpy as np

def laplacian_kernel():
    kernel = torch.tensor([
        [[2, 3, 2], [3, 6 ,3], [2, 3, 2]],
        [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
        [[2, 3, 2], [3, 6, 3], [2, 3, 2]]
    ]) / 26
    return kernel

def log_kernel(sigma, kernel_size):
    mean = (kernel_size - 1) / 2
    range_1d = torch.arange(kernel_size) - mean
    grid_x, grid_y, grid_z = torch.meshgrid(range_1d, range_1d, range_1d)
    xyz_grid = torch.stack([grid_x, grid_y, grid_z], dim=3)
    xyz_square = torch.sum(torch.pow(xyz_grid, 2.0), dim=3)

    # this is laplacian of gaussian, NOT logarithm.
    xyz_square_varnorm = xyz_square / (2 * sigma ** 2.0)
    log_kernel = (-1 / (torch.pi * sigma ** 4.0)) * (
        1 - xyz_square_varnorm) * torch.exp(-xyz_square_varnorm)

    norm_log_kernel = log_kernel - torch.sum(
        log_kernel) / (kernel_size ** 3.0)

    return norm_log_kernel


def gaussian_kernel(sigma, kernel_size):
    mean = (kernel_size - 1) / 2
    range_1d = torch.arange(kernel_size) - mean
    grid_x, grid_y, grid_z = torch.meshgrid(range_1d, range_1d, range_1d)
    xyz_grid = torch.stack([grid_x, grid_y, grid_z], dim=3)
    xyz_square = torch.sum(torch.pow(xyz_grid, 2.0), dim=3)

    
    gaussian_kernel = (1./ torch.sqrt(
        torch.pow(torch.tensor(torch.pi), 3) * (sigma ** 3.0))) *\
                    torch.exp(
                        -xyz_square / (2 * sigma ** 2.0)
                  )

    norm_gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return norm_gaussian_kernel

def sobel_kernel(direction='x'):
    forward = torch.tensor([
            [1, 2, 1], [2, 4, 2], [1, 2, 1]
        ])
    if direction == "x":
        kernel = torch.zeros((3, 3, 3))
        kernel[0,:,:] = -forward 
        kernel[2,:,:] = forward
        return kernel
    elif direction == "y":
        kernel = torch.zeros((3, 3, 3))
        kernel[:,0,:] = -forward 
        kernel[:,2,:] = forward
        return kernel        
    elif direction == "z":
        kernel = torch.zeros((3, 3, 3))
        kernel[:,:,0] = -forward 
        kernel[:,:,2] = forward
        return kernel                
    else:
        raise ValueError("direction not supported.")

def convolve(phi, kernel):
    kernel_size = kernel.shape[0]
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    nn_log = torch.nn.Conv3d(1, 1, kernel_size, groups=1, bias=False, 
        padding='same', padding_mode='replicate')
    nn_log.weight.data = kernel
    nn_log.weight.requires_grad = False

    image = nn_log(phi.view(1, 1, phi.shape[0], phi.shape[1], phi.shape[2]))
    image = image[0, 0, :, :]

    return image

