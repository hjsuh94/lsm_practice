import torch

"""
Functions for computing kernels for computation.
"""

def laplacian_kernel_3d():
    """
    (3, 3, 3) Laplacian kernel for 3D convolution.
    """
    kernel = torch.tensor([
        [[2, 3, 2], [3, 6 ,3], [2, 3, 2]],
        [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
        [[2, 3, 2], [3, 6, 3], [2, 3, 2]]
    ]) / 26
    return kernel

def log_kernel_2d(sigma, kernel_size):
    """
    (kernel_size, kernel_size) LoG (Laplacian of Gaussian) kernel
    for 2D convolution. sigma sets the variance of the Gaussian.
    Note that bigger kernel sizes will result in slower computation.    
    """
    mean = (kernel_size - 1) / 2
    x_cord = torch.arange(kernel_size) - mean
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_square = torch.sum(torch.pow(xy_grid, 2.0), dim=2)

    # this is laplacian of gaussian, NOT logarithm.
    log_kernel = (-1 / (torch.pi * sigma ** 4.0)) * (
        1 - xy_square / (2 * sigma ** 2.0)) * (
            torch.exp(-xy_square / (2 * sigma ** 2.0)))

    norm_log_kernel = log_kernel - torch.sum(
        log_kernel) / (kernel_size ** 2.0)

    return norm_log_kernel


def log_kernel_3d(sigma, kernel_size):
    """
    (kernel_size, kernel_size, kernel_size) LoG (Laplacian of Gaussian) kernel
    for 3D convolution. sigma sets the variance of the Gaussian.
    Note that bigger kernel sizes will result in slower computation.
    """
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

def gaussian_kernel_2d(sigma, kernel_size):
    """
    (kernel_size, kernel_size) Gaussian kernel for 2D convolution.
    sigma sets the variance of the Gaussian.
    Note that bigger kernel sizes will result in slower computation.
    """    
    mean = (kernel_size - 1) / 2
    x_cord = torch.arange(kernel_size) - mean
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_square = torch.sum(torch.pow(xy_grid, 2.0), dim=2)

    # this is laplacian of gaussian, NOT logarithm.
    gaussian_kernel = (1./(2. * torch.pi * sigma ** 2.0)) *\
                    torch.exp(
                        -xy_square / (2 * sigma ** 2.0)
                  )    

    norm_gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return norm_gaussian_kernel

def gaussian_kernel_3d(sigma, kernel_size):
    """
    (kernel_size, kernel_size, kernel_size) Gaussian kernel for 3D convolution.
    sigma sets the variance of the Gaussian.
    Note that bigger kernel sizes will result in slower computation.
    """
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

def sobel_kernel_2d(direction='x'):
    """
    (3, 3) Sobel kernel for 2D convolution to image gradients. 
    Specify the gradient direction as 'x' or 'y'.
    """
    if direction == "x":
        return torch.Tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    elif direction == "y":
        return torch.Tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        raise ValueError("direction not supported.")

def sobel_kernel_3d(direction='x'):
    """
    (3, 3, 3) Sobel kernel for 3D convolution to image gradients.
    Specify the gradient direction as 'x' or 'y' or 'z'.
    """    
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

def convolve_2d(phi, kernel):
    """
    Convolve a (n_grid, n_grid) grid phi with a given kernel.
    Note that only square kernels are supported, so that the shape of the 
    kernel must be (kernel_size, kernel_size).
    """    
    kernel_size = kernel.shape[0]
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(1, 1, 1, 1)

    nn_log = torch.nn.Conv2d(1, 1, kernel_size, groups=1, bias=False, 
        padding='same', padding_mode='replicate')
    nn_log.weight.data = kernel
    nn_log.weight.requires_grad = False

    image = nn_log(phi.view(1, 1, phi.shape[0], phi.shape[1]))
    image = image[0, 0, :, :]

    return image

def convolve_3d(phi, kernel):
    """
    Convolve a (n_grid, n_grid, n_grid) grid phi with a given kernel.
    Note that only square kernels are supported, so that the shape of the 
    kernel must be (kernel_size, kernel_size, kernel_size).
    """
    kernel_size = kernel.shape[0]
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    nn_log = torch.nn.Conv3d(1, 1, kernel_size, groups=1, bias=False, 
        padding='same', padding_mode='replicate')
    nn_log.weight.data = kernel
    nn_log.weight.requires_grad = False

    image = nn_log(phi.view(1, 1, phi.shape[0], phi.shape[1], phi.shape[2]))
    image = image[0, 0, :, :]

    return image

class Derivatives():
    """
    Storage class for derivatives.
    """
    def __init__(self):
        self.smooth = None
        self.dx = None
        self.dx_norm = None
        self.n = None
        self.H = None # additive curvature.
        self.dHdn = None 
