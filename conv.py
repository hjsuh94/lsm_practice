import torch
import matplotlib.pyplot as plt

def log_kernel(sigma, kernel_size):
    """Create Laplacian of Gaussian kernel filter. Returns a function"""
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

def log_kernel_manual():
    """Test kernel for correctness of log_kernel derivation."""
    return torch.tensor([
        [0,1,1,2,2,2,1,1,0],
        [1,2,4,5,5,5,4,2,1],
        [1,4,5,3,0,3,5,4,1],
        [2,5,3,-12,-24,-12,3,5,2],
        [2,5,0,-24,-40,-24,0,5,2],
        [2,5,3,-12,-24,-12,3,5,2],
        [1,4,5,3,0,3,5,4,1],
        [1,2,4,5,5,5,4,2,1],
        [0,1,1,2,2,2,1,1,0],                        
        ], dtype=torch.float32)

def gaussian_kernel(sigma, kernel_size):
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

def sobel_kernel(direction='x'):
    if direction == "x":
        return torch.Tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    elif direction == "y":
        return torch.Tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    else:
        raise ValueError("direction not supported.")

def convolve(phi, kernel):
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

def get_boundary_phi(phi, thres):
    boundary = phi
    boundary[boundary > thres] = 0.0
    boundary[boundary < -thres] = 0.0
    return boundary

def get_interior_phi(phi, thres):
    interior = phi.clone()
    interior[interior > thres] = 1.0
    interior[interior < thres] = -1.0
    return interior

def compute_volume(phi):
    """make sure thres is very close to 0."""
    interior = get_interior_phi(phi, 0.0)
    interior[interior > 0.0] = 0.0
    interior[interior < 0.0] = 1.0
    return torch.sum(interior)

def plot_image(phi):
    plt.figure()
    plt.imshow(phi.T, cmap='gray', origin='lower')
    plt.show()