import torch
import conv_2d
import matplotlib.pyplot as plt

"""
Functions here must color the interior of the shape with -1, and 
exterior with 1.
"""

def generate_ellipse(a, b, n_grid):
    grid = torch.ones(n_grid, n_grid)
    center = n_grid / 2
    for i in range(n_grid):
        for j in range(n_grid):
            if ((i - center) / a) ** 2 + ((j - center) / b) ** 2 <= 1:
                grid[i,j] = -1
    return grid

def generate_ellipse_off_center(a, b, mu, n_grid):
    grid = torch.ones(n_grid, n_grid)
    center = n_grid / 2
    for i in range(n_grid):
        for j in range(n_grid):
            if ((i - mu[0]) / a) ** 2 + ((j - mu[1]) / b) ** 2 <= 1:
                grid[i,j] = -1
    return grid

def generate_half_plane(normal, point, n_grid):
    """
    Generate half-plane for shape primitive. Defines an obstacle. The normal 
    points away from the obstacle region, and the boundary of the half plane 
    passes through the point.
    """
    normal = torch.tensor(normal)
    point = torch.tensor(point)
    grid = torch.ones(n_grid, n_grid)
    pos_x, pos_y = torch.meshgrid(
        torch.range(0, n_grid-1), torch.range(0, n_grid-1))
    pos = torch.stack((pos_x, pos_y), dim=2)

    neg_ind = torch.einsum('ijk,k->ij', pos-point, normal) > 0
    
    grid[neg_ind] = 1.0
    grid[~neg_ind] = -1.0
    return grid

def generate_rectangle(center, l_x, l_y, n_grid):
    """
    Generate rectangle. 
    points away from the obstacle region, and the boundary of the half plane 
    passes through the point.
    """
    grid = torch.ones(n_grid, n_grid)
    center_x = center[0]
    center_y = center[1]
    hl_x = int(l_x / 2)
    hl_y = int(l_y / 2)
    print(center_x - hl_x)
    print(center_x + hl_x)
    print(center_y - hl_y)
    print(center_y + hl_y)
    grid[center_x - hl_x : center_x + hl_x,
         center_y - hl_y : center_y + hl_y] = -1.0
    return grid
    
def add_shape(grid_lst, n_grid):
    """
    For each grid, assume that the filled parts are -1, unfilled parts are 1.
    Creates a union of all shapes.
    """
    grid = torch.ones(n_grid, n_grid)
    for i in range(len(grid_lst)):
        grid[grid_lst[i] == -1] = -1
    return grid

def generate_star(a, b, k, n_grid):
    grid = torch.ones(n_grid, n_grid)
    center = n_grid / 2
    for i in range(n_grid):
        for j in range(n_grid):
            nx = i - center
            ny = j - center
            r = torch.norm(torch.tensor([nx, ny]))
            theta = torch.atan2(torch.tensor(ny), torch.tensor(nx))
            if r < (a + b * torch.cos(k * theta)):
                grid[i,j] = -1
    return grid

def get_boundary_phi(phi, thres):
    boundary = phi
    boundary[boundary > thres] = 0.0
    boundary[boundary < -thres] = 0.0
    return boundary

def get_interior_phi(phi, thres, up=1.0, down=-1.0):
    interior = phi.clone()
    interior[phi > thres] = up
    interior[phi < thres] = down
    return interior

def compute_volume(phi):
    """make sure thres is very close to 0."""
    interior = get_interior_phi(phi, 0.0)
    interior[interior > 0.0] = 0.0
    interior[interior < 0.0] = 1.0
    return torch.sum(interior)

def plot_image(phi, show=True):
    if (show):
        plt.figure()
        plt.imshow(phi.T, cmap='gray', origin='lower')
        plt.show()    
    else:
        plt.imshow(phi.T, cmap='gray', origin='lower')        
    
def render_scene(phi, psi, n_grid):
    """
    Create an RGBA image.
    """
    image = torch.zeros((n_grid, n_grid, 4)).to(torch.int32)
    phi_ind = get_interior_phi(phi, 0.0, 0.0, 1.0).to(torch.bool)
    psi_ind = get_interior_phi(psi, 0.0, 0.0, 1.0).to(torch.bool)
    image[psi_ind] = torch.Tensor([0, 0, 0, 200]).to(torch.int32)
    image[phi_ind] = image[phi_ind] + torch.Tensor([25, 128, 128, 150]).to(torch.int32)
    return image.transpose(0,1)
    

