import numpy as np
import torch
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure 
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf 

"""
Functions for shape primitives.
TODO(terry-suh): There should be a more general framework.
"""

def generate_ellipse_2d(a, b, n_grid):
    grid = torch.ones(n_grid, n_grid)
    center = n_grid / 2
    for i in range(n_grid):
        for j in range(n_grid):
            if ((i - center) / a) ** 2 + ((j - center) / b) ** 2 <= 1:
                grid[i,j] = -1
    return grid

def generate_ellipse_off_center_2d(a, b, mu, n_grid):
    grid = torch.ones(n_grid, n_grid)
    center = n_grid / 2
    for i in range(n_grid):
        for j in range(n_grid):
            if ((i - mu[0]) / a) ** 2 + ((j - mu[1]) / b) ** 2 <= 1:
                grid[i,j] = -1
    return grid

def generate_half_plane_2d(normal, point, n_grid):
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

def generate_rectangle_2d(center, l_x, l_y, n_grid):
    """
    Generate axis-aligned rectangle. 
    points away from the obstacle region, and the boundary of the half plane 
    passes through the point.
    TODO(terry-suh): change this to support rigid body transforms.
    """
    grid = torch.ones(n_grid, n_grid)
    center_x = center[0]
    center_y = center[1]
    hl_x = int(l_x / 2)
    hl_y = int(l_y / 2)
    grid[center_x - hl_x : center_x + hl_x,
         center_y - hl_y : center_y + hl_y] = -1.0
    return grid
    
def add_shape(grid_lst, n_grid, dim=2):
    """
    For each grid, assume that the filled parts are -1, unfilled parts are 1.
    Creates a union of all shapes. Constructive solid geometry?
    """
    if dim == 2:
        grid = torch.ones(n_grid, n_grid)
    elif dim == 3:
        grid = torch.ones(n_grid, n_grid, n_grid)
    else:
        raise ValueError("add_shape encountered invalid dimension.")

    for i in range(len(grid_lst)):
        grid[grid_lst[i] == -1] = -1
    return grid

def generate_star_2d(a, b, k, n_grid):
    """
    Generate a star shape in 2d given by polar equations.
    """
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
    """
    Get the boundary of a signed distance function phi based on thresholding.
    """
    boundary = phi.clone()
    boundary[boundary > thres] = 0.0
    boundary[boundary < -thres] = 0.0
    return boundary

def get_interior_phi(phi, thres, up=1.0, down=-1.0):
    """
    Given phi, threshold the image so that x s.t. phi(x) > 0 evaluates to up
    and x s.t. phi(x) < thres evaluates 
    """
    interior = phi.clone()
    interior[phi > thres] = up
    interior[phi < thres] = down
    return interior

def compute_volume(phi):
    """
    Compute the volume of an sdf.
    """
    interior = get_interior_phi(phi, 0.0, 0.0, 1.0)
    return torch.sum(interior)

def plot_image_2d(phi, show=True):
    """
    Given a 2d grid, plot on imagespace using matplotlib imshow.
    """
    if (phi.is_cuda): 
        phi_cpu = torch.clone(phi).cpu()

    if (show):
        plt.figure()
        plt.imshow(phi_cpu.T, cmap='gray', origin='lower')
        plt.show()    
    else:
        plt.imshow(phi_cpu.T, cmap='gray', origin='lower')
    
def render_scene(phi, psi, n_grid):
    """
    Given a object sdf and obstacle sdf psi, render an image.
    """
    image = torch.zeros((n_grid, n_grid, 4)).to(torch.int32)
    phi_ind = get_interior_phi(phi, 0.0, 0.0, 1.0).to(torch.bool)
    psi_ind = get_interior_phi(psi, 0.0, 0.0, 1.0).to(torch.bool)
    image[psi_ind] = torch.Tensor([0, 0, 0, 200]).to(torch.int32)
    image[phi_ind] = image[phi_ind] + torch.Tensor([25, 128, 128, 150]).to(torch.int32)
    return image.transpose(0,1)

def generate_ellipsoid_3d(a, b, c, n_grid):
    grid = torch.ones(n_grid, n_grid, n_grid)
    center = n_grid / 2
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                if (((i - center) / a) ** 2 + ((j - center) / b) ** 2 + \
                    ((k - center) / c) ** 2 <= 1):
                    grid[i,j,k] = -1
    return grid

def generate_ellipsoid_off_center_3d(a, b, c, mu, n_grid):
    grid = torch.ones((n_grid, n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                if ((i - mu[0]) / a) ** 2 + (
                    (j - mu[1]) / b) ** 2 + ((k - mu[2]) / c) ** 2 <= 1:
                    grid[i,j,k] = -1
    return grid

def generate_half_plane_3d(normal, point, n_grid):
    """
    Generate half-plane for shape primitive. Defines an obstacle. The normal 
    points away from the obstacle region, and the boundary of the half plane 
    passes through the point.
    """
    normal = torch.tensor(normal)
    point = torch.tensor(point)
    grid = torch.ones(n_grid, n_grid, n_grid)
    pos_x, pos_y, pos_z = torch.meshgrid(
        torch.range(0, n_grid-1), torch.range(0, n_grid-1),
        torch.range(0, n_grid-1))
    pos = torch.stack((pos_x, pos_y, pos_z), dim=3)

    neg_ind = torch.einsum('ijkl,l->ijk', pos-point, normal) > 0
    
    grid[neg_ind] = 1.0
    grid[~neg_ind] = -1.0
    return grid

def plot_image_plt(phi, show=True):
    ax = plt.figure().add_subplot(projection='3d')
    x, y, z = torch.meshgrid(torch.arange(phi.shape[0] + 1),
                             torch.arange(phi.shape[1] + 1),
                             torch.arange(phi.shape[2] + 1))

    verts, faces, _, _ = measure.marching_cubes(phi.numpy(), 0, step_size=2)

    mesh = Poly3DCollection(verts[faces])
    mesh.set_facecolor('springgreen')
    mesh.set_edgecolor('black')
    mesh.set_alpha(0.3)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, phi.shape[0] + 1)
    ax.set_ylim(0, phi.shape[1] + 1)
    ax.set_zlim(0, phi.shape[2] + 1)
    plt.show()

def plot_image_meshcat(phi, name, vis, color="position"):
    interior = torch.clone(phi)
    indices = interior.nonzero().cpu().numpy().T / phi.shape[0]

    if color == "position":
        vis[name].set_object(g.Points(
            g.PointsGeometry(indices, color=indices),
            g.PointsMaterial(size= 1/phi.shape[0] + 0.01)
        ))
    # TODO(terry-suh): implement scaling with color.
    elif color == "scale":
        color = (phi - phi.min()) / (phi.max() - phi.min())
        color = color.nonzero().cpu().numpy().T
        vis[name].set_object(g.Points(
            g.PointsGeometry(indices, color=color),
            g.PointsMaterial(size= 1/phi.shape[0] + 0.01)
        ))
    else:
        raise ValueError("unsupported color scheme.")

def plot_interior_meshcat(phi, name, vis, color="position"):
    interior = get_interior_phi(phi, 0, up=0, down=1)
    plot_image_meshcat(interior.to(torch.int), name, vis, color=color)


def plot_dual_meshcat(dual, name, vis, color="scale"):
    plot_image_meshcat(dual, name, vis, color=color)
        
    
def plot_half_plane_meshcat(normals, point, scale, name, vis):
    # Compute the size of the box
    box = g.Box([1.0, 1.0, 0.5])
    vis[name].set_object(box, g.MeshLambertMaterial(
        color=0xffffff, opacity=0.2))
    vis[name].set_transform(
        tf.translation_matrix(
            scale * np.array(point) - np.array([0, 0, normals[2] * 0.25])))

def plot_ellipsoid_meshcat(a, b, c, mu, scale, name, vis):
    # Compute the size of the box
    ellipsoid = g.Ellipsoid(scale * np.array([a, b, c]))
    vis[name].set_object(ellipsoid, g.MeshLambertMaterial(
        color=0xffffff, opacity=0.2))
    vis[name].set_transform(
        tf.translation_matrix(scale * mu))
