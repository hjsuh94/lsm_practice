import os, time

import skfmm
import meshcat
import meshcat.geometry as g
import matplotlib.pyplot as plt
from matplotlib import cm

from conv_3d import *
from shape_generator_3d import *
from shape_generator_2d import plot_image


n_grid = 64

def gaussian_smoothing(phi, sigma=2.0, ksize_gaussian=5):
    kernel_g = gaussian_kernel(sigma, ksize_gaussian).cuda()
    phi_smooth = convolve(phi, kernel_g)
    return phi_smooth

def test_gaussian_filter():
    phi = generate_ellipsoid(25, 18, 18, n_grid).cuda()
    phi_interior = get_interior_phi(phi, 0, 0.0, 1.0)
    phi_interior_smooth = gaussian_smoothing(phi_interior)
    phi_interior_smooth_more = gaussian_smoothing(
        phi_interior, sigma=10.0, ksize_gaussian=10)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(phi_interior_smooth[:,:,32].cpu(),
        origin='lower', cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(phi_interior[:,:,32].cpu(),
        origin='lower', cmap='gray')
    plt.subplot(1,3,3)        
    plt.imshow(phi_interior_smooth_more[:,:,32].cpu(),
        origin='lower', cmap='gray')        
    plt.show()

def test_sobel_filter_slice():
    phi = generate_ellipsoid(25, 18, 18, n_grid).cuda()
    phi_smooth = gaussian_smoothing(phi)
    kernel_x = sobel_kernel(direction='x').cuda()
    kernel_y = sobel_kernel(direction='y').cuda()
    kernel_z = sobel_kernel(direction='z').cuda()

    phi_x = convolve(phi_smooth, kernel_x)
    phi_y = convolve(phi_smooth, kernel_y)
    phi_z = convolve(phi_smooth, kernel_z)

    plt.figure()

    plt.subplot(4,3,1)
    plt.title('x-slice')
    plot_image(phi_smooth[32,:,:].cpu(), show=False)
    plt.subplot(4,3,2)
    plt.title('y-slice')
    plot_image(phi_smooth[:,32,:].cpu(), show=False)
    plt.subplot(4,3,3)
    plt.title('z-slice')
    plot_image(phi_smooth[:,:,32].cpu(), show=False)    

    plt.subplot(4,3,4)
    plt.title("grad_x, x-slice")
    plot_image(phi_x[32,:,:].cpu(), show=False)
    plt.subplot(4,3,5)
    plt.title("grad_x, y-slice")
    plot_image(phi_x[:,32,:].cpu(), show=False)
    plt.subplot(4,3,6)
    plt.title("grad_x, z-slice")
    plot_image(phi_x[:,:,32].cpu(), show=False)

    plt.subplot(4,3,7)
    plt.title("grad_y, x-slice")
    plot_image(phi_y[32,:,:].cpu(), show=False)
    plt.subplot(4,3,8)
    plt.title("grad_y, y-slice")
    plot_image(phi_y[:,32,:].cpu(), show=False)
    plt.subplot(4,3,9)
    plt.title("grad_y, z-slice")
    plot_image(phi_y[:,:,32].cpu(), show=False)

    plt.subplot(4,3,10)
    plt.title("grad_z, x-slice")
    plot_image(phi_z[32,:,:].cpu(), show=False)
    plt.subplot(4,3,11)
    plt.title("grad_z, y-slice")
    plot_image(phi_z[:,32,:].cpu(), show=False)
    plt.subplot(4,3,12)
    plt.title("grad_z, z-slice")
    plot_image(phi_z[:,:,32].cpu(), show=False)
    plt.show()

def test_sobel_filter_meshcat():

    vis = meshcat.Visualizer().open()

    phi = generate_ellipsoid(25, 18, 18, n_grid).cuda()
    phi_smooth = gaussian_smoothing(phi)
    kernel_x = sobel_kernel(direction='x').cuda()
    kernel_y = sobel_kernel(direction='y').cuda()
    kernel_z = sobel_kernel(direction='z').cuda()

    phi_x = convolve(phi_smooth, kernel_x)
    phi_y = convolve(phi_smooth, kernel_y)
    phi_z = convolve(phi_smooth, kernel_z)

    phi_grad = torch.stack((phi_x, phi_y, phi_z), dim=3)

    scale = 0.01
    interior = get_interior_phi(phi, 0, up=0, down=1)
    indices = interior.nonzero().cpu().numpy().T * scale
    vis["ellipse"].set_object(g.Points(
        g.PointsGeometry(indices, color=indices),
        g.PointsMaterial(size=scale)
    ))

    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                if (interior[i,j,k]):
                    normal = phi_grad[i,j,k,:].cpu().numpy()

                    name = "line_{}_{}_{}".format(i,j,k)
                    start_point = scale * np.array([i, j, k])
                    end_point = scale * np.array([i, j, k]) + scale * normal

                    vis[name].set_object(g.Line(g.PointsGeometry(
                        np.vstack((start_point, end_point)).T
                    )))

def test_laplace_filter_slice():
    grid = generate_ellipsoid(25, 18, 18, n_grid)
    phi = torch.Tensor(skfmm.distance(grid)).cuda()
    kernel_log = log_kernel(2.0, 5).cuda()

    phi_curvature = convolve(phi, kernel_log)
    plt.figure()
    plt.imshow(phi_curvature[:,:,32].cpu(), origin='lower', cmap='gray')
    plt.show()

def test_laplace_filter_meshcat():
    vis = meshcat.Visualizer().open()
    
    grid = generate_ellipsoid(25, 18, 18, n_grid)
    phi = torch.Tensor(skfmm.distance(grid)).cuda()
    kernel_log = log_kernel(2.0, 5).cuda()
    phi_curvature = convolve(phi, kernel_log)

    # normalize curvature to 0-1.
    phi_curvature = 1./ (torch.max(phi_curvature) - torch.min(phi_curvature)) * (
        phi_curvature - torch.min(phi_curvature))

    scale = 0.01
    interior = get_interior_phi(phi, 0, up=0, down=1)
    indices = interior.nonzero().cpu().numpy().T

    colormap = cm.get_cmap('jet')
    colors = np.zeros((4, indices.shape[1]))

    for i in range(indices.shape[1]):
        colors[:,i] = colormap(phi_curvature[
            indices[0,i], indices[1,i], indices[2,i]].cpu().numpy())

    vis["ellipse"].set_object(g.Points(
        g.PointsGeometry(indices * scale, color=colors),
        g.PointsMaterial(size=scale)
    ))
    