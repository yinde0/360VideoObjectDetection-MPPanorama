# import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
# import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = './ffmpeg'
from tqdm import tqdm

def map_to_sphere(x, y, z, radius, yaw_radian, pitch_radian, distance=1.):




    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)
    r = radius

    # Equations to get the original point before projected onto plane
    # This point is the intersection of the line through the points (0, 0, -R) (x_p, y_p, z_p) and the sphere x^2 + y^2 + z^2 = r^2
    denominator = x ** 2 + y ** 2 + 4 * (r ** 2)


    x_circle = (4 * r ** 2) * x / denominator
    y_circle = (4 * r ** 2) * y / denominator
    z_circle = -r + (8 * r ** 3) / denominator

    print_vals = np.sqrt(x_circle ** 2 + y_circle ** 2 + z_circle ** 2)
    theta = np.arccos(z_circle / np.sqrt(x_circle ** 2 + y_circle ** 2 + z_circle ** 2))
    phi = np.arctan2(y_circle, x_circle)


    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    Panorama_W_angle, Panorama_H_angle = FOV
    W, H = output_size

    # Number of pixels covered by FOV angle in the panorama image
    pano_pixel_W_range = int(pano_width * (Panorama_W_angle / 360))
    pano_pixel_H_range = int(pano_height * (Panorama_H_angle / 180))
    print("\n" + str(pano_pixel_W_range))
    print("\n" + str(pano_pixel_H_range))
    

    # Step size for how much each output pixel changes relative to panorama pixel range
    W_step_size = pano_pixel_W_range/ W
    H_step_size = pano_pixel_H_range/ H 

    # Create arrays of length W and H, with values from 1 to W and 1 to H
    u, v = np.rint(np.meshgrid(np.arange(pano_pixel_W_range, step=W_step_size), np.arange(pano_pixel_H_range, step=H_step_size), indexing='xy'))

    # Set array of x values from -W/2 to W/2 so that 0 is in center
    x = u - pano_pixel_W_range / 2
    # Set array of y values from -H/2 to H/2 so that 0 is in center
    y = pano_pixel_H_range / 2 - v
    # z distance of plane to center of sphere (plane is parallel to xy-plane and assuming we will be using 180-degree stereographic projection)
    z = pano_pixel_W_range / 4
    # The radius of the sphere is the same distance as the distance of the plane from the xy plane
    radius = z

    theta, phi = map_to_sphere(x, y, z, radius, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')

    return output_image



