
'''

Coordinate System:
    coordinate system is defined as:
    x -> south-direction axis
    y -> zenith-direction axis
    z -> west-direction axis
    theta -> zenith angle from positive y-axis
    phi -> azimuth angle from positive z-axis

'''

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from tqdm import tqdm
from draw_boxes import draw_bounding_box


def map_to_sphere(x, y, z, radius, yaw_radian, pitch_radian):
    
    '''
    Function:
        Take input coordinates (x, y, z)
        and find what coordinates on the sphere project
        onto them using stereographic projection.

        This means finding the points on the sphere defined by:
            x^2 + y^2 + z^2 = radius^2
        that are along the vector line passing through the points:
            (0, 0, -R) and (x, y, z)

    Input:
        x, y, z (np.ndarray, np.ndarray, np.ndarray): 3 np.ndarrays input point coordinates
        radius (float): input radius of the sphere (this sphere is what will be used to wrap the panorama around)
        yaw_radians (float): how much to rotate around y axis
        pitch_radians (float): how much to rotate around z axis

    Output:
        theta, phi: 2 np.ndarrays of coordinates in (theta, phi) to use
        to map pixel colors from the panorama to the pixels in the output image
    '''

    r = radius

    # Equations to get the original point before projected onto plane
    # This point is the intersection of the line through the points (0, 0, -R) (x_p, y_p, z_p) and the sphere x^2 + y^2 + z^2 = r^2
    denominator = x ** 2 + y ** 2 + 4 * (r ** 2)
    x_circle = (4 * r ** 2) * x / denominator
    y_circle = (4 * r ** 2) * y / denominator
    z_circle = -r + (8 * r ** 3) / denominator

    # Calculate theta (azimuth angle) and phi (zenith angle) 
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
    '''
    Function:
        Take input coordinates and use the input image to
        get the color values at each coordinate
        (interpolate color values between
        coordinate values if necessary).

    Input:
        coords (np.ndarray): a np.ndarray of coordinates
        img (np.ndarray): a np.ndarray of image color values
        method (str/int: optional): type of interpolation

    Output:
        np.ndarray of color values of each related coordinate,
        and an array shape of coords
    '''
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')

    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    '''
    Function:
        Take an input panorama image (360-degree),
        a set Field of View, a file size in pixels-by-pixels for the output plane image,
        and the rotational transformation yaw and pitch and get the
        stereographic projection onto a plane tangeant to the sphere
        created by the panorama.

    Input:
        panorama_path (str): file path to panorama image
        FOV (float, float): (width-angle, height-angle) -> (based on theta from 0 to 360, based on phi from 0 to 180)
        output_size (int, int): (width-pixels, height-pixels)
        yaw_radians (float): how much to rotate around y axis
        pitch_radians (float): how much to rotate around z axis

    Output:
        output_image_array: np.ndarray of color values over the output plane
        image with shape (height, width, 3): 3 for Red, Green, Blue
    '''

    # Open panorama and get info about its size in pixels
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)

    # Convert to radians for numpy math
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)
    print(panorama_path)

    # Get FOV angles and size of output in pixels
    Panorama_W_angle, Panorama_H_angle = FOV
    W, H = output_size

    # Number of pixels covered by FOV angle in the panorama image
    pano_pixel_W_range = int(pano_width * (Panorama_W_angle / 360))
    pano_pixel_H_range = int(pano_height * (Panorama_H_angle / 180))
    

    # Step size for how much each output pixel changes relative to panorama pixel range
    W_step_size = pano_pixel_W_range / W
    H_step_size = pano_pixel_H_range / H 

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

    # Get an np.ndarray of theta and phi angles that are associated with each point (x, y, z) in the stereographic projected plane
    theta, phi = map_to_sphere(x, y, z, radius, yaw_radian, pitch_radian)

    # Convert theta and phi angles to associated pixels in the panorama
    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    # Format the pixels appropriately
    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    # Get the pixel values in the for the coordinates of (theta, phi) associated with the stereographic projected plane
    output_image_array = interpolate_color(coords, pano_array)

    # reshape the output to be in the format of the output image
    output_image_array = output_image_array.reshape((H, W, 3)).astype('uint8')

    return output_image_array

def panorama_to_stereo_multiprojections(panorama_path, stereographic_image_size, FOV):
    '''
    Function:
        Take a panorama image and convert it to 4 stereographic projections,
        based on "Object Detection in Equirectangular Panorama" paper.

    Input:
        panorama_path (str): file path to panorama image
        FOV (float, float): (width-angle, height-angle) -> (based on theta from 0 to 360, based on phi from 0 to 180)
        output_image_size (int, int): (width-pixels, height-pixels)
    
    Output:
        frames_with_meta_np: a np.ndarray with 4 tuples for each projection:
            (1) an np.ndarray of a stereographic projection plane with shape:
            (output_image_size.height, output_image_size.width, 3) -> 3 for Red, Green, Blue
            (2) its associated yaw and pitch rotations
    '''

    frames_with_meta = []

    for i in tqdm(range(4)):
        yaw_rotation = i * 90
        pitch_rotation = 90
        W, H = stereographic_image_size
        output_image_array = panorama_to_plane(panorama_path, FOV, stereographic_image_size, yaw_rotation, pitch_rotation)
        
        frame_with_meta = (output_image_array, yaw_rotation, pitch_rotation)
        frames_with_meta.append(frame_with_meta)

        output_image = Image.fromarray(output_image_array.reshape((H, W, 3)).astype('uint8'), 'RGB')
        
        extension_passed = False
        file_name = ""
        for char in reversed(panorama_path):
            if extension_passed:
                file_name = char + file_name
            elif extension_passed == False and char == ".":
                extension_passed = True

        output_name = file_name + "_stereographic-Face" + str(i) + ".jpg"
        output_image.save(output_name)
        output_image.show()
    

    frames_with_meta_np = np.array(frames_with_meta, dtype=np.dtype([('image', np.ndarray), ('yaw', int), ('pitch', int)]))

    return frames_with_meta_np

def stereo_bounding_boxes_to_panorama(frame_detections_with_meta, panorama_path, stereographic_image_size, FOV):
    '''
    Function:
        Take a set of detections on the stereographic projections and
        convert them to coordinates on the panorama, then return the
        panorama with detections on it.

    Input:
        detections_with_meta (np.ndarray of tuple with (np.ndarray, (int, int), int, int)):
            (bounding boxes, (stereographic image width, height), yaw, pitch))
            -> bounding boxes info (see yolov8 model run), the height and width in pixels
            of the image used in detections, and the yaw and pitch of frame used in detections.
        panorama_path (str): file path to panorama image
        stereographic_image_size (int, int): (pixel_W_size, pixel_H_size)
        FOV (int, int): (W_angle, H_angle)
    
    Output:
        annotated_panorama (np.ndarray): The output panorama with annotated bounding boxes
    '''

    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)

    Panorama_W_angle, Panorama_H_angle = FOV
    W, H = stereographic_image_size

    # Number of pixels covered by FOV angle in the panorama image
    pano_pixel_W_range = int(pano_width * (Panorama_W_angle / 360))
    pano_pixel_H_range = int(pano_height * (Panorama_H_angle / 180))
    

    # Step size for how much each output pixel changes relative to panorama pixel range
    W_step_size = pano_pixel_W_range / W
    H_step_size = pano_pixel_H_range / H


    # z distance of plane to center of sphere (plane is parallel to xy-plane and assuming we will be using 180-degree stereographic projection)
    z = pano_pixel_W_range / 4
    # The radius of the sphere is the same distance as the distance of the plane from the xy plane
    radius = z


    # Iterate through each frame
    for frame_detections in frame_detections_with_meta:
        
        x_array = []
        y_array = []
        # Get all of the detections in the frame and get the coordinates of the 4 corners of the box
        for detection in frame_detections['image_detections']:
            print(detection)
            # box -> {x, y, width, height}
            box = detection['box']
            # scale is conversion factor to scale up to original image (stereographic projection plane) size
            # from downscaled YOLOv8 image (needed to process)
            scale = detection['scale']

            # Get box coordinates for x (x, x + w) and y (y, y + h) from the stereographic image
            box_coord_x = np.array([(box[0] * scale), (box[0] + box[2]) * scale])
            box_coord_y = np.array([(box[1] * scale), (box[1] + box[3]) * scale])
            
            
            print("Box coordinates for x:")
            print(box_coord_x)
            print("Box coordinates for y:")
            print(box_coord_y)

            # Get the stereographic projection plane coordinates from the stereographic image
            # Set array of x values from -W/2 to W/2 so that 0 is in center of plane
            x_plane = box_coord_x * (W_step_size) - pano_pixel_W_range / 2

            # Set array of y values from -H/2 to H/2 so that 0 is in center of plane
            y_plane = pano_pixel_H_range / 2 - box_coord_y * (H_step_size)

            x_grid, y_grid = np.meshgrid(x_plane, y_plane, indexing="xy")
            
            print("X grid: ", x_grid)
            print("Y grid: ", y_grid)

            x_array.append(x_grid)
            y_array.append(y_grid)

        print("Frame detection yaw: " + str(frame_detections['yaw']))
        print("Frame detection pitch: " + str(frame_detections['pitch']))

        print(x_array)
        print(y_array)

        # For each detection, draw the corresponding bounding box with the associated pixels on the panorama
        for i in range(len(x_array)):
            # Get associated theta and phi angles
            print("x, y, z, radius, yaw, pitch: " ,x_array[i], y_array[i], z, radius, frame_detections['yaw'], frame_detections['pitch'])

            theta, phi = map_to_sphere(x_array[i], y_array[i], z, radius, np.radians(frame_detections['yaw']), np.radians(frame_detections['pitch']))

            print("Phi: ", np.degrees(phi))
            print("Theta: ", np.degrees(theta))

            # Convert theta and phi angles to associated pixels in the panorama
            U = phi * pano_width / (2 * np.pi)
            V = theta * pano_height / np.pi

            print("U: " + str(U))
            print("V: " + str(V))

            # Get label info from frame_detections
            frame_detection = frame_detections['image_detections'][i]
            class_id = frame_detection['class_id']
            confidence = frame_detection['confidence']

            for j in range(len(U)):

                # Take the smallest x and y value for top left corner and largest x and y value for the bottom right corner 
                x_pano = min(U)
                y_pano = min(V)
                x_plus_w_pano = max(U)
                y_plus_h_pano = max(V)

                # Round to nearest integer for convert for pixels
                x_pano = round(x_pano)
                y_pano = round(y_pano)
                x_plus_w_pano = round(x_plus_w_pano)
                y_plus_h_pano = round(y_plus_h_pano)

                draw_bounding_box(
                    pano_array,
                    class_id,
                    confidence,
                    x_pano,
                    y_pano,
                    x_plus_w_pano,
                    y_plus_h_pano,
                )

    # Return the final annotated pano_array
    return pano_array



