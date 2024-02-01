import numpy as np

# Given values
x = 0
y = 0
z = 760
radius = 760
yaw = 270
pitch = 90

yaw_radian = np.radians(270)
pitch_radian = np.radians(90)

r = radius

# Equations to get the original point before projected onto plane
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

# Print the results
print("x, y, z, radius, yaw, pitch", x_circle, y_circle, z_circle, radius, yaw, pitch)
print("Theta Prime:", np.degrees(theta_prime))
print("Phi Prime:", np.degrees(phi_prime))