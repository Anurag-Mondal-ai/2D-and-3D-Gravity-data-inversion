import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

# Define the Gaussian Legendre weights and nodes
def lgwt(N, a, b):
    x, w = np.polynomial.legendre.leggauss(N)
    x = 0.5 * (x + 1) * (b - a) + a
    w = 0.5 * (b - a) * w
    return x, w

# Function for gravity calculation
def poly_gravityrho(x_obs, z_obs, x, y, rho_func, t_leg, c_leg):
    grav = np.zeros_like(x_obs)
    epsilon = 1e-10  # Small value to prevent division by zero
    for i in range(len(x_obs)):
        integrand = np.zeros(len(t_leg))
        for j in range(len(t_leg)):
            for k in range(1, len(x)):
                dx = x[k] - x[k-1]
                dy = y[k] - y[k-1]
                dist = np.sqrt((x_obs[i] - x[k])**2 + (z_obs - y[k])**2)
                dist = max(dist, epsilon)  # Prevent division by zero
                
                rho = rho_func(y[k])
                integrand[j] += rho * dx * dy / dist
        grav[i] = np.sum(integrand * c_leg)
    return grav

# Synthetic function for depth of the basin
def f(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

# Creating synthetic depth profile
xx = np.linspace(5.5, 8.5, 100)
yy = 1000 * f(xx, 7, 2)

# Legendre Gaussian quadrature points for numerical integration
t_leg, c_leg = lgwt(10, 0, 1)

# Synthetic depth and observation points
x_obs = np.linspace(0, 5000, 100)
depth = yy

# Plotting x_obs vs depth
plt.figure(1)
plt.subplot(2, 1, 2)
plt.plot(x_obs, depth, linewidth=1.25)  # Set linewidth directly in plot function
plt.gca().invert_yaxis()
plt.xlabel('Distance in meter')
plt.ylabel('Depth in meter')
plt.title('True Model and Prismatic Approximation of depth profile for synthetic basin (Model 1)')

# Finding Gravity field of the basin for density rho(z)= (-0.55-2.5*10^-3.*z).*1000 kg/m^3
density = -800
z_obs = 0  # height of observation point is on the surface

# Polygonic profile of the basin
xx1 = np.concatenate([x_obs, [x_obs[-1], 0]])
yy1 = np.concatenate([depth, [0, 0]])

zz1 = poly_gravityrho(x_obs, z_obs, xx1, yy1, lambda z: (-0.55 - 2.5e-3 * z) * 1000, t_leg, c_leg)

# Plotting the Gravity field anomaly
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(x_obs, zz1, linewidth=1.25)  # Set linewidth directly in plot function
plt.xlabel('Observation points in meter')
plt.ylabel('Gravity anomaly in mGal')
plt.title('Gravity anomaly for Model 1')

# Prismatic model
nn = 50  # number of prisms
xx1 = np.linspace(0, 5000, nn)
yy1 = CubicSpline(x_obs, depth)(xx1)

# Loop for plotting all prisms along the depth profile
grav = np.zeros_like(x_obs)
for ii in range(len(xx1) - 1):
    # Vertices of each prism
    x1 = [xx1[ii], xx1[ii], xx1[ii+1], xx1[ii+1]]
    y1 = [0, yy1[ii+1], yy1[ii+1], 0]
    
    # Plotting each prism
    plt.figure(1)
    plt.subplot(2, 1, 2)
    polygon = plt.Polygon(list(zip(x1, y1)), color='red', alpha=0.1)
    plt.gca().add_patch(polygon)
    
    # Gravity anomaly for each prism at each observation point
    zz11 = poly_gravityrho(x_obs, z_obs, x1, y1, lambda z: (-0.55 - 2.5e-3 * z) * 1000, t_leg, c_leg)
    
    # Sum of gravity anomaly of each prism
    grav += zz11

# Plotting the gravity anomaly
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(x_obs, grav, '-o', linewidth=1.25)  # Set linewidth directly in plot function
plt.legend(['Observed', 'Prismatic'], loc='best')
plt.xlim([0, 5000])

# Initialization of loop
nn = 5  # number of prisms
RMSE_g = 10

# Loop for prismatic model
while RMSE_g >= 0.19:  # Desired stopping criterion
    nn += 1  # Increment in number of prisms
    xx1 = np.linspace(0, 5000, nn)
    yy1 = CubicSpline(x_obs, depth)(xx1)
    
    # Gravity field due to nn prisms
    grav = np.zeros_like(x_obs)
    for ii in range(len(xx1) - 1):
        x1 = [xx1[ii], xx1[ii], xx1[ii+1], xx1[ii+1]]
        y1 = [0, yy1[ii+1], yy1[ii+1], 0]
        zz11 = poly_gravityrho(x_obs, z_obs, x1, y1, lambda z: (-0.55 - 2.5e-3 * z) * 1000, t_leg, c_leg)
        grav += zz11
    
    # RMSE error calculation
    N_g = len(grav)
    
    # Check if grav contains valid values
    if np.any(np.isnan(grav)) or np.any(np.isinf(grav)):
        print("Error: grav contains NaN or infinite values")
        break  # Exit if invalid values are found
    
    # Avoid division by zero in RMSE calculation
    grav_range = np.max(grav) - np.min(grav)
    if grav_range == 0:
        print("Warning: Zero range in gravity anomalies, RMSE cannot be calculated")
        RMSE_g = np.nan  # Assign NaN if there's a division by zero
        break  # Exit the loop if division by zero occurs
    else:
        RMSE_g = (np.sqrt(np.sum((grav - zz1)**2) / N_g) / grav_range) * 100
        # Printing RMSE for each number of prisms
        print(f"For n={nn} RMSE in gravity field={RMSE_g:.6f}")

# RMSE of Prismatic model having desired accuracy
print(f'For n={nn} RMSE in gravity field={RMSE_g:.6f}')
