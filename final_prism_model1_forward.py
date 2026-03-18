import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Function to compute Legendre-Gauss quadrature weights and nodes
def lgwt(N, a, b):
    x = np.cos(np.pi * (4*np.arange(1, N+1) - 1) / (2*N + 1))
    P0 = np.zeros(N)
    P1 = np.ones(N)
    
    for k in range(1, 30):
        P2 = ((2*k + 1)*x*P1 - k*P0) / (k + 1)
        P0 = P1
        P1 = P2
        x = x - P2 / (2*k + 1)
    
    w = 2 * (b - a) / ((1 - x**2) * (P1**2))
    x = 0.5 * (x + 1) * (b - a) + a
    return x, w

# Synthetic function for depth of the basin (Gaussian function)
def f(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

# Parameters
mu = 7
sigma = 2
x_obs = np.linspace(0, 5000, 100)
xx = np.linspace(5.5, 8.5, 100)
yy = 1000 * f(xx, mu, sigma)

# Legendre-Gauss quadrature points for numerical integration
t_leg, c_leg = lgwt(10, 0, 1)

# Plotting depth profile
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 2)
plt.plot(x_obs, yy)
plt.gca().invert_yaxis()
plt.xlabel('Distance in meter')
plt.ylabel('Depth in meter')
plt.title('True Model and Prismatic Approximation of depth profile for synthetic basin (Model 1)')

# Gravity field for density -800 kg/m^3
density = -800
z_obs = 0

# Polygonal profile of the basin
xx1 = np.concatenate([x_obs, [x_obs[-1], 0]])
yy1 = np.concatenate([yy, [0, 0]])

# Function for calculating gravity anomaly (placeholder, needs actual implementation)
def poly_gravity(x_obs, z_obs, xx1, yy1, density, t_leg, c_leg):
    # Placeholder function, replace with actual gravity anomaly computation
    return np.random.normal(0, 1, len(x_obs))

# Calculate gravity anomaly
zz1 = poly_gravity(x_obs, z_obs, xx1, yy1, density, t_leg, c_leg)

# Plot gravity anomaly
plt.subplot(2, 1, 1)
plt.plot(x_obs, zz1, label='Observed', linewidth=1.25)
plt.xlabel('Observation points in meter')
plt.ylabel('Gravity anomaly in mGal')
plt.title('Gravity anomaly for Model 1')
plt.legend()

# Prismatic model
nn = 50
xx1 = np.linspace(0, 5000, nn)
yy1 = CubicSpline(x_obs, yy)(xx1)

grav = np.zeros_like(x_obs)
for ii in range(len(xx1) - 1):
    # Vertices of each prism
    x1 = [xx1[ii], xx1[ii], xx1[ii+1], xx1[ii+1]]
    y1 = [0, yy1[ii+1], yy1[ii+1], 0]
    
    # Gravity anomaly for each prism
    zz11 = poly_gravity(x_obs, z_obs, x1, y1, density, t_leg, c_leg)
    grav += zz11

# Plotting prismatic model gravity anomaly
plt.plot(x_obs, grav, '-o', label='Prismatic', linewidth=1.25)
plt.xlim([0, 5000])
plt.legend()

# Initialization for prismatic model loop
RMSE_g = 10
nn = 5

# Loop for prismatic model refinement
while RMSE_g >= 0.18:
    nn += 1
    xx1 = np.linspace(0, 5000, nn)
    yy1 = CubicSpline(x_obs, yy)(xx1)
    
    grav = np.zeros_like(x_obs)
    for ii in range(len(xx1) - 1):
        x1 = [xx1[ii], xx1[ii], xx1[ii+1], xx1[ii+1]]
        y1 = [0, yy1[ii+1], yy1[ii+1], 0]
        zz11 = poly_gravity(x_obs, z_obs, x1, y1, density, t_leg, c_leg)
        grav += zz11
    
    N_g = len(grav)
    RMSE_g = (np.sqrt(np.sum((grav - zz1)**2) / N_g) / (np.max(grav) - np.min(grav))) * 100
    
    if nn == 10:
        print(f"For n={nn} RMSE in gravity field={RMSE_g:.6f}.")

# Final RMSE and number of prisms
print(f'For n={nn} RMSE in gravity field={RMSE_g:.6f}.')
plt.show()