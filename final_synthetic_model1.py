import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import make_interp_spline
from numpy.linalg import norm
import poly_gravity as pg
import b_spline_basis as bsb
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

# ------------------------- Synthetic Model -------------------------
def synthetic_model1():
    f = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    xx = np.linspace(0, 5000, 100)
    yy = 150 * f((xx - 2500) / 1000, 0, 1.5)
    t_leg, c_leg = np.polynomial.legendre.leggauss(10)
    x_obs = xx
    depth = yy
    density = -800
    z_obs = 0

    xx1 = np.concatenate((x_obs, [x_obs[-1], 0]))
    yy1 = np.concatenate((depth, [0, 0]))
    zz1 = pg.poly_gravity(x_obs, z_obs, xx1, yy1, density, t_leg, c_leg)
    zz2 = zz1.copy()
    zz1 += np.sqrt(0.05) * np.random.randn(len(zz1))

    return x_obs, depth, zz1, zz2, t_leg, c_leg, density

# ------------------------- Cost Function -------------------------
def my_cost_function(x, data, n, k, t_leg, c_leg):
    xx = np.linspace(0, 5000, 100)
    t = np.linspace(0, 1, 100)
    N_final = bsb.b_spline_basis(t, n, k)
    yy = np.dot(N_final, x[:n+1])
    xx1 = np.linspace(np.min(xx), np.max(xx), 30)
    spline_interp = make_interp_spline(xx, yy)
    yy1 = spline_interp(xx1)
    x1 = np.concatenate([xx1, [5000, 0]])
    y1 = np.concatenate([yy1, [0, 0]])
    zz1 = pg.poly_gravity(xx, 0, x1, y1, -800, t_leg, c_leg)
    return norm(data - zz1)

def constrained1(x, n, k):
    t = np.linspace(0, 1, 100)
    N_final = bsb.b_spline_basis(t, n, k)
    yy = np.dot(N_final, x[:n+1])
    return max(0, -np.min(yy))**2

def constrained2(x, n, k):
    t = np.linspace(0, 1, 100)
    N_final = bsb.b_spline_basis(t, n, k)
    yy = np.dot(N_final, x[:n+1])
    return max(0, np.max(yy) - 150)**2

# ------------------------- Inversion -------------------------
x_obs, depth, zz1, zz2, t_leg, c_leg, density = synthetic_model1()

n = 9
k = 4
nVar = n + 1
Cross_rate = 0.8
K = 0.5
F = 0.8
nPoP = 30
MaxIt = 1000
data = zz1

def objective_function(x, data):
    return my_cost_function(x, data, n, k, t_leg, c_leg) + \
           1000 * (constrained1(x, n, k) + constrained2(x, n, k))

bounds = [(-500, 500)] * nVar
result = differential_evolution(objective_function, bounds, args=(data,),
                                popsize=nPoP, maxiter=MaxIt, mutation=(K, F),
                                recombination=Cross_rate)

best_var = result.x
error_energy = result.fun ** 2

# ------------------------- Evaluate Model -------------------------
xx = x_obs
t = np.linspace(0, 1, 100)
N_final = bsb.b_spline_basis(t, n, k)
yy_eval = N_final @ best_var[:n+1]

xx1 = np.linspace(min(xx), max(xx), 30)
yy1 = make_interp_spline(xx, yy_eval)(xx1)
yy2 = make_interp_spline(x_obs, depth)(xx1)

x1 = np.concatenate((xx1, [x_obs[-1], 0]))
y1 = np.concatenate((yy1, [0, 0]))
zz1_pred = pg.poly_gravity(xx, 0, x1, y1, density, t_leg, c_leg)

# ------------------------- RMSE -------------------------
RMSE_g = np.sqrt(np.mean((data - zz1_pred) ** 2)) / (max(data) - min(data))
RMSE_true = np.sqrt(np.mean((data - zz2) ** 2)) / (max(data) - min(data))
RMSE_d = np.sqrt(np.mean((yy2 - yy1) ** 2)) / (max(yy2) - min(yy2))

print(f'RMSE in gravity field = {RMSE_g:.6f}')
print(f'True RMSE in gravity field = {RMSE_true:.6f}')
print(f'RMSE in depth profile = {RMSE_d:.6f}')

# ------------------------- Plotting -------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Panel a
axs[0, 0].plot(x_obs, zz2, 'k--', linewidth=1.25, label='Observed data')
axs[0, 0].plot(x_obs, zz1_pred, 'r-', linewidth=1.25, label='Inverted data')
axs[0, 0].set_xlabel('Observation points (m)')
axs[0, 0].set_ylabel('Gravity anomaly (mGal)')
axs[0, 0].set_title('a)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Panel b
axs[0, 1].plot(x_obs, zz1, 'b-', linewidth=1.25, label='Observed data')
axs[0, 1].plot(x_obs, zz2, 'r-', linewidth=1.25, label='Inverted data')
axs[0, 1].set_xlabel('Observation points (m)')
axs[0, 1].set_ylabel('Gravity anomaly (mGal)')
axs[0, 1].set_title('b)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Panel c
axs[1, 0].fill_between(xx1, yy2, 150, color='royalblue')
axs[1, 0].plot(xx1, yy2, 'k-', linewidth=1.25, label='True model')
axs[1, 0].plot(xx1, yy1, 'r-', linewidth=1.25, label='Best model')
axs[1, 0].set_ylim(150, 0)
axs[1, 0].set_xlabel('Distance (m)')
axs[1, 0].set_ylabel('Depth (m)')
axs[1, 0].set_title('c)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Panel d
axs[1, 1].fill_between(xx1, yy2, 150, color='royalblue')
axs[1, 1].plot(xx1, yy2, 'k-', linewidth=1.25, label='True model')
axs[1, 1].plot(xx1, yy1, 'r-', linewidth=1.25, label='Best model')
axs[1, 1].set_ylim(150, 0)
axs[1, 1].set_xlabel('Distance (m)')
axs[1, 1].set_ylabel('Depth (m)')
axs[1, 1].set_title('d)')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Colorbar
norm = mcolors.Normalize(vmin=-850, vmax=-600)
sm = ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal', pad=0.08)
cbar.set_label('Density (kg/m³)')

plt.tight_layout()
plt.show()

