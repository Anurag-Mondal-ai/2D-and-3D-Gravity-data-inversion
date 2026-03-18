import numpy as np
from scipy.interpolate import BSpline, splev
from scipy.optimize import differential_evolution
import time

# Synthetic function for depth of the basin
def f(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2 * sigma**2))

# Generate synthetic data
xx = np.linspace(5.5, 11.5, 100)
yy = 1000 * (f(xx, 7, 2) + f(xx, 10, 1.5))

x_obs = np.linspace(0, 5000, 100)
depth = yy
density = -800
z_obs = 0

# Replace with actual gravity function
def poly_gravity(x_obs, z_obs, xx1, yy1, density, t_leg, c_leg):
    return np.zeros_like(x_obs)  # Dummy placeholder

# Replace with actual Legendre-Gauss function
def lgwt(N, a, b):
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(N)
    x_mapped = 0.5 * (x + 1) * (b - a) + a
    w_mapped = 0.5 * (b - a) * w
    return x_mapped, w_mapped

t_leg, c_leg = lgwt(10, 0, 1)
zz1 = poly_gravity(x_obs, z_obs, np.concatenate([x_obs, [x_obs[-1], 0]]),
                   np.concatenate([depth, [0, 0]]), density, t_leg, c_leg)

# B-spline basis function (simplified)
def b_spline_basis(t, n, k):
    knots = np.linspace(0, 1, n - k + 2)
    knots = np.concatenate(([0]*k, knots, [1]*k))
    basis = np.array([BSpline.basis_element(knots[i:i+k+1])(t) for i in range(len(knots)-k-1)]).T
    return basis

# Cost and constraints
def myCostFunction(x, data, n, k, t_leg, c_leg):
    t = np.linspace(0, 1, 100)
    N = b_spline_basis(t, n, k)
    yy = N @ x
    xx1 = np.linspace(0, 5000, 20)
    yy1 = np.interp(xx1, np.linspace(0, 5000, 100), yy)
    x1 = np.concatenate([xx1, [5000, 0]])
    y1 = np.concatenate([yy1, [0, 0]])
    zz1 = poly_gravity(np.linspace(0, 5000, 100), 0, x1, y1, -800, t_leg, c_leg)
    return np.linalg.norm(data - zz1)

def constraint_upper(x, n, k):
    t = np.linspace(0, 1, 100)
    N = b_spline_basis(t, n, k)
    yy = N @ x
    return max(0, np.min(yy))**2

def constraint_lower(x, n, k):
    t = np.linspace(0, 1, 100)
    N = b_spline_basis(t, n, k)
    yy = N @ x
    return max(0, np.max(yy) - 1500)**2

# Optimization loop
nn = [8, 9, 10, 11, 12, 13, 14, 15]
kk = [3, 4, 5, 6, 7]
value_data = []

for n in nn:
    for k in kk:
        nVar = n + 1
        bounds = [(-500, 500)] * nVar
        data = zz1

        def total_cost(x):
            return myCostFunction(x, data, n, k, t_leg, c_leg) + 1000 * (
                constraint_upper(x, n, k) + constraint_lower(x, n, k))

        start_time = time.time()
        result = differential_evolution(total_cost, bounds, maxiter=1000, popsize=15, recombination=0.8)
        duration = time.time() - start_time
        print(f"Total iterations for n={n}, k={k}: {result.nit}, time: {duration:.2f}s")

        value_data.append([n, k, result.nit, duration])
