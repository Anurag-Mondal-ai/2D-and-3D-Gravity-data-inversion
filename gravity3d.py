import numpy as np
from numba import njit, prange

# Gravitational constant (m^3/kg/s^2) scaled to mGal
G = 6.67430e-11
G_mGal = G * 1e5

@njit
def prism_gz_center(xo, yo, zo, xc, yc, zc, dx, dy, dz, drho):
    """
    Compute gz due to a rectangular prism centered at (xc, yc, zc)
    with dimensions dx, dy, dz and density contrast drho.
    """
    x1, x2 = xc - dx/2, xc + dx/2
    y1, y2 = yc - dy/2, yc + dy/2
    z1, z2 = zc - dz/2, zc + dz/2

    gz = 0.0
    for i in [x1, x2]:
        for j in [y1, y2]:
            for k in [z1, z2]:
                dx_ = i - xo
                dy_ = j - yo
                dz_ = k - zo
                R = np.sqrt(dx_**2 + dy_**2 + dz_**2 + 1e-10)
                sign = (-1) ** ((i == x2) + (j == y2) + (k == z2))
                gz += sign * np.arctan2(dx_ * dy_, dz_ * R + 1e-10)
    return G_mGal * drho * gz

@njit(parallel=True)
def compute_gravity_center(Xobs, Yobs, Zobs, x, y, z, dx, dy, dz, rho_model):
    """
    Compute gz at observation grid (Xobs, Yobs, Zobs) due to 3D density model
    using center-based prism contributions.
    """
    nx, ny, nz = rho_model.shape
    nobs_x, nobs_y = Xobs.shape
    gz_obs = np.zeros((nobs_x, nobs_y))

    for i in prange(nobs_x):
        for j in range(nobs_y):
            xo, yo, zo = Xobs[i, j], Yobs[i, j], Zobs[i, j]
            gz = 0.0
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        drho = rho_model[ix, iy, iz]
                        if drho == 0.0:
                            continue
                        xc, yc, zc = x[ix], y[iy], z[iz]
                        gz += prism_gz_center(xo, yo, zo, xc, yc, zc, dx, dy, dz, drho)
            gz_obs[i, j] = gz
    return gz_obs
