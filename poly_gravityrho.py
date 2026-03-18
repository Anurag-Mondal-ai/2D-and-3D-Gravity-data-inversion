import numpy as np

def poly_gravityrho(x_obs, z_obs, x, z, rho, t, c):
    """
    Calculates the z-component of gravity field due to a 2D polygonal body
    with depth-varying density contrast using Gauss-Legendre quadrature.

    Parameters:
    x_obs : ndarray
        Array of observation x-coordinates.
    z_obs : float
        Depth of observation (positive downward).
    x : ndarray
        x-coordinates of polygon vertices (counterclockwise).
    z : ndarray
        z-coordinates of polygon vertices (counterclockwise).
    rho : callable
        A function that returns density contrast at given depth (z).
    t : ndarray
        Gauss-Legendre quadrature points in [0, 1].
    c : ndarray
        Gauss-Legendre quadrature weights.

    Returns:
    grav : ndarray
        Gravity field in mGal at observation points.
    """

    G = 6.67408e-11  # Gravitational constant (m^3/kg/s^2)
    x = np.append(x, x[0])  # Close polygon
    z = np.append(z, z[0])
    n_poly = len(x) - 1

    grav = np.zeros_like(x_obs, dtype=np.float64)

    for i in range(len(x_obs)):
        value = np.zeros(n_poly)
        for j in range(n_poly):
            # Parametric points along the polygon side
            xj_t = x[j] * (1 - t) + x[j + 1] * t
            zj_t = z[j] * (1 - t) + z[j + 1] * t

            ax1 = xj_t - x_obs[i]
            ax2 = zj_t - z_obs

            rr = rho(zj_t)  # depth-dependent density
            ax = -2 * rr * G * np.arctan2(ax1, ax2) * (z[j + 1] - z[j])

            value[j] = np.sum(c * ax)

        grav[i] = 1e5 * np.sum(value)  # Convert to mGal

    return grav
