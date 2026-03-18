import numpy as np

def poly_gravity(x_obs, z_obs, x, z, roh, t, c):
    """
    Calculates the vertical (z) component of gravity field for a 2D polygon body.

    Parameters:
    - x_obs: array of x-coordinates of observation points
    - z_obs: scalar, z-level of observations (typically 0)
    - x, z: vertices of the polygon (must be in counter-clockwise order)
    - roh: density contrast (kg/m^3)
    - t, c: Gauss-Legendre points and weights (0 to 1 interval)

    Returns:
    - grav: gravity anomaly at observation points (in mGal)
    """
    x = np.array(x)
    z = np.array(z)
    G = 6.67408e-11  # Gravitational constant in m^3/kg/s^2
    n_poly = len(x)

    # Close the polygon
    x = np.append(x, x[0])
    z = np.append(z, z[0])

    grav = np.zeros_like(x_obs)

    for i in range(len(x_obs)):
        value = np.zeros(n_poly)

        for j in range(n_poly):
            # Parametrize line segment with t in [0, 1]
            x_t = x[j] * (1 - t) + x[j + 1] * t
            z_t = z[j] * (1 - t) + z[j + 1] * t

            ax1 = x_t - x_obs[i]
            ax2 = z_t - z_obs

            # Avoid divide-by-zero warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                integrand = -2 * roh * G * np.arctan2(ax1, ax2) * (z[j + 1] - z[j])
                integrand = np.nan_to_num(integrand)  # Convert NaNs/infs to 0

            value[j] = np.sum(c * integrand)

        grav[i] = 1e5 * np.sum(value)  # Convert to mGal

    return grav

