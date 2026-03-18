
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.special import roots_legendre
from math import factorial

G = 6.67430e-11  # gravitational constant (m³/kg/s²)

def gauss_fft_gravity_3d(dx, dy, dz, zc, rho_model, z0=0.0, n_terms=6, nq=8):
    """
    Gauss-FFT forward gravity modeling for general 3D density contrast.

    Parameters
    ----------
    dx, dy, dz : float
        Grid spacing in x, y, z (in meters).
    zc : 1D array
        Center depths of vertical layers (nz,).
    rho_model : 3D array
        Density contrast model (nx, ny, nz) in kg/m³.
    z0 : float
        Observation height (typically 0 for surface).
    n_terms : int
        Number of Taylor series terms.
    nq : int
        Number of Gauss–Legendre quadrature points for depth integration.

    Returns
    -------
    gz_mGal : 2D array
        Vertical gravity anomaly (nx, ny) in mGal.
    """
    nx, ny, nz = rho_model.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    # Wavenumber grid
    kx = 2 * np.pi * fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    Kmod = np.sqrt(KX**2 + KY**2)
    Kmod[0, 0] = 1e-10  # avoid division by zero

    # Gaussian quadrature nodes and weights on [-1, 1]
    xi, wi = roots_legendre(nq)

    # Depth slab midpoints and layer thicknesses
    dz_array = np.full(nz, dz)

    # Compute FFT of moment integrals
    gz_fft = np.zeros((nx, ny), dtype=np.complex128)

    for n in range(n_terms):
        moment_n = np.zeros((nx, ny))

        for k in range(nz):
            z_k = zc[k]
            z_half = 0.5 * dz_array[k]
            z_nodes = z_k + z_half * xi
            weights = wi * z_half

            for q in range(nq):
                zq = z_nodes[q]
                weight = weights[q]
                phi = (zq - z_k) ** n
                moment_n += rho_model[:, :, k] * phi * weight

        # FFT of nth moment
        F_n = fft2(moment_n)
        coeff = ((-Kmod) ** n) / factorial(n)
        gz_fft += coeff * F_n

    gz_fft *= 2 * np.pi * G * np.exp(Kmod * z0)
    gz = np.real(ifft2(gz_fft)) * 1e5  # Convert to mGal

    return gz
