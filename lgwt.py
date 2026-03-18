import numpy as np

def lgwt(N, a, b):
    """
    Computes the Legendre-Gauss nodes and weights on the interval [a, b] with N points.

    Parameters:
    N (int): Number of quadrature points
    a (float): Lower bound of the interval
    b (float): Upper bound of the interval

    Returns:
    x (ndarray): Legendre-Gauss nodes
    w (ndarray): Legendre-Gauss weights
    """

    N = N - 1
    N1 = N + 1
    N2 = N + 2

    xu = np.linspace(-1, 1, N1)

    # Initial guess using Chebyshev-Gauss nodes
    y = np.cos((2 * np.arange(N1) + 1) * np.pi / (2 * N + 2)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)

    # Initialize Legendre-Gauss Vandermonde matrix and its derivative
    L = np.zeros((N1, N2))
    Lp = np.zeros((N1, N2))

    y0 = 2 * np.ones_like(y)

    # Newton-Raphson method
    while np.max(np.abs(y - y0)) > np.finfo(float).eps:
        L[:, 0] = 1
        Lp[:, 0] = 0

        L[:, 1] = y
        Lp[:, 1] = 1

        for k in range(1, N1):
            L[:, k+1] = ((2 * k + 1) * y * L[:, k] - k * L[:, k-1]) / (k + 1)

        Lp = N2 * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y ** 2)

        y0 = y
        y = y0 - L[:, N2 - 1] / Lp

    # Map from [-1, 1] to [a, b]
    x = (a * (1 - y) + b * (1 + y)) / 2

    # Compute the weights
    w = (b - a) / ((1 - y ** 2) * Lp ** 2) * (N2 / N1) ** 2

    return x, w
