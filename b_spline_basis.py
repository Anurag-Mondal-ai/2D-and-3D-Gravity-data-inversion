import numpy as np
import matplotlib.pyplot as plt

def b_spline_basis(t, n, k):
    """
    Generate B-spline basis functions.

    Parameters:
        t (numpy.ndarray): 1D array of equally spaced data points from [0, 1].
        n (int): Number of knots.
        k (int): Order of the polynomial (degree = k - 1).

    Returns:
        numpy.ndarray: B-spline basis matrix of shape (len(t), n + 1).
    """
    len_t = len(t)
    r = n + k  # total number of segments
    u = np.linspace(t[0], t[-1], r + 1)

    # Set first k and last k values to boundary values
    u[:k] = t[0]
    u[n+1:] = t[-1]

    # Initialize basis function matrix
    N = np.zeros((len_t, r, k))

    # Compute N(i,1)
    for tt in range(len_t):
        for ii in range(r):
            if u[ii] <= t[tt] < u[ii+1]:
                N[tt, ii, 0] = 1
            else:
                N[tt, ii, 0] = 0
    N[-1, n, 0] = 1  # Ensure the last value is 1 as in MATLAB

    # Compute N(i,k) for k > 1
    for kk in range(1, k):
        for ii in range(r - kk):
            left_denom = u[ii + kk] - u[ii]
            right_denom = u[ii + kk + 1] - u[ii + 1]

            left = np.divide(t - u[ii], left_denom, out=np.zeros_like(t), where=left_denom != 0)
            right = np.divide(u[ii + kk + 1] - t, right_denom, out=np.zeros_like(t), where=right_denom != 0)

            N[:, ii, kk] = left * N[:, ii, kk - 1] + right * N[:, ii + 1, kk - 1]

    N_final = N[:, :n+1, k - 1]  # Extract final basis
    return N_final



# Parameters
t = np.linspace(0, 1, 100)
n = 4
k = 3

# Compute B-spline basis
N_final = b_spline_basis(t, n, k)
N_final.shape
# Plot the B-spline basis functions
plt.figure(figsize=(10, 6))
for i in range(N_final.shape[1]):
    plt.plot(t, N_final[:, i], label=f'B-spline basis {i+1}')
plt.title('B-spline Basis Functions')
plt.xlabel('t')
plt.ylabel('Basis value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
