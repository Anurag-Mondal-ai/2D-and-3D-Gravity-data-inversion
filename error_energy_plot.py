import numpy as np
import matplotlib.pyplot as plt

# Error Energy plot

# Noise-free data
y1 = np.loadtxt('model1_without_noise_fixed.dat')
y2 = np.loadtxt('model1_without_noise_varying.dat')
y3 = np.loadtxt('model2_without_noise_fixed.dat')
y4 = np.loadtxt('model2_without_noise_varying.dat')

# Plot for noise-free data (Figure 6(a))
plt.figure(1)
plt.semilogy(range(1, len(y1) + 1), y1, linewidth=2, label='Model1 Fixed density')
plt.semilogy(range(1, len(y2) + 1), y2, linewidth=2, label='Model1 Varying density')
plt.semilogy(range(1, len(y3) + 1), y3, linewidth=2, label='Model2 Fixed density')
plt.semilogy(range(1, len(y4) + 1), y4, linewidth=2, label='Model2 Varying density')
plt.ylim([1e-4, 1e4])
plt.xlabel('Generation')
plt.ylabel('Error Energy in mGal²')
plt.title('Error Energy plot for Noise Free data')
plt.legend()
plt.grid(True)

# Noisy data
y1 = np.loadtxt('model1_with_noise_fixed.dat')
y2 = np.loadtxt('model1_with_noise_varying.dat')
y3 = np.loadtxt('model2_with_noise_fixed.dat')
y4 = np.loadtxt('model2_with_noise_varying.dat')

# Plot for noisy data (Figure 6(b))
plt.figure(2)
plt.semilogy(range(1, len(y1) + 1), y1, linewidth=2, label='Model1 Fixed density')
plt.semilogy(range(1, len(y2) + 1), y2, linewidth=2, label='Model1 Varying density')
plt.semilogy(range(1, len(y3) + 1), y3, linewidth=2, label='Model2 Fixed density')
plt.semilogy(range(1, len(y4) + 1), y4, linewidth=2, label='Model2 Varying density')
plt.ylim([1e0, 1e4])
plt.xlabel('Generation')
plt.ylabel('Error Energy in mGal²')
plt.title('Error Energy plot for Noisy data')
plt.legend()
plt.grid(True)

plt.show()
