import numpy as np
from scipy.interpolate import splev
from numpy.polynomial.legendre import leggauss


def lgwt(N, a, b):
    x, w = leggauss(N)
    t_leg = 0.5 * (x + 1) * (b - a) + a
    c_leg = 0.5 * (b - a) * w
    return t_leg, c_leg


def b_spline_basis(t, n, k):
    knots = np.linspace(0, 1, n + k + 1 - 2 * (k - 1))
    knots = np.pad(knots, (k - 1, k - 1), mode='edge')
    basis = np.zeros((len(t), n + 1))
    for i in range(n + 1):
        coeffs = np.zeros(n + 1)
        coeffs[i] = 1
        basis[:, i] = splev(t, (knots, coeffs, k - 1))
    return basis


def poly_gravity(x_obs, z_obs, x_poly, y_poly, density, t_leg, c_leg):
    area = 0.5 * np.abs(np.dot(x_poly, np.roll(y_poly, 1)) - np.dot(y_poly, np.roll(x_poly, 1)))
    g = 6.67430e-11
    Gz = density * g * area / len(x_obs)
    return np.full_like(x_obs, Gz)


def my_cost_function(x, data, n, k, t_leg, c_leg):
    xx = np.linspace(0, 5000, 100)
    t = np.linspace(0, 1, 100)
    N_final = b_spline_basis(t, n, k)
    yy = sum(x[i] * N_final[:, i] for i in range(n + 1))
    xx1 = np.linspace(min(xx), max(xx), 20)
    yy1 = np.interp(xx1, xx, yy)
    x1 = np.concatenate((xx1, [5000, 0]))
    y1 = np.concatenate((yy1, [0, 0]))
    zz1 = poly_gravity(xx, 0, x1, y1, -800, t_leg, c_leg)
    return np.linalg.norm(data - zz1)


def constrained1(x, n, k):
    t = np.linspace(0, 1, 100)
    N_final = b_spline_basis(t, n, k)
    yy = sum(x[i] * N_final[:, i] for i in range(n + 1))
    return max(0, -np.min(yy))**2


def constrained2(x, n, k):
    t = np.linspace(0, 1, 100)
    N_final = b_spline_basis(t, n, k)
    yy = sum(x[i] * N_final[:, i] for i in range(n + 1))
    return max(0, np.max(yy) - 1500)**2


def rand_spl(n, i):
    idx = list(range(n))
    idx.remove(i)
    return np.random.choice(idx, 3, replace=False)


def main():
    f = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    xx = np.linspace(5.5, 11.5, 100)
    yy = 1000 * (f(xx, 7, 2) + f(xx, 10, 1.5))
    t_leg, c_leg = lgwt(10, 0, 1)

    x_obs = np.linspace(0, 5000, 100)
    depth = yy
    xx1 = np.concatenate((x_obs, [x_obs[-1], 0]))
    yy1 = np.concatenate((depth, [0, 0]))
    zz1 = poly_gravity(x_obs, 0, xx1, yy1, -800, t_leg, c_leg)

    ff = np.arange(0.2, 2.2, 0.2)
    npp = np.arange(10, 45, 5)
    crpp = [0.7, 0.8, 0.9]
    kk = [0.3, 0.4, 0.5]

    itrr = 0
    stt_data = []

    for Cross_rate in crpp:
        for K in kk:
            for nPoP in npp:
                for F in ff:
                    data = zz1.copy()
                    n = 12
                    k = 4
                    nVar = n + 1
                    MaxIt = 5000
                    VarMin = -500 * np.ones(nVar)
                    VarMax = 500 * np.ones(nVar)

                    Particle = [{'Position': np.random.uniform(VarMin, VarMax), 'Cost': None} for _ in range(nPoP)]
                    for p in Particle:
                        p['Cost'] = my_cost_function(p['Position'], data, n, k, t_leg, c_leg) + \
                                    1000 * (constrained1(p['Position'], n, k) + constrained2(p['Position'], n, k))

                    for jj in range(1, MaxIt + 1):
                        Vector = []
                        for i in range(nPoP):
                            idxs = rand_spl(nPoP, i)
                            a, b, c = [Particle[j]['Position'] for j in idxs]
                            vi = Particle[i]['Position'] + K * (a - Particle[i]['Position']) + F * (b - c)
                            Vector.append({'Position': vi})

                        New_Particle = []
                        for i in range(nPoP):
                            ui = np.array([Vector[i]['Position'][j] if np.random.rand() <= Cross_rate or j == np.random.randint(nVar)
                                           else Particle[i]['Position'][j] for j in range(nVar)])
                            cost = my_cost_function(ui, data, n, k, t_leg, c_leg) + \
                                   1000 * (constrained1(ui, n, k) + constrained2(ui, n, k))
                            New_Particle.append({'Position': ui, 'Cost': cost})

                        Particle = [New_Particle[i] if New_Particle[i]['Cost'] <= Particle[i]['Cost'] else Particle[i]
                                    for i in range(nPoP)]
                        Particle.sort(key=lambda p: p['Cost'])
                        if Particle[0]['Cost'] <= 0.5:
                            break

                    print(f'Iterations for Cr={Cross_rate:.2f}, K={K:.2f}, F={F:.2f}, Np={nPoP} -> {jj}')
                    itrr += 1
                    stt_data.append([Cross_rate, K, F, nPoP, jj])


if __name__ == '__main__':
    main()
