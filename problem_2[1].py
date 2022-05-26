import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags


def f(x):
    index = len(x[x <= 1])
    f_vector = np.zeros(len(x))
    f_vector[:index] = np.exp(-8*(x[:index]-0.5)**2)*np.sin(40*np.pi*x[:index])
    return f_vector


def u_exact(x, T=2):
    return f(x - c * T)


def Interpolation(h, k, L=6, T=2):
    M = int(L/h)
    N = int(T/k)

    sigma = (c * k) / h

    x = np.linspace(0, L, M+1)
    t = np.linspace(0, T, N+1)

    w0 = f(x)
    bc_1 = f(x[0] - c * t)
    bc_2 = f(x[-1] - c * t)

    w = np.zeros(len(x))
    for j in range(N):
        w[1:-1] = w0[1:-1] - 1/2*sigma * \
                (w0[2:] - w0[:-2]) + 1/2*(sigma**2) * \
                (w0[2:] - 2*w0[1:-1] + w0[:-2])
        w[0] = bc_1[j+1]
        w[-1] = bc_2[j+1]
        w0 = w.copy()

    return w  # Solution at t=T


def CrankNicolson(h, k, L=6, T=2):
    M = int(L/h)
    N = int(T/k)

    sigma = (c * k) / h

    x = np.linspace(0, L, M+1)
    t = np.linspace(0, T, N+1)

    w0 = f(x)
    bc_1 = f(x[0] - c * t)
    bc_2 = f(x[-1] - c * t)

    diag = [-sigma/4 * np.ones(M-2), np.ones(M-1), sigma/4 * np.ones(M-2)]
    A = diags(diag, [-1, 0, 1]).toarray()

    diag0 = [sigma/4 * np.ones(M-2), np.ones(M-1), -sigma/4 * np.ones(M-2)]
    A0 = diags(diag0, [-1, 0, 1]).toarray()

    b_bc = np.zeros(len(x) - 2)
    for j in range(N):
        b_bc[0] = sigma/4 * bc_1[j+1]
        b_bc[-1] = -sigma/4 * bc_2[j+1]

        b = np.matmul(A0, w0[1:-1]) + b_bc
        aux = np.linalg.solve(A, b)

        w = np.concatenate(([bc_1[j+1]], aux, [bc_2[j+1]]))

        w0 = w.copy()

    return w  # Solution at t=T


def UpwindLeapfrog(h, k, L=6, T=2):

    M = int(L/h)
    N = int(T/k)

    sigma = (c * k) / h

    x = np.linspace(0, L, M+1)
    t = np.linspace(0, T, N+1)

    w0 = f(x)
    bc_1 = f(x[0] - c * t)
    bc_2 = f(x[-1] - c * t)

    #
    # BTCS to compute w(x,t_1)
    #
    diag = [-sigma/2 * np.ones(M-2), np.ones(M-1), sigma/2 * np.ones(M-2)]
    A = diags(diag, [-1, 0, 1]).toarray()

    b_bc = np.zeros(len(x) - 2)
    b_bc[0] = sigma/2 * bc_1[1]
    b_bc[-1] = -sigma/2 * bc_2[1]

    b = w0[1:-1] + b_bc
    aux = np.linalg.solve(A, b)
    w1 = np.concatenate(([bc_1[1]], aux, [bc_2[1]]))

    w = np.zeros(len(x))
    for j in range(1, N):
        w[1:-1] = w0[:-2] + (1 - 2*sigma) * (w1[1:-1] - w1[:-2])
        w[0] = bc_1[j+1]
        w[-1] = bc_2[j+1]
        w0 = w1.copy()
        w1 = w.copy()

    return w  # Solution at t=T

#
# Parameters
#

c = 2

h = 0.01
k = 0.004

L = 6.0
T = 2.0

M = int(L/h)

x = np.linspace(0, L, M+1)

# Exact solution
u = u_exact(x, T)

############
# Part (b) #
############

#
# Interpolation
#

w_scheme_1 = Interpolation(h, k, L, T)

plt.plot(x, u)
plt.plot(x, w_scheme_1, '-')
plt.legend(['u', 'w_scheme_1'])
plt.title(f'Interpolation (t={T:.1f})')
plt.savefig('problem_2_interpolation.png')
plt.show()
plt.close()

############
# Part (c) #
############

#
# Crank-Nicolson
#

w_cn = CrankNicolson(h, k, L, T)

plt.plot(x, u)
plt.plot(x, w_cn, '-')
plt.legend(['u', 'w_cn'])
plt.title(f'Crank-Nicolson (t={T:.1f})')
plt.savefig('problem_2_CrankNicolson.png')
plt.show()
plt.close()

#
# Upwind leapfrog
#

w_upw_leap = UpwindLeapfrog(h, k, L, T)

plt.plot(x, u)
plt.plot(x, w_upw_leap, '-')
plt.legend(['u', 'w_upw_leap'])
plt.title(f'Upwind leapfrog (t={T:.1f})')
plt.savefig('problem_2_UpwindLeapfrog.png')
plt.show()
plt.close()
