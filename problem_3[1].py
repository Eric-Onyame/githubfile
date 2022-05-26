import numpy as np
import matplotlib.pyplot as plt


def init_condition(x):
    return 3 * np.sin(x)


def u_exact(x, T=1):
    return 3 * (np.e**(-2*T)) * np.sin(x - 2*T)


def FTCS(h, k, L=2*np.pi, T=1, verbosity=0):
    M = int(L/h)
    N = int(T/k)

    beta = (mu * k) / (h**2)
    sigma = (c * k) / h

    if verbosity:
        print('CFL diffusion (beta <= 1/2) -- beta =', beta)
        print('CFL advection (sigma <= 1) -- sigma =', sigma)
        print('Coupling tability condition (sigma^2 - 2beta <= 0) -- condition =',
              sigma**2 - 2*beta)
        print('')
    if beta > 1/2:
        raise(Exception('CFL for diffusion not satisfied.'))
    if sigma > 1:
        raise(Exception('CFL for advection not satisfied.'))
    if sigma**2 > 2*beta:
        raise(Exception('Coupling stability condition not satisfied.'))

    x = np.linspace(0, L, M+1)
    t = np.linspace(0, T, N+1)

    w0 = init_condition(x)
    bc_1 = u_exact(x[0], t)
    bc_2 = u_exact(x[-1], t)

    w = np.zeros(len(x))

    for j in range(N):
        w[1:-1] = w0[1:-1] - 1/2*sigma * \
            (w0[2:] - w0[:-2]) + beta*(w0[2:] - 2*w0[1:-1] + w0[:-2])
        w[0] = bc_1[j+1]
        w[-1] = bc_2[j+1]
        w0 = w.copy()

    return w  # Solution at t=T

#
# Parameters
#
c = 2
mu = 2

############
# Part (b) #
############

h = 0.05
k = 2e-4

L = 2.0 * np.pi
T = 1.0

M = int(L/h)

x = np.linspace(0, L, M+1)

# Exact solution
u = u_exact(x, T)

w_FTCS = FTCS(h, k, L, T, verbosity=1)

plt.plot(x, u)
plt.plot(x, w_FTCS, '-')
plt.legend(['u', 'w_FTCS'])
plt.title(f'FTCS (t={T:.1f})')
plt.savefig('problem_3_FTCS.png')
plt.show()
plt.close()

############
# Part (c) #
############

#
# Order of accuracy in space (norm L2)
#
L = 0.1
T = 0.01

h = 0.01
k = 5e-8
max_iter = 5

print('Order of accuracy in space (norm L2):')
iter = 0
u_vector = []
w_vector = []
h_vector = []
while iter < max_iter:
    M = int(L/h)
    x = np.linspace(0, L, M+1)
    h_vector.append(h)

    u_vector.append(u_exact(x, T))
    w_vector.append(FTCS(h, k, L, T))

    h /= 2
    iter += 1

for i in range(max_iter-1):
    p = np.log((np.sqrt(h_vector[i]) * np.linalg.norm(u_vector[i] - w_vector[i]))
               / (np.sqrt(h_vector[i+1]) * np.linalg.norm(u_vector[i+1] - w_vector[i+1]))) \
               / np.log(2)
    print(p)

#
# Order of accuracy in time (extrapolation)
#
L = 0.1
T = 0.01

h = 0.001
k = 1e-7

M = int(L/h)
x = np.linspace(0, L, M+1)
index = round(M/2)
max_iter = 5

print('\nOrder of accuracy in time (extrapolation):')
iter = 0
u_vector = []
w_vector = []
while iter < max_iter:
    u_vector.append(u_exact(x, T)[index])
    w_vector.append(FTCS(h, k, L, T)[index])

    k /= 2
    iter += 1

for i in range(max_iter-2):
    p = np.log(abs(w_vector[i] - w_vector[i+1]) / abs(w_vector[i+1] - w_vector[i+2])) / np.log(2)
    print(p)
