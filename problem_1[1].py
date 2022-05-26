import numpy as np


def f(t, y):
    return 2*y + 4*t


def y_exact(t=1):
    return -1 - 2*t + 2*np.e**(2*t)


def Scheme(h, T=1):
    N = int(T/h)

    t0 = 0
    t1 = t0 + h

    w0 = 1

    #
    # Explicit Trapezoid
    #
    w1 = w0 + h/2*(f(t0, w0) + f(t1, w0 + h * f(t0, w0)))

    for i in range(N-1):
        w = 1/3*w1 + 2/3*w0 + 11*h/6*f(t1, w1) - h/6*f(t0, w0)
        w0 = w1
        w1 = w
        t0 += h
        t1 += h

    return w  # Solution at t=T

#
# Solution at t=1
#
y = y_exact(1)
print('w at t=1 (h=0.001):', Scheme(0.001, 1))
print('y at t=1:', y)

#
# Order of accuracy
#
h = 1/10
iter = 0
w_vector = []
while iter < 5:
    w_vector.append(Scheme(h))
    h /= 2
    iter += 1

print('\nOrder of accuracy:')
for i in range(4):
    p = np.log(abs(y - w_vector[i]) / abs(y - w_vector[i+1])) / np.log(2)
    print(p)
