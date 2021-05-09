from math import exp, sin, log
from matplotlib import pyplot as plt
import numpy as np

f = lambda x: exp(sin(x) * log(x))


def parabola(f, epsilon, x0):
    x = [x0]
    h = epsilon
    while True:
        x_prev = x[-1]
        x_next = x_prev - 0.5 * h * (f(x_prev + h) - f(x_prev - h)) / (f(x_prev + h) - 2 * f(x_prev) + f(x_prev - h))
        x.append(x_next)

        if abs(x_next - x_prev) < epsilon:
            break

    return x


def brent(f, epsilon, a, b):
    x_arr = []
    cg = (3 - 5 ** 0.5) / 2
    x = w = v = (a + b) / 2
    fw = fv = fx = f(x)

    deltax = 0.0
    while True:
        tol1 = epsilon * np.abs(x)
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if np.abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x
            else:
                deltax = b - x
            rat = cg * deltax
        else:
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat

            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x
                else:
                    deltax = b - x
                rat = cg * deltax

        if np.abs(rat) < tol1:
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = f(u)

        if fu > fx:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        x_arr.append(x)
    return x_arr


# x = parabola(f, 1e-6, 5.5)
x = brent(f, 1e-6, 3, 8)
print(x)
print(len(x))
r = np.arange(0.1, 10, 0.001)
plt.plot(r, list(map(f, r)))
plt.scatter(x, list(map(f, x)))
plt.scatter(x[-1], f(x[-1]))
plt.show()
