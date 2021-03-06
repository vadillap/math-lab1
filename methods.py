import numpy as np
from math import log


def dichotomy(f, epsilon, a, b):
    x = []

    while b - a > epsilon:
        m = (a + b) / 2
        if f(m + epsilon) < f(m - epsilon):
            a = m
        else:
            b = m

        x.append(m)

    return x


def gold(f, epsilon, a, b):
    x = []
    cg = (3 - 5 ** 0.5) / 2

    x1 = a + (b - a) * cg
    x2 = b - (b - a) * cg
    f1 = f(x1)
    f2 = f(x2)
    while b - a > epsilon:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) * cg
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - (b - a) * cg
            f2 = f(x2)
        x.append((x1 + x2) / 2)

    return x


def fibonacci(f, epsilon, a, b):
    x = []
    cg = (1 + 5 ** 0.5) / 2
    fib = lambda n: (cg ** n - (-cg) ** -n) / (2 * cg - 1)

    # вычислим количество итераций, которые дают заданную точность
    n = round((log((b - a) / epsilon) + log(5 ** 0.5)) / log(cg))

    x1 = a + fib(n - 2) / fib(n) * (b - a)
    x2 = a + fib(n - 1) / fib(n) * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    k = 1

    while k < n - 2:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + fib(n - k - 2) / fib(n - k) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + fib(n - k - 1) / fib(n - k) * (b - a)
            f2 = f(x2)
        x.append((x1 + x2) / 2)
        k += 1

    return x


def parabola(f, epsilon, a, b):
    x = []

    x1 = a
    x2 = (a + b) / 2
    x3 = b

    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    x.append(x2)

    while True:
        u = x2 - 0.5 * ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
                (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1))

        fu = f(u)

        if fu < f2:
            if u < x2:
                x3 = x2
                f3 = f2
            else:
                x1 = x2
                f1 = f2
            x2 = u
            f2 = fu
        else:
            if u < x2:
                x3 = u
                f3 = fu
            else:
                x1 = u
                f1 = fu

        x.append(u)

        if abs(x[-2] - x[-1]) < epsilon:
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
