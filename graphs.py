from math import exp, sin, log
import numpy as np
from matplotlib import pyplot as plt
from methods import *


# обертка, чтобы считать кол-во вызовов функции
class Counter:
    def __init__(self, f):
        self._f = f
        self._calls = 0

    def call(self, x):
        self._calls += 1
        return self._f(x)

    def func(self):
        return lambda x: self.call(x)

    def count(self):
        return self._calls

    def reset(self):
        self._calls = 0


def epsilon_dependency(f, method, epsilon_range=np.arange(1e-10, 1e-1, 1e-5), x_range=(0.1, 10)):
    counter = Counter(f)
    wrapped_f = counter.func()

    epsilon_x = []
    iterations_y = []
    calls_y = []

    for epsilon in epsilon_range:
        x = method(wrapped_f, epsilon, *x_range)

        epsilon_x.append(epsilon)
        iterations_y.append(len(x))
        calls_y.append(counter.count())
        counter.reset()

    return epsilon_x, iterations_y, calls_y


def draw_individual(f, method, title):
    epsilon_x, iterations_y, calls_y = epsilon_dependency(f, method)

    plt.plot(epsilon_x, iterations_y, label="Итерации")
    plt.plot(epsilon_x, calls_y, label="Вызовы функций")

    plt.xscale("log")
    plt.gca().invert_xaxis()

    plt.title(title)
    plt.xlabel("Точность")
    plt.ylabel("Количество")
    plt.legend()
    plt.grid()
    plt.show()


def draw_all(f, methods):
    for method in methods:
        epsilon_x, iterations_y, calls_y = epsilon_dependency(f, method)
        plt.plot(epsilon_x, iterations_y)
        # plt.plot(epsilon_x, calls_y)

    plt.xscale("log")
    plt.gca().invert_xaxis()

    plt.xlabel("Точность")
    plt.ylabel("Количество")
    plt.grid()
    plt.show()


def task2():
    f = lambda x: exp(sin(x) * log(x))
    draw_individual(f, gold, "Золтое сечение")

    draw_all(f, (gold, dichotomy, fibonacci, brent))


if __name__ == '__main__':
    task2()
