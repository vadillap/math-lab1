from math import exp, sin, log
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


def epsilon_dependency(f, method, epsilon_range=np.arange(1e-10, 1e-1, 1e-5), x_range=(2, 7)):
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


def draw_individual(f, method):
    epsilon_x, iterations_y, calls_y = epsilon_dependency(f, method["method"])

    plt.plot(epsilon_x, iterations_y, label="Итерации")
    plt.plot(epsilon_x, calls_y, label="Вызовы функций")

    plt.xscale("log")
    plt.gca().invert_xaxis()

    plt.title(method["name"])
    plt.xlabel("Точность")
    plt.ylabel("Количество")
    plt.legend()
    plt.grid()
    plt.show()


def draw_all(f, methods):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    fig.set_figheight(fig.get_figheight() * 1.5)
    for method in methods:
        epsilon_x, iterations_y, calls_y = epsilon_dependency(f, method["method"])
        ax1.plot(epsilon_x, iterations_y, label=method["name"])
        ax2.plot(epsilon_x, calls_y, label=method["name"])

    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.grid()
    ax2.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=3)

    plt.xlabel("Точность")
    ax1.set_ylabel("Итерации")
    ax2.set_ylabel("Вызовы функции")
    fig.tight_layout()
    plt.show()


def task2():
    f = lambda x: exp(sin(x) * log(x))

    methods = (
        {
            "method": dichotomy,
            "name": "Дихотомия",
        },
        {
            "method": gold,
            "name": "Золотое сечение",
        },
        {
            "method": fibonacci,
            "name": "Фиббоначи",
        },
        {
            "method": parabola,
            "name": "Парабола",
        },
        {
            "method": brent,
            "name": "Брент",
        },
    )

    for method in methods:
        draw_individual(f, method)

    draw_all(f, methods)


if __name__ == '__main__':
    task2()
