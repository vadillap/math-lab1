from math import exp, sin, log
from numpy import average
from queue import Queue

from matplotlib import pyplot as plt, cycler
from methods import *
from threading import Thread

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

    for method in methods:
        draw_individual(f, method)

    draw_all(f, methods)


def task3():
    f = lambda x: 5 * x ** 2 + x ** 3 + sin(5 * x) * 7

    epsilon = 1e-7
    x_range = np.array((-2, 2))

    fig, ax = plt.subplots()
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])

    x_values = np.arange(*(x_range * 2), 1e-3)
    y_values = list(map(f, x_values))
    ax.plot(x_values, y_values)

    markercycle = cycler(marker=['o', '+', 'x', '*', 'X'])
    colorcycle = cycler(color=['blue', 'orange', 'green', 'magenta', "red"])
    plt.gca().set_prop_cycle(colorcycle + markercycle)
    axins.set_prop_cycle(colorcycle + markercycle)

    x_mins = []
    y_mins = []

    for method in methods:
        q = Queue()
        t = Thread(
            target=lambda q, args: q.put(method["method"](*args)),
            args=(q, (f, epsilon, *x_range)))
        t.daemon = True
        t.start()
        t.join(timeout=2.0)

        if t.is_alive():
            print(f'{method["name"]} не сходится')
            continue

        x = q.get()
        x_min, y_min = x[-1], f(x[-1])
        x_mins.append(x_min)
        y_mins.append(y_min)
        ax.plot(x_min, y_min, label=method["name"], linestyle="")
        axins.plot(x_min, y_min, label=method["name"], linestyle="")

    x_avg, y_avg = average(x_mins), average(y_mins)
    x_len = abs(max(x_mins) - min(x_mins))
    y_len = abs(max(y_mins) - min(y_mins))

    x1, x2, y1, y2 = x_avg - x_len, x_avg + x_len, y_avg - y_len, y_avg + y_len
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    x_values = np.arange(x1, x2, (x2 - x1) * 1e-3)
    y_values = list(map(f, x_values))
    axins.plot(x_values, y_values, "-", color="lightblue", zorder=0)
    ax.indicate_inset_zoom(axins)

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    task2()
    task3()
