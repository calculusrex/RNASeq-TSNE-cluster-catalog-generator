import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def take(c, gen):
    xs = []
    for _ in range(c):
        xs.append(next(gen))
    return xs

def value_gen(n):
    values = np.linspace(0.5, 0.85, n)
    c = 0
    while True:
        c = (c+1) % n
        yield values[c]

def saturation_gen(n):
    values = np.linspace(0.5, 1, n)
    c = 0
    while True:
        c = (c+1) % n
        yield values[c]

def repeated_values(n):
    gen = value_gen(3)
    return take(n, gen)

def repeated_saturations(n):
    gen = saturation_gen(4)
    return take(n, gen)

def generate_rgb_colors(n):
    hues = np.linspace(1/n, 1, n)
    saturations = repeated_saturations(n)
    values = repeated_values(n)
    colors = list(map(lambda h_sv: matplotlib.colors.hsv_to_rgb([h_sv[0], h_sv[1][0], h_sv[1][1]]),
                      zip(hues,
                          zip(saturations, values))))
    return colors


def showcase_colormap(n):
    values = np.linspace([0, 0], [1, 1], n)
    colors = generate_rgb_colors(n)

    for val, col in zip(values, colors):
        x = val[0]
        y = val[1]
        plt.plot(x, y, '.', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=64)

    plt.show()


if __name__ == '__main__':
    print('Colors')

    showcase_colormap(40)
