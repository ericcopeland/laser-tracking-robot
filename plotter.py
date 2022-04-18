from matplotlib import pyplot as plt
import numpy as np


def plot_data(filename, title, y_label, x_label, name):
    x = []
    y = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x_point, y_point = line.split('\t')
            x.append(float(x_point) / 1000)
            y.append(float(y_point))

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ylim([0, 2.75])
    plt.text(min(x), 2.5, f'Mean = {round(np.mean(y), 2)} s')
    plt.plot(x, y)
    plt.show()
    plt.savefig(f'{name}.png')


if __name__ == '__main__':
    plot_data('latency_bag.txt', 'Latency vs. Time Running [Bag]', 'Latency (s)', 'Time Running (s)', 'bag')
    plot_data('latency_nobag.txt', 'Latency vs. Time Running [No Bag]', 'Latency (s)', 'Time Running (s)', 'no_bag')
