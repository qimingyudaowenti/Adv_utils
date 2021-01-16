import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Sequence


def quantify_vector(v: Sequence):
    return min(v), max(v), max(v) - min(v)


def compare_two_logits(vector_a: Sequence, vector_b: Sequence, normalize=False):
    assert len(vector_a) == len(vector_b) == 10
    min_a, max_a, range_a = quantify_vector(vector_a)
    min_b, max_b, range_b = quantify_vector(vector_b)

    if normalize:
        vector_a = (vector_a - min_a) / range_a
        vector_b = (vector_b - min_b) / range_b
        min_a, min_b = 0, 0
        max_a, max_b = 1, 1
        range_a, range_b = 1, 1

    t = np.arange(0, 10, 1)

    fig, axs = plt.subplots(3, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    y_axis_interval = 3

    # Plot each graph, and manually set the y tick values
    axs[0].plot(t, vector_a, marker='o')
    axs[0].set_yticks(np.linspace(min_a, max_a, y_axis_interval))
    axs[0].set_ylim(min_a - range_a / (2 * y_axis_interval),
                    max_a + range_a / (2 * y_axis_interval))
    axs[0].xaxis.set_ticks_position('none')

    axs[1].plot(t, vector_b, marker='o')
    axs[1].set_yticks(np.linspace(min_b, max_b, y_axis_interval))
    axs[1].set_ylim(min_b - range_b / (2 * y_axis_interval),
                    max_b + range_b / (2 * y_axis_interval))
    axs[1].xaxis.set_ticks_position('none')

    min_t = min(min_a, min_b)
    max_t = max(max_a, max_b)
    range_t = max_t - min_t
    axs[2].fill_between(t, vector_a, vector_b)
    axs[2].set_yticks(np.linspace(min_t, max_t, y_axis_interval))
    axs[2].set_ylim(min_t - range_t / 10, max_t + range_t / 10)

    axs[0].grid()
    axs[1].grid()
    axs[2].set_xticks(t)

    plt.tight_layout(h_pad=0)
    plt.show()


if __name__ == '__main__':
    a = [i for i in range(10)]
    b = [i for i in range(10, 20)]
    compare_two_logits(a, b)
