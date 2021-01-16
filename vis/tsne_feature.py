import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
# https://distill.pub/2016/misread-tsne/

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=plt.cm.Set3(y[i]), s=20)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_tsne(x, y, title):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x)
    plot_embedding(x_tsne, y, title)
    plt.show()
