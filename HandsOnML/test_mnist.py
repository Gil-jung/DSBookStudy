from plot_mnist import *
from sklearn.datasets import fetch_openml
import numpy as np

np.random.seed(42)

if __name__ == '__main__':
    ##########  Fetch Mnist  ##########
    mnist = fetch_openml('MNIST_784', version=1)
    X, y = mnist["data"], mnist['target']
    print(X.shape)
    print(y.shape)
    y = y.astype(np.int)

    ##########  Plot Mnist  ##########
    # plot_digit(X[36000])

    plt.figure(figsize=(9, 9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)
