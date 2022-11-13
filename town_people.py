"""
This script generates a scatter plot representing 'town people'.

There are two centers: A and B that people will cluster
A and B have different gaussian widths.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

def gaussian_points(mu, sigma, n):
    """
    :param mu: center of gaussian
    :param sigma: width of guassian
    :param n: int number of samples
    :return: points

    """
    points = np.ones((n, 2))

    points[:, 0] = np.random.normal(mu[0], sigma, n)
    points[:, 1] = np.random.normal(mu[1], sigma, n)

    return points


def plot_town(A, B, points_A, points_B):
    plt.scatter(A[0], A[1], marker="^", label="Business: A", c="red", s=80)
    plt.scatter(B[0], B[1], marker="o", label="Business: B", c="blue", s=80)
    plt.scatter(points_A[:, 0], points_A[:, 1], marker="x", c="green", s=20, label="A people")
    plt.scatter(points_B[:, 0], points_B[:, 1], marker="x", c="green", s=20, label="B people")
    plt.legend()
    #plt.show()


def define_points():
    mu_A = (1, 1)
    mu_B = (6, 3)
    sigma_A = 3
    sigma_B = 1
    A_points = gaussian_points(mu_A, sigma_A, 150)
    B_points = gaussian_points(mu_B, sigma_B, 150)
    plot_town(mu_A, mu_B, A_points, B_points)
    return A_points, B_points, mu_A, mu_B


def kmeans_plot(points, kmeans):
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(8, 8), sharex = True,
                                   sharey=True)
    fig.suptitle(f"K-means Clustering Algorithm", fontsize=16)
    fte_colors = {0: "#008fd5",
                 1: "#fc4f30",
                 }
    # Truth
    truth = np.concatenate((np.ones(150), np.zeros(150)), axis=0)
    km_colors = [fte_colors[label] for label in truth]
    ax1.scatter(points[:, 0], points[:, 1], c=km_colors)
    ax1.set_title(f"Truth", fontdict={"fontsize": 22}
                  )
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax2.scatter(points[:, 0], points[:, 1], c=km_colors)
    ax2.set_title(f"k-means", fontdict={"fontsize": 22})
    plt.show()


def gmm_plot(points, kmeans, gmm_labels):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                   figsize=(8, 8), sharex=True,
                                   sharey=True)
    fig.suptitle(f"GMM and K-means Clustering Algorithm", fontsize=16)
    fte_colors = {0: "#008fd5",
                 1: "#fc4f30", }
    # Truth
    truth = np.concatenate((np.ones(150), np.zeros(150)), axis=0)
    km_colors = [fte_colors[label] for label in truth]
    ax1.scatter(points[:, 0], points[:, 1], c=km_colors)
    ax1.set_title(f"Truth", fontdict={"fontsize": 22}
                  )
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax2.scatter(points[:, 0], points[:, 1], c=km_colors)
    ax2.set_title(f"k-means", fontdict={"fontsize": 22})

    # The GMM plot
    km_colors = [fte_colors[label] for label in gmm_labels]
    ax3.scatter(points[:, 0], points[:, 1], c=km_colors)
    ax3.set_title(f"GMM", fontdict={"fontsize": 22})
    plt.show()


if __name__ == "__main__":
    A_points, B_points, mu_A, mu_B = define_points()
    kmeans = KMeans(init="random", n_clusters=2, n_init=10, max_iter=1000, random_state=42)
    points = np.concatenate((A_points, B_points), axis=0)
    kmeans.fit(points)
    print(kmeans.labels_[:])
    # kmeans_plot(points, kmeans)
    print("Real Centers", mu_A, mu_B)
    print("Predicted Centers", kmeans.cluster_centers_)

    gmm = GMM(n_components=2, random_state=0)
    gmm.fit(points)
    print("GMM", gmm.means_)
    gmm_labels = gmm.predict(points)
    print(gmm_labels)
    gmm_plot(points, kmeans, gmm_labels)





