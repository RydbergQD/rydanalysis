import numpy as np


def pca_2d(im):
    av = im.mean(axis=0)
    phi2 = im - av
    phi = np.reshape(phi2, (phi2.shape[0], -1))
    L = np.array([[phi_m.T @ phi_n for phi_m in phi] for phi_n in phi])

    val, v = np.linalg.eigh(L)

    u = v.T @ phi

    norm = np.sqrt(np.sum(u ** 2, axis=(-1)))
    u = np.array([u_ / norm_ for u_, norm_ in zip(u, norm)])

    u2 = u.reshape(phi2.shape)

    return v[::-1], u2[::-1]
