"""
TODO: Current method of grabbing the probabilities is not pedagogical.
Need to define a method of converting to a mixture distribution ->
then use the mixture pdf defined on [0 ... 255] to make hists.
"""

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

from bregman.base import LAMBDA_COORDS, Point
from bregman.manifold.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.manifold.manifold import ETA_COORDS

warnings.filterwarnings("error")


def image_red_hist(image_path: pathlib.Path) -> np.ndarray:
    im_array = plt.imread(image_path)
    r_pixel_array = 255 * im_array.reshape(-1, 4)[:, 0]  # Get "r" in "rgba"
    r_pixel_array = np.rint(r_pixel_array)

    hist = np.zeros(256)
    for p in r_pixel_array:
        hist[int(p)] += 1

    hist += 1e-8
    hist = hist / np.sum(hist)

    return hist


# def category_to_coord(cat: int, dim=256) -> np.ndarray:
#    return np.eye(dim)[cat]
#
#
# def wrap_categorical_pdf(pdf, dim=256):
#    def wrapped(cat: int):
#        v = category_to_coord(cat, dim)
#        return pdf(v)
#
#    return wrapped


if __name__ == "__main__":

    im1 = pathlib.Path("examples/images/Barbara-gray.png")
    im2 = pathlib.Path("examples/images/Lena-gray.png")

    hist1 = image_red_hist(im1)
    hist2 = image_red_hist(im2)

    assert len(hist1) == len(hist2)
    manifold = CategoricalManifold(k=len(hist1))

    values = np.arange(manifold.k)

    barbara = Point(LAMBDA_COORDS, hist1)
    lena = Point(LAMBDA_COORDS, hist2)

    js_centroid = manifold.eta_skew_burbea_rao_barycenter(
        [barbara, lena], [0.5, 0.5], [1.0, 1.0]
    )
    js_centroid = manifold.convert_coord(LAMBDA_COORDS, js_centroid)

    jef_centroid = manifold.theta_skew_burbea_rao_barycenter(
        [barbara, lena], [0.5, 0.5], [1.0, 1.0]
    )
    jef_centroid = manifold.convert_coord(LAMBDA_COORDS, jef_centroid)

    plt.plot(values, barbara.data, c="blue")
    plt.plot(values, lena.data, c="red")
    plt.plot(values, js_centroid.data, c="black")
    plt.plot(values, jef_centroid.data, c="grey")
    plt.show()
