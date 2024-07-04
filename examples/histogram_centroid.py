# Replication of https://arxiv.org/pdf/1303.7286 Figure 1.
# Calculates centroids of gray intensity histograms of two images.
# The histograms are encoded as categorical distributions.

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np

from bregman.application.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.barycenter.bregman import SkewBurbeaRaoBarycenter
from bregman.base import LAMBDA_COORDS, DualCoords, Point

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

    js_centroid = SkewBurbeaRaoBarycenter(manifold, dcoords=DualCoords.ETA)(
        [barbara, lena], weights=[0.5, 0.5], alphas=[1.0, 1.0]
    )
    js_centroid = manifold.convert_coord(LAMBDA_COORDS, js_centroid)

    jef_centroid = SkewBurbeaRaoBarycenter(manifold, dcoords=DualCoords.THETA)(
        [barbara, lena], weights=[0.5, 0.5], alphas=[1.0, 1.0]
    )
    jef_centroid = manifold.convert_coord(LAMBDA_COORDS, jef_centroid)

    plt.plot(values, barbara.data, c="red", label="Image 1 Pixels")
    plt.plot(values, lena.data, c="blue", label="Image 2 Pixels")
    plt.plot(values, js_centroid.data, c="black", label="JS Centroid")
    plt.plot(values, jef_centroid.data, c="grey", label="Jeffreys Centroid")

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")

    plt.legend()
    plt.show()
