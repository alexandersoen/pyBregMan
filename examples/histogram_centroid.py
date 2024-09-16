# Replication of https://arxiv.org/pdf/1303.7286 Figure 1.
# Calculates centroids of gray intensity histograms of two images.
# The histograms are encoded as categorical distributions.

try:
    import requests
except ImportError:
    print("This example requires `requests` library to be installed.")
    exit()

import matplotlib.pyplot as plt
import jax.numpy as jnp

from jax import Array
from PIL import Image


def image_gray_hist(image_url: str, ratio: float = 1.0) -> Array:
    # Get image from url
    image_path = requests.get(image_url, stream=True).raw
    image = Image.open(image_path)
    size = image.size
    image = image.resize((int(size[0] * ratio), int(size[1] * ratio)))
    im_array = jnp.array(image)

    # Discritize pixel intensity
    pixel_array = jnp.mean(im_array.reshape(-1, 3), axis=1)
    pixel_array = jnp.rint(pixel_array)
    pixel_array = pixel_array.astype(int)

    # Make intensity array
    _hist = [0] * 256
    for p in pixel_array.tolist():
        _hist[p] += 1

    # Ensures we are in the interiors of the simplex
    _hist = jnp.array(_hist)
    hist = _hist + 1e-8

    return hist / jnp.sum(hist)


image_url_1 = "https://raw.githubusercontent.com/alexandersoen/pyBregMan/main/img/coco_1.jpg"
image_url_2 = "https://raw.githubusercontent.com/alexandersoen/pyBregMan/main/img/coco_2.jpg"

hist_1 = image_gray_hist(image_url_1)
hist_2 = image_gray_hist(image_url_2)

from bregman.application.distribution.exponential_family.categorical import \
    CategoricalManifold
from bregman.base import LAMBDA_COORDS, DualCoords, Point

cat_manifold = CategoricalManifold(k=256)

values = jnp.arange(cat_manifold.k)

cute_cat_1 = Point(LAMBDA_COORDS, hist_1)
cute_cat_2 = Point(LAMBDA_COORDS, hist_2)

# Let us work in the discrete mixture manifold space now.
mix_manifold = cat_manifold.to_discrete_mixture_manifold()
cute_mix_1 = cat_manifold.point_to_mixture_point(cute_cat_1)
cute_mix_2 = cat_manifold.point_to_mixture_point(cute_cat_2)


from bregman.barycenter.bregman import SkewBurbeaRaoBarycenter

# Define barycenter objects
js_centroid_obj = SkewBurbeaRaoBarycenter(
    mix_manifold, dcoords=DualCoords.THETA
)
jef_centroid_obj = SkewBurbeaRaoBarycenter(
    mix_manifold, dcoords=DualCoords.ETA
)

# Centroids can be calculated
js_centroid = js_centroid_obj([cute_mix_1, cute_mix_2])
jef_centroid = jef_centroid_obj([cute_mix_1, cute_mix_2])

# Convert from theta-/eta-parameterization back to histograms (lambda)
js_cat = mix_manifold.point_to_categorical_point(js_centroid)
jef_cat = mix_manifold.point_to_categorical_point(jef_centroid)

js_hist = cat_manifold.convert_coord(LAMBDA_COORDS, js_cat).data
jef_hist = cat_manifold.convert_coord(LAMBDA_COORDS, jef_cat).data


# Plot intensity histograms and centroids
with plt.style.context("bmh"):
    plt.plot(values, hist_1, c="red", label="Image 1 Pixels")
    plt.plot(values, hist_2, c="blue", label="Image 2 Pixels")
    plt.plot(values, js_hist, c="black", label="Jensen-Shannon Centroid")
    plt.plot(values, jef_hist, c="grey", label="Jeffreys Centroid")

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")

    plt.legend()
    plt.show()
