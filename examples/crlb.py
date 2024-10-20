# CRLB visualization of 1D Gaussians
# Based on visualization from:
# "An Elementary Introduction to Information Geometry", Frank Nielsen

import jax
import jax.numpy as jnp

from bregman.application.distribution.exponential_family.gaussian import (
    GaussianManifold,
)
from bregman.base import LAMBDA_COORDS, THETA_COORDS, Point

from bregman.visualizer.matplotlib.callback import (
    Visualize2DTissotIndicatrix,
)
from bregman.visualizer.matplotlib.matplotlib import MatplotlibVisualizer


def to_vector(mu, sigma):
    if type(mu) is float:
        return jnp.array([mu, sigma])
    return jnp.concatenate([mu, sigma])


def gen_samples(key, mu, sigma):
    if type(mu) is float:
        samples = jax.random.normal(key, (N,))
        samples = mu + sigma * samples
    else:
        samples = jax.random.multivariate_normal(key, mu, sigma, (N,))

    return samples


if __name__ == "__main__":
    # Set parameters
    # VISUAL_COORD = THETA_COORDS
    VISUAL_COORD = LAMBDA_COORDS

    N = 1_000
    mu, sigma = 0.0, 1.0

    key = jax.random.PRNGKey(0)
    samples = gen_samples(key, mu, sigma)

    # Define Univariate-Normal Manifold
    manifold = GaussianManifold(input_dimension=1)

    # Define data
    dist_point = Point(LAMBDA_COORDS, to_vector(mu, sigma))

    # These objects can be visualized through matplotlib
    visualizer = MatplotlibVisualizer(manifold, (0, 1))

    visualizer.plot_object(dist_point, c="black")
    visualizer.add_callback(Visualize2DTissotIndicatrix(scale=1.0))

    visualizer.visualize(VISUAL_COORD)  # Display coordinate type
