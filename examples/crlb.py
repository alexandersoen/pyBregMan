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
from bregman.visualizer.matplotlib.matplotlib import (
    MultiMatplotlibVisualizer,
    MatplotlibVisualizer,
)


def to_vector(mu, sigma):
    if type(mu) is float or mu.size == 1:
        return jnp.array([mu, sigma])
    return jnp.concatenate([mu, sigma])


def gen_samples(key, mu, sigma):
    if type(mu) is float:
        samples = jax.random.normal(key, (N,))
        samples = mu + sigma * samples
    else:
        samples = jax.random.multivariate_normal(key, mu, sigma, (N,))

    return samples


def from_samples(samples: jax.Array):
    """MLE estimate from samples"""
    mu = samples.mean()
    sigma = jnp.sqrt(jnp.square(samples).mean() - jnp.square(mu))

    return to_vector(mu, sigma)


if __name__ == "__main__":
    # Set parameters
    # VISUAL_COORD = THETA_COORDS
    VISUAL_COORD = LAMBDA_COORDS

    N = 1_000
    TRIALS = 100
    mu, sigma = 0.0, 1.0

    key = jax.random.key(0)

    # Define Univariate-Normal Manifold
    manifold = GaussianManifold(input_dimension=1)

    # These objects can be visualized through matplotlib
    visualizer = MatplotlibVisualizer(manifold, (0, 1))

    # True
    true_dist_point = Point(LAMBDA_COORDS, to_vector(mu, sigma))
    visualizer.plot_object(true_dist_point, facecolor="black", c="black", s=40)

    # Sampled
    agg_sampled = to_vector(0.0, 0.0)
    for _ in range(TRIALS):
        key, subkey = jax.random.split(key)
        samples = gen_samples(subkey, mu, sigma)
        sampled_param = from_samples(samples)
        fit_dist_point = Point(LAMBDA_COORDS, sampled_param)

        visualizer.plot_object(fit_dist_point, facecolor="black", c="blue", s=3)

        agg_sampled += sampled_param

    agg_mle_point = Point(LAMBDA_COORDS, agg_sampled / TRIALS)
    visualizer.plot_object(agg_mle_point, facecolor="red", c="red", s=40)

    # visualizer.add_callback(Visualize2DTissotIndicatrix(scale=1.0))

    visualizer.visualize(VISUAL_COORD)  # Display coordinate type
