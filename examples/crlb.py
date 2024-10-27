# CRLB visualization of 1D Gaussians
# Based on visualization from:
# "An Elementary Introduction to Information Geometry", Frank Nielsen

import jax
import jax.numpy as jnp

from itertools import product

from bregman.application.distribution.exponential_family.gaussian import (
    GaussianManifold,
)
from bregman.base import LAMBDA_COORDS, Point

from bregman.visualizer.matplotlib.callback import (
    Visualize2DTissotIndicatrix,
    EllipseAtPoint,
)
from bregman.visualizer.matplotlib.matplotlib import (
    MatplotlibVisualizer,
)


def to_vector(mu, var):
    if type(mu) is float or mu.size == 1:
        return jnp.array([mu, var])
    return jnp.concatenate([mu, var])


def gen_samples(key, mu, var, N):
    sigma = jnp.sqrt(var)
    if type(mu) is float:
        samples = jax.random.normal(key, (N,))
        samples = mu + sigma * samples
    else:
        samples = jax.random.multivariate_normal(key, mu, sigma, (N,))

    return samples


def from_samples(samples: jax.Array):
    """MLE estimate from samples"""
    mu = samples.mean()
    var = samples.var()  # jnp.square(samples).mean() - jnp.square(mu)

    return to_vector(mu, var)


if __name__ == "__main__":
    # Set parameters
    # VISUAL_COORD = THETA_COORDS
    VISUAL_COORD = LAMBDA_COORDS

    N = 200
    TRIALS = 100
    # mu, var = 0.0, 1.0

    min_mu, max_mu = -5.0, 5.0
    min_var, max_var = 1.0, 100.0
    steps = 5

    key = jax.random.key(0)

    # Define Univariate-Normal Manifold
    manifold = GaussianManifold(input_dimension=1)

    # These objects can be visualized through matplotlib
    visualizer = MatplotlibVisualizer(
        manifold, (0, 1), dim_names=(r"$\mu$", r"$\sigma^2$")
    )

    colors = "bgrcmyk"
    for i, j in product(range(steps), range(steps)):
        mu = min_mu + (i / (steps - 1)) * (max_mu - min_mu)
        var = min_var + (j / (steps - 1)) * (max_var - min_var)

        # True
        true_dist_point = Point(LAMBDA_COORDS, to_vector(mu, var))
        visualizer.plot_object(
            true_dist_point,
            callbacks=Visualize2DTissotIndicatrix(scale=1 / N, inverse=True),
            facecolor="black",
            c="black",
            s=40,
        )

        # Sampled
        sampled_params = []
        cur_c = colors[(i * steps + j) % len(colors)]
        for _ in range(TRIALS):
            key, subkey = jax.random.split(key)
            samples = gen_samples(subkey, mu, var, N)
            sampled_param = from_samples(samples)
            fit_dist_point = Point(LAMBDA_COORDS, sampled_param)

            visualizer.plot_object(
                fit_dist_point, facecolor=cur_c, c=cur_c, s=3, alpha=0.4
            )

            sampled_params.append(sampled_param)

        # Calculate aggreated samples
        sampled_params = jnp.stack(sampled_params)
        agg_sampled = sampled_params.mean(axis=0)
        agg_cov = jnp.cov(sampled_params.T, bias=True)

        agg_mle_point = Point(LAMBDA_COORDS, agg_sampled)
        visualizer.plot_object(
            agg_mle_point,
            callbacks=EllipseAtPoint(agg_cov, LAMBDA_COORDS, scale=1.0),
            facecolor="red",
            c="red",
            s=40,
        )

    visualizer.save(VISUAL_COORD, "img/crlb.png")
    # visualizer.visualize(VISUAL_COORD)  # Display coordinate type
