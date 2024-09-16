# Calculate various centroids of two points on the Gaussian manifold. Example
# from README.md.

import jax.numpy as jnp

from bregman.application.distribution.exponential_family.gaussian import (
    GaussianManifold,
)
from bregman.application.distribution.exponential_family.gaussian.geodesic import (
    FisherRaoKobayashiGeodesic,
)
from bregman.base import LAMBDA_COORDS, DualCoords, Point

if __name__ == "__main__":

    # Define Bivariate Normal Manifold
    manifold = GaussianManifold(input_dimension=2)

    # Define data
    def to_vector(mu, sigma):
        return jnp.concatenate([mu, sigma.flatten()])

    mu_1, sigma_1 = jnp.array([0.0, 1.0]), jnp.array([[1.0, 0.5], [0.5, 2.0]])
    mu_2, sigma_2 = jnp.array([1.0, 2.0]), jnp.array([[2.0, 1.0], [1.0, 1.0]])

    point_1 = Point(LAMBDA_COORDS, to_vector(mu_1, sigma_1))
    point_2 = Point(LAMBDA_COORDS, to_vector(mu_2, sigma_2))

    # KL divergence can be calculated
    kl = manifold.kl_divergence(point_1, point_2)
    rkl = manifold.kl_divergence(point_2, point_1)

    print("KL(point_1 || point_2):", kl)
    print("KL(point_2 || point_1):", rkl)

    from bregman.barycenter.bregman import (
        BregmanBarycenter,
        SkewBurbeaRaoBarycenter,
    )

    # We can define and calculate centroids
    theta_barycenter = BregmanBarycenter(manifold, DualCoords.THETA)
    eta_barycenter = BregmanBarycenter(manifold, DualCoords.ETA)
    br_barycenter = SkewBurbeaRaoBarycenter(manifold)
    dbr_barycenter = SkewBurbeaRaoBarycenter(manifold, DualCoords.ETA)

    theta_centroid = theta_barycenter([point_1, point_2])
    eta_centroid = eta_barycenter([point_1, point_2])
    br_centroid = br_barycenter([point_1, point_2])
    dbr_centroid = dbr_barycenter([point_1, point_2])

    # Mid point of Fisher-Rao Geodesic is its corresponding centroid of two points
    fr_geodesic = FisherRaoKobayashiGeodesic(manifold, point_1, point_2)
    fr_centroid = fr_geodesic(t=0.5)

    print("Right-Sided Centroid:", manifold.convert_to_display(theta_centroid))
    print("Left-Sided Centroid:", manifold.convert_to_display(eta_centroid))
    print("Bhattacharyya Centroid:", manifold.convert_to_display(br_centroid))
    print("Fisher-Rao Centroid:", manifold.convert_to_display(fr_centroid))

    from bregman.visualizer.matplotlib.callback import (
        VisualizeGaussian2DCovariancePoints,
    )
    from bregman.visualizer.matplotlib.matplotlib import MatplotlibVisualizer

    # These objects can be visualized through matplotlib
    visualizer = MatplotlibVisualizer(manifold, (0, 1))
    visualizer.plot_object(point_1, c="black")
    visualizer.plot_object(point_2, c="black")
    visualizer.plot_object(
        theta_centroid, c="red", label="Right-Sided Centroid"
    )
    visualizer.plot_object(eta_centroid, c="blue", label="Left-Sided Centroid")
    visualizer.plot_object(
        br_centroid, c="purple", label="Bhattacharyya Centroid"
    )
    visualizer.plot_object(fr_centroid, c="pink", label="Fisher-Rao Centroid")
    visualizer.add_callback(VisualizeGaussian2DCovariancePoints())

    visualizer.visualize(LAMBDA_COORDS)  # Display coordinate type
