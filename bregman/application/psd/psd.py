import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike

from bregman.application.application import ApplicationManifold
from bregman.base import DisplayPoint
from bregman.manifold.generator import AutoDiffGenerator


class PSDPoint(DisplayPoint):
    """DisplayPoint type for PSD manifold"""

    @property
    def dimension(self) -> int:
        """Matrix shape dimension of the PSD data.

        Returns:
            Matrix shape dimension of the PSD data (i.e, d in dxd PSD matrix).
        """
        return int(0.5 * (jnp.sqrt(8 * len(self.data) + 1) - 1))

    @property
    def m(self) -> Array:
        """Calculates matrix representation of PSD data (which is a vector).

        Returns:
            Matrix format of the PSD data.
        """
        return psd_data_to_matrices(self.data, self.dimension)

    def display(self) -> str:
        """Display string of matrix format of PSD data.

        Returns:
            Matrix string of PSD data.
        """
        return str(self.m)


class PSDPrimalGenerator(AutoDiffGenerator):
    """PSD manifold primal Bregman generator.

    Parameters:
        selector: Exacts indices from matrix to yield vector PSD data.
    """

    def __init__(self, dimension: int):
        """Initialize PSD manifold primal Bregman generator.

        Args:
            dimension: Matrix shape dimension.
        """
        super().__init__(dimension)

        triu_indices = jnp.triu_indices(self.dimension)
        selector = jnp.arange(self.dimension * self.dimension).reshape(
            self.dimension, self.dimension
        )[triu_indices]

        self.selector = selector

    def _pre_autodiff(self, x: ArrayLike) -> Array:
        """Converts vector data into flattened matrix data format.

        Args:
            x: Minimal PSD vector data.

        Returns:
            Flattened matrix data of x.
        """
        return psd_data_to_matrices(x, self.dimension).flatten()

    def _post_grad(self, x: ArrayLike) -> Array:
        """Extracts the gradients corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[self.selector]

    def _post_hess(self, x: ArrayLike) -> Array:
        """Extracts the Hessian corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[jnp.ix_(self.selector, self.selector)]

    def _F(self, x: ArrayLike) -> Array:
        """PSD primal Bregman generator function.

        Args:
            x: Flattened PSD matrix input.

        Returns:
            PSD manifold primal Bregman generator value.
        """
        m = x.reshape(self.dimension, self.dimension)
        return -jnp.log(jnp.linalg.det(m))


class PSDDualGenerator(AutoDiffGenerator):
    """PSD manifold dual Bregman generator.

    Parameters:
        selector: Exacts indices from matrix to yield vector PSD data.
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)

        triu_indices = jnp.triu_indices(self.dimension)
        selector = jnp.arange(self.dimension * self.dimension).reshape(
            self.dimension, self.dimension
        )[triu_indices]

        self.selector = selector

    def _pre_autodiff(self, x: ArrayLike) -> Array:
        """Converts vector data into flattened matrix data format.

        Args:
            x: Minimal PSD vector data.

        Returns:
            Flattened matrix data of x.
        """
        return psd_data_to_matrices(x, self.dimension).flatten()

    def _post_grad(self, x: ArrayLike) -> Array:
        """Extracts the gradients corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[self.selector]

    def _post_hess(self, x: ArrayLike) -> Array:
        """Extracts the Hessian corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[jnp.ix_(self.selector, self.selector)]

    def _F(self, x: ArrayLike) -> Array:
        """PSD dual Bregman generator function.

        Args:
            x: Flattened PSD matrix input.

        Returns:
            PSD manifold dual Bregman generator value.
        """
        m = x.reshape(self.dimension, self.dimension)
        return jnp.log(jnp.linalg.det(jnp.linalg.inv(m))) - self.dimension


class PSDManifold(ApplicationManifold[PSDPoint]):
    r"""PSD Manifold. Points on the PSD manifold are represented by a minimal
    set of values. For instance, in the manifold of 2x2 PSD matrices, each
    point only needs to be represented by 3 values (rather than the full 4
    values of the matrix).
    """

    def __init__(self, n_dimension: int):
        """Initialize PSD manifold.

        Args:
            n_dimension: Shape dimension of the PSD matrices (i.e, d in dxd PSD matrix).
        """
        F_gen = PSDPrimalGenerator(n_dimension)
        G_gen = PSDDualGenerator(n_dimension)

        super().__init__(
            theta_generator=F_gen,
            eta_generator=G_gen,
            display_factory_class=PSDPoint,
            dimension=int(n_dimension * (n_dimension + 1) / 2),
        )

        self.eta_generator = G_gen  # Fix typing

    def _lambda_to_theta(self, lamb: ArrayLike) -> Array:
        return lamb

    def _lambda_to_eta(self, lamb: ArrayLike) -> Array:
        return self._theta_to_eta(lamb)

    def _theta_to_lambda(self, theta: ArrayLike) -> Array:
        return theta

    def _eta_to_lambda(self, eta: ArrayLike) -> Array:
        return self._eta_to_theta(eta)


def psd_data_from_matrices(m: ArrayLike, dimension: int) -> Array:
    """Function to convert a PSD matrix into a minimal vector PSD data format.

    Args:
        m: PSD matrix.
        dimension: Shape dimension of the matrix m (i.e, d in dxd PSD matrix).

    Returns:
        Minimal vector represented a PSD matrix.
    """
    return m[jnp.triu_indices(dimension)]


def psd_data_to_matrices(d: ArrayLike, dimension: int) -> Array:
    """Function to convert minimal vector PSD data format into a PSD matrix.

    Args:
        d: Minimal PSD vector data.
        dimension: Shape dimension of the matrix m (i.e, d in dxd PSD matrix).

    Returns:
        PSD matrix constructed from minimal data.
    """
    m = jnp.empty((dimension, dimension))
    m = m.at[jnp.triu_indices(dimension)].set(d)
    m = m.at[jnp.tril_indices(dimension)].set(d)

    return m
