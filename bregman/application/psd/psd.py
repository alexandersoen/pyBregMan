import autograd.numpy as anp
import numpy as np

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
        return int(0.5 * (np.sqrt(8 * len(self.data) + 1) - 1))

    @property
    def m(self) -> np.ndarray:
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

        triu_indices = np.triu_indices(self.dimension)
        selector = np.arange(self.dimension * self.dimension).reshape(
            self.dimension, self.dimension
        )[triu_indices]

        self.selector = selector

    def _pre_autodiff(self, x: np.ndarray) -> np.ndarray:
        """Converts vector data into flattened matrix data format.

        Args:
            x: Minimal PSD vector data.

        Returns:
            Flattened matrix data of x.
        """
        return psd_data_to_matrices(x, self.dimension).flatten()

    def _post_grad(self, x: np.ndarray) -> np.ndarray:
        """Extracts the gradients corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[self.selector]

    def _post_hess(self, x: np.ndarray) -> np.ndarray:
        """Extracts the Hessian corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[np.ix_(self.selector, self.selector)]

    def _F(self, x: np.ndarray) -> np.ndarray:
        """PSD primal Bregman generator function.

        Args:
            x: Flattened PSD matrix input.

        Returns:
            PSD manifold primal Bregman generator value.
        """
        m = x.reshape(self.dimension, self.dimension)
        return -anp.log(anp.linalg.det(m))


class PSDDualGenerator(AutoDiffGenerator):
    """PSD manifold dual Bregman generator.

    Parameters:
        selector: Exacts indices from matrix to yield vector PSD data.
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)

        triu_indices = np.triu_indices(self.dimension)
        selector = np.arange(self.dimension * self.dimension).reshape(
            self.dimension, self.dimension
        )[triu_indices]

        self.selector = selector

    def _pre_autodiff(self, x: np.ndarray) -> np.ndarray:
        """Converts vector data into flattened matrix data format.

        Args:
            x: Minimal PSD vector data.

        Returns:
            Flattened matrix data of x.
        """
        return psd_data_to_matrices(x, self.dimension).flatten()

    def _post_grad(self, x: np.ndarray) -> np.ndarray:
        """Extracts the gradients corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[self.selector]

    def _post_hess(self, x: np.ndarray) -> np.ndarray:
        """Extracts the Hessian corresponding to the original minimal PSD
        vector data format.

        Args:
            x: Gradient of the PSD generator.

        Returns:
            Extracted elements corresponding to the original PSD data's vector format.
        """
        return x[np.ix_(self.selector, self.selector)]

    def _F(self, x: np.ndarray) -> np.ndarray:
        """PSD dual Bregman generator function.

        Args:
            x: Flattened PSD matrix input.

        Returns:
            PSD manifold dual Bregman generator value.
        """
        m = x.reshape(self.dimension, self.dimension)
        return anp.log(anp.linalg.det(anp.linalg.inv(m))) - self.dimension


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

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        return lamb

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        return self._theta_to_eta(lamb)

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        return theta

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        return self._eta_to_theta(eta)


def psd_data_from_matrices(m: np.ndarray, dimension: int) -> np.ndarray:
    """Function to convert a PSD matrix into a minimal vector PSD data format.

    Args:
        m: PSD matrix.
        dimension: Shape dimension of the matrix m (i.e, d in dxd PSD matrix).

    Returns:
        Minimal vector represented a PSD matrix.
    """
    return m[np.triu_indices(dimension)]


def psd_data_to_matrices(d: np.ndarray, dimension: int) -> np.ndarray:
    """Function to convert minimal vector PSD data format into a PSD matrix.

    Args:
        d: Minimal PSD vector data.
        dimension: Shape dimension of the matrix m (i.e, d in dxd PSD matrix).

    Returns:
        PSD matrix constructed from minimal data.
    """
    m = np.empty((dimension, dimension))
    m[np.triu_indices(dimension)] = d
    m[np.tril_indices(dimension)] = d

    return m
