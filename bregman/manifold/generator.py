from abc import ABC, abstractmethod

import autograd
import numpy as np


class Generator(ABC):
    """Abstract class for Bregman manifolds.

    Parameters:
        dimension: Dimension of input data.
    """

    def __init__(self, dimension: int):
        """Initialize Bregman generator.

        Args:
            dimension: Dimension of input data.
        """
        self.dimension = dimension

    @abstractmethod
    def F(self, x: np.ndarray) -> np.ndarray:
        """Function of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator evaluated at x.
        """
        pass

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator's gradient evaluated at x.
        """
        pass

    @abstractmethod
    def hess(self, x: np.ndarray) -> np.ndarray:
        """Hessian of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator's hessian evaluated at x.
        """
        pass

    def bergman_divergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bregman divergence on raw data defined by generator.

        Args:
            x: Left argument of Bregman divergence.
            y: Right argument of Bregman divergence.

        Returns:
            Bregman divergence between x and y.
        """
        return self.F(x) - self.F(y) - np.inner(self.grad(y), x - y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Function of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator evaluated at x.
        """
        return self.F(x)


class AutoDiffGenerator(Generator, ABC):
    """Bregman generator defined via autograd's auto-differentiation. As a
    result, only the function definition of the Bregman generator needs to be
    defined. The gradient and hessian will be automatically defined.

    autograd's auto-differentiation requires functions to take vectors as
    inputs. As such, to utilize this class one may need to define wrapper
    functions (reshape operations) before and after autograd is utilized.
    """

    def _pre_autodiff(self, x: np.ndarray) -> np.ndarray:
        """Pre-wrapper function which transforms x before it
        auto-differentiation.

        Args:
            x: Input for generator.

        Returns:
            Transformed x which the generator will take as an input.
        """
        return x

    @abstractmethod
    def _F(self, x: np.ndarray) -> np.ndarray:
        """Generator function which takes in the transformed x input.
        Assumes that the input of this function is the output of
        self._pre_auto_diff.

        Args:
            x: Input for generator.

        Returns:
            Generator evaluated at x.
        """
        pass

    def F(self, x: np.ndarray) -> np.ndarray:
        """Function of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator evaluated at x.
        """
        y = self._pre_autodiff(x)
        return self._F(y)

    def _post_grad(self, x: np.ndarray) -> np.ndarray:
        """Post-wrapper function which transforms the gradient of the generator
        after auto-grad operations.

        Args:
            x: A gradient output of self.F.

        Returns:
            Transformed gradient to align with original input.
        """
        return x

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator's gradient evaluated at x.
        """
        y = self._pre_autodiff(x)
        z = autograd.grad(self._F)(y)
        return self._post_grad(z)

    def _post_hess(self, x: np.ndarray) -> np.ndarray:
        """Post-wrapper function which transforms the hessian of the generator
        after auto-Hessian calculation.

        Args:
            x: A Hessian output of self.F.

        Returns:
            Transformed hessian to align with original input.
        """
        return x

    def hess(self, x: np.ndarray) -> np.ndarray:
        """Hessian of generator.

        Args:
            x: Input for generator.

        Returns:
            Generator's hessian evaluated at x.
        """
        y = self._pre_autodiff(x)
        z = autograd.hessian(self._F)(y)
        return self._post_hess(z)
