# TODOs

 - Split the univariate and multivariate Gaussian class into separate classes.

 - Proper errors for visualization class. Basically need to have warnings about
   visualzing certain things when they shouldn't work. Some examples:
   1. Tissot and Gaussian covariance not being in 2d
   2. Bisectors being miss-specified when manifold not in 2d for visualization.

 - Not sure how to define parallel transport.

 - I think that abstracting "points" might still be nice. Then the manifold
   class can "identify" points into l-, e-, m-parameters.
