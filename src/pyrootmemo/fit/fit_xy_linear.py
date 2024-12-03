# import packages
import numpy as np
from pyrootmemo.fit.fit import FitBase

############################
#### LINEAR REGRESSION  ####
############################

# linear regression between x and y
class LinearFit(FitBase):

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            weights = None
    ):
        # check and set x and y parameters
        self._check_input(x, finite = True)
        self.x = x
        self._check_input(y, finite = True)
        self.y = y
        # check x and y have the same length
        if len(x) != len(y):
            ValueError('lengths of x and y not compatible')
        # set fit weights
        if weights is None:
            self.weights = np.ones(len(x))
        elif np.isscalar(weights):
            self.weights = weights * np.ones(len(x))
        else:
            if len(weights) == len(x):
                self.weights = weights
            else:
                ValueError('length of x and weight arrays not compatible')
        # set reference values
        self.x0 = self.get_reference(x)
        self.y0 = self.get_reference(y)
        # generate fit
        self.intercept, self.gradient = self._generate_fit()


    # generate fit from data
    def _generate_fit(self):
        """
        Least-squares linear fit between x and y

        Returns
        -------
        Intercept and gradient of best fit.

        """
        # non-dimensionalise data
        xn = self.nondimensionalise(self.x, self.x0)
        yn = self.nondimensionalise(self.y, self.y0)
        # exception - only one unique x-value
        if len(np.unique(xn)) == 1:
            intercept_nondimensional = yn.mean()
            gradient_nondimensional = 0.0
        # multiple unique x-values
        else:
            # calculate coefficients
            c = np.sum(self.weights)
            cx = np.sum(self.weights * xn)
            cx2 = np.sum(self.weights * xn ** 2)
            cy = np.sum(self.weights * yn)
            cxy = np.sum(self.weights * xn * yn)
            # calculate determinant
            D = c*cx2 - cx**2
            # calculate gradient and intercept using least-squares regression
            intercept_nondimensional = (cx2 * cy - cx * cxy) / D
            gradient_nondimensional = (-cx * cy + c * cxy) / D
        # redimensionalise data
        intercept = self.redimensionalise(intercept_nondimensional, self.y0)
        gradient = self.redimensionalise(gradient_nondimensional, self.y0 / self.x0)
        # return results
        return(intercept, gradient)

