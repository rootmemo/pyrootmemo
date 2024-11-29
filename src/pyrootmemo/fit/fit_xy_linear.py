# import packages
import numpy as np


#########################################
#### BASE FITTING CLASS FOR X,Y DATA ####
#########################################

class FitBase:
    
    def __init__(
            self,
            x,
            y,
            weights = None
    ):
        # set parameters
        self.x = x
        self.y = y
        if weights is None:
            self.weights = np.ones(len(x))
        else:
            self.weights = weights


############################
#### LINEAR REGRESSION  ####
############################

# linear regression between x and y
class LinearFit(FitBase):

    def __init__(
            self,
            x,
            y,
            weights = None
    ):
        # call initialiser of parent class
        super().__init__(x, y, weights)
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
        # exception - only one unique x-value
        if len(np.unique(self.x)) == 1:
            intercept = self.y.mean()
            gradient = 0.0
        # multiple unique x-values
        else:
            # calculate coefficients
            c = np.sum(self.weights)
            cx = np.sum(self.weights * self.x)
            cx2 = np.sum(self.weights * self.x ** 2)
            cy = np.sum(self.weights * self.y)
            cxy = np.sum(self.weights * self.x * self.y)
            # calculate determinant
            D = c*cx2 - cx**2
            # calculate gradient and intercept using least-squares regression
            intercept = (cx2 * cy - cx * cxy) / D
            gradient = (-cx * cy + c * cxy) / D
        # return results
        return(intercept, gradient)

