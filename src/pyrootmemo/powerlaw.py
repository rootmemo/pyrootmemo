import numpy as np

class Powerlaw:

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            weights = None,
            x0 = 1.0
    ):
        
        # check length of vectors
        nx = len(x)
        ny = len(y)
        if nx != ny:
            ValueError('length of x and y vectors must be equal')
        # check input for x and y
        if any(x <= 0) | any(np.isinf(x)):
            ValueError('all x values must be finite and larger than zero')
        if any(y <= 0) | any(np.isinf(y)):
            ValueError('all y values must be finite and larger than zero')
        # assign weights, if not defined
        if weights is None:
            self.weights = np.ones(nx)
        elif np.isscalar(weights):
            self.weights = np.ones(nx) * weights
        elif isinstance(weights, np.ndarray):
            if len(weights) == nx:
                self.weights = weights
            else:
                ValueError('length of weights vector does not match length of x and y')
        else:
            ValueError('weights must be defined as an np.ndarray or a scalar')
        # check weights
        if any(self.weights < 0) | any(np.isinf(self.weights)):
            ValueError('all weights must finite and equal or larger than zero')
        # assign values
        self.x = x
        self.y = y
        self.x0 = x0
        # check for colinearity
        self.colinear, self.multiplier, self.exponent = self._check_colinearity()


    # check for colinearity - all points exactly on power law --> zero variation
    def _check_colinearity(self):
        # bind x and y data together, and logtransform
        log_xy = np.column_stack((self.x / self.x0, self.y))
        # unique pairs of log(x) and log(y) only
        log_xy_unique = np.log(np.unique(log_xy, axis = 0))
        # check
        if len(log_xy_unique) == 1:
            # only one point --> colinear and fit as horizontal line
            check = True
            exponent = 0.0
            multiplier = np.exp(log_xy_unique[0, 1])            
        else:
            # get difference between subsequently defined points
            diff_log_xy_unique = np.diff(log_xy_unique, axis = 0)
            if len(log_xy_unique) == 2:
                # two unique points --> colinear, and fit as powerlaw
                check = True
                exponent = diff_log_xy_unique[0, 1] / diff_log_xy_unique[0,0]
                multiplier = np.exp(log_xy_unique[0, 1] - exponent*log_xy_unique[0, 0])
            else:
                # calculate cross-products of vectors connecting subsequent data points
                cross_prod = (
                    diff_log_xy_unique[:-1, 0] * diff_log_xy_unique[1:, 1]
                    - diff_log_xy_unique[1:, 0] * diff_log_xy_unique[:-1, 1]
                    )
                check = all(np.isclose(cross_prod, 0.0))
                if check is True:
                    exponent = np.mean(diff_log_xy_unique[:, 1] / diff_log_xy_unique[:, 0])
                    multiplier = np.mean(np.exp(log_xy_unique[:, 1] - exponent * log_xy_unique[:, 0]))
                else:
                    exponent = None
                    multiplier = None
        # return
        return(check, multiplier, exponent)
    
    