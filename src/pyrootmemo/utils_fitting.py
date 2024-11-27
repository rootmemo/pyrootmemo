# import packages
import numpy as np
from scipy.optimize import root_scalar


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



#######################################
#### BASE LIKELIHOOD FITTING CLASS ####
#######################################

class FitLikelihoodBase():

    def __init__(
        self,
        x,
        weights = None,
        start = None,
        root_method = 'newton'
    ):
        # set parameters
        self.x = x
        if weights is None:
            self.weights = np.ones(len(x))
        # starting guess for fit
        self.start = start
        # solver routing
        self.root_method = root_method

    # Kolmogorov-Smirnov distance of fit
    def ks_distance(self):
        # sort data
        xs = np.sort(self.x)
        # cumulative density of data
        y0 = np.cumsum(self.y) / np.sum(self.y)
        # cumulative density of fit
        y1 = self.density(xs, cumulative = True)
        # differences between cumulatives curves: top and bottom of data
        d_top = np.max(np.abs(y1 - y0))
        d_bot = np.max(np.abs(y1 - np.append(0.0, y0[:-1])))
        # return
        return(max(d_top, d_bot))

    # covariance
    def covariance(
            self, 
            method: str = 'fisher', 
            n: int = 100
            ):
        # hessian of loglikelihood
        J = self.loglikelihood(deriv = 2)
        # method
        if J is None or method == 'bootstrap':
            # bootstrapping
            rng = np.random.default_rng()
            xyw = np.column_stack((self.x, self.y, self.weights))
            data = rng.choice(xyw, (n, len(self.x)), axis = 0, replace = True)
            fits = np.array([self.fit(zip(*datai)) for datai in data])
            return(np.cov(fits.transpose()))
        else:
            fisher = -J
            if np.isscalar(fisher):
                return(1. / fisher)
            else:
                return(np.linalg.inv(fisher))


#################################
#### FIT GUMBEL DISTRIBUTION ####
#################################

class GumbelFit(FitLikelihoodBase):

    def __init__(
            self,
            x,
            weights = None,
            start = None,
            root_method = 'newton'
    ):
        # call initialiser of parent class
        super().__init__(x, weights, start, root_method)
        # generate fit
        self.location, self.scale = self._generate_fit()

    # generate MLE parameters
    def _generate_fit(
            self
            ):
        # initial guess
        if self.start is None:
            self.start = self._initialguess_scale()
        # find scale parameter using root solving
        ft = root_scalar(
            self._root,
            method = self.root_method,
            fprime = True,
            x0 = self.start
        )
        scale = ft.root
        # find location parameter
        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * np.exp(-self.x / scale))
        location = scale*np.log(c1/c3)
        # return
        return(location, scale)
    
    # initial guess for scale parameter
    def _initialguess_scale(self):
        # guess from 1st method of moments - lognormal fit
        muL = np.sum(self.weights* np.log(self.x)) / np.sum(self.weights)
        sdL = np.sqrt(np.sum(self.weights * (np.log(self.x) - muL)**2) / np.sum(self.weights))
        var = (np.exp(sdL**2) - 1.) * np.exp(2. * muL + sdL**2)
        return(np.sqrt(6. * var) / np.pi)

    # root function to solve
    def _root(
            self, 
            theta, 
            jacobian = False
            ):
        # coefficients
        c1 = np.sum(self.weights)
        c2 = np.sum(self.weights * self.x)
        c3 = np.sum(self.weights * np.exp(-self.x / theta))
        c4 = np.sum(self.weights * self.x * np.exp(-self.x / theta))
        # root
        root = 1. / theta**2 * (c2 - c1 * c4 / c3) - c1 / theta
        if jacobian is False:
            return(root)
        else:
            # additional coefficients
            c5 = np.sum(self.weights * self.x**2 * np.exp(-self.x / theta))
            root_jacobian = (
                c1 / (c3 * theta**4) * (c4**2 / c3 - c5) 
                - 2. / theta**3 * (c2 - c1 * c4 / c3) + c1 / theta**2
                )
            return(root, root_jacobian)
        
    # random draw
    def random(
            self, 
            n
            ):
        y = np.random.rand(n)
        return(self.location - self.scale * np.log(-np.log(y)))
        
    # density + cumulative density
    def density(
            self, 
            x = None,
            log = False, 
            cumulative = False
            ):
        # x not defined --> use x values used when construction the fit
        if x is None:
            x = self.x
        # gumbel reduced parameter
        z = (x - self.location) / self.scale
        # calculate and return
        if cumulative is True:
            if log is True:
                return(-np.exp(-z))
            else:
                return(np.exp(-np.exp(-z)))
        else:
            if log is True:
                return(-np.log(self.scale) - z - np.exp(-z))
            else:
                return(1./self.scale*np.exp(-z - np.exp(-z)))
            

##############################
#### WEIBULL DISTRIBUTION ####
##############################

class WeibullFit(FitLikelihoodBase):

    def __init__(
            self,
            x,
            weights = None,
            start = None,
            root_method = 'newton'
    ):
        # call initialiser of parent class
        super().__init__(x, weights, start, root_method)
        # generate fit
        self.shape, self.scale = self._generate_fit()

    def _generate_fit(self):
        # initial guess
        if self.start is None:
            self.start = self._initialguess_shape()
        # root solve for shape parameter
        ft = root_scalar(
            self._root,
            fprime = True,
            fprime2 = True,
            x0 = self.start,
            method = self.root_method
            )
        shape = ft.root
        # calcualte scale parameter
        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * self.x**shape)
        scale = (c3 / c1) ** (1. / shape)
        # return
        return(shape, scale)
    
    # initial guess for shape function
    def _initialguess_shape(self):
        x = np.sort(self.x)
        n = len(x)
        yp = (2. * np.arange(n, 0, -1) - 1.) / (2. * n)
        # linear fit of transformed cumulative density function
        linear_fit = LinearFit(
            np.log(x), 
            np.log(-np.log(yp)), 
            weights = self.weights
            )
        # return shape
        return(linear_fit.gradient)
    
    # root solving function
    def _root(
            self,
            shape, 
            jacobian = True, 
            hessian = True
            ):
        # coefficients
        c1 = np.sum(self.weights)
        c2 = np.sum(self.weights * np.log(self.x))
        c3 = np.sum(self.weights * self.x**shape)
        c4 = np.sum(self.weights * self.x**shape * np.log(self.x))
        # root
        root = c1 / shape - c1 * c4 / c3 + c2
        if jacobian is True:
            # additional coefficients
            c5 = np.sum(self.weights * self.x**shape * np.log(self.x)**2)
            # jacobian
            root_jacobian = (
                -c1 / shape**2 
                - c1 * c5 / c3 
                + c1 * c4**2 / c3**2
            )
            if hessian is True:
                # additional coefficients
                c6 = np.sum(self.weights * self.x**shape * np.log(self.x)**3)
                # hessian
                root_hessian = (
                    2. * c1 / shape**3 
                    - c1 * c6 / c3 
                    + 3. * c1 * c4 * c5 / c3**2 
                    - 2. * c1 * c4**3 / c3**3
                )
                return(root, root_jacobian, root_hessian)
            else:
                return(root, root_jacobian)
        else:
            return(root)
        
    # random draw
    def random(
            self, 
            n
            ):
        y = np.random.rand(n)
        return(self.scale * (-np.log(1. - y)) ** (1. / self.shape))
    
    # calculate probability density
    def density(
            self, 
            x = None, 
            log = False, 
            cumulative = False
            ):
        # get data
        if x is None:
            x = self.x
        # get densities
        if cumulative is False:
            # probability density
            if log is True:
                return(np.log(self.shape) 
                       - self.shape * np.log(self.scale)
                       + (self.shape - 1.) * np.log(x) 
                       - (x / self.scale) ** self.shape
                       )
            else:
                return(self.shape / self.scale
                       * (x / self.scale) ** (self.shape - 1.)
                       * np.exp(-(x / self.scale) ** self.shape)
                       )
        else:
            # cumulative density
            if log is True:
                return(np.log(1. - np.exp(-(x / self.scale) ** self.shape)))
            else:
                return(1. - np.exp(-(x / self.scale) ** self.shape))
