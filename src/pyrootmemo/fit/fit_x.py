## FIT DISTRIBUTIONS TO AN ARRAY OF x-DATA, BASED ON (WEIGHTED) LIKELIHOOD FITTING
## * Gumbel distribution (class "GumbelFit")
## * Weibull distribution (class "WeibullFit")
## * Power law distribution (class "PowerFit")

# import packages
import numpy as np
from scipy.optimize import root_scalar
from pyrootmemo.fit.fit_xy_linear import LinearFit


####################################################
#### BASE LIKELIHOOD FITTING CLASS FOR 1-D DATA ####
####################################################

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
        else:
            self.weights = weights
        # starting guess for fit
        self.start = start
        # solver routing
        self.root_method = root_method

    # Kolmogorov-Smirnov distance of fit
    def ks_distance(self):
        # sort data
        xs = np.sort(self.x)
        # cumulative density of data
        y0 = np.cumsum(self.weights) / np.sum(self.weights)
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
        self.location, self.scale = self.generate_fit()

    # generate MLE parameters
    def generate_fit(
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
            jacobian = True
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
            
    # calculate loglikelihood and derivatives
    def loglikelihood(
            self, 
            x = None, 
            location = None, 
            scale = None,
            deriv = 0,
            weights = None
            ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        # unpack input parameters
        location = self.location if location is None else location
        scale = self.scale if scale is None else scale 
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * x)
        c3 = np.sum(weights * np.exp(-x / scale))
        c4 = np.sum(weights * x * np.exp(-x / scale))
        c5 = np.sum(weights * x**2 * np.exp(-x / scale))
        # loglikehihood
        if deriv == 0:
            return(c1 * (location / scale - np.log(scale)) 
                   - c2 / scale 
                   - c3 * np.exp(location / scale)
                   )
        elif deriv == 1:
            dlogL_dmu = c1 / scale - c3 / scale * np.exp(location / scale)
            dlogL_dtheta = (
                -c1 * (location / scale**2 + 1./scale) 
                + c2 / scale**2 
                + (c3 * location - c4) / scale**2 * np.exp(location / scale)
                )
            return(np.array([dlogL_dmu, dlogL_dtheta]))
        elif deriv == 2:
            d2logL_dmu2 = -c3 / scale**2 * np.exp(location / scale)
            d2logL_dmudtheta = (
                1. / scale**2 * (c3 * (1. + location / scale) - c4 / scale)
                * np.exp(location / scale) 
                - c1 / scale**2
                )
            d2logL_dtheta2 = (
                c1 / scale**2 
                + 2. / scale**3 * (c1 * location - c2) 
                - 1. / scale**3 * (c3 * location - c4) 
                * np.exp(location / scale) * (2. + location / scale) 
                + 1. / scale**4 * (c4 * location - c5) * np.exp(location / scale)
                )
            return(np.array([
                [d2logL_dmu2, d2logL_dmudtheta], 
                [d2logL_dmudtheta, d2logL_dtheta2]
                ]))  


##############################
#### WEIBULL DISTRIBUTION ####
##############################

class WeibullFit(FitLikelihoodBase):

    def __init__(
            self,
            x,
            weights = None,
            start = None,
            root_method = 'halley'
    ):
        # call initialiser of parent class
        super().__init__(x, weights, start, root_method)
        # generate fit
        self.shape, self.scale = self.generate_fit()

    def generate_fit(self):
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
            fprime = True, 
            fprime2 = True
            ):
        # coefficients
        c1 = np.sum(self.weights)
        c2 = np.sum(self.weights * np.log(self.x))
        c3 = np.sum(self.weights * self.x**shape)
        c4 = np.sum(self.weights * self.x**shape * np.log(self.x))
        # root
        root = c1 / shape - c1 * c4 / c3 + c2
        if fprime is True or fprime2 is True:
            # additional coefficients
            c5 = np.sum(self.weights * self.x**shape * np.log(self.x)**2)
            # jacobian
            root_jacobian = (
                -c1 / shape**2 
                - c1 * c5 / c3 
                + c1 * c4**2 / c3**2
            )
            if fprime2 is True:
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

    def loglikelihood(
            self,
            x = None, 
            weights = None,
            shape = None, 
            scale = None, 
            deriv = 0      
            ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        # unpack input parameters
        shape = self.shape if shape is None else shape
        scale = self.scale if scale is None else scale 
        # coefficients
        c1 = np.sum(weights * np.log(x))
        c2 = np.sum(weights * x**shape)
        c3 = np.sum(weights * x**shape * np.log(x))
        c4 = np.sum(weights * x**shape * np.log(x)**2)
        c5 = np.sum(weights * x**shape * np.log(x)**3)
        # derivatives
        if deriv == 0:
            return(
                c1 * np.log(shape) 
                - c1 * shape * np.log(scale) 
                + c2 * (shape - 1.) 
                - c3 * scale**(-shape)
                )
        elif deriv == 1:
            dlogL_dshape = (
                c1 / shape 
                - c1*np.log(scale) 
                + c2 
                + scale**(-shape) * (c3 * np.log(scale) - c4)
                )
            dlogL_dscale = (
                -c1 * shape / scale 
                + c3 * shape * scale**(-shape - 1.)
                )
            return(np.array([dlogL_dshape, dlogL_dscale]))
        elif deriv == 2:
            d2logL_dshape2 = (
                -c1 / shape**2 
                - scale**(-shape)
                * (c5 - 2. * c4 * np.log(scale) + c3 * np.log(scale)**2)
                )
            d2logL_dshapedscale = (
                -c1 / scale 
                + scale**(-shape - 1.)
                * ( c3 - c3 * shape * np.log(scale) + shape * c4)
                )
            d2logL_dscale2 = (
                c1 * shape / scale**2 
                - c3 * shape * (shape + 1.) * scale**(-shape - 2.)
                )
            return(np.array([
                [d2logL_dshape2, d2logL_dshapedscale],
                [d2logL_dshapedscale, d2logL_dscale2]
                ]))
        

################################
#### POWER DISTRIBUTION FIT ####
################################

class PowerFit(FitLikelihoodBase):

    def __init__(
            self,
            x,
            weights = None,
            lower = None,
            upper = None,
            start = None,
            root_method = 'newton'
    ):
        # call initialiser of parent class
        super().__init__(x, weights, start, root_method)
        # set limits
        if lower is None:
            self.lower = min(x)
        else:
            self.lower = lower
        if lower is None:
            self.upper = max(x)
        else:
            self.upper = upper
        # generate fit
        self.exponent = self.generate_fit()
        # calculate power function multiplier
        self.multiplier = self.get_multiplier(self.exponent, self.lower, self.upper)

    def generate_fit(self):
        # initial guess
        if self.start is None:
            self.start = self._initialguess_exponent()
        # root solve for shape parameter
        ft = root_scalar(
            lambda b: self.loglikelihood(exponent = b, deriv = 1),
            x0 = self.start,
            fprime = lambda b: self.loglikelihood(exponent = b, deriv = 2),
            method = self.root_method,
            )
        # return
        return(ft.root)
        
    def get_multiplier(
        self,
        exponent, 
        lower,
        upper,
        deriv = 0
        ):
        # asymptotic case where exponent is equal to -1
        if np.isclose(exponent, -1.):
            t = np.log(upper / lower)
            if deriv == 0:
                return(1. / t)
            elif deriv == 1:
                return(0.5 - np.log(upper) / t)
            elif deriv == 2:
                return(
                    (np.log(lower)**2 
                    + np.log(upper)**2 
                    + 4. * np.log(lower) * np.log(upper)) / (6. * t)
                    )
        # all other cases (exponent not equal to -1)
        else:
            t = upper**(exponent + 1.) - lower**(exponent + 1.)
            if deriv == 0:
                return((exponent + 1.)/t)
            elif deriv == 1:
                dt = (
                    np.log(upper) * upper**(exponent + 1.) 
                    - np.log(lower) * lower**(exponent + 1.)
                )
                return(1. / t - (exponent + 1.) * dt / t**2)
            elif deriv == 2:
                dt = (
                    np.log(upper) * upper**(exponent + 1.) 
                    - np.log(lower) * lower**(exponent + 1.)
                )
                ddt = (
                    np.log(upper)**2 * upper**(exponent + 1.) 
                    - np.log(lower)**2 * lower**(exponent + 1.)
                )
                return(
                    -2. * dt / t**2 
                    - (exponent + 1.) * (ddt / t**2 - 2. * dt**2 / t**3)
                    )
    
    def _initialguess_exponent(self):
        return(0.)

    def loglikelihood(
            self, 
            x = None, 
            weights = None,
            exponent = None, 
            lower = None, 
            upper = None, 
            deriv = 0
        ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        exponent = self.exponent if exponent is None else exponent
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper 
        # calculate multiplier
        multiplier = self.get_multiplier(exponent, lower, upper)
        # calculate
        if deriv == 0:
            # calculate log-probability
            logpi = np.log(multiplier) + exponent * np.log(x)
            # return weighted loglikelihood
            return(np.sum(weights * logpi))
        elif deriv == 1:
            # calculate derivatives of multiplier
            dmultiplier_dexponent = self.get_multiplier(exponent, lower, upper, deriv = 1)
            # calculate derivative of log-probabilities
            dlogpi_dexponent = dmultiplier_dexponent / multiplier + np.log(x)
            # return derivative of loglikelihoood
            return(np.sum(weights * dlogpi_dexponent))
        elif deriv == 2:
            # calculate derivatives of multiplier
            dmultiplier_dexponent = self.get_multiplier(exponent, lower, upper, deriv = 1)
            d2multiplier_dexponent2 = self.get_multiplier(exponent, lower, upper, deriv = 2)
            # calculate second derivative of log-probabilities
            d2logpi_dexponent2 = (
                d2multiplier_dexponent2 / multiplier 
                - (dmultiplier_dexponent / multiplier)**2
                )
            # return second derivative of loglikelihoood
            return(np.sum(weights * d2logpi_dexponent2))

    def random(
            self, 
            n
        ):
        y = np.random.rand(n)
        if np.isclose(self.exponent, -1.0):
            return(self.lower * (self.upper / self.lower)**y)
        else:
            return((
                self.lower**(self.exponent + 1.)
                + y * (self.upper**(self.exponent + 1.) - self.lower**(self.exponent + 1.))
                ) ** (1.0 / (self.exponent + 1.)))
    
    def density(
            self, 
            x = None, 
            weights = None,
            log = False, 
            cumulative = False
        ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        # get densities
        if cumulative is False:
            # probability density
            y = self.multiplier * x**self.exponent
            y[x < self.lower] = 0.
            y[x > self.upper] = 0.
            return(y)
        else:
            # cumulative density
            if np.isclose(self.exponent, -1.0):
                y = np.log(x / self.lower) / np.log(self.upper / self.lower)
            else:
                y = (
                    (x**(self.exponent + 1.) - self.lower**(self.exponent + 1.))
                    / (self.upper**(self.exponent + 1.) - self.lower**(self.exponent + 1.))
                    )
            y[x < self.lower] = 0.
            y[x > self.upper] = 1.
            return(y) 