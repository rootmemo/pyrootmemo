## FIT DISTRIBUTIONS TO AN ARRAY OF x-DATA, BASED ON (WEIGHTED) LIKELIHOOD FITTING
## * Gumbel distribution (class "GumbelFit")
## * Weibull distribution (class "WeibullFit")
## * Power law distribution (class "PowerFit")

# import packages
import numpy as np
from scipy.optimize import root_scalar
from pyrootmemo.fit.fit import FitBase
from pyrootmemo.fit.fit_xy_linear import LinearFit
from pint import Quantity


####################################################
#### BASE LIKELIHOOD FITTING CLASS FOR 1-D DATA ####
####################################################

class FitBaseX(FitBase):

    def __init__(
        self,
        x,
        weights = None,
        start = None,
        root_method = 'newton',
        x0 = 1.0,
        xmin = None,
        xmin_include = True
    ):
        # check and set x and y parameters
        self._check_input(x, finite = True, min = xmin, min_include = xmin_include, label = 'x')
        self.x = x
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
        self._check_input(self.weights, finite = True, min = 0.0, min_include = True, label = 'weights')
        # set reference values
        self.x0 = self.get_reference(x, x0)
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
        # distance
        ks = max(d_top, d_bot)
        # return non-dimensional value
        if isinstance(ks, Quantity):
            return(ks.magnitude)
        else:
            return(ks)

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
            # nondimensionalise data
            xn = self.nondimensionalise(self.x, self.x0)
            # select random data indices
            rng = np.random.default_rng()
            indices = rng.choice(
                np.arange(len(self.x), dtype = int),
                (n, len(self.x)),
                replace = True
                )
            # generate fit results
            fits = np.array([
                np.array(self.generate_fit(
                    xn[i], self.weights[i], 
                    nondimensional_input = True, nondimensional_output = True
                    ))
                for i in indices
                ])
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

class GumbelFit(FitBaseX):

    def __init__(
            self,
            x,
            weights = None,
            start = None,
            root_method = 'newton'
    ):
        # call initialiser of parent class
        super().__init__(
            x, 
            weights = weights, 
            start = start, 
            root_method = root_method      
            )
        # generate fit
        self.location, self.scale = self.generate_fit(self.x, self.weights)

    # generate MLE parameters
    def generate_fit(
            self,
            x,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):
        # make input data nondimensional
        if nondimensional_input is True:
            xn = x
        else:
            xn = self.nondimensionalise(x, self.x0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_scale_nondimensional(xn, weights)
        # find scale parameter using root solving
        ft = root_scalar(
            self._root_nondimensional,
            method = self.root_method,
            fprime = True,
            x0 = self.start,
            args = (xn, self.weights)
        )
        scale_nondimensional = ft.root
        # find location parameter
        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * np.exp(-xn / scale_nondimensional))
        location_nondimensional = scale_nondimensional * np.log(c1 / c3)
        # return
        if nondimensional_output is True:
            return(location_nondimensional, scale_nondimensional)
        else:
            return(
                self.redimensionalise(location_nondimensional, self.x0),
                self.redimensionalise(scale_nondimensional, self.x0),
            )
    
    # initial guess for (nondimensional) scale parameter
    def _initialguess_scale_nondimensional(
            self,
            xn,
            weights
            ):
        # guess from 1st method of moments - lognormal fit
        muL = np.sum(weights* np.log(xn)) / np.sum(weights)
        sdL = np.sqrt(np.sum(weights * (np.log(xn) - muL)**2) / np.sum(weights))
        var = (np.exp(sdL**2) - 1.) * np.exp(2. * muL + sdL**2)
        return(np.sqrt(6. * var) / np.pi)

    # root function to solve
    def _root_nondimensional(
            self,
            scale_nondimensional,
            xn,
            weights,
            jacobian = True
            ):
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * xn)
        c3 = np.sum(weights * np.exp(-xn / scale_nondimensional))
        c4 = np.sum(weights * xn * np.exp(-xn / scale_nondimensional))
        # root
        root = 1. / scale_nondimensional**2 * (c2 - c1 * c4 / c3) - c1 / scale_nondimensional
        if jacobian is False:
            return(root)
        else:
            # additional coefficients
            c5 = np.sum(weights * xn**2 * np.exp(-xn / scale_nondimensional))
            root_jacobian = (
                c1 / (c3 * scale_nondimensional**4) * (c4**2 / c3 - c5) 
                - 2. / scale_nondimensional**3 * (c2 - c1 * c4 / c3) + c1 / scale_nondimensional**2
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
                return(1. / self.scale * np.exp(-z - np.exp(-z)))
            
    # calculate loglikelihood and derivatives
    def loglikelihood(
            self, 
            x = None, 
            location = None, 
            scale = None,
            deriv = 0,
            weights = None,
            nondimensional_input = False
            ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        # unpack input parameters
        location = self.location if location is None else location
        scale = self.scale if scale is None else scale
        # data in nondimensional form
        if nondimensional_input is True:
            xn = self.x
        else:
            xn = self.nondimensionalise(x, self.x0)
            location = self.nondimensionalise(location, self.x0)
            scale = self.nondimensionalise(scale, self.x0)
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * xn)
        c3 = np.sum(weights * np.exp(-xn / scale))
        c4 = np.sum(weights * xn * np.exp(-xn / scale))
        c5 = np.sum(weights * xn**2 * np.exp(-xn / scale))
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

class WeibullFit(FitBaseX):

    def __init__(
            self,
            x,
            weights = None,
            start = None,
            root_method = 'halley'
    ):
        # call initialiser of parent class
        super().__init__(
            x, 
            weights = weights, 
            start = start, 
            root_method = root_method,
            xmin = 0.0,
            xmin_include = True            
        )
        # generate fit
        self.shape, self.scale = self.generate_fit(self.x, self.weights)

    # generate a fit, using nondimensional values
    def generate_fit(
            self,
            x,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):
        # make input data nondimensional
        if nondimensional_input is True:
            xn = x
        else:
            xn = self.nondimensionalise(x, self.x0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_shape_nondimensional(xn, weights)
        # root solve for shape parameter
        ft = root_scalar(
            self._root_nondimensional,
            fprime = True,
            fprime2 = True,
            x0 = self.start,
            method = self.root_method,
            args = (xn, weights)
            )
        shape = ft.root
        # calculate scale parameter
        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * xn**shape)
        scale_nondimensional = (c3 / c1) ** (1. / shape)
        # return
        if nondimensional_output is True:
            return(shape, scale_nondimensional)
        else:
            return(
                shape,
                self.redimensionalise(scale_nondimensional, self.x0)
            )
    
    # initial guess for shape function
    def _initialguess_shape_nondimensional(
            self,
            xn,
            weights
            ):
        xn_sort = np.sort(xn)
        n = len(xn)
        yp = (2. * np.arange(n, 0, -1) - 1.) / (2. * n)
        # linear fit of transformed cumulative density function
        linear_fit = LinearFit(
            np.log(xn_sort), 
            np.log(-np.log(yp)), 
            weights = weights
            )
        # return shape
        return(linear_fit.gradient)
    
    # root solving function
    def _root_nondimensional(
            self,
            shape,
            xn,
            weights, 
            fprime = True, 
            fprime2 = True
            ):
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * xn**shape)
        c4 = np.sum(weights * xn**shape * np.log(xn))
        # root
        root = c1 / shape - c1 * c4 / c3 + c2
        if fprime is True or fprime2 is True:
            # additional coefficients
            c5 = np.sum(weights * xn**shape * np.log(xn)**2)
            # jacobian
            root_jacobian = (
                -c1 / shape**2 
                - c1 * c5 / c3 
                + c1 * c4**2 / c3**2
            )
            if fprime2 is True:
                # additional coefficients
                c6 = np.sum(weights * xn**shape * np.log(xn)**3)
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
            deriv = 0,
            nondimensional_input = False
            ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        # unpack input parameters
        shape = self.shape if shape is None else shape
        scale = self.scale if scale is None else scale
        # make nondimensional
        if nondimensional_input is True:
            xn = x
        else:
            xn = self.nondimensionalise(x, self.x0)
            scale = self.nondimensionalise(scale, self.x0)
        # coefficients
        c1 = np.sum(weights * np.log(xn))
        c2 = np.sum(weights * xn**shape)
        c3 = np.sum(weights * xn**shape * np.log(xn))
        c4 = np.sum(weights * xn**shape * np.log(xn)**2)
        c5 = np.sum(weights * xn**shape * np.log(xn)**3)
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

class PowerFit(FitBaseX):

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
        super().__init__(
            x, 
            weights = weights, 
            start = start, 
            root_method = root_method,
            xmin = 0.0,
            xmin_include = False            
        )
        # generate fit
        self.multiplier, self.exponent, self.lower, self.upper = self.generate_fit(
            self.x, self.weights, 
            lower = lower, upper = upper
            )

    def generate_fit(
            self,
            x,
            weights,
            lower = None,
            upper = None,
            nondimensional_input = False,
            nondimensional_output = False
            ):
        # set lower and upper
        if lower is None:
            lower = np.min(x)
        if upper is None:
            upper = np.max(x)
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
        else:
            xn = self.nondimensionalise(x, self.x0)
            lower = self.nondimensionalise(lower, self.x0)
            upper = self.nondimensionalise(upper, self.x0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_exponent_nondimensional()
        # root solve for power law exponent
        ft = root_scalar(
            lambda b: self.loglikelihood(
                exponent = b, x = xn, weights = weights,
                lower = lower, upper = upper,
                nondimensional_input = True,
                deriv = 1
                ),
            x0 = self.start,
            fprime = lambda b: self.loglikelihood(
                exponent = b, x = xn, weights = weights, 
                lower = lower, upper = upper,
                nondimensional_input = True,
                deriv = 2
                ),
            method = self.root_method,
            )
        exponent = ft.root
        # multiplier - nondimensionalised
        multiplier_nondimensional = self.get_multiplier_nondimensional(exponent, lower, upper)
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, lower, upper)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, 1. / self.x0),
                exponent,
                self.redimensionalise(lower, self.x0),
                self.redimensionalise(upper, self.x0)
            )
        
    # power law multiplier (asssuming nondimensionalised lower and upper limits)
    def get_multiplier_nondimensional(
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
                return((exponent + 1.) / t)
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
    
    def _initialguess_exponent_nondimensional(
            self
            ):
        return(0.)

    def loglikelihood(
            self, 
            x = None, 
            weights = None,
            exponent = None, 
            lower = None, 
            upper = None, 
            deriv = 0,
            nondimensional_input = False
        ):
        # get data
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        exponent = self.exponent if exponent is None else exponent
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper 
        # make data nondimensional
        if nondimensional_input is True:
            xn = x
        else:
            xn = self.nondimensionalise(x, self.x0)
            lower = self.nondimensionalise(lower, self.x0)
            upper = self.nondimensionalise(upper, self.x0)
        # calculate (dimensionless) multiplier
        multiplier = self.get_multiplier_nondimensional(exponent, lower, upper)
        # calculate
        if deriv == 0:
            # calculate log-probability
            logpi = np.log(multiplier) + exponent * np.log(xn)
            # return weighted loglikelihood
            return(np.sum(weights * logpi))
        elif deriv == 1:
            # calculate derivatives of multiplier
            dmultiplier_dexponent = self.get_multiplier_nondimensional(exponent, lower, upper, deriv = 1)
            # calculate derivative of log-probabilities
            dlogpi_dexponent = dmultiplier_dexponent / multiplier + np.log(xn)
            # return derivative of loglikelihoood
            return(np.sum(weights * dlogpi_dexponent))
        elif deriv == 2:
            # calculate derivatives of multiplier
            dmultiplier_dexponent = self.get_multiplier_nondimensional(exponent, lower, upper, deriv = 1)
            d2multiplier_dexponent2 = self.get_multiplier_nondimensional(exponent, lower, upper, deriv = 2)
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
            cumulative = False
        ):
        # get data
        x = self.x if x is None else x
        # get densities
        if cumulative is False:
            # probability density
            y = self.multiplier * (x / self.x0)**self.exponent
            y[x < self.lower] = 0. * y[x < self.lower]
            y[x > self.upper] = 0. * y[x > self.upper]
            return(y)
        else:
            # cumulative density
            x = self.nondimensionalise(x, self.x0)
            lower = self.nondimensionalise(self.lower, self.x0)
            upper = self.nondimensionalise(self.upper, self.x0)
            if np.isclose(self.exponent, -1.0):
                y = np.log(x / lower) / np.log(upper / lower)
            else:
                y = (
                    (x**(self.exponent + 1) - lower**(self.exponent + 1))
                    / (upper**(self.exponent + 1) - lower**(self.exponent + 1))
                    )
            y[x < lower] = 0.
            y[x > upper] = 1.
            return(y) 