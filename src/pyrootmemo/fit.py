# FITTING MODULE
# 
# - Probabiliy density curve fitting (array of x-data), based on weighted loglikelihood fitting
#   * Gumbel distribution (class "Gumbel")
#   * Weibull distribution (class "Weibull")
#   * Power law distribution (class "Power")
#
# - x versus y fitting
#   * Linear regression (class "Linear")
#   * Power law regression (function "Powerlaw()")
#     This function is a wrapper for different power law fit types, 
#     depending on different intra-diameter variation assumptions:
#     - gamma (class "PowerlawGamma")
#     - gumbel (class "PowerlawGumbel")
#     - logistic (class "PowerlawLogistic")
#     - lognormal (class "PowerlawLognormal")
#     - lognormal_corrected (class "PowerlawLognormalCorrected")
#     - normal (class "PowerlawNormal")
#     - normal_force (class "PowerlawNormalForce")
#     - norma_freesd(class "PowerlawNormalFreesd")
#     - normal_scaled(class "PowerlawNormalScaled")
#     - uniform (class "PowerlawUniform")
#     - weibull (class "PowerlawWeibull")
#    see Meijer (2024) for more details on models
#    (https://www.doi.org/10.1007/s11104-024-07007-9)


# import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, root_scalar, bracket
from scipy.special import gamma, digamma, polygamma, loggamma, erf, erfinv, gammainc, gammaincinv
from scipy.spatial import ConvexHull
from pint import Quantity
from pint.errors import DimensionalityError
from pyrootmemo.tools.utils_plot import round_range
from pyrootmemo.tools.helpers import units
from pint import Quantity
import warnings


#######################################
#######################################
### DISTRIBUTION FITTING (1-D data) ###
#######################################
#######################################

# Base fit class for distribution fitting
class _FitBase:

    # check input values
    def _check_input(
            self,
            x: list | np.ndarray | Quantity,
            finite: bool = True,
            min: float | int | Quantity | None = None,
            max: float | int | Quantity | None = None,
            min_include: bool = True,
            max_include: bool = True,
            label: str = 'x'
    ):  
        # x-values
        if isintance(x, list):
            x = np.array(x)
        else:
            if not (isinstance(x, np.ndarray) | isintance(x, Quantity)):
                raise TypeError('{label} must be list, np.ndarray or Quantity')
        # finite values
        if finite is True:
            if np.isinf(x).any() is True:
                raise ValueError('all {label} values must be finite')
        # minimum value
        if np.isscalar(min):
            if min_include is True:
                if (x < min).any():
                    raise ValueError(f'all {label} values must be larger or equal than {min}')
            else:
                if (x <= min).any():
                    raise ValueError(f'all {label} values must be larger than {min}')
        # maximum value
        if np.isscalar(max):
            if max_include is True:
                if (x > max).any():
                    raise ValueError(f'all {label} values must be smaller or equal than {max}')
            else:
                if (x >= min).any():
                    raise ValueError(f'all {label} values must be smaller than {max}')

    # get reference value for an array of data
    def get_reference(
            self,
            x: np.ndarray | Quantity,
            x0: int | float | Quantity | None = None
            ):
        if isinstance(x, Quantity):
            # x defined with units
            if x0 is None:
                return(1.0 * x.units)
            elif isinstance(x0, Quantity):
                if x.dimensionality == x0.dimensionality:
                    return(x0)
                else:
                    raise DimensionalityError('units of x and x0 are not compatible')
            elif np.isscalar(x0):
                warnings.warn(f'unit of x0 not defined - assumed {x.units}')
                return(x0 * x.units)
            else:
                raise TypeError('x0 must be int, float, Quantity or None')
        else:
            # x defined without units
            if x0 is None:
                return(1.0)
            elif isinstance(x0, Quantity) & (x0.dimensionless is False):
                raise DimensionalityError('units of x and x0 are not compatible')
            elif isinstance(x0, int) | isinstance(x0, float):
                return(x0)
            else:
                raise TypeError('x0 must be int, float, Quantity or None')
                    
    # generate non-dimensional data (+ apply scaling through magnitude of x0)
    def nondimensionalise(
            self,
            x,
            x0 = 1.0
            ):
        # convert list (or tuple) to numpy array
        if isinstance(x, list) or isinstance(x, tuple):
            if all([isinstance(xi, Quantity) for xi in x]) is True:
                # convert list of pint.Quantity elements to single array (using unit from first item)
                unit = x[0].units
                vals = np.array([xi.magnitude for xi in x])
                x = vals * unit
            else:
                # convert list to array
                x = np.array(x)
        # make nondimensional
        if isinstance(x, Quantity):
            if isinstance(x0, Quantity):
                # convert x to units of reference value, and return magnitude
                return((x.to(x0.units)).magnitude / x0.magnitude)
            else:
                return(x.magnitude / x0)
        else:
            return(x / x0)

    # add units (+ scaling through magnitude of x0) to nondimensionalised values
    def redimensionalise(
            self,
            x,
            x0 = 1.0
    ):
        return(x * x0)
    

####################################################
#### BASE LIKELIHOOD FITTING CLASS FOR 1-D DATA ####
####################################################

class _FitBaseLoglikelihood(_FitBase):

    def __init__(
        self,
        x: np.ndarray | Quantity,
        weights: np.ndarray = None,
        start: int | float | tuple | None = None,
        root_method: str = 'newton',
        x0: float | int | Quantity = 1.0,
        xmin: float | int | Quantity | None = None,
        xmin_include: bool = True
    ):
        # check and set x and y parameters
        self._check_input(
            x, 
            finite = True, 
            min = xmin, 
            min_include = xmin_include, 
            label = 'x'
            )
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
        self._check_input(
            self.weights, 
            finite = True, 
            min = 0.0, 
            min_include = True, 
            label = 'weights'
            )
        # set reference values
        self.x0 = self.get_reference(x, x0)
        # starting guess for fit
        self.start = start
        # solver routing
        self.root_method = root_method
        
    # Kolmogorov-Smirnov distance of fit
    def ks_distance(self):
        # sort data
        order = np.argsort(self.x)
        xs = self.x[order]
        weights = self.weights[order]
        # cumulative density of data
        y0 = np.cumsum(weights) / np.sum(weights)
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

class Gumbel(_FitBaseLoglikelihood):

    def __init__(
            self,
            x: np.ndarray | Quantity,
            weights: np.ndarray | None = None,
            start: int | float | None = None,
            root_method: str = 'newton'
    ):
        # call initialiser of parent class
        super().__init__(
            x, 
            weights = weights, 
            start = start, 
            root_method = root_method      
            )
        # generate fit
        self.location, self.scale = self._generate_fit(self.x, self.weights)

    # generate MLE parameters
    def _generate_fit(
            self,
            x: np.ndarray | Quantity,
            weights: np.ndarray,
            nondimensional_input: bool = False,
            nondimensional_output: bool = False
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

class Weibull(_FitBaseLoglikelihood):

    def __init__(
            self,
            x: np.ndarray | Quantity,
            weights: np.ndarray = None,
            start: int | float | None = None,
            root_method: str = 'halley'
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
        self.shape, self.scale = self._generate_fit(self.x, self.weights)

    # generate a fit, using nondimensional values
    def _generate_fit(
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
        linear_fit = Linear(
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

class Power(_FitBaseLoglikelihood):

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
        self.multiplier, self.exponent, self.lower, self.upper = self._generate_fit(
            self.x, self.weights, 
            lower = lower, upper = upper
            )

    def _generate_fit(
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


##############################
### FIT GAMMA DISTRIBUTION ###
##############################

class Gamma(_FitBaseLoglikelihood):

    def __init__(
            self,
            x: np.ndarray | Quantity,
            weights: np.ndarray | Quantity = None,
            start: float | int | Quantity | None = None,
            root_method: str = 'halley'
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
        self.shape, self.scale = self._generate_fit(self.x, self.weights)

    # generate a fit, using nondimensional values
    def _generate_fit(
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
        c3 = np.sum(self.weights * xn)
        scale_nondimensional = c3 / (shape * c1)
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
        # mean and variance
        #mean = np.sum(weights * xn) / np.sum(weights)
        #variance = np.sum(weights * (xn - mean)**2) / np.sum(weights)
        # return guess for shape
        #return(mean**2 / variance)
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * xn)
        return(c1 / (2. * c1 * np.log(c3 / c1) - 2. * c2))
    
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
        c3 = np.sum(weights * xn)
        # root
        root = (
            c1 * (np.log(shape) - np.log(c3 / c1))
            - c1 * digamma(shape)
            + c2
            )
        if fprime is True or fprime2 is True:
            # jacobian
            root_jacobian = c1 / shape  - c1 * polygamma(1, shape)
            if fprime2 is True:
                # hessian
                root_hessian = -c1 / shape**2 - c1 * polygamma(2, shape)
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
        return(self.scale * gammaincinv(self.shape, y))
    
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
        # make dimensionless
        xn = self.nondimensionalise(x, self.x0)
        scale = self.nondimensionalise(self.scale, self.x0)
        # get densities
        if cumulative is False:
            # probability density
            if log is True:
                out = (
                    (self.shape - 1.) * np.log(xn)
                    - np.log(gamma(self.shape)) - self.shape * np.log(scale)
                    - xn / scale
                    )
            else:
                out = (
                    xn**(self.shape - 1.)
                    / (gamma(self.shape) * scale**self.shape)
                    * np.exp(-xn / scale)
                    )
            return(self.redimensionalise(out, 1. / self.x0))
        else:
            # cumulative density
            P = gammainc(self.shape, xn / scale)
            if log is True:
                return(np.log(P))
            else:
                return(P)
            
    # calculate loglikelihood
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
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * xn)
        # derivatives
        if deriv == 0:
            return(
                - c1 * shape * np.log(scale) 
                - c1 * np.log(gamma(shape))
                + c2 * (shape - 1.)
                - c3 / scale
                )
        elif deriv == 1:
            dlogL_dshape = (
                - c1 * np.log(scale)
                - c1 * digamma(shape)
                + c2
                )
            dlogL_dscale = (
                - c1 * shape / scale
                + c3 / scale**2
                )
            return(np.array([dlogL_dshape, dlogL_dscale]))
        elif deriv == 2:
            d2logL_dshape2 = -c1 * polygamma(1, shape)
            d2logL_dshapedscale = -c1 / scale
            d2logL_dscale2 = (
                c1 * shape / scale**2
                - 2. * c3 / scale**3
                )
            return(np.array([
                [d2logL_dshape2, d2logL_dshapedscale],
                [d2logL_dshapedscale, d2logL_dscale2]
                ]))
        




###################
###################
### 2-D FITTING ###
###################
###################


# linear regression between x and y
class Linear(_FitBase):

    def __init__(
            self,
            x: np.ndarray | Quantity,
            y: np.ndarray | Quantity,
            weights: np.ndarray | None = None
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






###############################
###############################
### POWER LAW FITTING (2-D) ###
###############################
###############################




#############################################
### WRAPPER FUNCTION FOR POWERLAW FITTING ###
#############################################

def Powerlaw(
        x: list | np.ndarray | Quantity,
        y: list | np.ndarray | Quantity, 
        weights: list | np.ndarray | None = None,
        model: str = 'normal',
        x0: float | int | Quantity = 1.0
        ):
    # check model type
    if not isinstance(model, str):
        raise TypeError('model must be defined as a string')
    # convert model name to lowercase, to catch cases where model defined with uppercase letters
    model = model.lower()
    # return fit
    if model == 'gamma':
        return(PowerlawGamma(x, y, weights = weights, x0 = x0))
    elif model == 'gumbel':
        return(PowerlawGumbel(x, y, weights = weights, x0 = x0))
    elif model == 'logistic':
        return(PowerlawLogistic(x, y, weights = weights, x0 = x0))
    elif (model == 'lognormal') | (model == 'lognormal_corrected'):
        return(PowerlawLognormal(x, y, weights = weights, x0 = x0))
    elif model == 'lognormal_uncorrected':
        return(PowerlawLognormalUncorrected(x, y, weights = weights, x0 = x0))
    elif (model == 'normal') | (model == 'normal_strength'):
        return(PowerlawNormal(x, y, weights = weights, x0 = x0))
    elif model == 'normal_force':
        return(PowerlawNormalForce(x, y, weights = weights, x0 = x0))
    elif model == 'normal_freesd':
        return(PowerlawNormalFreesd(x, y, weights = weights, x0 = x0))
    elif model == 'normal_scaled':
        return(PowerlawNormalScaled(x, y, weights = weights, x0 = x0))
    elif model == 'uniform':
        return(PowerlawUniform(x, y, weights = weights, x0 = x0))
    elif model == 'weibull':
        return(PowerlawWeibull(x, y, weights = weights, x0 = x0))
    else:
        raise ValueError('model not recognised')


#################################
#### BASE CLASS - POWER LAWS ####
#################################

class _PowerlawFitBase(_FitBase):

    # initialise - default
    def __init__(
            self, 
            x,
            y,
            weights = None,
            x0 = 1.0
            ):
        # x and y data
        self._check_input(x, finite = True, min = 0.0, min_include = False, label = 'x')
        self.x = x
        self._check_input(y, finite = True, min = 0.0, min_include = False, label = 'y')
        self.y = y
        if len(x) != len(y):
            ValueError('length of x and y arrays not compatible')
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
        # set reference x and y-value
        self.x0 = self.get_reference(x, x0)
        self.y0 = self.get_reference(y)
        # check for zero variance case (perfect fit - all points lie on power law curve)
        self.colinear, self.multiplier, self.exponent = self.check_colinearity()
    
    # predict
    def predict(
            self, 
            x = None
            ):
        if x is None:
            x = self.x
        xn = self.nondimensionalise(x, self.x0)
        return(self.multiplier * xn**self.exponent)

    # kolmogorov-smirnov distance
    def ks_distance(self):
        if self.colinear is True:
            return(0.0)  # colinear case - perfect fit
        else:
            # cumulative density of fit
            cumul_dens = self.density(cumulative = True)
            # sort data in increasing order of occurance
            order = np.argsort(cumul_dens)
            weights = self.weights[order]
            xd = self.x[order]
            yd = self.y[order]
            # cumulative density of data
            cumul_data = np.cumsum(weights) / np.sum(weights)
            # cumulative density of fit
            cumul_fit = self.density(x = xd, y = yd, cumulative = True)
            # differences between cumulatives curves: top and bottom of data
            diff_top = np.max(np.abs(cumul_fit - cumul_data))
            diff_bot = np.max(np.abs(cumul_fit - np.append(0.0, cumul_data[:-1])))
            # return
            return(max(diff_top, diff_bot))
      
    # covariance matrix, from MLE/fisher information, or bootstrapping
    # returns in non-dimensionalised units
    def covariance(
            self, 
            method = 'fisher', 
            n = 100
            ):
        if self.colinear is True:
            # colinear case
            return None
        else:
            # calculate second partial derivative of loglikelihood
            J = self.loglikelihood(deriv = 2, nondimensional_input = False)
            if J is None or method == 'bootstrap':
                # bootstrapping
                rng = np.random.default_rng()
                # nondimensionalise data
                xn = self.nondimensionalise(self.x, self.x0)
                yn = self.nondimensionalise(self.y, self.y0)
                # select random data indices
                indices = rng.choice(
                    np.arange(len(xn), dtype = int),
                    (n, len(xn)),
                    replace = True
                )
                # generate fit results
                fits = np.array([
                    np.array(self.generate_fit(
                        xn[i], 
                        yn[i], 
                        self.weights[i],
                        nondimensional_input = True, 
                        nondimensional_output = True
                    ))
                    for i in indices
                    ])
                # return covariance matrix, in non-dimensional units
                return(np.cov(fits.transpose()))
            else:
                fisher = -J
                if np.isscalar(fisher):
                    return(1. / fisher)
                else:
                    return(np.linalg.inv(fisher))
              
    # default range for prediction and confidence intervals
    def xrange(
            self, 
            n = 101
            ):
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        if xmax > xmin:
            return(np.linspace(xmin, xmax, n))
        else:
            return(np.array([xmin]))
                
    # confidence_interval
    def confidence_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get range of x values
        if x is None:
            x = self.xrange(n = n)
        # power-law prediction
        y_pred = self.predict(x)
        if self.colinear is True:
            # colinear case
            return(x, np.column_stack(y_pred, y_pred))
        else:
            # nondimensionalise
            xn = self.nondimensionalise(x, self.x0)
            yn_pred = self.nondimensionalise(y_pred, self.y0)
            mult = self.nondimensionalise(self.multiplier, self.y0)
            # derivatives of power-law function
            dyn_dmult = xn**self.exponent
            dyn_dexponent = mult * np.log(xn) * xn**self.exponent
            # get covariance matrix
            cov = self.covariance()
            # get confidence interval using delta method
            var = (
                dyn_dmult**2 * cov[0, 0]
                + 2. * dyn_dmult * dyn_dexponent * cov[0, 1]
                + dyn_dexponent**2 * cov[1, 1]
                )
            # multiplier of standard deviation (from gaussian distribution)
            P = 0.5 + 0.5 * level
            sd_mult = np.sqrt(2.) * erfinv(2. * P - 1.)
            # lower and upper values
            yn_lower = yn_pred - sd_mult * np.sqrt(var)
            yn_upper = yn_pred + sd_mult * np.sqrt(var)
            # add scaling and units
            y_lower = self.redimensionalise(yn_lower, self.y0)
            y_upper = self.redimensionalise(yn_upper, self.y0)
            # return
            return(x, np.column_stack((y_lower, y_upper)))
                        
    # check for colinearity - zero variance in residuals
    def check_colinearity(
            self,
            ):
        # non-dimensionalise data - remove units
        xn = self.nondimensionalise(self.x, self.x0)
        yn = self.nondimensionalise(self.y, self.y0)
        # unique pairs of x and y only (with non-zero weights) - logtransform
        mask = (self.weights > 0.0)
        log_xyn = np.log(np.unique(
            np.column_stack((xn, yn))[mask, ...], 
            axis = 0
            ))
        # check
        if len(log_xyn) == 1:
            check = True
            exponent = 0.0
            multiplier_nondimensional = np.exp(log_xyn[0, 1])            
        else:
            diff_log_xyn = np.diff(log_xyn, axis = 0)
            if len(log_xyn) == 2:
                check = True
                exponent = diff_log_xyn[0, 1] / diff_log_xyn[0, 0]
                multiplier_nondimensional = np.exp(log_xyn[0, 1] - exponent * log_xyn[0, 0])
            else:
                # calculate cross-products of vectors connecting subsequent data points
                cross_prod = (
                    diff_log_xyn[:-1, 0] * diff_log_xyn[1:, 1] 
                    - diff_log_xyn[1:, 0] * diff_log_xyn[:-1, 1]
                    )
                check = all(np.isclose(cross_prod, 0.0))
                # in case of colinearity, get fitting parameters
                if check is True:
                    exponent = np.mean(diff_log_xyn[:, 1] / diff_log_xyn[:, 0])
                    multiplier_nondimensional = np.mean(np.exp(log_xyn[:, 1] - exponent * log_xyn[:, 0]))
                else:
                    exponent = None
                    multiplier_nondimensional = None
        # reintroduce scaling/units into multiplier
        if multiplier_nondimensional is not None:
            multiplier = self.redimensionalise(multiplier_nondimensional, self.y0)
        else:
            multiplier = None
        # return
        return(check, multiplier, exponent)
    
    def plot(
            self,
            xunit = 'mm',
            yunit = 'MPa',
            xlabel = 'Diameter',
            ylabel = 'Tensile strength',
            data = True,
            fit = True,
            n = 101,
            confidence = True,
            confidence_level = 0.95,
            prediction = False,
            prediction_level = 0.95,
            legend = True,
            legend_location = 'best',
            axis_expand = 0.05          
            ):
        # initiate plot
        fig, ax = plt.subplots(1, 1)
        # add measured data
        if isinstance(self.x, Quantity):
            if self.x.dimensionality != units(xunit).dimensionality:
                warnings.warn(
                    'xunit (' 
                    + xunit 
                    + ') incompatible with data (' 
                    + str(self.x.units)
                    + '. Set to data units'
                )
                xunit = self.x.units
            x_data = self.x.to(xunit).magnitude
        else:
            x_data = self.x
        if isinstance(self.y, Quantity):
            if self.y.dimensionality != units(yunit).dimensionality:
                warnings.warn(
                    'yunit (' 
                    + yunit 
                    + ') incompatible with data (' 
                    + str(self.y.units)
                    + '. Set to data units'
                )
                yunit = self.y.units
            y_data = self.y.to(yunit).magnitude
        else:
            y_data = self.y
        if data is True:
            ax.plot(x_data, y_data, 'x', label = 'Data')
        # add fit
        if fit is True:
            x_fit = np.linspace(np.min(self.x), np.max(self.x), n)
            y_fit = self.predict(x_fit)
            if isinstance(x_fit, Quantity):
                x_fit = x_fit.to(xunit).magnitude
            if isinstance(y_fit, Quantity):
                y_fit = y_fit.to(yunit).magnitude
            ax.plot(x_fit, y_fit, '-', label = 'Fit')
        # add prediction interval
        if prediction is True and hasattr(self, 'prediction_interval'):
            xp, yp = self.prediction_interval(
                level = prediction_level,
                n = n
                )
            if isinstance(xp, Quantity):
                xp = xp.to(xunit).magnitude
            if isinstance(yp, Quantity):
                yp = yp.to(yunit).magnitude
            labelp = str(round(prediction_level * 100)) + '% Prediction interval' 
            ax.fill_between(
                xp, yp[:, 0], yp[:, 1], 
                label = labelp, alpha = 0.25
                )
        # add confidence interval
        if confidence is True and hasattr(self, 'confidence_interval'):
            xc, yc = self.confidence_interval(
                level = confidence_level,
                n = n
                )
            if isinstance(xc, Quantity):
                xc = xc.to(xunit).magnitude
            if isinstance(yc, Quantity):
                yc = yc.to(yunit).magnitude
            labelc = str(round(confidence_level * 100)) + '% Confidence interval' 
            ax.fill_between(
                xc, yc[:, 0], yc[:, 1], 
                label = labelc, alpha = 0.25
                )
        # set axis limits (based on measured data)
        ax.set_xlim(round_range(
            x_data * (1.0 + axis_expand), 
            limits = [0.0, None]
            )['limits'])
        ax.set_ylim(round_range(
            y_data * (1.0 + axis_expand), 
            limits = [0.0, None]
            )['limits'])
        # set axes labels
        ax.set_xlabel(xlabel + ' [' + xunit + ']')
        ax.set_ylabel(ylabel + ' [' + yunit + ']')
        # add legend
        if legend is True:
            ax.legend(loc = legend_location)
        # return figure and axis objects
        return(fig, ax)


#################
#### WEIBULL ####
#################

# Power law Weibull class
class PowerlawWeibull(_PowerlawFitBase):
    
    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'hybr',
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawWeibull, self).__init__(x, y, weights, x0 = x0)
        # set other input arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.shape = self.generate_fit(
                self.x, self.y, self.weights)
        else: 
            self.shape = np.inf
        
        
    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess for root finding
        if self.start is None:
            self.start = self._initialguess_shape(xn, yn, weights)
        # fit for power law exponent, using root finding
        sol = root(  
            self._root,
            x0 = self.start,
            args = (xn, yn, weights),
            jac = True,
            method = self.root_method
        )
        exponent, shape = sol.x
        # calculate the power law multiplier
        c1 = np.sum(weights)
        c4 = np.sum(weights * xn**(-exponent * shape) * yn**shape)
        multiplier_nondimensional = gamma(1. + 1. / shape) * (c4 / c1)**(1. / shape)
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, shape)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                shape
            )
    
    def _initialguess_shape(
            self,
            xn,
            yn, 
            weights,
            ):
        # guess exponent from linear regression on log-data
        ftL = Linear(np.log(xn), np.log(yn), weights = weights)
        # scale data
        y_scaled = yn / (xn**ftL.gradient)
        # fit weibull distribution
        ftW = Weibull(y_scaled, weights = weights)
        # return
        return(ftL.gradient, ftW.shape)
    
    def _root(
            self,
            par, 
            xn, 
            yn, 
            weights,
            fprime = True
            ):
        # unpack input parameters
        exponent = par[0]  # power law exponent
        shape = par[1]  # Weibull shape parameter
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * xn**(-exponent * shape) * yn**shape)
        c5 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(xn))
        c6 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(yn))
        # roots
        dlogL_dexponent = shape * c1 * c5 / c4 - shape * c2
        dlogL_dshape = (1. / shape + (exponent * c5 - c6) / c4) * c1 - exponent * c2 + c3
        # root
        root = np.array([dlogL_dexponent, dlogL_dshape])
        # return
        if fprime is False:
            # return root
            return(root)
        else:
            # also get derivative
            # extra coefficients
            c7 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(xn)**2)
            c8 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(xn) * np.log(yn))
            c9 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(yn)**2)
            # derivatives
            d2logL_dexponent2 = c1 * shape**2 / c4 * (c5**2 / c4 - c7)
            d2logL_dexponentdshape = (
                c1 / c4 
                * (c5 - exponent * shape * c7 + shape*c8 + shape * c5 / c4 * (exponent * c5 - c6)) 
                - c2
                )
            d2logL_dshape2 = (
                (-1. / shape**2 - (exponent**2 * c7 - 2. * exponent * c8 + c9) / c4 
                 + (exponent * c5 - c6)**2 / c4**2) * c1
                 )
            root_jacobian = np.array([
                [d2logL_dexponent2, d2logL_dexponentdshape], 
                [d2logL_dexponentdshape, d2logL_dshape2]
                ])
            # return
            return(root, root_jacobian)

    def loglikelihood(
            self, 
            x = None,
            y = None,
            weights = None,
            multiplier = None, 
            exponent = None, 
            shape = None,
            deriv = 0,
            nondimensional_input = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        shape = self.shape if shape is None else shape
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
        # calculate
        if self.colinear is True:
            # colinear case
            if deriv == 0:
                return(np.inf)
            elif deriv == 1:
                return(np.full(3, -np.inf))
            elif deriv == 2:
                return(np.full((3, 3), -np.inf))
        else:
            # coefficients
            c1 = np.sum(weights)
            c2 = np.sum(weights * np.log(xn))
            c3 = np.sum(weights * np.log(yn))
            c4 = np.sum(weights * xn**(-exponent * shape) * yn**shape)
            c5 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(xn))
            c6 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(yn))
            c7 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(xn)**2)
            c8 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(xn) * np.log(yn))
            c9 = np.sum(weights * xn**(-exponent * shape) * yn**shape * np.log(yn)**2)
            # gamma functions
            g = gamma(1. + 1. / shape)
            p = digamma(1. + 1. / shape)
            q = polygamma(1, 1. + 1. / shape)
            # loglikelihood
            if deriv == 0:
                return(
                    (np.log(shape) - shape * np.log(multiplier) + shape * np.log(g)) * c1 
                    - exponent * shape * c2 
                    + (shape - 1.) * c3 
                    - (g / multiplier)**shape * c4
                    )
            # first partial derivative of loglikelihood
            elif deriv == 1:
                dlogL_dy0 = -c1 * shape / multiplier + c4 * shape * g**shape * multiplier**(-shape - 1.)
                dlogL_dexponent = -shape * c2 + shape * (g / multiplier)**shape * c5
                dlogL_dshape = (c1 * (1./shape + np.log(g / multiplier) - p / shape) - exponent * c2 + c3 
                    - (g / multiplier)**shape * (c4 * (np.log(g / multiplier) - p / shape) - exponent * c5 + c6))
                return(np.array([dlogL_dy0, dlogL_dexponent, dlogL_dshape]))
            # second partial derivative of loglikelihood
            elif deriv == 2:
                d2logL_dy02 = (
                    c1 * shape / multiplier**2 
                    - c4 * shape * (shape + 1.) * g**shape * multiplier**(-shape - 2.)
                    )
                d2logL_dy0dexponent = -c5 * shape**2 * g**shape * multiplier**(-shape - 1.)
                d2logL_dy0dshape = (
                    -c1 / multiplier 
                    + g**shape * multiplier**(-shape - 1.) 
                    * (c4 * (1. + shape * np.log(g / multiplier) - p) 
                       + shape * (c6 - exponent * c5)
                       )
                    )
                d2logL_dexponent2 = -shape**2 * (g / multiplier)**shape * c7
                d2logL_dexponentdshape = (
                    -c2 
                    + (g / multiplier)**shape 
                    * (c5 * (1. + shape * np.log(g / multiplier) - p) 
                       + shape*(c8 - exponent * c7)
                       )
                    )
                d2logL_dshape2 = (
                    c1 / shape**2 * (q / shape - 1) 
                    - (g / multiplier)**shape * (
                        2. * (np.log(g / multiplier) - p / shape) * (c6 - exponent * c5) 
                        + (np.log(g / multiplier) - p / shape)**2 * c4 
                        + (c4 * q / shape**3 + exponent**2 * c7 - 2. * exponent * c8 + c9)
                        )
                    )
                return(np.array([
                    [d2logL_dy02, d2logL_dy0dexponent, d2logL_dy0dshape], 
                    [d2logL_dy0dexponent, d2logL_dexponent2, d2logL_dexponentdshape], 
                    [d2logL_dy0dshape, d2logL_dexponentdshape, d2logL_dshape2]
                    ]))
        
    # calculate scale parameter (at new values of x)
    def get_scale(
            self, 
            x,
            ):
        xn = self.nondimensionalise(x, self.x0)
        return(self.multiplier * xn**self.exponent / gamma(1. + 1. / self.shape))
            
    # generate prediction intervals (at new values of x)
    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get values of x
        if x is None:
            x = self.xrange(n = n)
        # generate
        if self.colinear is True:
            # colinear case
            y_lower = self.predict(x)
            y_upper = y_lower
        else:
            # cumulative fraction
            P_lower = 0.5 - 0.5 * level
            P_upper = 0.5 + 0.5 * level
            # calculate scale parameter
            scale = self.get_scale(x)
            # interval
            y_lower = scale * (-np.log(1. - P_lower))**(1. / self.shape)
            y_upper = scale * (-np.log(1. - P_upper))**(1. / self.shape)
        # return
        return(x, np.column_stack((y_lower, y_upper)))
        
    # probability density
    def density(
            self, 
            x = None,
            y = None,
            cumulative = False,
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # calculate density
        if self.colinear is True:
            # colinear case
            if cumulative is False:
                out = np.where(
                    np.isclose(y, self.predict(x)),
                    np.inf,
                    0.0)                             
            else:   
                out = np.where(
                    y < self.predict(x),
                    0.0,
                    1.0)
        else:
            # other cases
            scale = self.get_scale(x)
            if cumulative is False:
                out = (self.shape 
                       / scale
                       * (y / scale)**(self.shape - 1.)
                       * np.exp(-(y / scale)**self.shape)
                       )
            else:
                out = 1. - np.exp(-(y / scale)**self.shape)
        # strip units and return
        if isinstance(out, Quantity):
            return(out.magnitude)
        else:
            return(out)

        
    # generate random y-data at new x-values
    def random(
            self, 
            x
            ):
        # scale
        scale = self.get_scale(x)
        # variation
        P = np.random.rand(*x.shape)
        # return
        return(scale * (-np.log(1. - P))**(1. / self.shape))
        

#############
### GAMMA ###
#############

class PowerlawGamma(_PowerlawFitBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'newton',
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawGamma, self).__init__(x, y, weights, x0 = x0)
        # set other input arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.shape = self.generate_fit(
                self.x, self.y, self.weights)
        else: 
            self.shape = np.inf

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess for root finding
        if self.start is None:
            self.start = list(self._initialguess_shape_nondimensional(xn, yn, weights))
        # find best fitting power law exponent
        ft_exponent = root_scalar(
            self._root_exponent_nondimensional,
            x0 = self.start[0],
            fprime = True,
            args = (xn, yn, weights, True),
            method = self.root_method
            )
        exponent = ft_exponent.root
        # initial guess for shape parameter k
        if self.start[1] is None:
            c1 = np.sum(weights)
            c2 = np.sum(weights * np.log(xn))
            c3 = np.sum(weights * np.log(yn))
            c4 = np.sum(weights * yn * xn**(-exponent))
            self.start[1] = 0.5 * c1 / (exponent * c2 - c3 + c1 * np.log(c4) - c1 * np.log(c1))
        # find shape parameter k, using root solving
        ft_shape = root_scalar(
            self._root_shape_nondimensional,
            x0 = self.start[1],
            fprime = True,
            args = (exponent, xn, yn, weights, True),
            method = self.root_method
            )
        shape = ft_shape.root
        # calculate the power law multiplier
        c1 = np.sum(weights)
        c4 = np.sum(weights * yn * xn**(-exponent))
        multiplier_nondimensional = c4 / c1
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, shape)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                shape
            )
    
    def _initialguess_shape_nondimensional(
            self,
            xn,
            yn,
            weights
    ):
        # guess for power law exponent
        ftL = Linear(xn, yn, weights = weights)
        exponent = ftL.gradient
        # guess for shape
        shape = None
        # return
        return(exponent, shape)
    
    def _root_exponent_nondimensional(
            self,
            exponent,
            xn,
            yn,
            weights,
            fprime = True
    ):
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c4 = np.sum(weights * yn * xn**(-exponent))
        c5 = np.sum(weights * yn * xn**(-exponent) * np.log(xn))
        # root
        root = c1 * c5 / c4 - c2
        # return
        if fprime is False:
            return(root)
        else:
            # additional coefficients
            c6 = np.sum(weights * yn * xn**(-exponent) * np.log(xn)**2)
            # derivative of root
            diff_root = c1 / c4 * (c5**2 / c4 - c6)
            # return
            return(root, diff_root)

    def _root_shape_nondimensional(
            self,
            shape,
            exponent,
            xn,
            yn,
            weights,
            fprime = True
    ):
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * yn * xn**(-exponent))
        # root
        root = (c1 * (np.log(shape) - digamma(shape) - np.log(c4) + np.log(c1)) 
                - exponent * c2 + c3
                )
        # return
        if fprime is False:
            return(root)
        else:
            # derivative
            diff_root = c1 * (1. / shape - polygamma(1, shape))
            # return
            return(root, diff_root)

    def loglikelihood(
            self, 
            x = None, 
            y = None,
            weights = None,
            multiplier = None, 
            exponent = None, 
            shape = None,
            deriv = 0,
            nondimensional_input = False
    ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        shape = self.shape if shape is None else shape
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
        else:
            xn = x
            yn = y
        # calculate
        if self.colinear is True:
            # colinear case
            if deriv == 0:
                return(np.inf)
            elif deriv == 1:
                return(np.full(3, -np.inf))
            elif deriv == 2:
                return(np.full((3,3), -np.inf))
        else:
            # coefficients
            c1 = np.sum(weights)
            c2 = np.sum(weights * np.log(xn))
            c3 = np.sum(weights * np.log(yn))
            c4 = np.sum(weights * xn**(-exponent) * yn)
            c5 = np.sum(weights * xn**(-exponent) * yn * np.log(xn))
            c6 = np.sum(weights * xn**(-exponent) * yn * np.log(xn)**2)
            # gamma functions
            logg = loggamma(shape)
            p = digamma(shape)
            q = polygamma(1, shape)
            # loglikelihood
            if deriv == 0:
                return(c1 * (shape * np.log(shape) - logg - shape * np.log(multiplier)) 
                       - exponent * shape * c2 
                       + (shape - 1.) * c3 
                       - shape * c4 / multiplier
                       )
            elif deriv == 1:
                dlogL_dmultiplier = shape / multiplier * (c4 / multiplier - c1)
                dlogL_dexponent = shape * (c5 / multiplier - c2)
                dlogL_dshape = (
                    c1 * (1. + np.log(shape) - p - np.log(multiplier)) 
                    - exponent * c2 
                    + c3 
                    - c4 / multiplier
                )
                return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dshape]))
            elif deriv == 2:
                d2logL_dmultiplier2 = shape / multiplier**2 * (c1 - 2. * c4 / multiplier)
                d2logL_dmultiplierdexponent = -shape * c5 / multiplier**2
                d2logL_dmultiplierdshape = 1. / multiplier * (c4 / multiplier - c1)
                d2logL_dexponent2 = -shape * c6 / multiplier
                d2logL_dexponentdshape = c5 / multiplier - c2
                d2logL_dshape2 = c1 * (1. / shape - q)
                return(np.array([
                    [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdshape],
                    [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdshape],
                    [d2logL_dmultiplierdshape, d2logL_dexponentdshape, d2logL_dshape2]
                    ]))

    def get_scale(
            self, 
            x
            ):
        return(
            self.multiplier / self.shape
            * (x / self.x0)**self.exponent 
            )
    
    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get values of x
        if x is None:
            x = self.xrange(n = n)
        # generate
        if self.colinear is True:
            # colinear case
            y_lower = self.predict(x)
            y_upper = y_lower
        else:
            # cumulative fraction
            P_lower = 0.5 - 0.5*level
            P_upper = 0.5 + 0.5*level
            # scale parameter
            scale = self.get_scale(x)
            # interval
            y_lower = scale * gammaincinv(self.shape, P_lower)
            y_upper = scale * gammaincinv(self.shape, P_upper)
        # return
        return(x, np.column_stack((y_lower, y_upper)))

    def density(
            self, 
            x = None,
            y = None,
            cumulative = False,
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # scale parameter
        scale = self.get_scale(x)
        # make nondimensional, to get around numpy functions that do not accept Quantities
        xn = self.nondimensionalise(x, self.x0)
        yn = self.nondimensionalise(y, self.y0)
        multiplier_nondimensional = self.nondimensionalise(self.multiplier, self.y0)
        scale_nondimensional = self.nondimensionalise(scale, self.y0)
        # calculate density
        if self.colinear is True:
            # colinear case
            yn_predict = multiplier_nondimensional * xn**self.exponent
            if cumulative is False:
                out = np.where(
                    np.isclose(yn, yn_predict),
                    np.inf,
                    0.0)                             
            else:   
                out = np.where(
                    y < yn_predict,
                    0.0,
                    1.0)
        else:
            # other cases
            if cumulative is False:
                out = (
                    yn**(self.shape - 1.)
                    / (gamma(self.shape) * scale_nondimensional**self.shape)
                    * np.exp(-yn / scale_nondimensional))
            else:
                out = gammainc(self.shape, yn / scale_nondimensional)
        # strip units and return
        if isinstance(out, Quantity):
            return(out.magnitude)
        else:
            return(out)        
        
    def random(
            self, 
            x
            ):
        # scale
        scale = self.get_scale(x)
        # variation
        P = np.random.rand(*x.shape)
        # return
        return(scale * gammaincinv(self.shape, P))


##############
### GUMBEL ###
##############

class PowerlawGumbel(_PowerlawFitBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'hybr',
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawGumbel, self).__init__(x, y, weights, x0 = x0)
        # set other input arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.scale0 = self.generate_fit(
                self.x, self.y, self.weights)
        else: 
            self.scale0 = self.redimensionalise(0.0, self.y0)

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess for root finding
        if self.start is None:
            self.start = list(self._initialguess_exponent_scale_nondimensional(xn, yn, weights))
        # find best fitting power law exponent
        ft = root(
            self._root_nondimensional,
            x0 = self.start,
            jac = True,
            args = (xn, yn, weights, True),
            method = self.root_method
            )
        exponent, scale0 = ft.x
        # calculate power law multiplier
        c1 = np.sum(weights)
        c6 = np.sum(weights * np.exp(-yn / (scale0 * xn**exponent)))
        multiplier_nondimensional = scale0 * (np.log(c1) - np.log(c6) + np.euler_gamma)
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, scale0)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                self.redimensionalise(scale0, self.y0)
            )

    def _initialguess_exponent_scale_nondimensional(
            self,
            xn,
            yn,
            weights
    ):
        # initial guess for power law exponent - linear regression on log data
        ftL = Linear(np.log(xn), np.log(yn), weights = weights)
        exponent = ftL.gradient
        # get scale parameter based gumbel fitting of y/x^exponent
        ftG = Gumbel(yn / (xn**exponent), weights = weights)
        scale0 = ftG.scale
        # return
        return(exponent, scale0)
    
    def _root_nondimensional(
            self,
            par,
            xn,
            yn,
            weights,
            jac = True
    ):
        # unpack parameters
        exponent, shape0 = par
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * yn / (xn**exponent))
        c4 = np.sum(weights * yn * np.log(xn) / (xn**exponent))
        c5 = np.sum(weights * yn * np.log(xn)**2 / (xn**exponent))
        c6 = np.sum(weights * np.exp(-yn / (shape0 * xn**exponent)))
        c7 = np.sum(weights * yn / (xn**exponent) * np.exp(-yn / (shape0 * xn**exponent)))
        c8 = np.sum(weights * yn * np.log(xn) / (xn**exponent) * np.exp(-yn / (shape0 * xn**exponent)))
        # roots
        dlogL_dexponent = c4 / shape0 - c1 * c8 / (shape0 * c6) - c2
        dlogL_dshape0 = c3 / shape0**2 - c1 * c7 / (shape0**2 * c6) - c1 / shape0
        root = np.array([dlogL_dexponent, dlogL_dshape0])
        # return
        if jac is False:
            return(root)
        else:
            # additional coefficients
            c9 = np.sum(
                weights * yn * np.log(xn)**2 
                / (xn**exponent) 
                * np.exp(-yn / (shape0 * xn**exponent))
                )
            c10 = np.sum(
                weights * yn**2 / (xn**(2. * exponent))
                * np.exp(-yn / (shape0 * xn**exponent))
                )
            c11 = np.sum(
                weights * yn**2 * np.log(xn) / (xn**(2. * exponent))
                * np.exp(-yn / (shape0 * xn**exponent))
                )
            c12 = np.sum(
                weights * yn**2 * np.log(xn)**2 / (xn**(2. * exponent))
                * np.exp(-yn / (shape0 * xn**exponent))
                )
            # roots
            d2logL_dexponent2 = (
                -c5 / shape0 
                - c1 * c12 / (shape0**2 * c6) 
                + c1 * c9 / (shape0 * c6) 
                + c1 * c8**2 / (shape0**2 * c6**2)
                )
            d2logL_dexponentdshape0 = (
                -c1 * c11 / (shape0**3 * c6) 
                + c1 * c7 * c8 / (shape0**3 * c6**2) 
                - c4 / shape0**2 
                + c1 * c8 / (shape0**2 * c6)
                )
            d2logL_dshape02 = (
                -2. * c3 / shape0**3 
                + 2. * c1 * c7 / (shape0**3 * c6) 
                - c1 * c10 / (shape0**4 * c6) + c1 * c7**2 / (shape0**4 * c6**2) 
                + c1 / shape0**2
                )
            diff_root = np.array([
                [d2logL_dexponent2, d2logL_dexponentdshape0],
                [d2logL_dexponentdshape0, d2logL_dshape02]
            ])
            # return
            return(root, diff_root)

    def loglikelihood(
            self, 
            x = None,
            y = None,
            weights = None,
            multiplier = None, 
            exponent = None, 
            scale0 = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        scale0 = self.scale0 if scale0 is None else scale0
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
            scale0 = self.nondimensionalise(scale0, self.y0)
        else:
            xn = x
            yn = y
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * yn / (xn**exponent))
        c4 = np.sum(weights * yn * np.log(xn) / (xn**exponent))
        c5 = np.sum(weights * yn * np.log(xn)**2 / (xn**exponent))
        c6 = np.sum(weights * np.exp(-yn / (scale0 * xn**exponent)))
        c7 = np.sum(weights * yn / (xn**exponent) * np.exp(-yn / (scale0 * xn**exponent)))
        c8 = np.sum(weights * yn * np.log(xn) / (xn**exponent) * np.exp(-yn / (scale0 * xn**exponent)))
        c9 = np.sum(weights * yn * np.log(xn)**2 / (xn**exponent) * np.exp(-yn / (scale0 * xn**exponent)))
        c10 = np.sum(weights * yn**2 / (xn**(2. * exponent)) * np.exp(-yn / (scale0 * xn**exponent)))
        c11 = np.sum(weights * yn**2 * np.log(xn) / (xn**(2.*exponent)) * np.exp(-yn / (scale0 * xn**exponent)))
        c12 = np.sum(weights * yn**2 * np.log(xn)**2 / (xn**(2.*exponent)) * np.exp(-yn / (scale0 * xn**exponent)))
        # loglikelihood
        if deriv == 0:
            return(
                c1 * (multiplier / scale0 - np.log(scale0) - np.euler_gamma) 
                   - exponent*c2 - c3/scale0 
                   - np.exp(multiplier/scale0 - np.euler_gamma)*c6)
        elif deriv == 1:
            dlogL_dmultiplier = (
                c1 / scale0 
                - c6 / scale0 * np.exp(multiplier / scale0 - np.euler_gamma)
            )
            dlogL_dexponent = (
                c4 / scale0 
                - c2 
                - c8 / scale0 * np.exp(multiplier / scale0 - np.euler_gamma)
            )
            dlogL_dtheta0 = (
                (c3 - c1 * multiplier) / scale0**2 
                - c1 / scale0 
                + (c6 * multiplier - c7) / scale0**2 * np.exp(multiplier / scale0 - np.euler_gamma)
                )
            return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dtheta0]))
        elif deriv == 2:
            d2logL_dmultiplier2 = -c6 / scale0**2 * np.exp(multiplier / scale0 - np.euler_gamma)
            d2logL_dmultiplierdexponent = -c8/scale0**2 * np.exp(multiplier / scale0 - np.euler_gamma)
            d2logL_dmultiplierdscale0 = (
                1. / scale0**2
                * (c6 * (1. + multiplier / scale0) - c7 / scale0)
                * np.exp(multiplier / scale0 - np.euler_gamma) 
                - c1 / scale0**2
                )
            d2logL_dexponent2 = (
                1. / scale0 * (c9 - c12 / scale0) 
                * np.exp(multiplier / scale0 - np.euler_gamma) 
                - c5 / scale0
                )
            d2logL_dexponentdscale0 = (
                -c4 / scale0**2 
                + 1. / scale0**2 * (c8 * (1. + multiplier / scale0) - c11/scale0)
                * np.exp(multiplier / scale0 - np.euler_gamma)
                )
            d2logL_dscale02 = (
                c1 / scale0**2 
                - 2. * (c3 - c1 * multiplier) / scale0**3 
                - 2. * (c6 * multiplier - c7) / scale0**3 * np.exp(multiplier / scale0 - np.euler_gamma) 
                + (2. * c7 * multiplier - c10 - c6 * multiplier**2)
                / scale0**4 
                * np.exp(multiplier / scale0 - np.euler_gamma)
                )
            return(np.array([
                [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdscale0],
                [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdscale0],
                [d2logL_dmultiplierdscale0, d2logL_dexponentdscale0, d2logL_dscale02]
                ]))        
        
    def get_location_scale0(
            self, 
            x
            ):
        return(
            (self.multiplier - np.euler_gamma * self.scale0) * (x / self.x0)**self.exponent,
            self.scale0 * (x / self.x0)**self.exponent
            )

    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # x-data
        if x is None:
            x = self.xrange(n)
        # generate
        if self.colinear is True:
            # colinear case
            y_pred = self.predict(x)
            return(x, np.column_stack(y_pred, y_pred))
        else:
            # cumulative fraction
            P_lower = 0.5 - 0.5 * level
            P_upper = 0.5 + 0.5 * level
            # location and scale parameters
            location, scale0 = self.get_location_scale0(x)
            # return
            return(
                x, 
                np.column_stack((
                    location - scale0 * np.log(-np.log(P_lower)),
                    location - scale0 * np.log(-np.log(P_upper))
                ))
                )
        
    def density(
            self, 
            x = None,
            y = None, 
            cumulative = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # scale parameter
        location, scale0 = self.get_location_scale0(x)
        # make nondimensional, to get around numpy functions that do not accept Quantities
        yn = self.nondimensionalise(y, self.y0)
        location_nondimensional = self.nondimensionalise(location, self.y0)
        scale0_nondimensional = self.nondimensionalise(scale0, self.y0)
        # calclate
        z = (yn - location_nondimensional) / scale0_nondimensional
        if cumulative is False:
            out = np.exp(-z - np.exp(-z)) / scale0_nondimensional
        else:
            out = np.exp(-np.exp(-z))
        # strip any units
        if isinstance(out, Quantity):
            return(out.magnitude)
        else:
            return(out)   

    def random(
            self, 
            x
            ):
        # location and scale parameters
        location, scale0 = self.get_location_scale0(x)
        # variation
        P = np.random.rand(*x.shape)
        # return
        return(location - scale0 * np.log(-np.log(P)))
    

###############
### UNIFORM ###
############### 

class PowerlawUniform(_PowerlawFitBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            algorithm = 'convex_hull',
            root_method = 'bisect',
            offset = 1.0,
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawUniform, self).__init__(x, y, weights, x0 = x0)
        # set other input arguments
        self.start = start
        self.root_method = root_method
        self.algorithm = algorithm
        self.offset = offset
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.width = self.generate_fit(
                self.x, self.y, self.weights)
        else: 
            self.width = self.redimensionalise(0.0, self.y0)

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # fit using convex hull method
        if self.algorithm == 'convex_hull':
            # matrix with unique log-transformed x,y positions
            log_xyn = np.log(np.unique(np.column_stack((xn, yn)), axis = 0))
            # create a convex hull in log-space (clockwise order)
            hull = ConvexHull(log_xyn)
            # gradients of hull simplices
            d_log_xyn = np.diff(log_xyn[hull.simplices], axis = 1)
            grads = (d_log_xyn[:, :, 1] / d_log_xyn[:, :, 0]).flatten()
            # test each gradient (exponent) for loglikelihood - pick best
            fn = lambda b: self.loglikelihood(
                x = xn, y = yn, weights = weights,
                exponent = b, 
                nondimensional_input = True,
                deriv = 0
                )
            fts = np.array([fn(b) for b in grads])
            imax = np.argmax(fts)
            exponent = grads[imax]
        # root solving method, using bracketing or gradient
        elif self.algorithm == 'root':
            # intial guess - log-log transform
            if self.start is None:
                ftL = Linear(np.log(xn), np.log(yn), weights = weights)
                self.start = ftL.gradient
            # make a guess for root solving bracket
            fn0 = lambda b: -self.loglikelihood(
                x = xn, y = yn, weights = weights,
                exponent = b, 
                nondimensional_input = True,
                deriv = 0
                )
            br = bracket(
                fn0,
                xa = self.start - self.offset,
                xb = self.start + self.offset
                )
            # find exponent using root solving
            fn1 = lambda b: self.loglikelihood(
                x = xn, y = yn, weights = weights,
                exponent = b, 
                nondimensional_input = True,
                deriv = 1
            )
            sol = root_scalar(
                fn1,
                bracket = (br[0], br[2]),
                x0 = self.start,
                method = self.root_method
                )
            exponent = sol.root
        # find min and max of domain (at x = 1)
        lower_nondimensional = np.min(yn / xn**exponent)
        upper_nondimensional = np.max(yn / xn**exponent)
        width_nondimensional = upper_nondimensional - lower_nondimensional
        multiplier_nondimensional = 0.5*(lower_nondimensional + upper_nondimensional)
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, width_nondimensional)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                self.redimensionalise(width_nondimensional, self.y0)
            )
        
    def loglikelihood(
            self, 
            x = None,
            y = None,
            weights = None,
            exponent = None, 
            deriv = 0,
            nondimensional_input = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        exponent = self.exponent if exponent is None else exponent
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        else:
            xn = x
            yn = y
        # calculate logklikelihoods, or its derivatives
        if deriv == 0:
            lower = np.min(yn / xn**exponent)
            upper = np.max(yn / xn**exponent)
            return(
                -np.log(upper - lower) * np.sum(weights) 
                - exponent * np.sum(weights * np.log(xn))
                )
        elif deriv == 1:
            il = np.argmin(yn / xn**exponent)
            iu = np.argmax(yn / xn**exponent)
            lower = yn[il] / xn[il]**exponent
            upper = yn[iu] / xn[iu]**exponent
            dlower_dexponent = -yn[il] * np.log(xn[il]) / xn[il]**exponent
            dupper_dexponent = -yn[iu] * np.log(xn[iu]) / xn[iu]**exponent
            return(
                (dlower_dexponent - dupper_dexponent)
                / (upper - lower) * np.sum(weights) 
                - np.sum(weights * np.log(xn))
                )
        else:
            return(None)
        
    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get values of x
        if x is None:
            x = self.xrange(n=n)
        # lower and upper interval
        y_lower = (self.multiplier - 0.5 * level * self.width) * (x / self.x0)**self.exponent
        y_upper = (self.multiplier + 0.5 * level * self.width) * (x / self.x0)**self.exponent
        # return
        return(x, np.column_stack((y_lower, y_upper)))
    
    def density(
            self, 
            x = None,
            y = None,
            cumulative = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # make nondimensional, to get around numpy functions that do not accept Quantities
        xn = self.nondimensionalise(x, self.x0)
        yn = self.nondimensionalise(y, self.y0)
        multiplier_nondimensional = self.nondimensionalise(self.multiplier, self.y0)
        width_nondimensional = self.nondimensionalise(self.width, self.y0)
        # calculate
        lower = (multiplier_nondimensional - 0.5 * width_nondimensional) * xn**self.exponent
        upper = (multiplier_nondimensional + 0.5 * width_nondimensional) * xn**self.exponent
        if cumulative is False:
            p = 1. / (upper - lower)
            p[yn < lower] = 0.0
            p[yn > upper] = 0.0
        else:
            p = (yn - lower) / (upper - lower)
            p[yn < lower] = 0.0
            p[yn > upper] = 1.0
        # strip any units and return
        if isinstance(p, Quantity):
            return(p.magnitude)
        else:
            return(p)   

    def random(
            self, 
            x
            ):
        # lower and upper limits of distribution
        lower = (self.multiplier - 0.5 * self.width) * (x / self.x0)**self.exponent
        upper = (self.multiplier + 0.5 * self.width) * (x / self.x0)**self.exponent
        # variation
        P = np.random.rand(*x.shape)
        # return
        return(lower + P*(upper - lower))


################
### LOGISTIC ###
################

class PowerlawLogistic(_PowerlawFitBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'hybr',
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawLogistic, self).__init__(x, y, weights, x0 = x0)
        # set other input arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.scale0 = self.generate_fit(
                self.x, self.y, self.weights)
        else: 
            self.scale0 = self.redimensionalise(0.0, self.y0)

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess for root finding
        if self.start is None:
            self.start = list(self._initialguess_exponent_scale_nondimensional(xn, yn, weights))
        # find best fitting power law exponent
        f_root = lambda p: self.loglikelihood(
            x = xn, y = yn, weights = weights,
            multiplier = p[0], exponent = p[1], scale0 = p[2],
            nondimensional_input = True,
            deriv = 1
            )
        f_root_prime = lambda p: self.loglikelihood(
            x = xn, y = yn, weights = weights,
            multiplier = p[0], exponent = p[1], scale0 = p[2],
            nondimensional_input = True,
            deriv = 2
            )
        ft = root(
            f_root,
            x0 = self.start,
            jac = f_root_prime,
            method = self.root_method
            )
        multiplier_nondimensional, exponent, scale0_nondimensional = ft.x
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, scale0_nondimensional)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                self.redimensionalise(scale0_nondimensional, self.y0)
            )
    
    def loglikelihood(
            self,
            x = None,
            y = None,
            weights = None,
            multiplier = None, 
            exponent = None, 
            scale0 = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        scale0 = self.scale0 if scale0 is None else scale0
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
            scale0 = self.nondimensionalise(scale0, self.y0)
        else:
            xn = x
            yn = y
        # intermediate parameters
        eta = 1.0 / np.cosh((yn / xn**exponent - multiplier) / (2.*scale0))
        zeta = np.tanh((yn / xn**exponent - multiplier) / (2. * scale0))
        # weighted loglikelihood
        if (deriv == 0):
            logp = -np.log(4. * scale0) - exponent * np.log(x) + 2. * np.log(eta)
            return(np.sum(weights * logp))
        elif deriv == 1:
            # derivatives of loglikelihood
            dlogpL_dmultiplier = zeta / scale0
            dlogp_dexponent = yn * np.log(xn) * zeta / (scale0 * xn**exponent) - np.log(xn)
            dlogp_dscale0 = (yn / xn**exponent - multiplier) * zeta / scale0**2 - 1. / scale0
            dlogL_dmultiplier = np.sum(weights * dlogpL_dmultiplier)
            dlogL_dexponent = np.sum(weights * dlogp_dexponent)
            dlogL_dscale0 = np.sum(weights * dlogp_dscale0)
            return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dscale0]))
        elif deriv == 2:
            # derivatives of zeta
            dzeta_dmultiplier = -eta**2 / (2. * scale0)
            dzeta_dexponent = -yn * np.log(xn) * eta**2 / (2. * scale0 * xn**exponent)
            dzeta_dscale0 = -(yn / xn**exponent - multiplier) * eta**2 / (2. * scale0**2)
            # derivatives of probability
            d2p_dmultiplier2 = dzeta_dmultiplier / scale0
            d2p_dmultiplierdexponent = dzeta_dexponent / scale0
            d2p_dmultiplierdscale0 = (dzeta_dscale0 - zeta / scale0) / scale0
            d2p_dexponent2 = yn * np.log(xn) * (dzeta_dexponent - np.log(xn) * zeta) / (scale0 * xn**exponent)
            d2p_dexponentdscale0 = yn * np.log(xn) * (dzeta_dscale0 - zeta / scale0) / (scale0 * xn**exponent)
            d2p_dscale02 = (yn / xn**exponent - multiplier) * (dzeta_dscale0 - 2. * zeta / scale0) / scale0**2 + 1. / scale0**2
            # derivatives of np.loglikelihood
            d2logL_dmultiplier2 = np.sum(weights * d2p_dmultiplier2)
            d2logL_dmultiplierdexponent = np.sum(weights * d2p_dmultiplierdexponent)
            d2logL_dmultiplierdscale0 = np.sum(weights * d2p_dmultiplierdscale0)
            d2logL_dexponent2 = np.sum(weights * d2p_dexponent2)
            d2logL_dexponentdscale0 = np.sum(weights * d2p_dexponentdscale0)
            d2logL_dscale02 = np.sum(weights * d2p_dscale02)
            # return
            return(np.array([
              [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdscale0],
              [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdscale0],
              [d2logL_dmultiplierdscale0, d2logL_dexponentdscale0, d2logL_dscale02]
              ]))

    def _initialguess_nondimensional(
            self,
            xn,
            yn,
            weights
            ):
        # guess parameters from normal fit
        ftN = PowerlawNormalScaled(xn, yn, weights = weights)
        # return
        return(
            ftN.multiplier,
            ftN.exponent,
            ftN.sd_multiplier * np.sqrt(3.) / np.pi
        )

    def get_scale0(
            self, 
            x
            ):
        return(self.scale0 * (x / self.x0)**self.exponent)
        
    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get values of x
        if x is None:
            x = self.xrange(n=n)
        # cumulative fraction
        P_lower = 0.5 - 0.5 * level
        P_upper = 0.5 + 0.5 * level
        # scale and location parameter
        location = self.predict(x)
        scale0 = self.get_scale0(x)
        # upper and lower intervals
        y_lower = location + 2. * scale0 * np.arctanh(2. * P_lower - 1.)
        y_upper = location + 2. * scale0 * np.arctanh(2. * P_upper - 1.)
        # return
        return(x, np.column_stack((y_lower, y_upper)))

    def density(
            self, 
            x = None,
            y = None, 
            cumulative = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # make nondimensional, to avoid problems
        yn = self.nondimensionalise(y, self.y0)
        # calculate parameters
        location = self.nondimensionalise(self.predict(x), self.y0)
        scale0 = self.nondimensionalise(self.get_scale0(x), self.y0)
        # return
        if cumulative is False:
            return(1. / (4. * scale0 * np.cosh((yn - location)/(2. * scale0))**2))
        else:
            return(0.5 + 0.5 * np.tanh((yn - location) / (2. * scale0)))
                
    def random(
            self, 
            x
            ):
        # location and scale parameters
        location = self.predict(x)
        scale0 = self.get_scale(x)
        # variation
        P = np.random.rand(*x.shape)
        # return
        return(location + 2. * scale0 * np.arctanh(2. * P - 1.))
    

#################
### LOGNORMAL ###
#################

class PowerlawLognormal(_PowerlawFitBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawLognormal, self).__init__(x, y, weights, x0 = x0)
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.sdlog = self.generate_fit(
                self.x, self.y, self.weights)
        else: 
            self.sdlog = 0.0

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * np.log(xn)**2)
        c5 = np.sum(weights * np.log(xn) * np.log(yn))
        c6 = np.sum(weights * np.log(yn)**2)
        # solution
        exponent = -(c2 * c3 - c1 * c5) / (-c2**2 + c1 * c4)
        sdlog = np.sqrt(
            c6 / c1 
            - (c1 * c5**2 - 2. * c2 * c3 * c5 + c3**2 * c4)
            / (c1 * (c1 * c4 - c2**2))
            )
        multiplier_nondimensional = np.exp((c3 - exponent * c2) / c1 + sdlog**2 / 2.)
        # return
        if nondimensional_output is True:
            return(multiplier_nondimensional, exponent, sdlog)
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                sdlog
            )
        
    def loglikelihood(
            self, 
            x = None,
            y = None,
            weights = None,
            multiplier = None, 
            exponent = None, 
            sdlog = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        sdlog = self.sdlog if sdlog is None else sdlog
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
        else:
            xn = x
            yn = y
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * np.log(xn)**2)
        c5 = np.sum(weights * np.log(xn) * np.log(yn))
        c6 = np.sum(weights * np.log(yn)**2)
        # return result
        if deriv == 0:
            return(
                (np.log(multiplier) / sdlog**2 - 1.)
                * (c3 - exponent * c2 - c1 * np.log(multiplier) / 2.) 
                - c1 * (np.log(sdlog) + np.log(2. * np.pi) / 2. + sdlog**2 / 8.) 
                - (exponent * c2 + c3) /2. 
                - (exponent**2 * c4 - 2. * exponent * c5 + c6) / (2.*sdlog**2)
                )
        elif deriv == 1:
            dlogL_dmultiplier = (
                (c3 - exponent * c2 - c1 * np.log(multiplier))
                / (multiplier * sdlog**2) 
                + c1 / (2. * multiplier)
            )
            dlogL_dexponent = (
                0.5 * c2 
                - (exponent * c4 - c5 + c2 * np.log(multiplier))
                / (sdlog**2)
            )
            dlogL_dsdlog = (
                (c1 * np.log(multiplier)**2 
                 - 2. * np.log(multiplier) * (c3 - exponent * c2) 
                 + exponent**2 * c4 
                 - 2. * exponent * c5 
                 + c6)
                 / sdlog**3 
                 - c1 / sdlog 
                 - c1 * sdlog / 4.
                 )
            return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dsdlog]))
        elif deriv == 2:
            d2logL_dmultiplier2 = (
                -((c1 * sdlog**2) / 2. 
                  + c1 + c3 - exponent*c2 - c1*np.log(multiplier))
                  / (sdlog**2*multiplier**2)
            )
            d2logL_dmultiplierdexponent = -c2 / (sdlog**2 * multiplier)
            d2logL_dmultiplierdsdlog = (
                (2. * (exponent * c2 - c3 + c1 * np.log(multiplier)))
                / (sdlog**3 * multiplier)
            )
            d2logL_dexponent2 = -c4 / sdlog**2
            d2logL_dexponentdsdlog = (
                (2. * (exponent * c4 - c5 + c2 * np.log(multiplier)))
                / sdlog**3
            )
            d2logL_dsdlog2 = (
                -3. * (
                    c1 * np.log(multiplier)**2 
                    - 2. * np.log(multiplier) * (c3 - exponent * c2) 
                    + exponent**2 * c4 
                    - 2. * exponent * c5 
                    + c6
                ) 
                / sdlog**4 
                + c1 / sdlog**2 
                - c1 / 4.
            )
            return(np.array([
                [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdsdlog],
                [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdsdlog],
                [d2logL_dmultiplierdsdlog, d2logL_dexponentdsdlog, d2logL_dsdlog2]
                ]))
        
    def get_meanlog(
            self, 
            x
            ):
        # nondimensionalise
        xn = self.nondimensionalise(x, self.x0)
        multiplier = self.nondimensionalise(self.multiplier, self.y0)
        # return meanlog (mean of log(x/x0))
        return(
            np.log(multiplier)
            + self.exponent * np.log(xn) 
            - 0.5 * self.sdlog**2
        )    
    
    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get values of x
        if x is None:
            x = self.xrange(n=n)
        # cumulative fraction
        P_lower = 0.5 - 0.5*level
        P_upper = 0.5 + 0.5*level
        # mean-log parameter
        mulog = self.get_meanlog(x)
        # addition
        tmp_lower = np.sqrt(2.) * self.sdlog * erfinv(2. * P_lower - 1.)
        tmp_upper = np.sqrt(2.) * self.sdlog * erfinv(2. * P_upper - 1.)
        # interval
        y_lower_nondimensional = np.exp(mulog + tmp_lower)
        y_upper_nondimensional = np.exp(mulog + tmp_upper)
        # return
        return(
            x,
            self.redimensionalise(
                np.column_stack((y_lower_nondimensional, y_upper_nondimensional)),
                self.y0
            )
        )

    def density(
            self, 
            x = None,
            y = None,
            cumulative = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # make nondimensional, to avoid problems
        yn = self.nondimensionalise(y, self.y0)         
        # mean-log parameter
        mulog = self.get_meanlog(x)
        if cumulative is False:
            return(
                np.exp(-(np.log(yn) - mulog)**2 / (2. * self.sdlog**2)) 
                / (yn * self.sdlog * np.sqrt(2. * np.pi))
                )
        else:
            return(
                0.5 + 0.5 * erf(
                    (np.log(yn) - mulog)
                    / (self.sdlog * np.sqrt(2.))
                )
            )
                
    def random(
            self, 
            x
            ):
        # mean-log parameter
        mulog = self.get_meanlog(x)
        # variation
        P = np.random.rand(*x.shape)
        # return
        y_nondimensional = np.exp(mulog + np.sqrt(2.) * self.sdlog * erfinv(2. * P - 1.))
        return(self.redimensionalise(y_nondimensional, self.y0))


###############################
### LOGNORMAL (UNCORRECTED) ###
###############################

class PowerlawLognormalUncorrected(PowerlawLognormal):

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):        
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * np.log(xn)**2)
        c5 = np.sum(weights * np.log(xn) * np.log(yn))
        c6 = np.sum(weights * np.log(yn)**2)
        # solution
        exponent = (c1 * c5 - c2 * c3) / (c1 * c4 - c2**2)
        multiplier_nondimenisonal = np.exp((c3 - exponent * c2) / c1)
        sdlog = np.sqrt(
            np.log(multiplier_nondimenisonal)**2 
            + (c6 
               - 2. * exponent * c5 
               + exponent**2 * c4 
               + 2. * np.log(multiplier_nondimenisonal) * (exponent * c2 - c3)
               ) 
            / c1
        )
        # return
        if nondimensional_output is True:
            return(multiplier_nondimenisonal, exponent, sdlog)
        else:
            return(
                self.redimensionalise(multiplier_nondimenisonal, self.y0),
                exponent,
                sdlog
            )

    def loglikelihood(
            self, 
            x = None,
            y = None,
            weights = None,
            multiplier = None, 
            exponent = None, 
            sdlog = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        sdlog = self.sdlog if sdlog is None else sdlog
        # nondimensionalise data
        if nondimensional_input is False:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
        else:
            xn = x
            yn = y
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * np.log(xn)**2)
        c5 = np.sum(weights * np.log(xn) * np.log(yn))
        c6 = np.sum(weights * np.log(yn)**2)
        # return result
        if deriv == 0:
            return(
                -c1 * (np.log(sdlog) + 0.5 * np.log(2. * np.pi) + np.log(multiplier)**2 / (2. * sdlog**2)) 
                - c3 
                - 1. / (2. * sdlog**2) * (
                    c6 - 2. * exponent * c5 
                    + exponent**2 * c4 
                    + 2. * np.log(multiplier) * (exponent * c2 - c3)
                )
            )
        elif deriv == 1:
            dlogL_dmultiplier = (
                -1. / (sdlog**2 * multiplier)
                * (c1 * np.log(multiplier) + (exponent * c2 - c3))
            )
            dlogL_dexponent = -(exponent * c4 - c5 + c2 * np.log(multiplier)) / sdlog**2
            dlogL_dsdlog = (
                -c1 * (1. / sdlog - np.log(multiplier)**2 / sdlog**3) 
                + 1. / sdlog**3 
                * (c6 - 2. * exponent * c5 
                   + exponent**2 * c4 
                   + 2. * np.log(multiplier) * (exponent * c2 - c3)
                )
            )
            return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dsdlog]))
        elif deriv == 2:
            d2logL_dmultiplier2 = -(
                (c1 * (1. + np.log(multiplier)) - c3 + exponent * c2)
                / (multiplier**2 * sdlog**2)
            )
            d2logL_dmultiplierdexponent = -c2 / (sdlog**2 * multiplier)
            d2logL_dmultiplierdsdlog = (
                -2. / (sdlog**3 * multiplier) 
                * (c3 - exponent * c2 - c1 * np.log(multiplier))
            )
            d2logL_dexponent2 = -c4 / sdlog**2
            d2logL_dexponentdsdlog = 2. / sdlog**3 * (exponent * c4 - c5 + c2 * np.log(multiplier))
            d2logL_dsdlog2 = (
                -c1 * (-1. / sdlog**2 + 3. * np.log(multiplier)**2 / sdlog**4) 
                - 3. / sdlog**4 * (
                    c6 - 2. * exponent * c5 
                    + exponent**2 * c4 
                    + 2. * np.log(multiplier) * (exponent * c2 - c3)
                )
            )
            return(np.array([
                [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdsdlog],
                [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdsdlog],
                [d2logL_dmultiplierdsdlog, d2logL_dexponentdsdlog, d2logL_dsdlog2]
                ]))
    
    def get_meanlog(
            self, 
            x
        ):
        # nondimensionalise
        xn = self.nondimensionalise(x, self.x0)
        multiplier = self.nondimensionalise(self.multiplier, self.y0)
        # return meanlog (mean of log(x/x0))
        return(np.log(multiplier) + self.exponent * np.log(xn))
    

###########################
### NORMAL - BASE CLASS ###
###########################

class _PowerlawNormalBase(_PowerlawFitBase):
       
    def get_sd(
            self, 
            x
            ):
        return(self.sd_multiplier * (x / self.x0)**self.sd_exponent)
    
    def prediction_interval(
            self, 
            x = None, 
            level = 0.95, 
            n = 101
            ):
        # get values of x
        if x is None:
            x = self.xrange(n=n)
        # cumulative fraction
        P_lower = 0.5 - 0.5*level
        P_upper = 0.5 + 0.5*level
        # means and standard deviation
        mean = self.predict(x)
        sd = self.get_sd(x)
        # standard deviation multiplier
        tmp_lower = np.sqrt(2.) * erfinv(2. * P_lower - 1.)
        tmp_upper = np.sqrt(2.) * erfinv(2. * P_upper - 1.)
        # intervals
        y_lower = mean + sd * tmp_lower
        y_upper = mean + sd * tmp_upper
        # return
        return(x, np.column_stack((y_lower, y_upper)))
    
    def density(
            self, 
            x = None, 
            y = None,
            cumulative = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        # make nondimensional, to avoid problems
        yn = self.nondimensionalise(y, self.y0)   
        # calculate Gaussian parameters
        mean = self.nondimensionalise(self.predict(x), self.y0)
        sd = self.nondimensionalise(self.get_sd(x), self.y0)
        # return
        if cumulative is False:
            return(
                np.exp(-0.5 * ((yn - mean) / sd)**2)
                / (sd * np.sqrt(2. * np.pi))
                )
        else:
            return(0.5 + 0.5*erf((yn - mean) / (sd*np.sqrt(2.))))
        
    def random(
            self, 
            x
            ):
        # mean and standard deviation
        mu = self.predict(x)
        sd = self.get_sd(x)
        # variation
        P = np.random.rand(*x.shape)
        # return
        return(mu + np.sqrt(2.) * sd * erfinv(2. * P - 1.))
    
    def _root_bothexponents_nondimensional(
            self,
            par,
            xn,
            yn,
            weights,
            jac = True
    ):
        # unpack parameters
        exponent, sd_exponent = par
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent))
        c4 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent) * np.log(xn))
        c6 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn)
        c7 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn * np.log(xn))
        c9 = np.sum(weights * xn**(-2. * sd_exponent)* yn **2)
        c10 = np.sum(weights * xn**(-2. * sd_exponent) * yn**2 * np.log(xn))
        # temporary variables
        zeta = 1. / (c3**2 * c9 - c3 * c6**2)
        # roots
        dlogL_dexponent = c1 * c6 * (c3 * c7 - c4 * c6) * zeta
        dlogL_dsdexponent = c1 * (c3**2 * c10 - 2. * c3 * c6 * c7 + c4 * c6**2) * zeta - c2
        root = np.array([dlogL_dexponent, dlogL_dsdexponent])
        # return
        if jac is False:
            return(root)
        else:
            # additional coefficients
            c5 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent) * np.log(xn)**2)
            c8 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn * np.log(xn)**2)
            c11 = np.sum(weights * xn**(-2. * sd_exponent) * yn**2 * np.log(xn)**2)
            # temporary variables
            dzeta_dexponent = -(
                4. * c3 * c4 * c9 
                - 2. * c4 * c6**2 
                - 2. * c3 * c6 * c7
                ) * zeta**2
            dzeta_dsd_exponent = -(
                -4. * c3 * c4 * c9 
                - 2. * c3**2 * c10 
                + 2. * c4 * c6**2 
                + 4. * c3 * c6 * c7
                ) * zeta**2
            # derivatives
            d2logL_dexponent2 = (
                c1 
                * (c3 * c7 - c4 * c6)
                * (c6 * dzeta_dexponent + c7 * zeta) 
                + c1 * c6 * (c4 * c7 + c3 * c8 - 2. * c5 * c6) * zeta
                )
            d2logL_dexponentdsd_exponent = (
                c1 
                * (c3 * c7 - c4 * c6)
                * (c6 * dzeta_dsd_exponent - 2. * c7 * zeta) 
                + 2. * c1 * c6 * (c5 * c6 - c3 * c8) * zeta
                )
            d2logL_dsd_exponent2 = (
                c1 
                * (c3**2 * c10 - 2. * c3 * c6 * c7 + c4 * c6**2)
                * dzeta_dsd_exponent 
                + c1 * (
                    -4. * c3 * c4 * c10 
                    - 2. * c3**2 * c11 
                    + 4. * c3 * c7**2 
                    + 4. * c3 * c6 * c8 
                    - 2. * c5 * c6**2
                ) * zeta
            )
            # return matrix
            droot_dpar = np.array([
                [d2logL_dexponent2, d2logL_dexponentdsd_exponent],
                [d2logL_dexponentdsd_exponent, d2logL_dsd_exponent2]
                ])
            # return
            return(root, droot_dpar)
        
    def _loglikelihood_bothexponents(
            self,
            x,
            y,
            weights,
            multiplier,
            exponent,
            sd_multiplier,
            sd_exponent,
            nondimensional_input = False,
            deriv = 0
    ):
        # nondimensionalise input
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            multiplier = self.nondimensionalise(multiplier, self.y0)
            sd_multiplier = self.nondimensionalise(sd_multiplier, self.y0)
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent))
        c4 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent) * np.log(xn))
        c5 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent) * np.log(xn)**2)
        c6 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn)
        c7 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn * np.log(xn))
        c8 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn * np.log(xn)**2)
        c9 = np.sum(weights * xn**(-2. * sd_exponent) * yn**2)
        c10 = np.sum(weights * xn**(-2. * sd_exponent) * yn**2 * np.log(xn))
        c11 = np.sum(weights * xn**(-2. * sd_exponent) * yn**2 * np.log(xn)**2)
        # loglikelihood
        if deriv == 0:
            return(
                -c1 
                * (np.log(sd_multiplier) + 0.5 * np.log(2 * np.pi)) 
                - sd_exponent * c2 
                - (c9 - 2. * multiplier * c6 + c3 * multiplier**2)
                / (2. * sd_multiplier**2)
                )
        elif deriv == 1:
            dlogL_dmultiplier = (c6 - multiplier * c3) / sd_multiplier**2
            dlogL_dexponent = multiplier * (c7 - c4 * multiplier) / sd_multiplier**2
            dlogL_dsdmultiplier = (
                -c1 / sd_multiplier 
                + (c9 - 2. * c6 * multiplier + c3 * multiplier**2)
                / sd_multiplier**3
            )
            dlogL_dsdexponent = (
                c10 
                - 2. * c7 * multiplier 
                + c4 * multiplier**2
                ) / sd_multiplier**2 - c2
            return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dsdmultiplier, dlogL_dsdexponent]))
        elif deriv == 2:
            d2logL_dmultiplier2 = -c3 / sd_multiplier**2
            d2logL_dmultiplierdexponent = (c7 - 2. * c4 * multiplier) / sd_multiplier**2
            d2logL_dmultiplierdsdmultiplier = 2. * (c3 * multiplier - c6) / sd_multiplier**3
            d2logL_dmultiplierdsdexponent = 2. * (c4 * multiplier - c7) / sd_multiplier**2
            d2logL_dexponent2 = multiplier * (c8 - 2. * c5 * multiplier) / sd_multiplier**2
            d2logL_dexponentdsdmultiplier = 2. * multiplier * (c4 * multiplier - c7) / sd_multiplier**3
            d2logL_dexponentdsdexponent = 2. * multiplier * (c5 * multiplier - c8) / sd_multiplier**2
            d2logL_dsdmultiplier2 = (
                c1 / sd_multiplier**2 
                - 3. * (c9 - 2. * c6 * multiplier + c3 * multiplier**2)
                / sd_multiplier**4
            )
            d2logL_dsdmultiplierdsdexponent = (
                2. * (-c10 + 2. * c7 * multiplier - c4 * multiplier**2)
                / sd_multiplier**3
            )
            d2logL_dsdexponent2 = (
                (-2. * c11 + 4. * c8 * multiplier - 2. * c5 * multiplier**2)
                / sd_multiplier**2
            )
            return(np.array([
                [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdsdmultiplier, d2logL_dmultiplierdsdexponent],
                [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdsdmultiplier, d2logL_dexponentdsdexponent],
                [d2logL_dmultiplierdsdmultiplier, d2logL_dexponentdsdmultiplier, d2logL_dsdmultiplier2, d2logL_dsdmultiplierdsdexponent],
                [d2logL_dmultiplierdsdexponent, d2logL_dexponentdsdexponent, d2logL_dsdmultiplierdsdexponent, d2logL_dsdexponent2]
                ]))    


#########################
### NORMAL - STRENGTH ###
#########################

class PowerlawNormal(_PowerlawNormalBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'newton',
            x0 = 1.0,
            sd_exponent = 0.0
            ):
        # call initialisation from parent class
        super(PowerlawNormal, self).__init__(x, y, weights, x0 = x0)
        # set sd_exponent
        self.sd_exponent = sd_exponent
        # set other arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.sd_multiplier = self.generate_fit(
                self.x, self.y, self.weights,
                nondimensional_input = False,
                nondimensional_output = False
                )
        else: 
            self.sd_multiplier = self.redimensionalise(0.0, self.y0)

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y            
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_nondimensional(
                xn, yn, weights
            )
        # get exponents, solving root
        ft = root_scalar(
            self._root_nondimensional,
            x0 = self.start,
            fprime = True,
            args = (xn, yn, weights, True),
            method = self.root_method
            )
        exponent = ft.root
        # coefficients 
        c1 = np.sum(weights)
        c3 = np.sum(weights * xn**(2. * exponent - 2. * self.sd_exponent))
        c6 = np.sum(weights * xn**(exponent - 2. * self.sd_exponent) * yn)
        c9 = np.sum(weights * xn**(-2. * self.sd_exponent) * yn**2)
        # calculate power law multipliers
        multiplier_nondimensional = c6 / c3
        sd_multiplier_nondimensional = np.sqrt(c9 / c1 - c6**2 / (c1 * c3))
        # return
        if nondimensional_output is True:
            return(
                multiplier_nondimensional, 
                exponent, 
                sd_multiplier_nondimensional
                )
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                self.redimensionalise(sd_multiplier_nondimensional, self.y0)
                )
    
    def _root_nondimensional(
            self,
            exponent,
            xn,
            yn,
            weights,
            fprime = True            
            ):
        # calculate roots (and derivatives, if frime is True), 
        r = self._root_bothexponents_nondimensional(
            [exponent, self.sd_exponent],
            xn, yn, weights, 
            jac = fprime
            )
        # return
        if fprime is False:
            root = r
            return(root[0])
        else:
            root, droot_dpar = r
            return(root[0], droot_dpar[0, 0])

    def _initialguess_nondimensional(
            self,
            xn,
            yn,
            weights,
            ):
        # simple guess - linear regression of np.log transformed x and y data
        ftL = Linear(np.log(xn), np.log(yn), weights = weights)
        # return
        return(ftL.gradient) 

    def loglikelihood(
            self,
            x = None,
            y = None,
            weights = None,
            multiplier = None,
            exponent = None,
            sd_multiplier = None,
            sd_exponent = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # assign measurements
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # assign parameters
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        sd_multiplier = self.sd_multiplier if sd_multiplier is None else sd_multiplier
        sd_exponent = self.sd_exponent if sd_exponent is None else sd_exponent
        # loglikelihood with all parameters
        L = self._loglikelihood_bothexponents(
            x, y, weights, 
            multiplier, exponent, sd_multiplier, sd_exponent,
            nondimensional_input = nondimensional_input,
            deriv = deriv
        )
        # return
        if deriv == 0:
            return(L)
        elif deriv == 1:
            return(L[0:3])
        elif deriv == 2:
            return(L[0:3, 0:3])


######################
### NORMAL - FORCE ###
######################

class PowerlawNormalForce(PowerlawNormal):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'newton',
            x0 = 1.0,
            sd_exponent = -2.0
            ):
        # call initialisation from parent class
        super(PowerlawNormalForce, self).__init__(
            x, y, 
            weights = weights,
            start = start, 
            root_method = root_method,
            x0 = x0,
            sd_exponent = sd_exponent
            )


##########################################
### NORMAL - SCALED STANDARD DEVIATION ###
##########################################

class PowerlawNormalScaled(_PowerlawNormalBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'newton',
            x0 = 1.0,
            ):
        # call initialisation from parent class
        super(PowerlawNormalScaled, self).__init__(x, y, weights, x0 = x0)
        # set other arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.sd_multiplier = self.generate_fit(
                self.x, self.y, self.weights,
                nondimensional_input = False,
                nondimensional_output = False
                )
        else: 
            self.sd_multiplier = self.redimensionalise(0.0, self.y0)
        # set sd_exponent, equal to fitted exponent
        self.sd_exponent = self.exponent

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y            
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_nondimensional(
                xn, yn, weights
            )
        # get exponents, solving root
        ft = root_scalar(
            self._root_nondimensional,
            x0 = self.start,
            fprime = True,
            args = (xn, yn, weights, True),
            method = self.root_method
            )
        exponent = ft.root
        sd_exponent = exponent
        # coefficients 
        c1 = np.sum(weights)
        c3 = np.sum(weights * xn**(2.0 * exponent - 2.0 * sd_exponent))
        c6 = np.sum(weights * xn**(exponent - 2.0 * sd_exponent) * yn)
        c9 = np.sum(weights * xn**(-2.0 * sd_exponent) * yn**2)
        # calculate power law multipliers
        multiplier_nondimensional = c6 / c3
        sd_multiplier_nondimensional = np.sqrt(c9 / c1 - c6**2 / (c1 * c3))
        # return
        if nondimensional_output is True:
            return(
                multiplier_nondimensional, 
                exponent, 
                sd_multiplier_nondimensional
                )
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                self.redimensionalise(sd_multiplier_nondimensional, self.y0)
                )
    
    def _root_nondimensional(
            self,
            exponent,
            xn,
            yn,
            weights,
            fprime = True            
            ):
        # calculate roots (and derivatives, if frime is True), 
        r = self._root_bothexponents_nondimensional(
            [exponent, exponent],
            xn, yn, weights, 
            jac = fprime
            )
        # return
        if fprime is False:
            root = r
            return(np.sum(root))
        else:
            root, droot_dpar = r
            return(np.sum(root), np.sum(droot_dpar))

    def _initialguess_nondimensional(
            self,
            xn,
            yn,
            weights,
            ):
        # simple guess - linear regression of np.log transformed x and y data
        ftL = Linear(np.log(xn), np.log(yn), weights = weights)
        # return
        return(ftL.gradient) 

    def loglikelihood(
            self,
            x = None,
            y = None,
            weights = None,
            multiplier = None,
            exponent = None,
            sd_multiplier = None,
            sd_exponent = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # assign measurements
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # assign parameters
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        sd_multiplier = self.sd_multiplier if sd_multiplier is None else sd_multiplier
        sd_exponent = self.sd_exponent if sd_exponent is None else sd_exponent
        # loglikelihood with all parameters
        L = self._loglikelihood_bothexponents(
            x, y, weights, 
            multiplier, exponent, sd_multiplier, sd_exponent,
            nondimensional_input = nondimensional_input,
            deriv = deriv
        )
        # return
        if deriv == 0:
            return(L)
        elif deriv == 1:
            return(np.array([L[0], L[1] + L[3], L[2]]))
        elif deriv == 2:
            return(np.array([
                [L[0,0], L[0,1] + L[0,3], L[0,2]],
                [L[1,0] + L[3,0], L[1,1] + L[3,1] + L[1,3] + L[3,3], L[1,2] + L[3,2]],
                [L[2,0], L[2,1] + L[2,3], L[2,2]]
                ]))


####################################
### NORMAL - BOTH EXPONENTS FREE ###
####################################

class PowerlawNormalFreesd(_PowerlawNormalBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            start = None, 
            root_method = 'hybr',
            x0 = 1.0,
            ):
        # call initialisation from parent class
        super(PowerlawNormalFreesd, self).__init__(x, y, weights, x0 = x0)
        # set other arguments
        self.start = start
        self.root_method = root_method
        # get fit (if data not colinear in log-log space, in which case there is zero variance)
        if self.colinear is False:
            self.multiplier, self.exponent, self.sd_multiplier, self.sd_exponent = self.generate_fit(
                self.x, self.y, self.weights,
                nondimensional_input = False,
                nondimensional_output = False
                )
        else: 
            self.sd_multiplier = self.redimensionalise(0.0, self.y0)
            self.sd_exponent = 0.0

    def generate_fit(
            self,
            x,
            y,
            weights,
            nondimensional_input = False,
            nondimensional_output = False
            ):
        # nondimensionalise data
        if nondimensional_input is True:
            xn = x
            yn = y            
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_nondimensional(
                xn, yn, weights
            )
        # get exponents, solving root
        ft = root(
            self._root_nondimensional,
            x0 = self.start,
            jac = True,
            args = (xn, yn, weights, True),
            method = self.root_method
            )
        exponent, sd_exponent = ft.x
        # coefficients 
        c1 = np.sum(weights)
        c3 = np.sum(weights * xn**(2. * exponent - 2. * sd_exponent))
        c6 = np.sum(weights * xn**(exponent - 2. * sd_exponent) * yn)
        c9 = np.sum(weights * xn**(-2. * sd_exponent) * yn**2)
        # calculate power law multipliers
        multiplier_nondimensional = c6 / c3
        sd_multiplier_nondimensional = np.sqrt(c9 / c1 - c6**2 / (c1 * c3))
        # return
        if nondimensional_output is True:
            return(
                multiplier_nondimensional, 
                exponent, 
                sd_multiplier_nondimensional,
                sd_exponent
                )
        else:
            return(
                self.redimensionalise(multiplier_nondimensional, self.y0),
                exponent,
                self.redimensionalise(sd_multiplier_nondimensional, self.y0),
                sd_exponent
                )
    
    def _root_nondimensional(
            self,
            exponent,
            xn,
            yn,
            weights,
            jac = True            
            ):
        # calculate roots (and derivatives, if jac is True), 
        return(self._root_bothexponents_nondimensional(
            [exponent, exponent],
            xn, yn, weights, 
            jac = jac
            ))

    def _initialguess_nondimensional(
            self,
            xn,
            yn,
            weights,
            sd_exponent_offset = 1.0,
            n_range = 9
            ):
        # exponent - simple guess: linear regression of np.log transformed x and y data
        ftL = Linear(np.log(xn), np.log(yn), weights = weights)
        exponent = ftL.gradient
        # generate a number of guesses for sd_exponent
        sd_exponent_guess = exponent + np.linspace(-sd_exponent_offset, sd_exponent_offset, n_range)
        # fit using assumed values for delta
        fts = [PowerlawNormal(xn, yn, weights = weights, sd_exponent = d) 
               for d in sd_exponent_guess]
        # get guess with largest likelihood
        L = [ft.loglikelihood() for ft in fts]
        i_max = np.argmax(L)
        # return guesses for both exponents
        return(np.array([fts[i_max].exponent, fts[i_max].sd_exponent]))

    def loglikelihood(
            self,
            x = None,
            y = None,
            weights = None,
            multiplier = None,
            exponent = None,
            sd_multiplier = None,
            sd_exponent = None,
            nondimensional_input = False,
            deriv = 0
            ):
        # assign measurements
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # assign parameters
        multiplier = self.multiplier if multiplier is None else multiplier
        exponent = self.exponent if exponent is None else exponent
        sd_multiplier = self.sd_multiplier if sd_multiplier is None else sd_multiplier
        sd_exponent = self.sd_exponent if sd_exponent is None else sd_exponent
        # loglikelihood with all parameters
        L = self._loglikelihood_bothexponents(
            x, y, weights, 
            multiplier, exponent, sd_multiplier, sd_exponent,
            nondimensional_input = nondimensional_input,
            deriv = deriv
        )
        # return
        return(L)
