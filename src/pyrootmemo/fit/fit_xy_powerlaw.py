# import packages
import numpy as np
from scipy.optimize import root
from scipy.special import gamma, digamma, polygamma, erfinv
from pyrootmemo.fit.fit_x import WeibullFit
from pyrootmemo.fit.fit_xy_linear import LinearFit
from pint import Quantity


#################################
#### BASE CLASS - POWER LAWS ####
#################################

class PowerlawFitBase:

    # initialise - default
    def __init__(
            self, 
            x,
            y,
            weights = None,
            x0 = 1.0
            ):
        # data
        self.x = x
        self.y = y
        if weights is None:
            n_measurements = len(x)
            self.weights = np.ones(n_measurements)
        else:
            self.weights = weights
        # set reference x-value
        if isinstance(x, Quantity) and not isinstance(x0, Quantity):
            self.x0 = x0 * x.units
        else:
            self.x0 = x0
        # set reference y-value (required for correct handling of units --> set to 1 unit)
        if isinstance(y, Quantity):
            self.y0 = 1.0 * y.units
        else:
            self.y0 = 1.0
        # check for zero variance case (perfect fit)
        self.colinear, self.multiplier, self.exponent = self.check_colinearity()
    
    # generate non-dimensional data
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

    # add units + scaling again
    def redimensionalise(
            self,
            x,
            x0 = 1.0
    ):
        return(x * x0)

    # predict
    def predict(
            self, 
            x = None
            ):
        if x is None:
            x = self.x
        return(self.multiplier * (x / self.x0)**self.exponent)

    # kolmogorov-smirnov distance
    def ks_distance(self):
        if self.colinear is True:
            # colinear case
            return(0.0)
        else:
            # sort data in increasing order of occurance
            yp = self.predict()
            order = np.argsort(self.weights / yp)
            x = self.x[order]
            y = self.y[order]
            weights = self.weights[order]
            # cumulative density of data
            cumul_data = np.cumsum(weights) / np.sum(weights)
            # cumulative density of fit
            cumul_fit = self.density(x = x, y = y, cumulative = True)
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
            J = self.loglikelihood(deriv = 2)
            if J is None or method == 'bootstrap':
                # bootstrapping
                rng = np.random.default_rng()
                # select random data indices
                indices = rng.choice(
                    np.arange(len(self.x), dtype = int),
                    (n, len(self.x)),
                    replace = True
                )
                # generate fit results
                fits = [self.generate_fit(self.x[i], self.y[i], self.weights[i])
                        for i in indices]
                multiplier, exponent, shape = zip(*fits)
                # nondimensionalise                
                exponent = np.array(exponent)
                shape = np.array(shape)
                multiplier_nondimensional = self.nondimensionalise(multiplier, self.y0)
                # return covariance matrix, in non-dimensional units
                return(np.cov(np.stack((multiplier_nondimensional, exponent, shape))))
            else:
                fisher = -J
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
    

#################
#### WEIBULL ####
#################

# Power law Weibull class
class PowerlawFitWeibull(PowerlawFitBase):
    
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
        super(PowerlawFitWeibull, self).__init__(x, y, weights, x0 = x0)
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
            weights
            ):
        
        # nondimensionalise data
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
        # reintroduce scaling for y (and units)
        multiplier = self.redimensionalise(multiplier_nondimensional, self.y0)
        # return
        return(multiplier, exponent, shape)
    
    def _initialguess_shape(
            self,
            xn,
            yn, 
            weights,
            ):
        # guess exponent from linear regression on log-data
        ftL = LinearFit(np.log(xn), np.log(yn), weights = weights)
        # scale data
        y_scaled = yn / (xn**ftL.gradient)
        # fit weibull distribution
        ftW = WeibullFit(y_scaled, weights = weights)
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
        beta = par[0]  # power law exponent
        kappa = par[1]  # Weibull shape parameter
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(xn))
        c3 = np.sum(weights * np.log(yn))
        c4 = np.sum(weights * xn**(-beta * kappa) * yn**kappa)
        c5 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(xn))
        c6 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(yn))
        # roots
        dlogL_dbeta = kappa * c1 * c5 / c4 - kappa * c2
        dlogL_dkappa = (1. / kappa + (beta * c5 - c6) / c4) * c1 - beta * c2 + c3
        # root
        root = np.array([dlogL_dbeta, dlogL_dkappa])
        # return
        if fprime is False:
            # return root
            return(root)
        else:
            # also get derivative
            # extra coefficients
            c7 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(xn)**2)
            c8 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(xn) * np.log(yn))
            c9 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(yn)**2)
            # derivatives
            d2logL_dbeta2 = c1 * kappa**2 / c4 * (c5**2 / c4 - c7)
            d2logL_dbetadkappa = (
                c1 / c4 
                * (c5 - beta * kappa * c7 + kappa*c8 + kappa * c5 / c4 * (beta * c5 - c6)) 
                - c2
                )
            d2logL_dkappa2 = (
                (-1. / kappa**2 - (beta**2 * c7 - 2. * beta * c8 + c9) / c4 
                 + (beta * c5 - c6)**2 / c4**2) * c1
                 )
            root_jacobian = np.array([
                [d2logL_dbeta2, d2logL_dbetadkappa], 
                [d2logL_dbetadkappa, d2logL_dkappa2]
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
            deriv = 0
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # fit results
        multiplier = self.multiplier if multiplier is None else multiplier
        beta = self.exponent if exponent is None else exponent
        kappa = self.shape if shape is None else shape
        # nondimensionalise data
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
            c4 = np.sum(weights * xn**(-beta * kappa) * yn**kappa)
            c5 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(xn))
            c6 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(yn))
            c7 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(xn)**2)
            c8 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(xn) * np.log(yn))
            c9 = np.sum(weights * xn**(-beta * kappa) * yn**kappa * np.log(yn)**2)
            # gamma functions
            g = gamma(1. + 1. / kappa)
            p = digamma(1. + 1. / kappa)
            q = polygamma(1, 1. + 1. / kappa)
            # loglikelihood
            if deriv == 0:
                return(
                    (np.log(kappa) - kappa * np.log(multiplier) + kappa * np.log(g)) * c1 
                    - beta * kappa * c2 
                    + (kappa - 1.) * c3 
                    - (g / multiplier)**kappa * c4
                    )
            # first partial derivative of loglikelihood
            elif deriv == 1:
                dlogL_dy0 = -c1 * kappa / multiplier + c4 * kappa * g**kappa * multiplier**(-kappa - 1.)
                dlogL_dbeta = -kappa * c2 + kappa * (g / multiplier)**kappa * c5
                dlogL_dkappa = (c1 * (1./kappa + np.log(g / multiplier) - p / kappa) - beta * c2 + c3 
                    - (g / multiplier)**kappa * (c4 * (np.log(g / multiplier) - p / kappa) - beta * c5 + c6))
                return(np.array([dlogL_dy0, dlogL_dbeta, dlogL_dkappa]))
            # second partial derivative of loglikelihood
            elif deriv == 2:
                d2logL_dy02 = (
                    c1 * kappa / multiplier**2 
                    - c4 * kappa * (kappa + 1.) * g**kappa * multiplier**(-kappa - 2.)
                    )
                d2logL_dy0dbeta = -c5 * kappa**2 * g**kappa * multiplier**(-kappa - 1.)
                d2logL_dy0dkappa = (
                    -c1 / multiplier 
                    + g**kappa * multiplier**(-kappa - 1.) 
                    * (c4 * (1. + kappa * np.log(g / multiplier) - p) 
                       + kappa * (c6 - beta * c5)
                       )
                    )
                d2logL_dbeta2 = -kappa**2 * (g / multiplier)**kappa * c7
                d2logL_dbetadkappa = (
                    -c2 
                    + (g / multiplier)**kappa 
                    * (c5 * (1. + kappa * np.log(g / multiplier) - p) 
                       + kappa*(c8 - beta * c7)
                       )
                    )
                d2logL_dkappa2 = (
                    c1 / kappa**2 * (q / kappa - 1) 
                    - (g / multiplier)**kappa * (
                        2. * (np.log(g / multiplier) - p / kappa) * (c6 - beta * c5) 
                        + (np.log(g / multiplier) - p / kappa)**2 * c4 
                        + (c4 * q / kappa**3 + beta**2 * c7 - 2. * beta * c8 + c9)
                        )
                    )
                return(np.array([
                    [d2logL_dy02, d2logL_dy0dbeta, d2logL_dy0dkappa], 
                    [d2logL_dy0dbeta, d2logL_dbeta2, d2logL_dbetadkappa], 
                    [d2logL_dy0dkappa, d2logL_dbetadkappa, d2logL_dkappa2]
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
                return(np.where(
                    np.isclose(y, self.predict(x)),
                    np.inf,
                    0.0))                               
            else:   
                return(np.where(
                    y < self.predict(x),
                    0.0,
                    1.0))
        else:
            # other cases
            scale = self.get_scale(x)
            if cumulative is False:
                return(self.shape 
                       / scale
                       * (y / scale)**(self.shape - 1.)
                       * np.exp(-(y / scale)**self.shape)
                       )
            else:
                return(1. - np.exp(-(y / scale)**self.shape))
        
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
        
    
