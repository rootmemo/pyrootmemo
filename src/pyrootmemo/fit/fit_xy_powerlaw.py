# import packages
import numpy as np
from scipy.optimize import root, root_scalar, bracket
from scipy.special import gamma, digamma, polygamma, loggamma, erfinv, gammaincinv, gammainc
from scipy.spatial import ConvexHull
from pyrootmemo.fit.fit import FitBase
from pyrootmemo.fit.fit_x import WeibullFit, GumbelFit
from pyrootmemo.fit.fit_xy_linear import LinearFit
from pint import Quantity


#################################
#### BASE CLASS - POWER LAWS ####
#################################

class PowerlawFitBase(FitBase):

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
                        xn[i], yn[i], self.weights[i],
                        nondimensional_input = True, nondimensional_output = True
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
            deriv = 0,
            nondimensional_input = False
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

class PowerlawFitGamma(PowerlawFitBase):

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
        super(PowerlawFitGamma, self).__init__(x, y, weights, x0 = x0)
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
        ftL = LinearFit(xn, yn, weights = weights)
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
                dlogL_dy0 = shape / multiplier * (c4 / multiplier - c1)
                dlogL_dbeta = shape * (c5 / multiplier - c2)
                dlogL_dk = (
                    c1 * (1. + np.log(shape) - p - np.log(multiplier)) 
                    - exponent * c2 
                    + c3 
                    - c4 / multiplier
                )
                return(np.array([dlogL_dy0, dlogL_dbeta, dlogL_dk]))
            elif deriv == 2:
                d2logL_dy02 = shape / multiplier**2 * (c1 - 2. * c4 / multiplier)
                d2logL_dy0dbeta = -shape * c5 / multiplier**2
                d2logL_dy0dk = 1. / multiplier * (c4 / multiplier - c1)
                d2logL_dbeta2 = -shape * c6 / multiplier
                d2logL_dbetadk = c5 / multiplier - c2
                d2logL_dk2 = c1 * (1. / shape - q)
                return(np.array([
                    [d2logL_dy02, d2logL_dy0dbeta, d2logL_dy0dk],
                    [d2logL_dy0dbeta, d2logL_dbeta2, d2logL_dbetadk],
                    [d2logL_dy0dk, d2logL_dbetadk, d2logL_dk2]
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

class PowerlawFitGumbel(PowerlawFitBase):

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
        super(PowerlawFitGumbel, self).__init__(x, y, weights, x0 = x0)
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
        ftL = LinearFit(np.log(xn), np.log(yn), weights = weights)
        exponent = ftL.gradient
        # get scale parameter based gumbel fitting of y/x^beta
        ftG = GumbelFit(yn / (xn**exponent), weights = weights)
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
        c12 = np.sum(weights * yn**2 * np.log(xn)**2 / (xn**(2.*exponent)) * np.exp(-yn / (scale0 * x**exponent)))
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

class PowerlawFitUniform(PowerlawFitBase):

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
        super(PowerlawFitUniform, self).__init__(x, y, weights, x0 = x0)
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
                ftL = LinearFit(np.log(xn), np.log(yn), weights = weights)
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
    