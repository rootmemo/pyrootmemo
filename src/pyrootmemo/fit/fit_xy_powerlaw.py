# import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, root_scalar, bracket
from scipy.special import gamma, digamma, polygamma, loggamma, erf, erfinv, gammaincinv, gammainc
from scipy.spatial import ConvexHull
from pyrootmemo.fit.fit import FitBase
from pyrootmemo.fit.fit_x import WeibullFit, GumbelFit
from pyrootmemo.fit.fit_xy_linear import LinearFit
from pyrootmemo.utils_plot import round_range
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
            x_unit = 'mm',
            y_unit = 'MPa',
            x_label = 'Diameter',
            y_label = 'Tensile strength',
            data = True,
            fit = True,
            n = 101,
            confidence = True,
            confidence_level = 0.95,
            prediction = False,
            prediction_level = 0.95,
            legend = True,
            axis_expand = 0.05          
            ):
        # initiate plot
        fig, ax = plt.subplots(1, 1)
        # add measured data
        if isinstance(self.x, Quantity):
            x_data = self.x.to(x_unit).magnitude
        else:
            x_data = self.x
        if isinstance(self.y, Quantity):
            y_data = self.y.to(y_unit).magnitude
        else:
            y_data = self.y
        if data is True:
            ax.plot(x_data, y_data, 'x', label = 'Data')
        # add fit
        if fit is True:
            x_fit = np.linspace(np.min(self.x), np.max(self.x), n)
            y_fit = self.predict(x_fit)
            if isinstance(x_fit, Quantity):
                x_fit = x_fit.to(x_unit).magnitude
            if isinstance(y_fit, Quantity):
                y_fit = y_fit.to(y_unit).magnitude
            ax.plot(x_fit, y_fit, '-', label = 'Fit')
        # add prediction interval
        if prediction is True and hasattr(self, 'prediction_interval'):
            xp, yp = self.prediction_interval(
                level = prediction_level,
                n = n
                )
            if isinstance(xp, Quantity):
                xp = xp.to(x_unit).magnitude
            if isinstance(yp, Quantity):
                yp = yp.to(y_unit).magnitude
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
                xc = xc.to(x_unit).magnitude
            if isinstance(yc, Quantity):
                yc = yc.to(y_unit).magnitude
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
        ax.set_xlabel(x_label + ' [' + x_unit + ']')
        ax.set_ylabel(y_label + ' [' + y_unit + ']')
        # add legend
        if legend is True:
            ax.legend(loc = 'upper right')
        # return figure and axis objects
        return(fig, ax)


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
        # get scale parameter based gumbel fitting of y/x^exponent
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


################
### LOGISTIC ###
################

class PowerlawFitLogistic(PowerlawFitBase):

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
        super(PowerlawFitLogistic, self).__init__(x, y, weights, x0 = x0)
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
        ftN = PowerlawFitNormalScaled(xn, yn, weights = weights)
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

class PowerlawFitLognormal(PowerlawFitBase):

    def __init__(
            self, 
            x,
            y, 
            weights = None,
            x0 = 1.0
            ):
        # call initialisation from parent class
        super(PowerlawFitLognormal, self).__init__(x, y, weights, x0 = x0)
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

class PowerlawFitLognormalUncorrected(PowerlawFitLognormal):

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

class PowerlawFitNormalBase(PowerlawFitBase):
       
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

class PowerlawFitNormal(PowerlawFitNormalBase):

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
        super(PowerlawFitNormal, self).__init__(x, y, weights, x0 = x0)
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
        ftL = LinearFit(np.log(xn), np.log(yn), weights = weights)
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

class PowerlawFitNormalForce(PowerlawFitNormal):

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
        super(PowerlawFitNormalForce, self).__init__(
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

class PowerlawFitNormalScaled(PowerlawFitNormalBase):

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
        super(PowerlawFitNormalScaled, self).__init__(x, y, weights, x0 = x0)
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
        ftL = LinearFit(np.log(xn), np.log(yn), weights = weights)
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

class PowerlawFitNormalFreesd(PowerlawFitNormalBase):

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
        super(PowerlawFitNormalFreesd, self).__init__(x, y, weights, x0 = x0)
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
        ftL = LinearFit(np.log(xn), np.log(yn), weights = weights)
        exponent = ftL.gradient
        # generate a number of guesses for sd_exponent
        sd_exponent_guess = exponent + np.linspace(-sd_exponent_offset, sd_exponent_offset, n_range)
        # fit using assumed values for delta
        fts = [PowerlawFitNormal(xn, yn, weights = weights, sd_exponent = d) 
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
