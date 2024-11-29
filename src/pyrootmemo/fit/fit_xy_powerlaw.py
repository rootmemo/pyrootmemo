# import packages
import numpy as np
from scipy.optimize import root
from scipy.special import gamma, digamma, polygamma, erfinv
from pyrootmemo.fit.fit_x import WeibullFit
from pyrootmemo.fit.fit_xy_linear import LinearFit


#################################
#### BASE CLASS - POWER LAWS ####
#################################

class PowerlawFitBase:

    # initialise - default
    def __init__(
            self, 
            x,
            y,
            weights = None
            ):
        # data
        self.x = x
        self.y = y
        if weights is None:
            n_measurements = len(x)
            self.weights = np.ones(n_measurements)
        else:
            self.weights = weights
        # check for zero variance case (perfect fit)
        self.colinear, self.multiplier, self.exponent = self.check_colinearity()
            
    # predict
    def predict(
            self, 
            x = None
        ):
        if x is None:
            x = self.x
        return(self.multiplier * x**self.exponent)

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
            y = self.x[order]
            weights = self.weights[order]
            # cumulative density of data
            y0 = np.cumsum(weights) / np.sum(weights)
            # cumulative density of fit
            y1 = self.density(x = x, y = y, weights = weights, cumulative = True)
            # differences between cumulatives curves: top and bottom of data
            d_top = np.max(np.abs(y1 - y0))
            d_bot = np.max(np.abs(y1 - np.append(0.0, y0[:-1])))
            # return
            return(max(d_top, d_bot))
      
    # covariance matrix, from MLE/fisher information, or bootstrapping
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
                data = rng.choice(
                    np.column_stack((self.x, self.y, self.weights)),
                    (n, len(self.x)), 
                    axis = 0, 
                    replace = True
                    )
                fits = np.array([self.generate_fit(*datai.T) for datai in data])
                return(np.cov(fits.transpose()))
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
            return(np.column_stack((x, y_pred, y_pred)))
        else:
            # derivatives of power-law function
            dy_dmultiplier = x**self.exponent
            dy_dpower = self.multiplier * np.log(x) * x**self.exponent
            # get covariance matrix
            cov = self.covariance()
            # get confidence interval using delta method
            var = (
                dy_dmultiplier**2 * cov[0, 0]
                + 2. * dy_dmultiplier * dy_dpower * cov[0, 1]
                + dy_dpower**2 * cov[1, 1]
                )
            # multiplier of standard deviation (from gaussian distribution)
            P = 0.5 + 0.5 * level
            mult = np.sqrt(2.) * erfinv(2. * P - 1.)
            # return
            return(np.column_stack((
                x, 
                y_pred - mult * np.sqrt(var),
                y_pred + mult * np.sqrt(var)
                )))
                        
    # check for colinearity - zero variance in residuals
    def check_colinearity(self):
        # unique pairs of x and y only - logtransform
        mask = (self.weights > 0.0)
        log_xy = np.log(np.unique(np.column_stack((self.x[mask], self.y[mask])), axis = 0))
        # check
        if len(log_xy) == 1:
            check = True
            exponent = 0.0
            multiplier = np.exp(log_xy[0, 1])            
        else:
            diff_log_xy = np.diff(log_xy, axis = 0)
            if len(log_xy) == 2:
                check = True
                exponent = diff_log_xy[0, 1] / diff_log_xy[0, 0]
                multiplier = np.exp(log_xy[0, 1] - exponent * log_xy[0, 0])
            else:
                # calculate cross-products of vectors connecting subsequent data points
                cross_prod = (
                    diff_log_xy[:-1, 0] * diff_log_xy[1:, 1] 
                    - diff_log_xy[1:, 0] * diff_log_xy[:-1, 1]
                    )
                check = all(np.isclose(cross_prod, 0.0))
                # in case of colinearity, get fitting parameters
                if check is True:
                    exponent = np.mean(diff_log_xy[:, 1] / diff_log_xy[:, 0])
                    multiplier = np.mean(np.exp(log_xy[:, 1] - exponent * log_xy[:, 0]))
                else:
                    exponent = None
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
            root_method = 'hybr'
            ):
        # call initialisation from parent class
        super(PowerlawFitWeibull, self).__init__(x, y, weights)
        # set other input arguments
        self.start = start
        self.root_method = root_method
        # get fit
        if self.colinear is False:
            self.multiplier, self.exponent, self.shape = self.generate_fit()
        else: 
            self.shape = np.inf
        
        
    def generate_fit(
            self,
            x = None,
            y = None,
            weights = None
            ):
        # custom data?
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
        # initial guess for root finding
        if self.start is None:
            self.start = self._initialguess_shape(x, y, weights)
        # fit for power law exponent, using root finding
        sol = root(  
            self._root,
            x0 = self.start,
            args = (x, y, weights),
            jac = True,
            method = self.root_method
        )
        exponent, shape = sol.x
        # calculate the power law multiplier
        c1 = np.sum(weights)
        c4 = np.sum(weights * x**(-exponent * shape) * y**shape)
        multiplier = gamma(1. + 1. / shape) * (c4 / c1)**(1. / shape)
        # return
        return(multiplier, exponent, shape)
    
    def _initialguess_shape(
            self,
            x, 
            y, 
            weights
            ):
        # guess exponent from linear regression on log-data
        ftL = LinearFit(np.log(x), np.log(y), weights = weights)
        # scale data
        y_scaled = y / x**ftL.gradient
        # fit weibull distribution
        ftW = WeibullFit(y_scaled, weights = weights)
        # return
        return(ftL.gradient, ftW.shape)
    
    def _root(
            self,
            par, 
            x, 
            y, 
            weights,
            fprime = True
            ):
        # unpack input parameters
        beta = par[0]  # power law exponent
        kappa = par[1]  # Weibull shape parameter
        # coefficients
        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(x))
        c3 = np.sum(weights * np.log(y))
        c4 = np.sum(weights * x**(-beta * kappa) * y**kappa)
        c5 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(x))
        c6 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(y))
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
            c7 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(x)**2)
            c8 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(x) * np.log(y))
            c9 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(y)**2)
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
        y0 = self.multiplier if multiplier is None else multiplier
        beta = self.exponent if exponent is None else exponent
        kappa = self.shape if shape is None else shape
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
            c2 = np.sum(weights * np.log(x))
            c3 = np.sum(weights * np.log(y))
            c4 = np.sum(weights * x**(-beta * kappa) * y**kappa)
            c5 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(x))
            c6 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(y))
            c7 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(x)**2)
            c8 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(x) * np.log(y))
            c9 = np.sum(weights * x**(-beta * kappa) * y**kappa * np.log(y)**2)
            # gamma functions
            g = gamma(1. + 1. / kappa)
            p = digamma(1. + 1. / kappa)
            q = polygamma(1, 1. + 1. / kappa)
            # loglikelihood
            if deriv == 0:
                return(
                    (np.log(kappa) - kappa * np.log(y0) + kappa * np.log(g)) * c1 
                    - beta * kappa * c2 
                    + (kappa - 1.) * c3 
                    - (g / y0)**kappa * c4
                    )
            # first partial derivative of loglikelihood
            elif deriv == 1:
                dlogL_dy0 = -c1 * kappa / y0 + c4 * kappa * g**kappa * y0**(-kappa - 1.)
                dlogL_dbeta = -kappa * c2 + kappa * (g / y0)**kappa * c5
                dlogL_dkappa = (c1 * (1./kappa + np.log(g / y0) - p / kappa) - beta * c2 + c3 
                    - (g / y0)**kappa * (c4 * (np.log(g / y0) - p / kappa) - beta * c5 + c6))
                return(np.array([dlogL_dy0, dlogL_dbeta, dlogL_dkappa]))
            # second partial derivative of loglikelihood
            elif deriv == 2:
                d2logL_dy02 = (c1*kappa/y0**2 
                               - c4*kappa*(kappa + 1.)*g**kappa*y0**(-kappa - 2.))
                d2logL_dy0dbeta = -c5*kappa**2*g**kappa*y0**(-kappa - 1.)
                d2logL_dy0dkappa = (-c1/y0 + g**kappa*y0**(-kappa - 1.)*
                                    (c4*(1. + kappa*np.log(g/y0) - p) + kappa*(c6 - beta*c5)))
                d2logL_dbeta2 = -kappa**2*(g/y0)**kappa*c7
                d2logL_dbetadkappa = (-c2 + (g/y0)**kappa*
                                      (c5*(1. + kappa*np.log(g/y0) - p) + kappa*(c8 - beta*c7)))
                d2logL_dkappa2 = (c1/kappa**2*(q/kappa - 1) 
                                  - (g/y0)**kappa*(
                                      2.*(np.log(g/y0) - p/kappa)*(c6 - beta*c5) 
                                      + (np.log(g/y0) - p/kappa)**2*c4 
                                      + (c4*q/kappa**3 + beta**2*c7 - 2.*beta*c8 + c9)
                                      ))
                return(np.array([
                    [d2logL_dy02, d2logL_dy0dbeta, d2logL_dy0dkappa], 
                    [d2logL_dy0dbeta, d2logL_dbeta2, d2logL_dbetadkappa], 
                    [d2logL_dy0dkappa, d2logL_dbetadkappa, d2logL_dkappa2]
                    ]))
        
    # calculate scale parameter at any value of x
    def get_scale(
            self, 
            x
            ):
        return(self.multiplier * x**self.exponent / gamma(1. + 1. / self.shape))
            
    # generate prediction intervals
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
            y_pred = self.predict(x)
            return(np.column_stack((x, y_pred, y_pred)))
        else:
            # cumulative fraction
            P = np.array([0.5 - 0.5 * level, 0.5 + 0.5 * level])
            # calculate scale parameter
            scale = self.get_scale(x)
            # interval
            y_interval = np.outer(scale, (-np.log(1. - P))**(1. / self.shape))
            # return
            return(np.column_stack((x, y_interval)))
        
    # probability density
    def density(
            self, 
            x = None,
            y = None,
            weights = None, 
            cumulative = False
            ):
        # data
        x = self.x if x is None else x
        y = self.y if y is None else y
        weights = self.weights if weights is None else weights
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
        
    # generate random data based on known x-values
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
        
    




def _powerlaw_weibull_root(par, x, y, w):
    # unpack input parameters
    beta = par[0]
    kappa = par[1]
    # coefficients
    c1 = np.sum(w)
    c2 = np.sum(w*np.log(x))
    c3 = np.sum(w*np.log(y))
    c4 = np.sum(w*x**(-beta*kappa)*y**kappa)
    c5 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(x))
    c6 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(y))
    # roots
    dlogL_dbeta = kappa*c1*c5/c4 - kappa*c2
    dlogL_dkappa = (1./kappa + (beta*c5 - c6)/c4)*c1 - beta*c2 + c3
    # return
    return(np.array([dlogL_dbeta, dlogL_dkappa]))


def _powerlaw_weibull_root_jacobian(par, x, y, w):
    # unpack input parameters
    beta = par[0]
    kappa = par[1]
    # coefficients
    c1 = np.sum(w)
    c2 = np.sum(w*np.log(x))
    c4 = np.sum(w*x**(-beta*kappa)*y**kappa)
    c5 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(x))
    c6 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(y))
    c7 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(x)**2)
    c8 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(x)*np.log(y))
    c9 = np.sum(w*x**(-beta*kappa)*y**kappa*np.log(y)**2)
    # derivatives
    d2logL_dbeta2 = c1*kappa**2/c4*(c5**2/c4 - c7)
    d2logL_dbetadkappa = (c1/c4*(c5 - beta*kappa*c7 + kappa*c8
                                 + kappa*c5/c4*(beta*c5 - c6)) - c2)
    d2logL_dkappa2 = ((-1./kappa**2 - (beta**2*c7 - 2.*beta*c8 + c9)/c4 
                       + (beta*c5 - c6)**2/c4**2)*c1)
    # return matrix
    return(np.array([
        [d2logL_dbeta2, d2logL_dbetadkappa], 
        [d2logL_dbetadkappa, d2logL_dkappa2]
        ]))