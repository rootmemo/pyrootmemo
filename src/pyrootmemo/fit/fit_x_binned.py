## FIT DISTRIBUTIONS TO BINNED x-DATA, BASED ON (WEIGHTED) LIKELIHOOD FITTING
## * Power law distribution (class "PowerFitBinned")


# import packages
import numpy as np
from scipy.optimize import minimize


####################################
#### BASE CLASS FOR BINNED DATA ####
####################################

# Base class for binned probability density function fitting
class FitBinned:
    
    # initialise - default
    def __init__(
              self, 
              xmin, 
              xmax, 
              n = None, 
              weights = None, 
              start = None
              ):
        # number of bins
        n_bin = len(xmin)
        # set x, y data
        self.xmin = xmin
        self.xmax = xmax
        # number of observations
        if n is None:
            self.n = np.ones(n_bin)
        else:
            self.n = n
        # weighting of each class
        if weights is None:
            self.weights = np.ones(n_bin)
        else:
            self.weights = weights
        # starting guess for fit
        self.start = start
        # user-defined upper and lower values for min and max of distribution


############################
#### POWER DISTRIBUTION ####
############################

class PowerFitBinned(FitBinned):
        
    def __init__(
            self,
            xmin,
            xmax,
            n = None,
            weights = None,
            start = None,
            bin_min = 1.0e-6,
            bin_margin = 1.0e-6,
            lower = None,
            upper = None            
        ):
        # call initialiser of parent class
        super().__init__(xmin, xmax, n = n, weights = weights, start = start)
        # set user limits
        self.lower = lower
        self.upper = upper
        # set for each parameter (exponent, lower and upper) whether unknown
        self.unknown = [True, self.lower is None, self.upper is None]
        # generate fit
        self.exponent, self.lower, self.upper = self.generate_fit(bin_min = bin_min, bin_margin = bin_margin)
        self.multiplier = self.get_multiplier(self.exponent, self.lower, self.upper)

    def _initialguess(self):
        xmid = 0.5*(self.xmin + self.xmax)
        xmin = np.min(xmid[(self.weights * self.n) > 0])
        xmax = np.max(xmid[(self.weights * self.n) > 0])
        exponent = 0.0
        par = np.array([exponent, xmin, xmax])
        return(par[self.unknown])
    
    def generate_fit(
            self, 
            bin_min = 1.0e-6, 
            bin_margin = 1.0e-6
        ):
        # initial guess
        if self.start is None:
            self.start = self._initialguess()
        # bounds for all fitting parameters: b, margin_rel xmin and xmax
        mask = (self.weights * self.n) > 0.0
        xmin_max = np.min(self.xmax[mask] - bin_margin * (self.xmax - self.xmin)[mask])
        xmax_min = np.max(self.xmin[mask] + bin_margin * (self.xmax - self.xmin)[mask])
        bounds = np.array([(None, None), (bin_min, xmin_max), (xmax_min, None)])
        bounds = bounds[self.unknown, :]
        # root solve for shape parameter
        ft = minimize(
            self._minimize_function, 
            self.start, 
            jac = True,
            args = (self.xmin, self.xmax, self.n, self.weights, self.lower, self.upper),
            bounds = bounds
            )
        # parameter vector
        parall = np.zeros(3)
        parall[self.unknown] = ft.x
        if self.lower is not None:
            parall[1] = self.lower
        if self.upper is not None:
            parall[2] = self.upper
        # return
        return(parall)
    
    def get_multiplier(
            self, 
            exponent, 
            lower, 
            upper
        ):
        # asymptotic case where exponent is equal to -1
        if np.isclose(exponent, -1.0):
            t = np.log(upper / lower)
            return(1. / t)
        # all other cases (exponent not equal to -1)
        else:
            t = upper**(exponent + 1.) - lower**(exponent + 1.)
            return((exponent + 1.) / t)
        
    def loglikelihood(
            self, 
            xmin = None, 
            xmax = None,
            n = None,
            exponent = None, 
            lower = None, 
            upper = None, 
            deriv = 0
            ):
        # get data
        xmin = self.xmin if xmin is None else xmin
        xmax = self.xmax if xmax is None else xmax
        n = self.n if n is None else n
        exponent = self.exponent if exponent is None else exponent
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper
        # return
        par = np.array([exponent, lower, upper])
        L = self._minimize_function(par, self.xmin, self.xmax, self.n, self.weights)
        if deriv == 0:
            return(L[0])
        elif deriv == 1:
            return(L[1])
        
    def _minimize_function(
            self,
            par,
            xmin,
            xmax,
            n,
            weights,
            lower = None,
            upper = None
        ):
        # known mask
        unknown = np.array([True, lower is None, upper is None])
        # unpack fitting parameters
        parall = np.zeros(3)
        parall[unknown] = par
        if lower is not None:
            parall[1] = lower
        if upper is not None:
            parall[2] = upper
        exponent, lower, upper = parall
        # round class boundaries (min and max class may partially stick out of the distribution)
        dx1_dxmin = (lower > xmin)
        dx2_dxmax = (upper < xmax)
        x1 = np.maximum(xmin, lower)
        x2 = np.minimum(xmax, upper)
        # calculate probability
        if np.isclose(exponent, -1.0):
            # probability
            p = np.log(x2 / x1) / np.log(upper / lower)
            logp = np.log(p)
            # probability - first derivatives
            dp_db = (
                (np.log(x2 / x1) * (np.log(x1) + np.log(x2) - np.log(upper) - np.log(lower)))
                / (2. * np.log(upper / lower))
            )
            dp_dx1 = -1. / (x1 * np.log(upper / lower))
            dp_dx2 = 1. / (x2 * np.log(upper / lower))
            dp_dxmin = (
                np.log(x2 / x1) 
                / (lower * np.log(upper / lower)**2)
                + dp_dx1 * dx1_dxmin
            )
            dp_dxmax = (
                -np.log(x2 / x1)
                / (upper * np.log(upper/lower)**2) 
                + dp_dx2 * dx2_dxmax
            )
            dlogp_dp = 1. / p
            dlogp_db = dlogp_dp * dp_db
            dlogp_dxmin = dlogp_dp * dp_dxmin
            dlogp_dxmax = dlogp_dp * dp_dxmax
        else:
            # temporary variables
            t1 = x1**(exponent + 1.)
            t2 = x2**(exponent + 1.)
            tmin = lower**(exponent + 1.)
            tmax = upper**(exponent + 1.)
            # probability
            p = (t2 - t1) / (tmax - tmin)
            logp = np.log(p)
            # temporary variables - first derivative
            dt1_db = np.log(x1) * t1
            dt1_dx1 = (exponent + 1.) * x1**exponent
            dt2_db = np.log(x2) * t2
            dt2_dx2 = (exponent + 1.) * x2**exponent
            dtmin_db = np.log(lower) * tmin
            dtmin_dxmin = (exponent + 1.) * lower**exponent
            dtmax_db = np.log(upper)*tmax
            dtmax_dxmax = (exponent + 1.) * upper**exponent
            # probability - first derivatives
            dp_dt1 = -1. / (tmax - tmin)
            dp_dt2 = 1. / (tmax - tmin)
            dp_dtmin = (t2 - t1) / ((tmax - tmin)**2)
            dp_dtmax = -(t2 - t1) / ((tmax - tmin)**2)
            dp_db = dp_dt1 * dt1_db + dp_dt2 * dt2_db + dp_dtmin * dtmin_db + dp_dtmax * dtmax_db
            dp_dxmin = dp_dt1 * dt1_dx1 * dx1_dxmin + dp_dtmin * dtmin_dxmin
            dp_dxmax = dp_dt2 * dt2_dx2 * dx2_dxmax + dp_dtmax * dtmax_dxmax
            dlogp_dp = 1. / p
            dlogp_db = dlogp_dp * dp_db
            dlogp_dxmin = dlogp_dp * dp_dxmin
            dlogp_dxmax = dlogp_dp * dp_dxmax
        # calculate loglikelilhood
        logL = np.sum(weights * n * logp)
        # first derivative of loglikelihood
        dlogL_dpar = np.array([
            np.sum(weights * n * dlogp_db), 
            np.sum(weights * n * dlogp_dxmin), 
            np.sum(weights * n * dlogp_dxmax)
            ])
        # return negative results (in order to maximum likelihood using minimum solvers)
        return(-logL, -dlogL_dpar[unknown])

    # random draw
    def random(self, n):
        y = np.random.rand(n)
        if np.isclose(self.exponent, -1):
            return(self.lower * (self.upper / self.lower)**y)
        else:
            return((
                self.lower**(self.exponent + 1.)
                + y * (self.upper**(self.exponent + 1.) - self.lower**(self.exponent + 1.))
                ) ** (1.0 / (self.exponent + 1.))
                )
        
    # calculate probability density
    def density(
            self, 
            x, 
            cumulative = False
            ):
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