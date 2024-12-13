## FIT DISTRIBUTIONS TO BINNED x-DATA, BASED ON (WEIGHTED) LIKELIHOOD FITTING
## * Power law distribution (class "PowerFitBinned")


# import packages
import numpy as np
from scipy.optimize import minimize
from pyrootmemo.fit.fit import FitBase
from pint import Quantity


####################################
#### BASE CLASS FOR BINNED DATA ####
####################################

# Base class for binned probability density function fitting
class FitBaseXBinned(FitBase):
    
    # initialise - default
    def __init__(
              self, 
              x,
              y = None, 
              weights = None, 
              start = None,
              root_method = 'newton',
              x0 = None,
              xmin = None,
              xmin_include = True
              ):
        # check and set x bin parameters
        self.x = x
        self._check_input(x, finite = True, min = xmin, min_include = xmin_include, label = 'x')
        # check if upper limits always larger than lower limits
        if any(np.diff(x, axis = 1) <= 0.0):
            raise ValueError('x bins must be wider than zero')
        # number of bins
        n_bin = x.shape[0]
        # set number of observations in each bin
        if y is None:
            self.y = np.ones(n_bin)
        else:
            self.y = y
        self._check_input(self.y, finite = True, min = 0.0, min_include = True, label = 'y')  
        # get reference values
        self.x0 = self.get_reference(self.x, x0)
        self.y0 = self.get_reference(self.y)
        # weighting of each bin
        if weights is None:
            self.weights = np.ones(n_bin)
        else:
            self.weights = weights
        self._check_input(self.weights, finite = True, min = 0.0, min_include = True, label = 'weights')  
        # starting guess for fit
        self.start = start
        # set root method
        self.root_method = root_method


############################
#### POWER DISTRIBUTION ####
############################

class PowerFitBinned_old(FitBaseXBinned):
        
    def __init__(
            self,
            x,
            y = None,
            weights = None,
            start = None,
            bin_min = 1.0e-6,
            bin_margin = 1.0e-6,
            lower = None,
            upper = None,
            x0 = None  
        ):
        # call initialiser of parent class
        super().__init__(
              x, y = y, weights = weights,
              start = start, root_method = 'newton',
              x0 = x0,
              xmin = 0.0,
              xmin_include = False
        )
        # set user limits for lower x-limit of power law distribution
        # check:
        #   * units consistent with x input
        #   * lower must be within or below smallest bin with n*weights > 0
        #   * upper must be within or above largest bin with n*weights > 0
        if lower is not None or upper is not None:
            mask = (self.weights * self.y) > 0.0
            x_mask = self.x[mask, :]
        if lower is not None:
            self.check_dimensionality(lower, self.x)
            x_max = np.min(x_mask[:, 1])
            if lower > x_max:
                raise ValueError(f'lower must be smaller than {x_max} with given measurements')
        if upper is not None:
            self.check_dimensionality(upper, self.x)
            x_min = np.max(x_mask[:, 0])
            if upper < x_min:
                raise ValueError(f'lower must be smaller than {x_min} with given measurements')
        # generate fit
        self.multiplier, self.exponent, self.lower, self.upper = self.generate_fit(
            self.x, self.y, self.weights, 
            lower = lower, upper = upper,
            bin_min = bin_min, bin_margin = bin_margin
        )

    def _initialguess_exponent_nondimensional(
            self,
            xn,
            yn,
            weights,
            unknown = np.array([True, True, True])
            ):
        # midpoint of each class
        xn_mid = 0.5*np.sum(xn, axis = 1)
        # minimum and maximum, for classes with finite values
        mask = (weights * yn) > 0.0
        xn_min = np.min(xn_mid[mask])
        xn_max = np.max(xn_mid[mask])
        # initial guess for exponent
        exponent = 0.0
        # all parameters
        par = np.array([exponent, xn_min, xn_max])
        # only return guesses for non-fixed parameters
        return(par[unknown])
    
    def generate_fit(
            self,
            x,
            y,
            weights,
            lower = None,
            upper = None,
            bin_min = 1.0e-12, 
            bin_margin = 1.0e-6,
            nondimensional_input = False,
            nondimensional_output = False
        ):
        # set for each parameter (exponent, lower and upper) whether unknown
        unknown = [True, lower is None, upper is None]
        # make input nondimensional
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            if lower is not None:
                lower = self.nondimensionalise(lower, self.x0)
            if upper is not None:
                upper = self.nondimensionalise(upper, self.x0)
        # initial guess
        if self.start is None:
            self.start = self._initialguess_exponent_nondimensional(xn, yn, weights, unknown = unknown)
        # bounds for all fitting parameters: exponent, lower and upper
        mask = (weights * yn) > 0.0
        xmin_max = np.min(xn[mask, 1] - bin_margin * (xn[mask, 1] - xn[mask, 0])[mask])
        xmax_min = np.max(xn[mask, 0] + bin_margin * (xn[mask, 1] - xn[mask, 0])[mask])
        bounds = np.array([
            (None, None), 
            (bin_min, xmin_max), 
            (xmax_min, None)]
            )
        bounds = bounds[unknown, :]
        # root solve for shape parameter
        ft = minimize(
            self._minimize_function_nondimensional, 
            self.start, 
            jac = True,
            args = (xn, yn, self.weights, unknown, lower, upper),
            bounds = bounds
            )
        # parameter vector
        parall = np.zeros(3)
        parall[unknown] = ft.x
        if lower is not None:
            parall[1] = lower
        if upper is not None:
            parall[2] = upper
        # calculate (non-dimensional) multiplier
        multiplier = self.get_multiplier_nondimensional(*parall)
        # return (multiplier, exponent, lower and upper)
        if nondimensional_output is True:
            return(multiplier, *parall)
        else:
            return(
                self.redimensionalise(multiplier, self.y0),
                parall[0],
                self.redimensionalise(parall[1], self.x0),
                self.redimensionalise(parall[2], self.x0)
            )
    
    def get_multiplier_nondimensional(
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
    
    def _minimize_function_nondimensional(
            self,
            par,
            xn,
            yn,
            weights,
            unknown,
            lower = None,
            upper = None,
        ):
        # unpack fitting parameters (exponent, lower and upper)
        parall = np.zeros(3)
        parall[unknown] = par
        if lower is not None:
            parall[1] = lower
        if upper is not None:
            parall[2] = upper
        exponent, lower, upper = parall
        # round class boundaries (min and max class may partially stick out of the distribution)
        dx1_dxmin = (lower > xn[:, 0])
        dx2_dxmax = (upper < xn[:, 1])
        x1 = np.maximum(xn[:, 0], lower)
        x2 = np.minimum(xn[:, 1], upper)
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
        logL = np.sum(weights * yn * logp)
        # first derivative of loglikelihood
        dlogL_dpar = np.array([
            np.sum(weights * yn * dlogp_db), 
            np.sum(weights * yn * dlogp_dxmin), 
            np.sum(weights * yn * dlogp_dxmax)
            ])
        # return negative results (in order to maximum likelihood using minimum solvers)
        return(-logL, -dlogL_dpar[unknown])

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

    # random draw
    def random(self, n):
        y = np.random.rand(n)
        lower = self.nondimensionalise(self.lower, self.x0)
        upper = self.nondimensionalise(self.upper, self.x0)
        if np.isclose(self.exponent, -1):
            x = lower * (upper / lower)**y
        else:
            x = (
                lower**(self.exponent + 1.)
                + y * (upper**(self.exponent + 1.) - lower**(self.exponent + 1.))
                ) ** (1.0 / (self.exponent + 1.))
        return(self.redimensionalise(x, self.x0))
        
    # calculate probability density
    def density(
            self, 
            x, 
            cumulative = False
            ):
        # check dimensionality
        #self.check_dimensionality(x, self.x0)
        # make nondimensional
        xn = self.nondimensionalise(x, self.x0)
        multiplier = self.nondimensionalise(self.multiplier, self.y0)
        lower = self.nondimensionalise(self.lower, self.x0)
        upper = self.nondimensionalise(self.upper, self.x0)
        # get densities
        if cumulative is False:
            # probability density
            y = multiplier * xn**self.exponent
            y[xn < lower] = 0.0
            y[xn > upper] = 0.0
            # add dimension, and return
            return(self.redimensionalise(y, self.y0))
        else:
            # cumulative density
            if np.isclose(self.exponent, -1.0):
                y = np.log(xn / lower) / np.log(upper / lower)
            else:
                y = (
                    (xn**(self.exponent + 1.) - lower**(self.exponent + 1.))
                    / (upper**(self.exponent + 1.) - lower**(self.exponent + 1.))
                    )
            y[xn < lower] = 0.0
            y[xn > upper] = 1.0
            # add dimensions, and return
            return(self.redimensionalise(y, self.y0 * self.x0))
        


#############################################
### NEW ATTEMPT - WITH DIAMETER WEIGHTING ###
#############################################

class PowerFitBinned(FitBaseXBinned):
        
    def __init__(
            self,
            x,
            y = None,
            weights = None,
            weights_exponent = 0.0,
            start = None,
            bin_min = 1.0e-6,
            bin_margin = 1.0e-6,
            lower = None,
            upper = None,
            x0 = None
        ):
        # call initialiser of parent class
        super().__init__(
              x, y = y, weights = weights,
              start = start, root_method = 'newton',
              x0 = x0,
              xmin = 0.0,
              xmin_include = False
        )
        # set user limits for lower x-limit of power law distribution
        # check:
        #   * units consistent with x input
        #   * lower must be within or below smallest bin with n*weights > 0
        #   * upper must be within or above largest bin with n*weights > 0
        if lower is not None or upper is not None:
            mask = (self.weights * self.y) > 0.0
            x_mask = self.x[mask, :]
        if lower is not None:
            self.check_dimensionality(lower, self.x)
            x_max = np.min(x_mask[:, 1])
            if lower > x_max:
                raise ValueError(f'lower must be smaller than {x_max} with given measurements')
        if upper is not None:
            self.check_dimensionality(upper, self.x)
            x_min = np.max(x_mask[:, 0])
            if upper < x_min:
                raise ValueError(f'lower must be smaller than {x_min} with given measurements')
        # set weighting exponent
        self.weights_exponent = weights_exponent
        # generate fit
        self.multiplier, self.exponent, self.lower, self.upper = self.generate_fit(
            self.x, self.y, self.weights, 
            lower = lower, upper = upper,
            weights_exponent = weights_exponent,
            bin_min = bin_min, bin_margin = bin_margin
        )

    def generate_fit(
            self,
            x,
            y,
            weights = 1.0,
            lower = None,
            upper = None,
            weights_exponent = 0.0,
            bin_min = 1.0E-3,
            bin_margin = 1.0E-3,            
            nondimensional_input = False,
            nondimensional_output = False,
            ):
        # make nondimensional
        if nondimensional_input is True:
            xn = x
            yn = y
        else:
            xn = self.nondimensionalise(x, self.x0)
            yn = self.nondimensionalise(y, self.y0)
            if lower is not None:
                lower = self.nondimensionalise(lower, self.x0)
            if upper is not None:
                upper = self.nondimensionalise(upper, self.x0)
        # unknown parameter mask
        unknown = [True, lower is None, upper is None]
        # mask to select bins that contain more than 0 observations
        mask_bin = (weights * yn) > 0.0
        # get bounds for lower and upper - all (masked) bins must at least be partically within range
        lower_max = np.min(xn[mask_bin, 1] - bin_margin * np.diff(xn[mask_bin, :], axis = 1))
        upper_min = np.max(xn[mask_bin, 0] + bin_margin * np.diff(xn[mask_bin, :], axis = 1))
        bounds_all = np.array([
            (None, None), 
            (bin_min, lower_max), 
            (upper_min, None)]
            )
        bounds = bounds_all[unknown, :]
        # initial guess
        start = self._initialguess_nondimensional(xn, yn, weights, lower, upper, self.weights_exponent)
        # root solve for shape parameter
        ft = minimize(
            self._loglikelihood_nondimensional,
            start, 
            jac = True,
            args = (xn, yn, weights, lower, upper, weights_exponent, -1.0, True),
            bounds = bounds
            )
        # vector with all parameters (exponent with weighting effect, lower and upper)
        exponent_weighted, lower, upper = self._get_all_parameters(ft.x, upper)
        # calculate exponent, removing weighting effect
        exponent = exponent_weighted - weights_exponent
        # calculate (non-dimensional) multiplier
        multiplier = self._get_powerlaw_multiplier(exponent, lower, upper)
        # return all value
        if nondimensional_output is True:
            return(multiplier, exponent, lower, upper)
        else:
            return(
                self.redimensionalise(multiplier, self.y0),
                exponent,
                self.redimensionalise(lower, self.x0),
                self.redimensionalise(upper, self.x0)
            )

    def _initialguess_nondimensional(
            self,
            xn,
            yn,
            weights,
            lower = None,
            upper = None,
            weights_exponent = 0.0
            ):
        # midpoint of each class
        xn_mid = 0.5 * np.sum(xn, axis = 1)
        # minimum and maximum, for classes with finite values
        mask = (weights * yn) > 0.0
        xn_min = np.min(xn_mid[mask])
        xn_max = np.max(xn_mid[mask])
        # initial guess for exponent
        exponent = 0.0
        # all parameters
        par_all = np.array([exponent, xn_min, xn_max])
        # mask for unknown fitting parameters
        unknown = [True, lower is None, upper is None]
        # only return guesses for non-fixed parameters
        return(par_all[unknown])

    def _loglikelihood_nondimensional(
            self,
            par,
            xn,
            yn,
            weights,
            lower = None,
            upper = None,
            weights_exponent = 0.0,
            scale = 1.0,
            jac = True            
            ):
        # unknown parameter mask
        unknown = [True, lower is None, upper is None]
        # get all parameters
        exponent, lower, upper = self._get_all_parameters(par, lower, upper)
        # round class boundaries - x1 and x2 (min and max class may partially stick out of the distribution)
        xn1 = np.maximum(xn[:, 0], lower)
        xn2 = np.minimum(xn[:, 1], upper)
        # calculate probablity power law multiplier
        p = self._integrate_powerlaw_bounds(exponent, lower, upper, xn1, xn2, deriv = 0)     
        logp = np.log(p)
        # calculate rescaling of weights - based on powerlaw
        wnew = self._rescale_weights(
            exponent, lower, upper, xn1, xn2, yn,
            weights_exponent = weights_exponent,
            weights = weights,
            deriv = 0
            )
        # loglikelihood
        logL = np.sum(wnew * logp)
        # return (negative logL, as we use a solver that finds a MINIMUM)
        if (jac is False):
            return(scale * logL)
        else:
            # change in class boundaries, as power law limits change
            dxn1_dlower = (lower > xn[:, 0])
            dxn2_dupper = (upper < xn[:, 1])
            # derivatives of probability
            dp_dexponent, dp_dlower, dp_dupper, dp_dxn1, dp_dxn2 = self._integrate_powerlaw_bounds(
                exponent, lower, upper, xn1, xn2, deriv = 1).transpose()
            dlogp_dexponent = (dp_dexponent) / p
            dlogp_dlower = (dp_dlower + dp_dxn1 * dxn1_dlower) / p
            dlogp_dupper = (dp_dupper + dp_dxn2 * dxn2_dupper) / p
            # derivatives of newly scaled weights
            dwnew_dexponent, dwnew_dlower, dwnew_dupper, dwnew_dxn1, dwnew_dxn2 = self._rescale_weights(
                exponent, lower, upper, xn1, xn2, yn,
                weights_exponent = weights_exponent,
                weights = weights,
                deriv = 1
                ).transpose()        
            dwnew_dlower = dwnew_dlower + dwnew_dxn1 * dxn1_dlower
            dwnew_dupper = dwnew_dupper + dwnew_dxn2 * dxn2_dupper
            # derivatives of loglikelihood
            dlogL_dexponent = np.sum(dwnew_dexponent * logp + wnew * dlogp_dexponent)
            dlogL_dlower = np.sum(dwnew_dlower * logp + wnew * dlogp_dlower)
            dlogL_dupper = np.sum(dwnew_dupper * logp + wnew * dlogp_dupper)
            # derivatives with respect to unknown fitting parameters
            dlogL_dpar = np.array([dlogL_dexponent, dlogL_dlower, dlogL_dupper])[unknown]
            # return
            return(
                scale * logL,
                scale * dlogL_dpar
                )

    def _get_all_parameters(
            self,
            par,
            lower = None,
            upper = None
            ):
        unknown = [True, lower is None, upper is None]
        par_all = np.zeros(3)
        par_all[unknown] = par
        return(par_all)
    
    def _get_fitting_parameters(
            self,
            par_all,
            lower = None,
            upper = None
            ):
        unknown = [True, lower is None, upper is None]
        return(par_all[unknown])
    
    def _get_powerlaw_multiplier(
            self,
            exponent: float, 
            lower: np.ndarray, 
            upper: np.ndarray,
            deriv = 0
            ):
        # calculate integral, with assumed multiplier = 1.0
        if np.isclose(exponent, -1.0):
            total = np.log(upper / lower)
        else:
            total = (
                1.0 / (exponent + 1.)
                * (upper ** (exponent + 1.) - lower ** (exponent + 1.))
            )
        # return
        if deriv == 0:
            return(1. / total)
        
    def _integrate_powerlaw_bounds(
            self,
            exponent: float,
            lower: float,
            upper: float,
            xn1: np.ndarray,
            xn2: np.ndarray,
            deriv: int = 0
            ):
        if np.isclose(exponent, -1.):
            b = np.log(xn2 / xn1) 
            t = np.log(upper / lower)
        else:
            b = xn2**(exponent + 1.) - xn1**(exponent + 1.)
            t = upper**(exponent + 1.) - lower**(exponent + 1.)
        out = b / t
        if deriv == 0:
            return(out)
        elif deriv == 1:
            if np.isclose(exponent, -1.):
                db_dexponent = np.log(xn2 / xn1)
                db_dxn1 = -1. / xn1
                db_dxn2 = 1. / xn2
                dt_dexponent = np.log(upper / lower)
                dt_dlower = -1. / lower
                dt_dupper = 1. / upper
            else:
                db_dexponent = (
                    np.log(xn2) * xn2**(exponent + 1.)
                    - np.log(xn1) * xn1**(exponent + 1.)
                    )
                db_dxn1 = -(exponent + 1.) * xn1**exponent
                db_dxn2 = (exponent + 1.) * xn2**exponent
                dt_dexponent = (
                    np.log(upper) * upper**(exponent + 1.)
                    - np.log(lower) * lower**(exponent + 1.)
                    )
                dt_dlower = -(exponent + 1.) * lower**exponent
                dt_dupper = (exponent + 1.) * upper**exponent
            dout_db = 1. / t
            dout_dt = -b / t**2
            dout_dpar = np.column_stack((
                dout_db * db_dexponent + dout_dt * dt_dexponent,
                dout_dt * dt_dlower,
                dout_dt * dt_dupper,
                dout_db * db_dxn1,
                dout_db * db_dxn2
                ))
            return(dout_dpar)

    def _rescale_weights(
            self,
            exponent,
            lower,
            upper,
            xn1,
            xn2, 
            yn,
            weights_exponent = 2.0,
            weights = 1.0,
            deriv = 0
            ):
        # total old (input) weight, per bin
        w0 = weights * yn
        # total probability in each bin
        p = self._integrate_powerlaw_bounds(exponent, lower, upper, xn1, xn2)
        # average relative importance in each bin - using probability of occurance of x
        A = self._integrate_powerlaw_bounds(exponent + weights_exponent, lower, upper, xn1, xn2)
        B = A / p
        # new weights (ensure sum of new weights equal to old sum of weights)
        sumw0 = np.sum(w0)
        sumw0B = np.sum(w0 * B)
        w1 = sumw0 * (w0 * B) / sumw0B
        # return
        if deriv == 0:
            return(w1)
        elif deriv == 1:
            # differentiate
            dp_dexponent, dp_dlower, dp_dupper, dp_dxn1, dp_dxn2 = self._integrate_powerlaw_bounds(
                exponent, lower, upper, xn1, xn2, 
                deriv = 1).transpose()
            dA_dexponent, dA_dlower, dA_dupper, dA_dxn1, dA_dxn2 = self._integrate_powerlaw_bounds(
                exponent + weights_exponent, lower, upper, xn1, xn2, 
                deriv = 1).transpose()
            dB_dA = 1. / p
            dB_dp = -A / p**2
            dw1_dexponent = (
                sumw0 * w0 / sumw0B * (dB_dA * dA_dexponent + dB_dp * dp_dexponent)
                - sumw0 * (w0 * B) / sumw0B**2 * (
                    np.sum(w0 * dB_dA * dA_dexponent)
                    + np.sum(w0 * dB_dp * dp_dexponent)
                    )
                )
            dw1_dlower = (
                sumw0 * w0 / sumw0B * (dB_dA * dA_dlower + dB_dp * dp_dlower)
                - sumw0 * (w0 * B) / sumw0B**2 * (
                    np.sum(w0 * dB_dA * dA_dlower)
                    + np.sum(w0 * dB_dp * dp_dlower)
                    )
                )
            dw1_dupper = (
                sumw0 * w0 / sumw0B * (dB_dA * dA_dupper + dB_dp * dp_dupper)
                - sumw0 * (w0 * B) / sumw0B**2 * (
                    np.sum(w0 * dB_dA * dA_dupper)
                    + np.sum(w0 * dB_dp * dp_dupper)
                    )
                )
            dw1_dxn1 = (
                sumw0 * w0 / sumw0B * (dB_dA * dA_dxn1 + dB_dp * dp_dxn1)
                - sumw0 * (w0 * B) / sumw0B**2 * (
                    np.sum(w0 * dB_dA * dA_dxn1)
                    + np.sum(w0 * dB_dp * dp_dxn1)
                    )
                )
            dw1_dxn2 = (
                sumw0 * w0 / sumw0B * (dB_dA * dA_dxn2 + dB_dp * dp_dxn2)
                - sumw0 * (w0 * B) / sumw0B**2 * (
                    np.sum(w0 * dB_dA * dA_dxn2)
                    + np.sum(w0 * dB_dp * dp_dxn2)
                    )
                )
            dw1_dpar = np.column_stack((dw1_dexponent, dw1_dlower, dw1_dupper, dw1_dxn1, dw1_dxn2))
            # return
            return(dw1_dpar)
    
