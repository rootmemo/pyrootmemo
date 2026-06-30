import numpy as np
from pint import Quantity
from pyrootmemo import Parameter
from scipy.optimize import root, root_scalar, bracket
from scipy.spatial import ConvexHull
from scipy.special import loggamma, gamma, digamma, polygamma, loggamma, erf, erfinv, gammainc, gammaincinv
from .utils import create_quantity, create_reference_value, check_array_values, create_weights, nondimensionalise, redimensionalise
from .base_regression import _BaseRegression
from .linear_regression import LinearRegression
from .gumbel_distribution import GumbelDistribution
from .weibull_distribution import WeibullDistribution


class PowerRegression(_BaseRegression):
    """Powerlaw fitting class

    Fit a power-law function to a set of (x, y) data. The fit is defined as:

        y_fit = multiplier * (x / x0)**exponent

    where the power law multiplier and exponent are to be fitted. 'x0' is the 
    reference value, and must be specified to correctly deal with any units 
    (in the case of dimensional data). Usually `x0` will be set to one unit
    value of `x`. `multiplier' is the powerlaw multiplier, i.e. the value of 
    `y_fit` at `x = x0`, and `exponent` is the dimensionless powerlaw exponent.

    This class is able to deal with x and/or y data that has dimensions, e.g.
    millimetres or megapascals. Fitted values will be automatically assigned
    the correct units.

    Fitting is performed by maximising the weighted loglikelihood, i.e. by
    maximising log(L) = sum w * log(p), where 'w' is the weight and 'p' the 
    probability density associated with each observation. Different assumptions
    for the distribution (or density) of fit residuals can be made, see
    Meijer (2025) for more details (https://doi.org/10.1007/s11104-024-07007-9)

    Currently implemented fitting models are:

    * `gamma`: assumes the ratio y/y_fit is gamma distributed, with an average
        of one, and a fitted shape parameter `shape`.
    * `gumbel`: assumes the ratio y/y_fit is gumbel distributed, with an 
        average of one, and a fitted scale parameter `scale0` at `x = x0`.
    * `logistic`: assumes the ratio y/y_fit is logistic distributed, with an 
        average of one, and a fitted scale parameter `scale0` at `x = x0`.
    * `lognormal`: assumes the ratio y/y_fit is lognormal distributed, with an 
        average of one, and a log-standard deviation `sdlog`
    * `lognormal_uncorrected`: assumes the ratio y/y_fit is lognormal 
        distributed, where the average of log(y/y_fit) is equal to one, and a 
        log-standard deviation `sdlog`. Despite common in the literature, the 
        corresponding powerlaw is the geometric rather than arithmetic mean, 
        and thus underpredicts the average of y at each value of x. See Meijer
        (2025, https://doi.org/10.1007/s11104-024-07007-9) for more details
    * `normal` or `normal_stress`: assumes y - y_fit is normally (Gaussian) 
        distributed, with a standard deviation `sd_multiplier` describing 
        the residuals.
    * `normal_force`: assumes y - y_fit is normally (Gaussian) 
        distributed, with a standard deviation describing the residuals that 
        scales in magnitude with `x` according to: `sd = sd_multiplier * 
        (x/x0)**(-2)`
    * `normal_scaled`: assumes y - y_fit is normally (Gaussian) 
        distributed, with a standard deviation describing the residuals that
        scales with `x` according to: `sd = sd_multiplier * 
        (x/x0)**exponent`. In other words, the ratio y/y_fit is fitted 
        with a mean of one and a standard deviation `sd_multiplier/multiplier`.
    * `normal_freesd`: assumes y - y_fit is normally (Gaussian) 
        distributed, with a standard deviation describing the residuals that 
        scales in magnitude with an independently fitted power law for the 
        standard deviation: `sd = sd_multiplier * (x/x0)**sd_multiplier`
    * `uniform`: assumed the ratio y/y_fit is uniformly distributed. The width
        of the distribution scales with y_fit, so that the the width at any 
        value of `x` is defined as `width * (x/x0)**exponent`.
    * `weibull`: assumes the ratio y/y_fit is weibull distributed, with an average
        of one, and a fitted shape parameter `shape`. 

    The most common fitting model used in the literature is `model = normal`, 
    which corresponds to (weighted) non-linear least-square regression. 
    However, Meijer (2025, https://doi.org/10.1007/s11104-024-07007-9) showed
    that for root diameter--root tensile strength data, the `gamma` model 
    described the variation of root tensile strength data better. 
    
    Thus, care must be taken to select the best model, for example based on the 
    Kolmogorov-Smirnov distance describing the difference between observations
    and fitted data. These can easily be calculated using the `calc_ks()` 
    method that is implemented in this PowerlawFit class. 

    Attributes
    ----------
    x : np.ndarray | Quantity
        One-dimensional array with x-data
    y : np.ndarray | Quantity
        One-dimensional array with y-data
    weights : np.ndarray
        Array with (dimensionless) weighting for each (x, y) observation.
    x0 : float | int | Quantity
        Reference value for x-values
    y0 : float | int | Quantity
        Reference value for y-values
    model : str
        Fitting model used to generate the fit
    multiplier : float | Quantity
        Best-fitting powerlaw multiplier for average y-values as function of x
    exponent : float
        Best-fitting powerlaw exponent for average y-values as function of x
    zero_variance : bool
        Boolean indicating whether the fit is 'perfect', i.e all residuals are 
        zero (`zero_variance = True`) or not (`zero_variance = False`). 
    shape : float
        Best-fitting shape parameter for `'gamma'` and `'weibull'` models
    scale0 : float | Quantity
        Best-fitting scale parameter for `'gumbel'` and `'logistic'` models.
        Defined at x = x0.
    sdlog : float
        Best-fitting log-standard deviation for the various `'lognormal'` 
        models.
    sd_multiplier: float | Quantity
        Best-fitting standard deviation powerlaw multiplier, for the various 
        `'normal'` models.
    sd_exponent : float
        Assumed or best-fitting standard deviation powerlaw exponent, for the 
        various `'normal'` models.
    width : float | Quantity
        Best-fitting width parameter for `'uniform'` model. Defined at x = x0.
    
    Methods
    -------
    __init__(x, y, weights, model, x0, y0, algorithm, ...)
        Constructor
    calc_confidence_interval(x, level, n, method)
        Estimate the confidence interval of the powerlaw fit at a given
        confidence level
    calc_covariance_matrix(method, n)
        Estimate the covariance matrix of all fitting parameters
    calc_density(x, y, cumulative)
        Calculate the probability density, or the cumulative probability 
        density, for each (x, y) observation
    calc_ks()
        Calculate the Kolmogorov-Smirnov distance of the best-fitting powerlaw
        fit
    calc_loglikelihood(deriv, ...)
        Calculate the loglikelihood, or its first or second derivative with
        respect to the fitting parameters
    calc_prediction_interval(x, level, n)
        Estimate the prediction interval of the powerlaw fit at a given
        confidence level
    calc_quantile(x, quantile)
        Calculate the y-value for known x-positions, given a known cumulative 
        probability density quantile
    generate_random(x)
        Generate random values of y, given known x-positions, based on the fit
    plot(...):
        Plot the data, best fit, confidence intervals and/or prediction 
        intervals
    predict(x):
        Predict the fitted value of y_fit for each given x, using the 
        fitted power law

    """

    def __init__(
            self,
            x: np.ndarray | Quantity | Parameter,
            y: np.ndarray | Quantity | Parameter,
            model: str = 'normal_stress',
            weights: int | float | np.ndarray | None = None,
            x0: float | int | Quantity | Parameter | None = None,
            y0: float | int | Quantity | Parameter | None = None,
            algorithm: str = 'convex_hull',
            sd_exponent: float | int | None = None            
            ):
        """Initiate powerlaw fitting object

        Initialisation sets input data and will create the best fit.

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter
            array with x-data. Can either be defined as a (dimensionless)
            numpy array, a pint.Quantity object (array with units) or a 
            Parameter tuple (value array, unit string).
        y : np.ndarray | Quantity | Parameter
            array with y-data. Can either be defined as a (dimensionless)
            numpy array, a pint.Quantity object (array with units) or a 
            Parameter tuple (value array, unit string).
        model : str, optional
            fitting model, i.e. the assumed model for the variance in y-data, 
            by default 'normal'. See class description for more details
        weights : int | float | np.ndarray | None
            weighting for each (x, y) observation, by default None. If None,
            a weight of 1 is assumed for each observation. If defined as a 
            scalar value, this weight is assumed to each observation.
        x0 : float | int | Quantity | Parameter | None, optional
            the reference value for x, by default None
        y0 : float | int | Quantity | Parameter | None, optional
            the reference value for y, by default None
        algorithm : str, optional
            Algorithm to use for `model = 'uniform'`, by default 'convex_hull'.
            can be either `'convex_hull'` or `'root'`.
        sd_exponent : float | int | None, optional
            known value for the powerlaw exponent describing the variation in 
            data for 'normal' models, by default None. If None, it is 
            set accordingly to the model specified, i.e. `sd_exponent = 0` for
            `model = 'normal_stress'` and `sd_exponent = -2` for `model == 
            'normal_force'`.

        """        

        model_options = [
            'gamma', 'gumbel', 'logistic', 'lognormal', 'lognormal_uncorrected',
            'normal', 'normal_stress', 'normal_force', 'normal_scaled', 'normal_freesd',
            'uniform', 'weibull'
            ]
        self.model = model.lower()
        if not self.model in model_options:
            raise ValueError(f'model name not recognised. Must be one of {model_options}')

        self.x0 = create_reference_value(x, x0)
        self.x = create_quantity(x)
        check_array_values(self.x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        self.y0 = create_reference_value(y, y0)
        self.y = create_quantity(y)
        check_array_values(self.y, finite = True, xmin = 0.0 * self.y0, xmin_include = False)
        self.weights = create_weights(self.x, weights)

        if sd_exponent is not None:
            if isinstance(sd_exponent, int) | isinstance(sd_exponent, float):
                self.sd_exponent = sd_exponent
            else:
                raise ValueError('sd_exponent must be int or float (if explicitly defined) or None')

        x_nondimensional = nondimensionalise(self.x, self.x0)
        y_nondimensional = nondimensionalise(self.y, self.y0)

        mask_include = (self.weights > 0.0)
        log_xyn = np.log(np.unique(
            np.stack((x_nondimensional[mask_include], y_nondimensional[mask_include]), axis = -1),
            axis = -1
            ))
        if len(log_xyn) == 1:
            self.zero_variance = True
            self.exponent = 0.0
            self.multiplier = redimensionalise(np.exp(log_xyn[0, 1]), self.y0)
        elif len(log_xyn) == 2:
            self.zero_variance = True
            diff_log_xyn = np.diff(log_xyn, axis = 0)
            self.exponent = diff_log_xyn[0, 1] / diff_log_xyn[0, 0]
            self.multiplier = redimensionalise(
                np.exp(log_xyn[0, 1] - self.exponent * log_xyn[0, 0]),
                self.y0
                )
        else:
            diff_log_xyn = np.diff(log_xyn, axis = 0)    
            cross_product_consec_data = (   
                diff_log_xyn[:-1, 0] * diff_log_xyn[1:, 1] 
                - diff_log_xyn[1:, 0] * diff_log_xyn[:-1, 1]
                )
            if all(np.isclose(cross_product_consec_data, 0.0)) is True:
                self.zero_variance = True
                self.exponent = np.mean(diff_log_xyn[:, 1] / diff_log_xyn[:, 0])
                self.multiplier = redimensionalise(
                    np.mean(np.exp(log_xyn[:, 1] - self.exponent * log_xyn[:, 0])),
                    self.y0
                    )
            else:
                self.zero_variance = False

        if self.zero_variance is True:
            match self.model:
                case 'gamma' | 'weibull':
                    self.shape = redimensionalise(np.inf, self.y0)
                case 'gumbel' | 'logistic':
                    self.scale0 = redimensionalise(0.0, self.y0)
                case 'lognormal' | 'lognormal_corrected' | 'lognormal_uncorrected':
                    self.sdlog = 0.0
                case 'normal' | 'normal_stress' | 'normal_freesd' | 'normal_scaled':
                    self.sd_multiplier = redimensionalise(0.0, self.y0)
                    if not hasattr(self, 'sd_exponent'):
                        self.sd_exponent = 0.0
                case 'normal_force':
                    self.sd_multiplier = redimensionalise(0.0, self.y0)
                    if not hasattr(self, 'sd_exponent'):
                        self.sd_exponent = -2.0
                case 'uniform':
                    self.width = redimensionalise(0.0, self.y0)
        else:
            match self.model:
                case 'gamma':
                    fit_initial = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                    exponent_guess = fit_initial.gradient

                    def _powerlaw_fit_gamma_exponent_root(
                            exponent: int | float,
                            x: np.ndarray,
                            y: np.ndarray,
                            weights: np.ndarray,
                            return_jacobian: bool = True
                            ) -> float | tuple:
                        """Root function for powerlaw fit with gamma distributed variance

                        Root function to find the best-fitting powerlaw exponent 
                        
                        Parameters
                        ----------
                        exponent : int | float
                            power law exponent, to fit
                        x : np.ndarray
                            x-data (dimensionless)
                        y : np.ndarray
                            y-data (dimensionless)
                        weights : np.ndarray
                            weights per (x, y) observation
                        return_jacobian : bool, optional
                            if False, returns the root. If True, also returns the jacobian matrix, 
                            by default True

                        Returns
                        -------
                        float | tuple
                            root, or tuple with root and jacobian scalar
                        """
                        c1 = np.sum(weights)
                        c2 = np.sum(weights * np.log(x))
                        c4 = np.sum(weights * y * x**(-exponent))
                        c5 = np.sum(weights * y * x**(-exponent) * np.log(x))
                        root = c1 * c5 / c4 - c2
                        if return_jacobian is False:
                            return(root)
                        else:
                            c6 = np.sum(weights * y * x**(-exponent) * np.log(x)**2)
                            droot_dexponent = c1 / c4 * (c5**2 / c4 - c6)
                            return(root, droot_dexponent)

                    fit_exponent = root_scalar(
                        _powerlaw_fit_gamma_exponent_root,
                        x0 = exponent_guess,
                        fprime = True,
                        args = (x_nondimensional, y_nondimensional, self.weights, True),
                        method = 'newton'
                        )
                    self.exponent = fit_exponent.root
                    c1 = np.sum(self.weights)
                    c2 = np.sum(self.weights * np.log(x_nondimensional))
                    c3 = np.sum(self.weights * np.log(y_nondimensional))
                    c4 = np.sum(self.weights * y_nondimensional * x_nondimensional**(-self.exponent))
                    shape_guess = 0.5 * c1 / (self.exponent * c2 - c3 + c1 * np.log(c4) - c1 * np.log(c1))

                    def _powerlaw_fit_gamma_shape_root(
                            shape: int | float,
                            exponent: int | float,
                            x: np.ndarray,
                            y: np.ndarray,
                            weights: np.ndarray,
                            return_jacobian: bool = True
                            ):
                        """Root function for powerlaw fit with gamma distributed variance

                        Root function to find the best-fitting gamma shape parameter, given a
                        known value for the best powerlaw exponent
                        
                        Parameters
                        ----------
                        shape : int | float
                            gamma shape parameter, to fit
                        exponent : int | float
                            known law exponent
                        x : np.ndarray
                            x-data (dimensionless)
                        y : np.ndarray
                            y-data (dimensionless)
                        weights : np.ndarray
                            weights per (x, y) observation
                        return_jacobian : bool, optional
                            if False, returns the root. If True, also returns the jacobian matrix, 
                            by default True

                        Returns
                        -------
                        float | tuple
                            root, or tuple with root and jacobian scalar

                        """
                        c1 = np.sum(weights)
                        c2 = np.sum(weights * np.log(x))
                        c3 = np.sum(weights * np.log(y))
                        c4 = np.sum(weights * y * x**(-exponent))
                        root = (c1 * (np.log(shape) - digamma(shape) - np.log(c4) + np.log(c1)) 
                                - exponent * c2 + c3)
                        if return_jacobian is False:
                            return(root)
                        else:
                            droot_dshape = c1 * (1.0 / shape - polygamma(1, shape))
                            return(root, droot_dshape)

                    fit_shape = root_scalar(
                        _powerlaw_fit_gamma_shape_root,
                        x0 = shape_guess,
                        fprime = True,
                        args = (self.exponent, x_nondimensional, y_nondimensional, self.weights, True),
                        method = 'newton'
                        )
                    self.shape = fit_shape.root
                    c1 = np.sum(self.weights)
                    c4 = np.sum(self.weights * y_nondimensional * x_nondimensional**(-self.exponent))
                    self.multiplier = redimensionalise(c4 / c1, self.y0)
                    
                case 'gumbel':
                    fit_linear = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                    exponent = fit_linear.gradient
                    fit_gumbel = GumbelDistribution(y_nondimensional / (x_nondimensional**exponent), weights = self.weights)
                    exponent_scale0_guess = [exponent, fit_gumbel.scale]

                    def _powerlaw_fit_gumbel_root(
                            par: np.ndarray, 
                            x: np.ndarray,
                            y: np.ndarray,
                            weights: np.ndarray,
                            return_jacobian: bool = True
                            ) -> np.ndarray | tuple:
                        """Root function for powerlaw fit with gumbel distributed variance

                        Root function to find the best-fitting power law exponent and Gumbel shape
                        parameter.
                        
                        Parameters
                        ----------
                        par : np.ndarray
                            array with power law exponent and Gumbel shape parameter, to fit 
                            (dimensionless)
                        x : np.ndarray
                            x-data (dimensionless)
                        y : np.ndarray
                            y-data (dimensionless)
                        weights : np.ndarray
                            weights per (x, y) observation
                        return_jacobian : bool, optional
                            if False, returns the roots. If True, also returns the jacobian matrix, 
                            by default True

                        Returns
                        -------
                        np.ndarray | tuple
                            array with the two roots to solve, or tuple with roots and 2x2 
                            jacobian matrix

                        """
                        exponent, shape0 = par
                        c1 = np.sum(weights)
                        c2 = np.sum(weights * np.log(x))
                        c3 = np.sum(weights * y / (x**exponent))
                        c4 = np.sum(weights * y * np.log(x) / (x**exponent))
                        c5 = np.sum(weights * y * np.log(x)**2 / (x**exponent))
                        c6 = np.sum(weights * np.exp(-y / (shape0 * x**exponent)))
                        c7 = np.sum(weights * y / (x**exponent) * np.exp(-y / (shape0 * x**exponent)))
                        c8 = np.sum(weights * y * np.log(x) / (x**exponent) * np.exp(-y / (shape0 * x**exponent)))
                        dlogL_dexponent = c4 / shape0 - c1 * c8 / (shape0 * c6) - c2
                        dlogL_dshape0 = c3 / shape0**2 - c1 * c7 / (shape0**2 * c6) - c1 / shape0
                        root = np.array([dlogL_dexponent, dlogL_dshape0])
                        if return_jacobian is False:
                            return(root)
                        else:
                            c9 = np.sum(
                                weights * y * np.log(x)**2 
                                / (x**exponent) 
                                * np.exp(-y / (shape0 * x**exponent))
                                )
                            c10 = np.sum(
                                weights * y**2 / (x**(2. * exponent))
                                * np.exp(-y / (shape0 * x**exponent))
                                )
                            c11 = np.sum(
                                weights * y**2 * np.log(x) / (x**(2. * exponent))
                                * np.exp(-y / (shape0 * x**exponent))
                                )
                            c12 = np.sum(
                                weights * y**2 * np.log(x)**2 / (x**(2. * exponent))
                                * np.exp(-y / (shape0 * x**exponent))
                                )
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
                                -2.0 * c3 / shape0**3 
                                + 2.0 * c1 * c7 / (shape0**3 * c6) 
                                - c1 * c10 / (shape0**4 * c6) 
                                + c1 * c7**2 / (shape0**4 * c6**2) 
                                + c1 / shape0**2
                                )
                            droot_dpar = np.array([
                                [d2logL_dexponent2, d2logL_dexponentdshape0],
                                [d2logL_dexponentdshape0, d2logL_dshape02]
                            ])
                            return(root, droot_dpar)

                    fit = root(
                        _powerlaw_fit_gumbel_root,
                        x0 = exponent_scale0_guess,
                        jac = True,
                        args = (x_nondimensional, y_nondimensional, self.weights, True),
                        method = 'hybr'
                        )
                    self.exponent, scale0_nondimensional = fit.x
                    c1 = np.sum(self.weights)
                    c6 = np.sum(self.weights * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**self.exponent)))
                    multiplier_nondimensional = scale0_nondimensional * (np.log(c1) - np.log(c6) + np.euler_gamma)
                    self.scale0 = redimensionalise(scale0_nondimensional, self.y0)
                    self.multiplier = redimensionalise(multiplier_nondimensional, self.y0)
                
                case 'logistic':
                    fit_normal = PowerRegression(self.x, self.y, weights = self.weights, model = "normal_scaled")
                    guess = np.array([
                        nondimensionalise(fit_normal.multiplier, self.y0),
                        fit_normal.exponent,
                        nondimensionalise(fit_normal.sd_multiplier * np.sqrt(3.0) / np.pi, self.y0)
                        ])
                    fun_root = lambda p: self.calc_loglikelihood(
                        x = self.x,
                        y = self.y, 
                        weights = self.weights,
                        multiplier = redimensionalise(p[0], self.y0),
                        exponent = p[1], 
                        scale0 = redimensionalise(p[2], self.y0),
                        deriv = 1
                        )
                    fun_root_jacobian = lambda p: self.calc_loglikelihood(
                        x = self.x,
                        y = self.y, 
                        weights = self.weights,
                        multiplier = redimensionalise(p[0], self.y0), 
                        exponent = p[1], 
                        scale0 = redimensionalise(p[2], self.y0),
                        deriv = 2
                        )
                    fit = root(
                        fun_root,
                        x0 = guess,
                        jac = fun_root_jacobian,
                        method = 'hybr'
                        )
                    self.multiplier = redimensionalise(fit.x[0], self.y0)
                    self.exponent = fit.x[1]
                    self.scale0 = redimensionalise(fit.x[2], self.y0)

                case 'lognormal' | 'lognormal_corrected' | 'lognormal_uncorrected':
                    c1 = np.sum(self.weights)
                    c2 = np.sum(self.weights * np.log(x_nondimensional))
                    c3 = np.sum(self.weights * np.log(y_nondimensional))
                    c4 = np.sum(self.weights * np.log(x_nondimensional)**2)
                    c5 = np.sum(self.weights * np.log(x_nondimensional) * np.log(y_nondimensional))
                    c6 = np.sum(self.weights * np.log(y_nondimensional)**2)
                    self.exponent = (c1 * c5 - c2 * c3) / (c1 * c4 - c2**2)
                    if model == 'lognormal_uncorrected':
                        multiplier_nondimenisonal = np.exp((c3 - exponent * c2) / c1)
                        self.sdlog = np.sqrt(
                            np.log(multiplier_nondimenisonal)**2 
                            + (c6 - 2.0 * exponent * c5 + self.exponent**2 * c4 
                               + 2.0 * np.log(multiplier_nondimenisonal) * (exponent * c2 - c3)
                               ) 
                            / c1
                        )
                    else:        
                        self.sdlog = np.sqrt(
                            c6 / c1 
                            - (c1 * c5**2 - 2.0 * c2 * c3 * c5 + c3**2 * c4)
                            / (c1 * (c1 * c4 - c2**2))
                            )
                        multiplier_nondimensional = np.exp((c3 - self.exponent * c2) / c1 + self.sdlog**2 / 2.0)
                    self.multiplier = redimensionalise(multiplier_nondimensional, self.y0)

                case 'normal' | 'normal_stress' | 'normal_force' | 'normal_freesd' | 'normal_scaled':
            
                    def _powerlaw_fit_normal_freesd_root(
                            par: np.ndarray, 
                            x: np.ndarray,
                            y: np.ndarray,
                            weights: np.ndarray,
                            return_jacobian: bool = True
                            ):
                        """Root function for powerlaw fit with normal distributed variance (free sd)

                        Root function to find the best-fitting power law exponent and the standard
                        deviation powerlaw exponent.
                        
                        Parameters
                        ----------
                        par : np.ndarray
                            array with power law exponent and standard deviation powerlaw exponent, 
                            to fit (dimensionless)
                        x : np.ndarray
                            x-data (dimensionless)
                        y : np.ndarray
                            y-data (dimensionless)
                        weights : np.ndarray
                            weights per (x, y) observation
                        return_jacobian : bool, optional
                            if False, returns the two roots. If True, also returns the 2x2 jacobian
                            matrix, by default True

                        Returns
                        -------
                        np.ndarray | tuple
                            array with the two roots to solve, or tuple with roots and 2x2 
                            jacobian matrix

                        """
                        exponent, sd_exponent = par
                        c1 = np.sum(weights)
                        c2 = np.sum(weights * np.log(x))
                        c3 = np.sum(weights * x**(2.0 * exponent - 2.0 * sd_exponent))
                        c4 = np.sum(weights * x**(2.0 * exponent - 2.0 * sd_exponent) * np.log(x))
                        c6 = np.sum(weights * x**(exponent - 2.0 * sd_exponent) * y)
                        c7 = np.sum(weights * x**(exponent - 2.0 * sd_exponent) * y * np.log(x))
                        c9 = np.sum(weights * x**(-2.0 * sd_exponent)* y **2)
                        c10 = np.sum(weights * x**(-2.0 * sd_exponent) * y**2 * np.log(x))
                        zeta = 1.0 / (c3**2 * c9 - c3 * c6**2)
                        dlogL_dexponent = c1 * c6 * (c3 * c7 - c4 * c6) * zeta
                        dlogL_dsdexponent = c1 * (c3**2 * c10 - 2. * c3 * c6 * c7 + c4 * c6**2) * zeta - c2
                        root = np.array([dlogL_dexponent, dlogL_dsdexponent])
                        if return_jacobian is False:
                            return(root)
                        else:
                            c5 = np.sum(weights * x**(2.0 * exponent - 2.0 * sd_exponent) * np.log(x)**2)
                            c8 = np.sum(weights * x**(exponent - 2.0 * sd_exponent) * y * np.log(x)**2)
                            c11 = np.sum(weights * x**(-2.0 * sd_exponent) * y**2 * np.log(x)**2)
                            dzeta_dexponent = -(
                                4.0 * c3 * c4 * c9 
                                - 2.0 * c4 * c6**2 
                                - 2.0 * c3 * c6 * c7
                                ) * zeta**2
                            dzeta_dsdexponent = -(
                                -4.0 * c3 * c4 * c9 
                                - 2.0 * c3**2 * c10 
                                + 2.0 * c4 * c6**2 
                                + 4.0 * c3 * c6 * c7
                                ) * zeta**2
                            d2logL_dexponent2 = (
                                c1 
                                * (c3 * c7 - c4 * c6)
                                * (c6 * dzeta_dexponent + c7 * zeta) 
                                + c1 * c6 * (c4 * c7 + c3 * c8 - 2.0 * c5 * c6) * zeta
                                )
                            d2logL_dexponent_dsdexponent = (
                                c1 
                                * (c3 * c7 - c4 * c6)
                                * (c6 * dzeta_dsdexponent - 2.0 * c7 * zeta) 
                                + 2.0 * c1 * c6 * (c5 * c6 - c3 * c8) * zeta
                                )
                            d2logL_dsdexponent2 = (
                                c1 
                                * (c3**2 * c10 - 2.0 * c3 * c6 * c7 + c4 * c6**2)
                                * dzeta_dsdexponent 
                                + c1 * (
                                    -4.0 * c3 * c4 * c10 
                                    - 2.0 * c3**2 * c11 
                                    + 4.0 * c3 * c7**2 
                                    + 4.0 * c3 * c6 * c8 
                                    - 2.0 * c5 * c6**2
                                ) * zeta
                            )
                            droot_dpar = np.array([
                                [d2logL_dexponent2, d2logL_dexponent_dsdexponent],
                                [d2logL_dexponent_dsdexponent, d2logL_dsdexponent2]
                                ])
                            return(root, droot_dpar)

                    if self.model in ('normal', 'normal_stress', 'normal_force'):
                        if not hasattr(self, 'sd_exponent'):
                            if self.model == 'normal_force':
                                self.sd_exponent = -2.0
                            else:
                                self.sd_exponent = 0.0
                        fit_linear = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                        exponent_guess = fit_linear.gradient

                        def _powerlaw_fit_normal_fixedsd_root(
                                exponent: int | float,
                                x: np.ndarray,
                                y: np.ndarray,
                                weights: np.ndarray,
                                sd_exponent: int | float = 0.0,
                                return_jacobian: bool = True
                                ):
                            """Root function for powerlaw fit with normal distributed variance (fixed sd)

                            Root function to find the best-fitting power law exponent. The powerlaw 
                            exponent for the standard deviation is assumed to be known.
                            
                            Parameters
                            ----------
                            exponent : int | float
                                power law exponent, to fit (dimensionless)
                            x : np.ndarray
                                x-data (dimensionless)
                            y : np.ndarray
                                y-data (dimensionless)
                            weights : np.ndarray
                                weights per (x, y) observation
                            sd_exponent : int | float
                                known value of the standard deviation powerlaw exponent (dimensionless),
                                by default 0.0
                            return_jacobian : bool, optional
                                if False, returns the root. If True, also returns jacobian scalar, 
                                by default True

                            Returns
                            -------
                            float | tuple
                                root, or tuple with the root and the jacobian scalar

                            """
                            res = _powerlaw_fit_normal_freesd_root([exponent, sd_exponent], x, y, weights, return_jacobian = return_jacobian)
                            if return_jacobian is False:
                                return(res[0])
                            else:
                                return(res[0][0], res[1][0, 0])

                        fit_exponent = root_scalar(
                            _powerlaw_fit_normal_fixedsd_root,
                            x0 = exponent_guess,
                            fprime = True,
                            args = (x_nondimensional, y_nondimensional, self.weights, self.sd_exponent, True),
                            method = 'newton'
                            )
                        self.exponent = fit_exponent.root
                
                    elif self.model == 'normal_freesd':
                        fit_linear = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                        exponent_guess = fit_linear.gradient
                        sd_exponent_guesses = exponent_guess + np.linspace(-1.0, 1.0, 9)
                        fits_guesses = [
                            PowerRegression(self.x, self.y, weights = self.weights, model = 'normal', sd_exponent = d) 
                            for d in sd_exponent_guesses
                            ]
                        loglikelihood_guess = [ft.calc_loglikelihood() for ft in fits_guesses]
                        index_max = np.argmax(loglikelihood_guess)
                        exponents_guess = np.array([fits_guesses[index_max].exponent, fits_guesses[index_max].sd_exponent])
                        fit_exponents = root(
                            _powerlaw_fit_normal_freesd_root,
                            x0 = exponents_guess,
                            jac = True,
                            args = (x_nondimensional, y_nondimensional, self.weights, True),
                            method = 'hybr'
                            )
                        self.exponent, self.sd_exponent = fit_exponents.x

                    elif self.model == 'normal_scaled':
                        fit_linear = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                        exponent_guess = fit_linear.gradient

                        def _powerlaw_fit_normal_scaled_root(
                                exponent: int | float,
                                x: np.ndarray,
                                y: np.ndarray,
                                weights: np.ndarray,
                                return_jacobian: bool = True
                                ):
                            """Root function for powerlaw fit with normal distributed variance (scaled)

                            Root function to find the best-fitting power law exponent. The powerlaw 
                            exponent for the standard deviation is assumed equal to the powerlaw
                            exponent.
                            
                            Parameters
                            ----------
                            exponent : int | float
                                power law exponent, to fit (dimensionless)
                            x : np.ndarray
                                x-data (dimensionless)
                            y : np.ndarray
                                y-data (dimensionless)
                            weights : np.ndarray
                                weights per (x, y) observation
                            return_jacobian : bool, optional
                                if False, returns the root. If True, also returns jacobian scalar, 
                                by default True

                            Returns
                            -------
                            float | tuple
                                root, or tuple with the root and the jacobian scalar

                            """
                            res = _powerlaw_fit_normal_freesd_root([exponent, exponent], x, y, weights, return_jacobian = return_jacobian)
                            if return_jacobian is False:
                                return(np.sum(res))
                            else:
                                return(np.sum(res[0]), np.sum(res[1]))

                        fit_exponent = root_scalar(
                            _powerlaw_fit_normal_scaled_root,
                            x0 = exponent_guess,
                            fprime = True,
                            args = (x_nondimensional, y_nondimensional, self.weights, True),
                            method = 'newton'
                            )
                        self.exponent = fit_exponent.root
                        self.sd_exponent = fit_exponent.root
                
                    c1 = np.sum(self.weights)
                    c3 = np.sum(self.weights * x_nondimensional**(2.0 * self.exponent - 2.0 * self.sd_exponent))
                    c6 = np.sum(self.weights * x_nondimensional**(self.exponent - 2.0 * self.sd_exponent) * y_nondimensional)
                    c9 = np.sum(self.weights * x_nondimensional**(-2.0 * self.sd_exponent) * y_nondimensional**2)
                    self.multiplier = redimensionalise(c6 / c3, self.y0)
                    self.sd_multiplier = redimensionalise(np.sqrt(c9 / c1 - c6**2 / (c1 * c3)), self.y0)

                case 'uniform':
                    if algorithm == 'convex_hull':
                        log_xyn = np.log(np.unique(np.column_stack((x_nondimensional, y_nondimensional)), axis = 0))
                        hull = ConvexHull(log_xyn)
                        d_log_xyn = np.diff(log_xyn[hull.simplices], axis = 1)
                        gradients = (d_log_xyn[:, :, 1] / d_log_xyn[:, :, 0]).flatten()
                        fn = lambda b: self.calc_loglikelihood(
                            x = self.x, 
                            y = self.y, 
                            weights = self.weights,
                            exponent = b, 
                            deriv = 0
                            )
                        fts = np.array([fn(b) for b in gradients])
                        self.exponent = gradients[np.argmax(fts)]
                    elif algorithm == 'root':
                        fit_linear = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                        exponent_guess = fit_linear.gradient
                        fn0 = lambda b: -self.calc_loglikelihood(
                            x = self.x,
                            y = self.y, 
                            weights = self.weights,
                            exponent = b,
                            deriv = 0
                            )
                        br = bracket(
                            fn0,
                            xa = exponent_guess - 1.0,
                            xb = exponent_guess + 1.0
                            )
                        fn1 = lambda b: self.calc_loglikelihood(
                            x = self.x,
                            y = self.y, 
                            weights = self.weights,
                            exponent = b, 
                            deriv = 1
                        )
                        sol = root_scalar(
                            fn1,
                            bracket = (br[0], br[2]),
                            x0 = exponent_guess,
                            method = 'bisect'
                            )
                        self.exponent = sol.root
                    else:
                        raise ValueError("Algorithm for uniform powerlaw fit must be 'bisect' or 'root'")
                    lower_nondimensional = np.min(y_nondimensional / x_nondimensional**self.exponent)
                    upper_nondimensional = np.max(y_nondimensional / x_nondimensional**self.exponent)
                    self.width = redimensionalise(upper_nondimensional - lower_nondimensional, self.y0)
                    self.multiplier = redimensionalise(0.5*(lower_nondimensional + upper_nondimensional), self.y0)
                
                case 'weibull':
                    ft_linear = LinearRegression(np.log(x_nondimensional), np.log(y_nondimensional), weights = self.weights)
                    yn_scaled = y_nondimensional / (x_nondimensional**ft_linear.gradient)
                    ft_weibull = WeibullDistribution(yn_scaled, weights = self.weights)
                    exponent_shape_guess = (ft_linear.gradient, ft_weibull.shape)

                    def _powerlaw_fit_weibull_root(
                            par: np.ndarray, 
                            x: np.ndarray,
                            y: np.ndarray,
                            weights: np.ndarray,
                            return_jacobian: bool = True
                            ) -> np.ndarray | tuple:
                        """Root function for powerlaw fit with Weibull distributed variance

                        Parameters
                        ----------
                        par : np.ndarray
                            fitting parameters: power law exponent + weibull shape parameter
                        x : np.ndarray
                            x-data (dimensionless)
                        y : np.ndarray
                            y-data (dimensionless)
                        weights : np.ndarray
                            weights per (x, y) observation
                        return_jacobian : bool, optional
                            if False, returns the roots. If True, also returns the jacobian matrix, 
                            by default True

                        Returns
                        -------
                        np.ndarray | tuple
                            array with the two roots to solve, or tuple with roots and 2x2 
                            jacobian matrix

                        """
                        exponent, shape = par
                        c1 = np.sum(weights)
                        c2 = np.sum(weights * np.log(x))
                        c3 = np.sum(weights * np.log(y))
                        c4 = np.sum(weights * x**(-exponent * shape) * y**shape)
                        c5 = np.sum(weights * x**(-exponent * shape) * y**shape * np.log(x))
                        c6 = np.sum(weights * x**(-exponent * shape) * y**shape * np.log(y))
                        dlogL_dexponent = shape * c1 * c5 / c4 - shape * c2
                        dlogL_dshape = (1.0 / shape + (exponent * c5 - c6) / c4) * c1 - exponent * c2 + c3
                        root = np.array([dlogL_dexponent, dlogL_dshape])
                        if return_jacobian is False:
                            return(root)
                        else:
                            c7 = np.sum(weights * x**(-exponent * shape) * y**shape * np.log(x)**2)
                            c8 = np.sum(weights * x**(-exponent * shape) * y**shape * np.log(x) * np.log(y))
                            c9 = np.sum(weights * x**(-exponent * shape) * y**shape * np.log(y)**2)
                            d2logL_dexponent2 = c1 * shape**2 / c4 * (c5**2 / c4 - c7)
                            d2logL_dexponentdshape = (
                                c1 / c4 
                                * (c5 - exponent * shape * c7 + shape * c8 + shape * c5 / c4 * (exponent * c5 - c6)) 
                                - c2
                                )
                            d2logL_dshape2 = (
                                (-1.0 / shape**2 - (exponent**2 * c7 - 2.0 * exponent * c8 + c9) / c4 
                                    + (exponent * c5 - c6)**2 / c4**2) * c1
                                    )
                            droot_dpar = np.array([
                                [d2logL_dexponent2, d2logL_dexponentdshape], 
                                [d2logL_dexponentdshape, d2logL_dshape2]
                                ])
                            return(root, droot_dpar)

                    fit = root(  
                        _powerlaw_fit_weibull_root,
                        x0 = exponent_shape_guess,
                        args = (x_nondimensional, y_nondimensional, self.weights, True),
                        jac = True,
                        method = 'hybr'
                        )
                    self.exponent, self.shape = fit.x
                    c1 = np.sum(self.weights)
                    c4 = np.sum(self.weights * x_nondimensional**(-self.exponent * self.shape) * y_nondimensional**self.shape)
                    self.multiplier = redimensionalise(gamma(1.0 + 1.0 / self.shape) * (c4 / c1)**(1.0 / self.shape), self.y0)


    def calc_confidence_interval(
            self, 
            x: np.ndarray | Quantity | Parameter | None = None, 
            level: float = 0.95, 
            n: int = 101,
            method: str = 'fisher'
            ) -> tuple:
        """
        Predict the confidence interval for the fit

        Generate confidence intervals for power law fits, based on the
        first delta method, and a calculated variance-covariance matrix. This
        matrix is calculated using the class method `calc_covariance_matrix().

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter | None, optional
            x-values to predict the confidence interval at, by default None.
            If None, a range using `n` equally-spaced point is selected on the 
            range `min(x)` to `max(x)`, where `x` is the data used to create
            the fit.
        level : float, optional
            confidence level, by default 0.95
        n : int, optional
            number of equally-spaced x-points on the x-interval, by default 101
        method : str
            method to use to calculate covariance matrix. See method
            `calc_covariance_matrix()` for more details.

        Returns
        -------
        tuple
            tuple of three np.ndarrays, each with size n: x values, and 
            y-values of the lower and upper boundaries of the confidence 
            interval, respectively

        """
        if x is None:
            x = np.linspace(self.x.min(), self.x.max(), n)
        else:
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        y_fit = self.predict(x)
        if self.zero_variance is True:
            y_lower = y_fit
            y_upper = y_fit
        else:
            x_nondimensional = nondimensionalise(x, self.x0)
            y_fit_nondimensional = nondimensionalise(y_fit, self.y0)
            multiplier_nondimensional = nondimensionalise(self.multiplier, self.y0)
            dyn_dmultiplier = y_fit_nondimensional**self.exponent
            dyn_dexponent = multiplier_nondimensional * np.log(x_nondimensional) * x_nondimensional**self.exponent
            covariance_matrix = self.calc_covariance_matrix(method = method)
            var = (
                dyn_dmultiplier**2 * covariance_matrix[0, 0]
                + 2.0 * dyn_dmultiplier * dyn_dexponent * covariance_matrix[0, 1]
                + dyn_dexponent**2 * covariance_matrix[1, 1]
                )
            quantile = 0.5 + 0.5 * level
            sd_mult = np.sqrt(2.0) * erfinv(2.0 * quantile - 1.0)
            y_lower = redimensionalise(y_fit_nondimensional - sd_mult * np.sqrt(var), self.y0)
            y_upper = redimensionalise(y_fit_nondimensional + sd_mult * np.sqrt(var), self.y0)
        return(x, y_lower, y_upper)


    def calc_covariance_matrix(
            self, 
            method: str = 'fisher', 
            n: int = 1000
            ) -> np.ndarray:
        """
        Calculate the covariance matrix for all fitting parameters

        Return the covariance matrix for the fitting parameter. If the 
        method is set to 'fisher', determines the matrix based on the 
        Fisher information (negative second order partial differential of
        the loglikelihood with respect to the fitting parameters). 

        If method is 'bootstrap', use a bootstrap method instead. `n` fits
        are created through random selection of input data, and the covariance
        calculated from all fitting results.

        The covariance matrix is calculated in dimensionless terms, by using 
        the reference values for the x and y data (`x0` and `y0') in order
        to nondimensionalise any parameters.

        Parameters
        ----------
        method : str, optional
            Method used to estimate the covariance matrix, by default 'fisher'.
            Can be set to 'fisher' or 'bootstrap'.
        n : int, optional
            Number of bootstrap samples for method = 'bootstrap', by default 
            1000

        Returns
        -------
        np.ndarray
            the covariance matrix, a m*m square matrix, where 'm' is the 
            number of independent fitting parameters (`m=3` for most fitting 
            methods). Any values are nondimensionalised using 'x0' and 'y0'.

        """
        if self.zero_variance is True:
            return None
        if (method == 'bootstrap') or (self.model == 'uniform'):
            rng = np.random.default_rng()
            indices = rng.choice(
                np.arange(len(self.x), dtype = int),
                (n, len(self.x)),
                replace = True
            )
            fits = [PowerRegression(self.x[i], self.y[i], weights = self.weights[i], 
                                model = self.model, x0 = self.x0, y0 = self.y0)
                    for i in indices]
            match self.model:
                case 'gamma' | 'weibull':
                    vals = np.array([np.array([
                        nondimensionalise(f.multiplier, self.y0), 
                        f.exponent,
                        f.shape])
                        for f in fits])
                case 'gumbel' | 'logistic':
                    vals = np.array([np.array([
                        nondimensionalise(f.multiplier, self.y0), 
                        f.exponent,
                        nondimensionalise(f.scale0, self.y0)])
                        for f in fits])
                case 'lognormal' | 'lognormal_corrected' | 'lognormal_uncorrected':
                    vals = np.array([np.array([
                        nondimensionalise(f.multiplier, self.y0), 
                        f.exponent,
                        f.sdlog])
                        for f in fits])
                case 'normal' | 'normal_stress' | 'normal_force' | 'normal_scaled':
                    vals = np.array([np.array([
                        nondimensionalise(f.multiplier, self.y0), 
                        f.exponent,
                        nondimensionalise(f.sd_multiplier, self.y0)])
                        for f in fits])
                case 'normal_freesd':              
                    vals = np.array([np.array([
                        nondimensionalise(f.multiplier, self.y0), 
                        f.exponent,
                        nondimensionalise(f.sd_multiplier, self.y0),
                        f.sd_exponent])
                        for f in fits])
                case 'uniform':
                    vals = np.array([np.array([
                        nondimensionalise(f.multiplier, self.y0), 
                        f.exponent,
                        nondimensionalise(f.width, self.y0)])
                        for f in fits])               
            return(np.cov(vals.transpose()))
        elif method == 'fisher':
            fisher = -self.calc_loglikelihood(deriv = 2)
            if np.isscalar(fisher):
                return(1.0 / fisher)
            else:
                return(np.linalg.inv(fisher))
        else:
            raise ValueError("method not recognised. Must either be 'fisher' or 'bootstrap'")       


    def calc_density(
            self,
            x: np.ndarray | Quantity | Parameter | None = None,
            y: np.ndarray | Quantity | Parameter | None = None,
            cumulative: bool = False
            ) -> Quantity | np.ndarray:
        """Calculate (cumulative) probability densities.

        Calculate the probability density, or cumulative probability density
        for observations (x, y), given the best powerlaw fit

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter | None, optional
            x-data, by default None. If None, the x-data that was used to 
            generate the fit is used.
        y : np.ndarray | Quantity | Parameter | None, optional
            y-data, by default None. If None, the y-data that was used to 
            generate the fit is used.
        cumulative : bool, optional
            If `False`, return the probability density. If `True`, return
            cumulative probability densities instead. By default False

        Returns
        -------
        np.ndarray | Quantity
            An array of (cumulative) probability densities.

        """
        if x is None:
            x = self.x
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        if y is None:
            y = self.y
        else:
            y = create_quantity(y)
            check_array_values(y, finite = True, xmin = 0.0 * self.y0, xmin_include = False)
        x_nondimensional = nondimensionalise(x, self.x0)
        y_nondimensional = nondimensionalise(y, self.y0)
        y_fit_nondimensional = nondimensionalise(self.predict(x), self.y0)
        if self.zero_variance is True:
            if cumulative is False:
                return(np.inf * (y_nondimensional == y_fit_nondimensional))
            else:
                return((y_nondimensional >= y_fit_nondimensional).astype(float))
        else:
            match self.model:
                case 'gamma':
                    scale = self.multiplier / self.shape * (x / self.x0)**self.exponent
                    scale_nondimensional = nondimensionalise(scale, self.y0)
                    if cumulative is False:
                        log_density = (
                            (self.shape - 1.0) * np.log(y_nondimensional)
                            - loggamma(self.shape)
                            - self.shape * np.log(scale_nondimensional)
                            - y_nondimensional / scale_nondimensional
                            )
                        return(np.exp(log_density))
                    else:
                        return(gammainc(self.shape, y_nondimensional / scale_nondimensional))
                
                case 'gumbel':
                    location = (self.multiplier - np.euler_gamma * self.scale0) * (x / self.x0)**self.exponent
                    scale = self.scale0 * (x / self.x0)**self.exponent
                    location_nondimensional = nondimensionalise(location, self.y0)
                    scale_nondimensional = nondimensionalise(scale, self.y0)
                    z = (y_nondimensional - location_nondimensional) / scale_nondimensional
                    if cumulative is False:
                        return(np.exp(-z - np.exp(-z)) / scale_nondimensional)
                    else:
                        return(np.exp(-np.exp(-z)))

                case 'logistic':
                    scale = self.scale0 * (x / self.x0)**self.exponent
                    scale_nondimensional = nondimensionalise(scale, self.y0)
                    location_nondimensional = y_fit_nondimensional
                    if cumulative is False:
                        return(1.0 / (
                            4.0 * scale_nondimensional 
                            * np.cosh(
                                (y_nondimensional - location_nondimensional) 
                                / (2.0 * scale_nondimensional)
                                )**2
                            ))
                    else:
                        return(0.5 + 0.5 * np.tanh(
                            (y_nondimensional - location_nondimensional) 
                            / (2.0 * scale_nondimensional)
                            ))

                case 'lognormal' | 'lognormal_corrected' | 'lognormal_uncorrected':
                    multiplier_nondimensional = nondimensionalise(self.multiplier, self.y0)
                    if self.model == 'lognormal_uncorrected':
                        meanlog = (np.log(multiplier_nondimensional) 
                                   + self.exponent * np.log(x_nondimensional))
                    else:
                        meanlog = (np.log(multiplier_nondimensional) 
                                   + self.exponent * np.log(x_nondimensional)
                                   - 0.5 * self.sdlog**2)
                    if cumulative is False:
                        return(
                            np.exp(-(np.log(y_nondimensional) - meanlog)**2 / (2.0 * self.sdlog**2)) 
                            / (y_nondimensional * self.sdlog * np.sqrt(2.0 * np.pi))
                            )
                    else:
                        return(
                            0.5 + 0.5 * erf(
                                (np.log(y_nondimensional) - meanlog)
                                / (self.sdlog * np.sqrt(2.0))
                            )
                        )
                    
                case 'normal' | 'normal_stress' | 'normal_force' | 'normal_freesd' | 'normal_scaled':
                    mean_nondimensional = y_fit_nondimensional
                    sd_nondimensional = nondimensionalise(
                        self.sd_multiplier * x_nondimensional**self.sd_exponent,
                        self.y0
                    )
                    if cumulative is False:
                        return(np.exp(-0.5 * ((y_nondimensional - mean_nondimensional) / sd_nondimensional)**2)
                            / (sd_nondimensional * np.sqrt(2.0 * np.pi)))
                    else:
                        return(0.5 + 0.5*erf(
                            (y_nondimensional - mean_nondimensional) 
                            / (sd_nondimensional * np.sqrt(2.0))
                            ))
     
                case 'uniform':
                    multiplier_nondimensional = nondimensionalise(self.multiplier, self.y0)
                    width_nondimensional = nondimensionalise(self.width, self.y0)
                    lower = (multiplier_nondimensional - 0.5 * width_nondimensional) * x_nondimensional**self.exponent
                    upper = (multiplier_nondimensional + 0.5 * width_nondimensional) * x_nondimensional**self.exponent
                    if cumulative is False:
                        p = 1.0 / (upper - lower)
                        p[y_nondimensional < lower] = 0.0
                        p[y_nondimensional > upper] = 0.0
                    else:
                        p = (y_nondimensional - lower) / (upper - lower)
                        p[y_nondimensional < lower] = 0.0
                        p[y_nondimensional > upper] = 1.0
                    return(p)

                case 'weibull':
                    scale = self.multiplier * x_nondimensional**self.exponent / gamma(1.0 + 1.0 / self.shape)
                    scale_nondimensional = nondimensionalise(scale, self.y0)
                    if cumulative is False:
                        return(
                            self.shape 
                            / scale_nondimensional
                            * (y_nondimensional / scale_nondimensional)**(self.shape - 1.0)
                            * np.exp(-(y_nondimensional / scale_nondimensional)**self.shape)
                            )
                    else:
                        return(1.0 - np.exp(-(y_nondimensional / scale_nondimensional)**self.shape))


    def calc_ks(self) -> float:
        """
        Calculate the Kolmogorov-Smirnov distance for the fit.

        This is defined as the largest difference between the fitted cumulative
        density distribution curve and the cumulative curve for the input data.

        Returns
        -------
        float
            Kolmogorov-Smirnov distance

        """
        if self.zero_variance is True:
            return(0.0)
        else:
            cumul_density = self.calc_density(cumulative = True)
            order = np.argsort(cumul_density)
            weights_sorted = self.weights[order]
            x_sorted = self.x[order]
            y_sorted = self.y[order]
            cumul_data = np.cumsum(weights_sorted) / np.sum(weights_sorted)
            cumul_fit = self.calc_density(x = x_sorted, y = y_sorted, cumulative = True)
            distance_top = np.max(np.abs(cumul_fit - cumul_data))
            distance_bot = np.max(np.abs(cumul_fit - np.append(0.0, cumul_data[:-1])))
            return(max(distance_top, distance_bot))


    def calc_loglikelihood(
            self,
            deriv = 0,
            x: np.ndarray | Quantity | None = None,
            y: np.ndarray | Quantity | None = None,
            weights: np.ndarray | None = None,
            multiplier: float | Quantity | None = None,
            exponent: float | None = None,
            shape: float | None = None,
            scale0: float | Quantity | None = None,
            sdlog: float | None = None,
            sd_multiplier: float | Quantity | None = None,
            sd_exponent: float | None = None
            ) -> float | np.ndarray:
        """Calculate the (derivatives of the) loglikelihood

        Calculate the (weighted) loglikelihood of the powerlaw fit, or its 
        first or second derivative with respect to all fitting parameters.
        Parameters
        ----------
        deriv : int, optional
            Order of differentiation of the loglikelihood with respect to all
            fitting parameters, by default 0
        x : np.ndarray | Quantity | None, optional
            x-data, by default None. If `None`, the data used to generate
            the fit is used
        y : np.ndarray | Quantity | None, optional
            y-data, by default None. If `None`, the data used to generate
            the fit is used
        weights : np.ndarray | None, optional
            weights for each (x, y) observation, by default None, If `None`, 
            the data used to generate the fit is used
        multiplier : float | Quantity | None, optional
            powerlaw multiplier, by default None. If `None`, the fitted value
            is used
        exponent : float | None, optional
            powerlaw exponent, by default None. If `None`, the fitted value
            is used
        shape : float | None, optional
            shape parameter for `gamma` and weib`ull models, by default 
            None. If `None`, the fitted value is used
        scale0 : float | Quantity | None, optional
            scale parameter (at x = x0) for `gumbel` and `logistic` models, 
            by default None. If `None`, the fitted value is used
        sdlog : float | None, optional
            log-standard deviation in `lognormal` models, by default None. If
            `None`, the fitted value is used
        sd_multiplier : float | Quantity | None, optional
            standard deviation powerlaw multiplier, for `normal` models, by 
            default None. If `None`, the fitted value is used.
        sd_exponent : float | None, optional
            standard deviation powerlaw exponent, for `normal` models, by 
            default None. If `None`, the fitted value is used.

        Returns
        -------
        float | np.ndarray
            Loglikelihood (scalar) in case `deriv = 0`. Else a numpy array 
            with partial differentation of the weighted loglikelihood with
            respect to each of the fitting parameters (`deriv = 1` or 
            `deriv = 2`)
            
        """
        x = self.x if x is None else create_quantity(x)
        x_nondimensional = nondimensionalise(x, self.x0)
        y = self.y if y is None else create_quantity(y)
        y_nondimensional = nondimensionalise(y, self.y0)
        weights = self.weights if weights is None else weights
        if self.model != 'uniform':
            multiplier = self.multiplier if multiplier is None else multiplier
            multiplier_nondimensional = nondimensionalise(multiplier, self.y0)
        exponent = self.exponent if exponent is None else exponent

        if self.zero_variance is True:
            if deriv == 0:
                return(np.inf)
            elif deriv == 1:
                return(np.full(3, -np.inf))
            elif deriv == 2:
                return(np.full((3, 3), -np.inf))
        
        match self.model:
            case 'gamma':
                shape = self.shape if shape is None else shape
                c1 = np.sum(weights)
                c2 = np.sum(weights * np.log(x_nondimensional))
                c3 = np.sum(weights * np.log(y_nondimensional))
                c4 = np.sum(weights * x_nondimensional**(-exponent) * y_nondimensional)
                c5 = np.sum(weights * x_nondimensional**(-exponent) * y_nondimensional * np.log(x_nondimensional))
                c6 = np.sum(weights * x_nondimensional**(-exponent) * y_nondimensional * np.log(x_nondimensional)**2)
                logg = loggamma(shape)
                p = digamma(shape)
                q = polygamma(1, shape)
                if deriv == 0:
                    return(c1 * (shape * np.log(shape) - logg - shape * np.log(multiplier_nondimensional)) 
                        - exponent * shape * c2 
                        + (shape - 1.0) * c3 
                        - shape * c4 / multiplier_nondimensional
                        )
                elif deriv == 1:
                    dlogL_dmultiplier = shape / multiplier_nondimensional * (c4 / multiplier_nondimensional - c1)
                    dlogL_dexponent = shape * (c5 / multiplier_nondimensional - c2)
                    dlogL_dshape = (
                        c1 * (1.0 + np.log(shape) - p - np.log(multiplier_nondimensional)) 
                        - exponent * c2 
                        + c3 
                        - c4 / multiplier_nondimensional
                    )
                    return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dshape]))
                elif deriv == 2:
                    d2logL_dmultiplier2 = shape / multiplier_nondimensional**2 * (c1 - 2.0 * c4 / multiplier_nondimensional)
                    d2logL_dmultiplier_dexponent = -shape * c5 / multiplier_nondimensional**2
                    d2logL_dmultiplier_dshape = 1.0 / multiplier_nondimensional * (c4 / multiplier_nondimensional - c1)
                    d2logL_dexponent2 = -shape * c6 / multiplier_nondimensional
                    d2logL_dexponent_dshape = c5 / multiplier_nondimensional - c2
                    d2logL_dshape2 = c1 * (1.0 / shape - q)
                    return(np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplier_dexponent, d2logL_dmultiplier_dshape],
                        [d2logL_dmultiplier_dexponent, d2logL_dexponent2, d2logL_dexponent_dshape],
                        [d2logL_dmultiplier_dshape, d2logL_dexponent_dshape, d2logL_dshape2]
                        ]))

            case 'gumbel':
                scale0 = self.scale0 if scale0 is None else scale0
                scale0_nondimensional = nondimensionalise(scale0, self.y0)
                c1 = np.sum(weights)
                c2 = np.sum(weights * np.log(x_nondimensional))
                c3 = np.sum(weights * y_nondimensional / (x_nondimensional**exponent))
                c4 = np.sum(weights * y_nondimensional * np.log(x_nondimensional) / (x_nondimensional**exponent))
                c5 = np.sum(weights * y_nondimensional * np.log(x_nondimensional)**2 / (x_nondimensional**exponent))
                c6 = np.sum(weights * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                c7 = np.sum(weights * y_nondimensional / (x_nondimensional**exponent) * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                c8 = np.sum(weights * y_nondimensional * np.log(x_nondimensional) / (x_nondimensional**exponent) * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                c9 = np.sum(weights * y_nondimensional * np.log(x_nondimensional)**2 / (x_nondimensional**exponent) * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                c10 = np.sum(weights * y_nondimensional**2 / (x_nondimensional**(2.0 * exponent)) * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                c11 = np.sum(weights * y_nondimensional**2 * np.log(x_nondimensional) / (x_nondimensional**(2.*exponent)) * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                c12 = np.sum(weights * y_nondimensional**2 * np.log(x_nondimensional)**2 / (x_nondimensional**(2.*exponent)) * np.exp(-y_nondimensional / (scale0_nondimensional * x_nondimensional**exponent)))
                if deriv == 0:
                    return(
                        c1 * (multiplier_nondimensional / scale0_nondimensional - np.log(scale0_nondimensional) - np.euler_gamma) 
                        - exponent * c2 - c3 / scale0_nondimensional 
                        - np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma) * c6)
                elif deriv == 1:
                    dlogL_dmultiplier = (
                        c1 / scale0_nondimensional 
                        - c6 / scale0_nondimensional * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                    )
                    dlogL_dexponent = (
                        c4 / scale0_nondimensional 
                        - c2 
                        - c8 / scale0_nondimensional * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                    )
                    dlogL_dtheta0 = (
                        (c3 - c1 * multiplier_nondimensional) / scale0_nondimensional**2 
                        - c1 / scale0_nondimensional 
                        + (c6 * multiplier_nondimensional - c7) / scale0_nondimensional**2 * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                        )
                    return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dtheta0]))
                elif deriv == 2:
                    d2logL_dmultiplier2 = -c6 / scale0_nondimensional**2 * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                    d2logL_dmultiplier_dexponent = -c8 / scale0_nondimensional**2 * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                    d2logL_dmultiplier_dscale0 = (
                        1.0 / scale0_nondimensional**2
                        * (c6 * (1.0 + multiplier_nondimensional / scale0_nondimensional) - c7 / scale0_nondimensional)
                        * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma) 
                        - c1 / scale0_nondimensional**2
                        )
                    d2logL_dexponent2 = (
                        1.0 / scale0_nondimensional * (c9 - c12 / scale0_nondimensional) 
                        * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma) 
                        - c5 / scale0_nondimensional
                        )
                    d2logL_dexponent_dscale0 = (
                        -c4 / scale0_nondimensional**2 
                        + 1.0 / scale0_nondimensional**2 * (c8 * (1.0 + multiplier_nondimensional / scale0_nondimensional) - c11 / scale0_nondimensional)
                        * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                        )
                    d2logL_dscale02 = (
                        c1 / scale0_nondimensional**2 
                        - 2.0 * (c3 - c1 * multiplier_nondimensional) / scale0_nondimensional**3 
                        - 2.0 * (c6 * multiplier_nondimensional - c7) / scale0_nondimensional**3 * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma) 
                        + (2.0 * c7 * multiplier_nondimensional - c10 - c6 * multiplier_nondimensional**2)
                        / scale0_nondimensional**4 
                        * np.exp(multiplier_nondimensional / scale0_nondimensional - np.euler_gamma)
                        )
                    return(np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplier_dexponent, d2logL_dmultiplier_dscale0],
                        [d2logL_dmultiplier_dexponent, d2logL_dexponent2, d2logL_dexponent_dscale0],
                        [d2logL_dmultiplier_dscale0, d2logL_dexponent_dscale0, d2logL_dscale02]
                        ])) 
            
            case 'logistic':
                scale0 = self.scale0 if scale0 is None else scale0
                scale0_nondimensional = nondimensionalise(scale0, self.y0)
                eta = 1.0 / np.cosh((y_nondimensional / x_nondimensional**exponent - multiplier_nondimensional) / (2.0 * scale0_nondimensional))
                zeta = np.tanh((y_nondimensional / x_nondimensional**exponent - multiplier_nondimensional) / (2.0 * scale0_nondimensional))
                if (deriv == 0):
                    logp = -np.log(4.0 * scale0_nondimensional) - exponent * np.log(x) + 2.0 * np.log(eta)
                    return(np.sum(weights * logp))
                elif deriv == 1:
                    dlogpL_dmultiplier = zeta / scale0_nondimensional
                    dlogp_dexponent = y_nondimensional * np.log(x_nondimensional) * zeta / (scale0_nondimensional * x_nondimensional**exponent) - np.log(x_nondimensional)
                    dlogp_dscale0 = (y_nondimensional / x_nondimensional**exponent - multiplier_nondimensional) * zeta / scale0_nondimensional**2 - 1.0 / scale0_nondimensional
                    dlogL_dmultiplier = np.sum(weights * dlogpL_dmultiplier)
                    dlogL_dexponent = np.sum(weights * dlogp_dexponent)
                    dlogL_dscale0 = np.sum(weights * dlogp_dscale0)
                    return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dscale0]))
                elif deriv == 2:
                    dzeta_dmultiplier = -eta**2 / (2.0 * scale0_nondimensional)
                    dzeta_dexponent = -y_nondimensional * np.log(x_nondimensional) * eta**2 / (2.0 * scale0_nondimensional * x_nondimensional**exponent)
                    dzeta_dscale0 = -(y_nondimensional / x_nondimensional**exponent - multiplier_nondimensional) * eta**2 / (2.0 * scale0_nondimensional**2)
                    d2p_dmultiplier2 = dzeta_dmultiplier / scale0_nondimensional
                    d2p_dmultiplierdexponent = dzeta_dexponent / scale0_nondimensional
                    d2p_dmultiplierdscale0 = (dzeta_dscale0 - zeta / scale0_nondimensional) / scale0_nondimensional
                    d2p_dexponent2 = y_nondimensional * np.log(x_nondimensional) * (dzeta_dexponent - np.log(x_nondimensional) * zeta) / (scale0_nondimensional * x_nondimensional**exponent)
                    d2p_dexponentdscale0 = y_nondimensional * np.log(x_nondimensional) * (dzeta_dscale0 - zeta / scale0_nondimensional) / (scale0_nondimensional * x_nondimensional**exponent)
                    d2p_dscale02 = (y_nondimensional / x_nondimensional**exponent - multiplier_nondimensional) * (dzeta_dscale0 - 2.0 * zeta / scale0_nondimensional) / scale0_nondimensional**2 + 1.0 / scale0_nondimensional**2
                    d2logL_dmultiplier2 = np.sum(weights * d2p_dmultiplier2)
                    d2logL_dmultiplierdexponent = np.sum(weights * d2p_dmultiplierdexponent)
                    d2logL_dmultiplierdscale0 = np.sum(weights * d2p_dmultiplierdscale0)
                    d2logL_dexponent2 = np.sum(weights * d2p_dexponent2)
                    d2logL_dexponentdscale0 = np.sum(weights * d2p_dexponentdscale0)
                    d2logL_dscale02 = np.sum(weights * d2p_dscale02)
                    return(np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplierdexponent, d2logL_dmultiplierdscale0],
                        [d2logL_dmultiplierdexponent, d2logL_dexponent2, d2logL_dexponentdscale0],
                        [d2logL_dmultiplierdscale0, d2logL_dexponentdscale0, d2logL_dscale02]
                        ]))

            case 'lognormal' | 'lognormal_corrected':
                sdlog = self.sdlog if sdlog is None else sdlog
                c1 = np.sum(weights)
                c2 = np.sum(weights * np.log(x_nondimensional))
                c3 = np.sum(weights * np.log(y_nondimensional))
                c4 = np.sum(weights * np.log(x_nondimensional)**2)
                c5 = np.sum(weights * np.log(x_nondimensional) * np.log(y_nondimensional))
                c6 = np.sum(weights * np.log(y_nondimensional)**2)
                if deriv == 0:
                    return(
                        (np.log(multiplier_nondimensional) / sdlog**2 - 1.0)
                        * (c3 - exponent * c2 - c1 * np.log(multiplier_nondimensional) / 2.0) 
                        - c1 * (np.log(sdlog) + np.log(2.0 * np.pi) / 2.0 + sdlog**2 / 8.0) 
                        - (exponent * c2 + c3) / 2.0 
                        - (exponent**2 * c4 - 2.0 * exponent * c5 + c6) / (2.0 * sdlog**2)
                        )
                elif deriv == 1:
                    dlogL_dmultiplier = (
                        (c3 - exponent * c2 - c1 * np.log(multiplier_nondimensional))
                        / (multiplier_nondimensional * sdlog**2) 
                        + c1 / (2.0 * multiplier_nondimensional)
                    )
                    dlogL_dexponent = (
                        0.5 * c2 
                        - (exponent * c4 - c5 + c2 * np.log(multiplier_nondimensional))
                        / (sdlog**2)
                    )
                    dlogL_dsdlog = (
                        (c1 * np.log(multiplier_nondimensional)**2 
                        - 2. * np.log(multiplier_nondimensional) * (c3 - exponent * c2) 
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
                        -((c1 * sdlog**2) / 2.0 
                        + c1 + c3 - exponent * c2 - c1 * np.log(multiplier_nondimensional))
                        / (sdlog**2 * multiplier_nondimensional**2)
                    )
                    d2logL_dmultiplier_dexponent = -c2 / (sdlog**2 * multiplier_nondimensional)
                    d2logL_dmultiplier_dsdlog = (
                        (2.0 * (exponent * c2 - c3 + c1 * np.log(multiplier_nondimensional)))
                        / (sdlog**3 * multiplier_nondimensional)
                    )
                    d2logL_dexponent2 = -c4 / sdlog**2
                    d2logL_dexponent_dsdlog = (
                        (2.0 * (exponent * c4 - c5 + c2 * np.log(multiplier_nondimensional)))
                        / sdlog**3
                    )
                    d2logL_dsdlog2 = (
                        -3.0 * (
                            c1 * np.log(multiplier_nondimensional)**2 
                            - 2.0 * np.log(multiplier_nondimensional) * (c3 - exponent * c2) 
                            + exponent**2 * c4 
                            - 2.0 * exponent * c5 
                            + c6
                        ) 
                        / sdlog**4 
                        + c1 / sdlog**2 
                        - c1 / 4.0
                    )
                    return(np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplier_dexponent, d2logL_dmultiplier_dsdlog],
                        [d2logL_dmultiplier_dexponent, d2logL_dexponent2, d2logL_dexponent_dsdlog],
                        [d2logL_dmultiplier_dsdlog, d2logL_dexponent_dsdlog, d2logL_dsdlog2]
                        ]))
            
            case 'lognormal_uncorrected':
                sdlog = self.sdlog if sdlog is None else sdlog
                c1 = np.sum(weights)
                c2 = np.sum(weights * np.log(x_nondimensional))
                c3 = np.sum(weights * np.log(y_nondimensional))
                c4 = np.sum(weights * np.log(x_nondimensional)**2)
                c5 = np.sum(weights * np.log(x_nondimensional) * np.log(y_nondimensional))
                c6 = np.sum(weights * np.log(y_nondimensional)**2)
                if deriv == 0:
                    return(
                        -c1 * (np.log(sdlog) + 0.5 * np.log(2.0 * np.pi) + np.log(multiplier_nondimensional)**2 / (2.0 * sdlog**2)) 
                        - c3 
                        - 1.0 / (2.0 * sdlog**2) * (
                            c6 - 2.0 * exponent * c5 
                            + exponent**2 * c4 
                            + 2.0 * np.log(multiplier_nondimensional) * (exponent * c2 - c3)
                        )
                    )
                elif deriv == 1:
                    dlogL_dmultiplier = (
                        -1.0 / (sdlog**2 * multiplier_nondimensional)
                        * (c1 * np.log(multiplier_nondimensional) + (exponent * c2 - c3))
                    )
                    dlogL_dexponent = -(exponent * c4 - c5 + c2 * np.log(multiplier_nondimensional)) / sdlog**2
                    dlogL_dsdlog = (
                        -c1 * (1.0 / sdlog - np.log(multiplier_nondimensional)**2 / sdlog**3) 
                        + 1.0 / sdlog**3 
                        * (c6 - 2.0 * exponent * c5 
                        + exponent**2 * c4 
                        + 2.0 * np.log(multiplier_nondimensional) * (exponent * c2 - c3)
                        )
                    )
                    return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dsdlog]))
                elif deriv == 2:
                    d2logL_dmultiplier2 = -(
                        (c1 * (1.0 + np.log(multiplier_nondimensional)) - c3 + exponent * c2)
                        / (multiplier_nondimensional**2 * sdlog**2)
                    )
                    d2logL_dmultiplier_dexponent = -c2 / (sdlog**2 * multiplier_nondimensional)
                    d2logL_dmultiplier_dsdlog = (
                        -2.0 / (sdlog**3 * multiplier_nondimensional) 
                        * (c3 - exponent * c2 - c1 * np.log(multiplier_nondimensional))
                    )
                    d2logL_dexponent2 = -c4 / sdlog**2
                    d2logL_dexponent_dsdlog = 2.0 / sdlog**3 * (exponent * c4 - c5 + c2 * np.log(multiplier_nondimensional))
                    d2logL_dsdlog2 = (
                        -c1 * (-1.0 / sdlog**2 + 3.0 * np.log(multiplier_nondimensional)**2 / sdlog**4) 
                        - 3.0 / sdlog**4 * (
                            c6 - 2.0 * exponent * c5 
                            + exponent**2 * c4 
                            + 2.0 * np.log(multiplier_nondimensional) * (exponent * c2 - c3)
                        )
                    )
                    return(np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplier_dexponent, d2logL_dmultiplier_dsdlog],
                        [d2logL_dmultiplier_dexponent, d2logL_dexponent2, d2logL_dexponent_dsdlog],
                        [d2logL_dmultiplier_dsdlog, d2logL_dexponent_dsdlog, d2logL_dsdlog2]
                        ]))

            case 'normal' | 'normal_stress' | 'normal_force' | 'normal_freesd' | 'normal_scaled':
                if self.model == 'normal_freesd':
                    output_indices = np.arange(3)
                else:
                    output_indices = np.arange(3)
                sd_multiplier = self.sd_multiplier if sd_multiplier is None else sd_multiplier
                sd_exponent = self.sd_exponent if sd_exponent is None else sd_exponent
                sd_multiplier_nondimensional = nondimensionalise(sd_multiplier, self.y0)
                c1 = np.sum(weights)
                c2 = np.sum(weights * np.log(x_nondimensional))
                c3 = np.sum(weights * x_nondimensional**(2.0 * exponent - 2.0 * sd_exponent))
                c4 = np.sum(weights * x_nondimensional**(2.0 * exponent - 2.0 * sd_exponent) * np.log(x_nondimensional))
                c5 = np.sum(weights * x_nondimensional**(2.0 * exponent - 2.0 * sd_exponent) * np.log(x_nondimensional)**2)
                c6 = np.sum(weights * x_nondimensional**(exponent - 2.0 * sd_exponent) * y_nondimensional)
                c7 = np.sum(weights * x_nondimensional**(exponent - 2.0 * sd_exponent) * y_nondimensional * np.log(x_nondimensional))
                c8 = np.sum(weights * x_nondimensional**(exponent - 2.0 * sd_exponent) * y_nondimensional * np.log(x_nondimensional)**2)
                c9 = np.sum(weights * x_nondimensional**(-2.0 * sd_exponent) * y_nondimensional**2)
                c10 = np.sum(weights * x_nondimensional**(-2.0 * sd_exponent) * y_nondimensional**2 * np.log(x_nondimensional))
                c11 = np.sum(weights * x_nondimensional**(-2.0 * sd_exponent) * y_nondimensional**2 * np.log(x_nondimensional)**2)
                if deriv == 0:
                    return(
                        -c1 
                        * (np.log(sd_multiplier_nondimensional) + 0.5 * np.log(2 * np.pi)) 
                        - sd_exponent * c2 
                        - (c9 - 2.0 * multiplier_nondimensional * c6 + c3 * multiplier_nondimensional**2)
                        / (2.0 * sd_multiplier_nondimensional**2)
                        )
                elif deriv == 1:
                    dlogL_dmultiplier = (c6 - multiplier_nondimensional * c3) / sd_multiplier_nondimensional**2
                    dlogL_dexponent = multiplier_nondimensional * (c7 - c4 * multiplier_nondimensional) / sd_multiplier_nondimensional**2
                    dlogL_dsdmultiplier = (
                        -c1 / sd_multiplier_nondimensional 
                        + (c9 - 2.0 * c6 * multiplier_nondimensional + c3 * multiplier_nondimensional**2)
                        / sd_multiplier_nondimensional**3
                    )
                    dlogL_dsdexponent = (
                        c10 
                        - 2.0 * c7 * multiplier_nondimensional 
                        + c4 * multiplier_nondimensional**2
                        ) / sd_multiplier_nondimensional**2 - c2
                    dlogL_dpar = np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dsdmultiplier, dlogL_dsdexponent])
                    return(dlogL_dpar[output_indices])
                elif deriv == 2:
                    d2logL_dmultiplier2 = -c3 / sd_multiplier_nondimensional**2
                    d2logL_dmultiplier_dexponent = (c7 - 2.0 * c4 * multiplier_nondimensional) / sd_multiplier_nondimensional**2
                    d2logL_dmultiplier_dsdmultiplier = 2.0 * (c3 * multiplier_nondimensional - c6) / sd_multiplier_nondimensional**3
                    d2logL_dmultiplier_dsdexponent = 2.0 * (c4 * multiplier_nondimensional - c7) / sd_multiplier_nondimensional**2
                    d2logL_dexponent2 = multiplier_nondimensional * (c8 - 2.0 * c5 * multiplier_nondimensional) / sd_multiplier_nondimensional**2
                    d2logL_dexponent_dsdmultiplier = 2.0 * multiplier_nondimensional * (c4 * multiplier_nondimensional - c7) / sd_multiplier_nondimensional**3
                    d2logL_dexponent_dsdexponent = 2.0 * multiplier_nondimensional * (c5 * multiplier_nondimensional - c8) / sd_multiplier_nondimensional**2
                    d2logL_dsdmultiplier2 = (
                        c1 / sd_multiplier_nondimensional**2 
                        - 3.0 * (c9 - 2.0 * c6 * multiplier_nondimensional + c3 * multiplier_nondimensional**2)
                        / sd_multiplier_nondimensional**4
                    )
                    d2logL_dsdmultiplier_dsdexponent = (
                        2.0 * (-c10 + 2.0 * c7 * multiplier_nondimensional - c4 * multiplier_nondimensional**2)
                        / sd_multiplier_nondimensional**3
                    )
                    d2logL_dsdexponent2 = (
                        (-2.0 * c11 + 4.0 * c8 * multiplier_nondimensional - 2.0 * c5 * multiplier_nondimensional**2)
                        / sd_multiplier_nondimensional**2
                    )
                    d2logL_dpar2 = np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplier_dexponent, d2logL_dmultiplier_dsdmultiplier, d2logL_dmultiplier_dsdexponent],
                        [d2logL_dmultiplier_dexponent, d2logL_dexponent2, d2logL_dexponent_dsdmultiplier, d2logL_dexponent_dsdexponent],
                        [d2logL_dmultiplier_dsdmultiplier, d2logL_dexponent_dsdmultiplier, d2logL_dsdmultiplier2, d2logL_dsdmultiplier_dsdexponent],
                        [d2logL_dmultiplier_dsdexponent, d2logL_dexponent_dsdexponent, d2logL_dsdmultiplier_dsdexponent, d2logL_dsdexponent2]
                        ])
                    return(d2logL_dpar2[output_indices, :][:, output_indices])

            case 'uniform':
                if deriv == 0:
                    lower = np.min(y_nondimensional / x_nondimensional**exponent)
                    upper = np.max(y_nondimensional / x_nondimensional**exponent)
                    return(
                        -np.log(upper - lower) * np.sum(weights) 
                        - exponent * np.sum(weights * np.log(x_nondimensional))
                        )
                elif deriv == 1:
                    il = np.argmin(y_nondimensional / x_nondimensional**exponent)
                    iu = np.argmax(y_nondimensional / x_nondimensional**exponent)
                    lower = y_nondimensional[il] / x_nondimensional[il]**exponent
                    upper = y_nondimensional[iu] / x_nondimensional[iu]**exponent
                    dlower_dexponent = -y_nondimensional[il] * np.log(x_nondimensional[il]) / x_nondimensional[il]**exponent
                    dupper_dexponent = -y_nondimensional[iu] * np.log(x_nondimensional[iu]) / x_nondimensional[iu]**exponent
                    return(
                        (dlower_dexponent - dupper_dexponent)
                        / (upper - lower) * np.sum(weights) 
                        - np.sum(weights * np.log(x_nondimensional))
                        )
                else:
                    raise ValueError('deriv only defined for deriv = 0 or 1')

            case 'weibull':
                shape = self.shape if shape is None else shape
                c1 = np.sum(weights)
                c2 = np.sum(weights * np.log(x_nondimensional))
                c3 = np.sum(weights * np.log(y_nondimensional))
                c4 = np.sum(weights * x_nondimensional**(-exponent * shape) * y_nondimensional**shape)
                c5 = np.sum(weights * x_nondimensional**(-exponent * shape) * y_nondimensional**shape * np.log(x_nondimensional))
                c6 = np.sum(weights * x_nondimensional**(-exponent * shape) * y_nondimensional**shape * np.log(y_nondimensional))
                c7 = np.sum(weights * x_nondimensional**(-exponent * shape) * y_nondimensional**shape * np.log(x_nondimensional)**2)
                c8 = np.sum(weights * x_nondimensional**(-exponent * shape) * y_nondimensional**shape * np.log(x_nondimensional) * np.log(y_nondimensional))
                c9 = np.sum(weights * x_nondimensional**(-exponent * shape) * y_nondimensional**shape * np.log(y_nondimensional)**2)
                g = gamma(1.0 + 1.0 / shape)
                p = digamma(1.0 + 1.0 / shape)
                q = polygamma(1, 1.0 + 1.0 / shape)
                if deriv == 0:
                    return(
                        (np.log(shape) - shape * np.log(multiplier_nondimensional) + shape * np.log(g)) * c1 
                        - exponent * shape * c2 
                        + (shape - 1.0) * c3 
                        - (g / multiplier_nondimensional)**shape * c4
                        )
                elif deriv == 1:
                    dlogL_dmultiplier = -c1 * shape / multiplier_nondimensional + c4 * shape * g**shape * multiplier_nondimensional**(-shape - 1.0)
                    dlogL_dexponent = -shape * c2 + shape * (g / multiplier_nondimensional)**shape * c5
                    dlogL_dshape = (c1 * (1./shape + np.log(g / multiplier_nondimensional) - p / shape) - exponent * c2 + c3 
                        - (g / multiplier_nondimensional)**shape * (c4 * (np.log(g / multiplier_nondimensional) - p / shape) - exponent * c5 + c6))
                    return(np.array([dlogL_dmultiplier, dlogL_dexponent, dlogL_dshape]))
                elif deriv == 2:
                    d2logL_dmultiplier2 = (
                        c1 * shape / multiplier_nondimensional**2 
                        - c4 * shape * (shape + 1.0) * g**shape * multiplier_nondimensional**(-shape - 2.0)
                        )
                    d2logL_dmultiplier_dexponent = -c5 * shape**2 * g**shape * multiplier_nondimensional**(-shape - 1.0)
                    d2logL_dmultiplier_dshape = (
                        -c1 / multiplier_nondimensional 
                        + g**shape * multiplier_nondimensional**(-shape - 1.0) 
                        * (c4 * (1.0 + shape * np.log(g / multiplier_nondimensional) - p) 
                        + shape * (c6 - exponent * c5)
                        )
                        )
                    d2logL_dexponent2 = -shape**2 * (g / multiplier_nondimensional)**shape * c7
                    d2logL_dexponent_dshape = (
                        -c2 
                        + (g / multiplier_nondimensional)**shape 
                        * (c5 * (1.0 + shape * np.log(g / multiplier_nondimensional) - p) 
                        + shape*(c8 - exponent * c7)
                        )
                        )
                    d2logL_dshape2 = (
                        c1 / shape**2 * (q / shape - 1.0) 
                        - (g / multiplier_nondimensional)**shape * (
                            2.0 * (np.log(g / multiplier_nondimensional) - p / shape) * (c6 - exponent * c5) 
                            + (np.log(g / multiplier_nondimensional) - p / shape)**2 * c4 
                            + (c4 * q / shape**3 + exponent**2 * c7 - 2.0 * exponent * c8 + c9)
                            )
                        )
                    return(np.array([
                        [d2logL_dmultiplier2, d2logL_dmultiplier_dexponent, d2logL_dmultiplier_dshape], 
                        [d2logL_dmultiplier_dexponent, d2logL_dexponent2, d2logL_dexponent_dshape], 
                        [d2logL_dmultiplier_dshape, d2logL_dexponent_dshape, d2logL_dshape2]
                        ]))


    def calc_prediction_interval(
            self, 
            x: np.ndarray | Quantity | Parameter | None = None, 
            level: float = 0.95, 
            n: int = 251
            ) -> tuple:
        """Generate prediction intervals for power law fits

        Calculate the upper and lower limits of the prediction interval for a
        power law fit.

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter | None, optional
            x-values to predict the confidence interval at, by default None.
            If None, a range using `n` equally-spaced point is selected on the 
            range `min(x)` to `max(x)`, where `x` is the data used to create
            the fit.
        level : float, optional
            confidence level, by default 0.95
        n : int, optional
            number of equally-spaced x-points on the x-interval, by default 101

        Returns
        -------
        tuple
            tuple of three np.ndarrays, each with size n: x values, and 
            y-values of the lower and upper boundaries of the prediction 
            interval, respectively

        """
        if x is None:
            x = np.linspace(self.x.min(), self.x.max(), n)
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        if self.zero_variance is True:
            y_fit = self.predict(x)
            return(x, y_fit, y_fit)
        else:
            return(
                x,
                self.calc_quantile(x, 0.5 - 0.5 * level),
                self.calc_quantile(x, 0.5 + 0.5 * level)
                )
        

    def calc_quantile(
            self,
            x: np.ndarray | Quantity | Parameter | None,
            quantile: int | float
            ) -> np.ndarray | Quantity:
        """Calculate y-values at known quantiles

        Calculate y-values for given x-positions, when knowing the cumulative
        density quantile for each point.

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter | None, optional
            x-values at which to predict y-values at chosen quantile, by 
            default None, in which case the x-data used to generate the fit
            is used
        quantile : int | float
            Cumulative density quantile

        Returns
        -------
        np.ndarray | Quantity
            y-values

        """
        if x is None:
            x = self.x
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        if self.zero_variance is True:
            return(self.predict(x))
        else:
            x_nondimensional = nondimensionalise(x, self.x0)
            match self.model:
                case 'gamma':
                    scale = self.multiplier / self.shape * x_nondimensional**self.exponent 
                    return(scale * gammaincinv(self.shape, quantile))
                case 'gumbel':
                    location = (self.multiplier - np.euler_gamma * self.scale0) * x_nondimensional**self.exponent
                    scale = self.scale0 * x_nondimensional**self.exponent
                    return(location - scale * np.log(-np.log(quantile)))
                case 'logistic':
                    location = self.predict(x)
                    scale = self.scale0 * x_nondimensional**self.exponent
                    return(location + 2.0 * scale * np.arctanh(2.0 * quantile - 1.0))
                case 'lognormal' | 'lognormal_corrected' | 'lognormal_uncorrected':
                    multiplier_nondimensional = nondimensionalise(self.multiplier, self.y0)
                    if self.model == 'lognormal_uncorrected':
                        meanlog = (np.log(multiplier_nondimensional) 
                                   + self.exponent * np.log(x_nondimensional))
                    else:
                        meanlog = (np.log(multiplier_nondimensional) 
                                   + self.exponent * np.log(x_nondimensional)
                                   - 0.5 * self.sdlog**2)
                    return(redimensionalise(
                        np.exp(meanlog + np.sqrt(2.0) * self.sdlog * erfinv(2.0 * quantile - 1.0)),
                        self.y0
                        ))                 
                case 'normal' | 'normal_stress' | 'normal_force' | 'normal_freesd' | 'normal_scaled':
                    mean = self.predict(x)
                    sd = self.sd_multiplier * x_nondimensional**self.sd_exponent
                    return(mean + sd * (np.sqrt(2.0) * erfinv(2.0 * quantile - 1.0)))
                case 'uniform':
                    return((self.multiplier + (quantile - 0.5) * self.width) * x_nondimensional**self.exponent)
                case 'weibull':
                    return(
                        self.predict(x)
                        * (-np.log(1.0 - quantile))**(1.0 / self.shape)
                        / gamma(1.0 + 1.0 / self.shape)
                        )


    def generate_random(
            self,
            x: np.ndarray | Quantity | Parameter
            ) -> np.ndarray | Quantity:
        """Generate random y-values at known x-values

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter
            Array with known x-values

        Returns
        -------
        np.ndarray | Quantity
            Array with random y-values

        """
        if x is None:
            x = self.x
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        return(self.calc_quantile(x, np.random.rand(*x.shape)))


    def predict(
            self,
            x: int | float | np.ndarray | Quantity | None = None
            ) -> float | np.ndarray | Quantity:
        """Predict average y-value at known x-values

        Calculate the average y-value at known x-positions using the fitted
        power law.

        Parameters
        ----------
        x : int | float | np.ndarray | Quantity | None, optional
            x-values, by default None. If None, the x-data used to generate
            the fit is used instead

        Returns
        -------
        float | np.ndarray | Quantity
            average y-values at each of the specified x-values

        """
        if x is None:
            x = self.x
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        x_nondimensional = nondimensionalise(x, self.x0)
        return(self.multiplier * x_nondimensional**self.exponent)
