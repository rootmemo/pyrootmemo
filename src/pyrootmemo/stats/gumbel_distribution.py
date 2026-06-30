import numpy as np
from pint import Quantity
from pyrootmemo import Parameter
from scipy.optimize import root_scalar
from .utils import create_quantity, create_reference_value, check_array_values, create_weights, nondimensionalise, redimensionalise
from .base_distribution import _BaseDistribution


class GumbelDistribution(_BaseDistribution):   
    """Gumbel distribution fitting class

    Fit a Gumbel distribution using (weighted) loglikehood 

    Probability density p(x):

        p(x) = exp(-(z + exp(-z))) / scale
        z = (x - location) / scale

    (Weighted) loglikelihood will be maximised, i.e.

        min sum_i weight * log p(x_i))
    
    This class is able to deal with data that has dimensions, e.g. millimetres 
    or megapascals. Fitted values will be automatically assigned the correct 
    units.

    Attributes
    ----------
    x : np.ndarray | Quantity
        One-dimensional array with x-data
    weights : np.ndarray
        Array with (dimensionless) weighting for each value in x
    x0 : float | int | Quantity
        Reference value for x-values
    location : float | Quantity
        best-fitting Gumbel distribution location parameter
    scale : float | Quantity
        best-fitting Gumbel distribution scale parameterr

    Methods
    -------
    __init__(x, weights, x0, shape_guess, root_method)
        Constructor
    generate_random(n)
        draw `n` new values from the fitted distribution
    calc_density(x, cumulative)
        calculate probability density or cumulative probablity density
    calc_loglikelihood(..., deriv)
        calculate (partial derivatives of) fit loglikehood
    plot(...)
        plot histogram of data, and the fitted distribution

    """

    def __init__(
            self,
            x: np.ndarray | Quantity | Parameter,
            weights: np.ndarray | Quantity = None,
            x0: int | float | Quantity | Parameter | None = None,
            scale_guess: float | int | None = None,
            root_method: str = 'newton'
            ):  
        
        self.x0 = create_reference_value(x, x0)
        self.x = create_quantity(x)
        check_array_values(self.x, finite = True)
        self.weights = create_weights(self.x, weights = weights)

        x_nondimensional = nondimensionalise(self.x, self.x0)
        if scale_guess is None:
            mu_log = np.sum(self.weights* np.log(x_nondimensional)) / np.sum(self.weights)  # # guess from 1st method of moments - lognormal fit
            sd_log = np.sqrt(np.sum(self.weights * (np.log(x_nondimensional) - mu_log)**2) / np.sum(self.weights))
            variance = (np.exp(sd_log**2) - 1.0) * np.exp(2.0 * mu_log + sd_log**2)
            scale_guess = np.sqrt(6.0 * variance) / np.pi

        def _gumbel_fit_root(
                scale: float | int,
                x: np.ndarray,
                weights: np.ndarray,
                return_jacobian: bool = True
                ):
            """
            Root function to solve to find best-fitting Gumbel scale parameter.
            Solves in terms of nondimensionalised data
            """
            c1 = np.sum(weights)
            c2 = np.sum(weights * x)
            c3 = np.sum(weights * np.exp(-x / scale))
            c4 = np.sum(weights * x * np.exp(-x / scale))
            root = 1.0 / scale**2 * (c2 - c1 * c4 / c3) - c1 / scale
            if return_jacobian is False:
                return(root)
            else:
                c5 = np.sum(weights * x**2 * np.exp(-x / scale))
                droot_dscale = (
                    c1 / (c3 * scale**4) * (c4**2 / c3 - c5) 
                    - 2.0 / scale**3 * (c2 - c1 * c4 / c3) + c1 / scale**2
                    )
                return(root, droot_dscale)

        fit = root_scalar(
            _gumbel_fit_root,
            method = root_method,
            fprime = True,
            x0 = scale_guess,
            args = (x_nondimensional, self.weights)
        )
        self.scale = redimensionalise(fit.root, self.x0)

        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * np.exp(-x_nondimensional / fit.root))
        self.location = self.scale * np.log(c1 / c3)


    def generate_random(
            self, 
            n: int
            ):
        """
        Generate new, random data based on the fitted distribution

        Parameters
        ----------
        n : int
            number of data points to create

        Returns
        -------
        np.ndarray | Quantity
            New data, in same units as input data
        """
        y = np.random.rand(n)
        return(self.location - self.scale * np.log(-np.log(y)))


    def calc_density(
            self, 
            x: np.ndarray | Quantity | None = None,
            log = False, 
            cumulative = False
            ):
        """
        Calculate probability densities

        Parameters
        ----------
        x : np.ndarray | Quantity | float | None, optional
            new x-data, by default None. Must have same dimensionality as x 
            data used to generate the fit. If None, calculations are made using
            the x-data used to generate the fit. 
        log : bool, optional
            if True, return log-transformed probabilities instead. By default 
            False
        cumulative : bool, optional
            if True, calculate the cumulative density instead of probability 
            density, by default False

        Returns
        -------
        np.ndarray | Quantity | float
            probability for each value in x
        """
        if x is None:
            x = self.x
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True)
        gumbel_reduced = (x - self.location) / self.scale
        if cumulative is True:
            if log is True:
                return(-np.exp(-gumbel_reduced))
            else:
                return(np.exp(-np.exp(-gumbel_reduced)))
        else:
            if log is True:
                return(-np.log(self.scale) - gumbel_reduced - np.exp(-gumbel_reduced))
            else:
                return(1.0 / self.scale * np.exp(-gumbel_reduced - np.exp(-gumbel_reduced)))
            

    def calc_loglikelihood(
            self, 
            x: np.ndarray | Quantity | None = None, 
            weights: np.ndarray = None,
            location: Quantity | float | None = None, 
            scale: Quantity | float | None = None,
            deriv: int = 0            
            ):
        """
        Calculate (partial derivatives of) fit loglikelihood

        Parameters
        ----------
        x : np.ndarray | Quantity | None, optional
            new x-data, by default None. Must have same dimensionality as x 
            data used to generate the fit. If None, calculations are made using
            the x-data used to generate the fit. 
        weights : np.ndarray | None, optional
            weighting for each data point in x, by default None. If None, same
            individual weightings as used to generate the fit are used
        location : float | Quantity | None, optional
            gumbel location parameter (must have same dimensionality as x), 
            by default None. If None, the fitted value is used
        scale : float | Quantity | None, optional
            gumbel scale parameter (must have same dimensionality as x), by
            default None. If None, the fitted value is used. M
        deriv : int, optional
            derivative order of loglikelihood with respect to location and scale 
            parameters, by default 0. 

        Returns
        -------
        float | np.ndarray
            loglikelihood (float), or partial derivatives of loglikelihood with
            respect to nondimensional location and scale parameters (np.ndarray)
        """
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        location = self.location if location is None else location
        scale = self.scale if scale is None else scale

        x_nondimensional = nondimensionalise(x, self.x0)
        location_nondimensional = nondimensionalise(location, self.x0)
        scale_nondimensional = nondimensionalise(scale, self.x0)

        c1 = np.sum(weights)
        c2 = np.sum(weights * x_nondimensional)
        c3 = np.sum(weights * np.exp(-x_nondimensional / scale_nondimensional))
        c4 = np.sum(weights * x_nondimensional * np.exp(-x_nondimensional / scale_nondimensional))
        c5 = np.sum(weights * x_nondimensional**2 * np.exp(-x_nondimensional / scale_nondimensional))

        if deriv == 0:
            return(c1 * (location_nondimensional / scale_nondimensional - np.log(scale_nondimensional)) 
                   - c2 / scale_nondimensional 
                   - c3 * np.exp(location_nondimensional / scale_nondimensional)
                   )
        elif deriv == 1:
            dlogL_dmu = c1 / scale_nondimensional - c3 / scale_nondimensional * np.exp(location_nondimensional / scale_nondimensional)
            dlogL_dtheta = (
                -c1 * (location_nondimensional / scale_nondimensional**2 + 1.0 / scale_nondimensional) 
                + c2 / scale_nondimensional**2 
                + (c3 * location_nondimensional - c4) / scale_nondimensional**2 * np.exp(location_nondimensional / scale_nondimensional)
                )
            return(np.array([dlogL_dmu, dlogL_dtheta]))
        elif deriv == 2:
            d2logL_dmu2 = -c3 / scale_nondimensional**2 * np.exp(location_nondimensional / scale_nondimensional)
            d2logL_dmudtheta = (
                1.0 / scale_nondimensional**2 * (c3 * (1.0 + location_nondimensional / scale_nondimensional) - c4 / scale_nondimensional)
                * np.exp(location_nondimensional / scale_nondimensional) 
                - c1 / scale_nondimensional**2
                )
            d2logL_dtheta2 = (
                c1 / scale_nondimensional**2 
                + 2.0 / scale_nondimensional**3 * (c1 * location_nondimensional - c2) 
                - 1.0 / scale_nondimensional**3 * (c3 * location_nondimensional - c4) 
                * np.exp(location_nondimensional / scale_nondimensional) * (2. + location_nondimensional / scale_nondimensional) 
                + 1.0 / scale_nondimensional**4 * (c4 * location_nondimensional - c5) * np.exp(location_nondimensional / scale_nondimensional)
                )
            return(np.array([
                [d2logL_dmu2, d2logL_dmudtheta], 
                [d2logL_dmudtheta, d2logL_dtheta2]
                ]))  
