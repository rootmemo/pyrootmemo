import numpy as np
from pint import Quantity
from pyrootmemo import Parameter
from scipy.optimize import root_scalar
from .utils import create_quantity, create_reference_value, check_array_values, create_weights, nondimensionalise, redimensionalise
from .base_distribution import _BaseDistribution


class PowerDistribution(_BaseDistribution):
    """Power distribution fitting class

    Fit a Power distribution using (weighted) loglikehood. Probability density
    is given by:

               [0                                 when x < lower
        p(x) = [multiplier * (x / x0)**exponent   when lower <= x <= upper
               [0                                 when x > upper

    where the multiplier follows from:

        integral p(x) = 1
    
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
    lower : float | Quantity
        lower limit
    upper : float | Quantity
        upper limit
    exponent: float
        power law exponent

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
    calc_multiplier()
        calculate the power law multiplier from the fitted solution
    plot(...)
        plot histogram of data, and the fitted distribution

    """

    def __init__(
            self,
            x: np.ndarray | Quantity | Parameter,
            weights: np.ndarray | Quantity = None,
            x0: int | float | Quantity | Parameter | None = None,
            lower: int | float | Quantity | Parameter | None = None,
            upper: int | float | Quantity | Parameter | None = None,
            exponent_guess: float | int | None = None,
            root_method: str = 'halley'
            ):  
        self.x0 = create_reference_value(x, x0)
        self.x = create_quantity(x)
        check_array_values(self.x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        self.weights = create_weights(self.x, weights = weights)
        if lower is None:
            self.lower = np.min(self.x)
        else:
            self.lower = create_quantity(lower, scalar_output = True)
        if upper is None:
            self.upper = np.max(self.x)
        else:
            self.upper = create_quantity(upper, scalar_output = True)

        if exponent_guess is None:
            exponent_guess = 0.0

        fit = root_scalar(
            lambda b: self.calc_loglikelihood(
                exponent = b, 
                x = self.x, 
                weights = self.weights,
                lower = self.lower, 
                upper = self.upper,
                deriv = 1
                ),
            x0 = exponent_guess,
            fprime = lambda b: self.calc_loglikelihood(
                exponent = b, 
                x = self.x, 
                weights = self.weights, 
                lower = self.lower, 
                upper = self.upper,
                deriv = 2
                ),
            fprime2 = lambda b: self.calc_loglikelihood(
                exponent = b, 
                x = self.x, 
                weights = self.weights, 
                lower = self.lower, 
                upper = self.upper,
                deriv = 3
                ),
            method = root_method,
            )
        self.exponent = fit.root


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
        if np.isclose(self.exponent, -1.0):
            return(self.lower * (self.upper / self.lower)**y)
        else:
            return((
                self.lower**(self.exponent + 1.0)
                + y * (self.upper**(self.exponent + 1.0) - self.lower**(self.exponent + 1.0))
                ) ** (1.0 / (self.exponent + 1.0)))
    

    def calc_density(
            self, 
            x = None, 
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
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = False)
        if cumulative is False:
            if np.isclose(self.exponent, -1.0):
                y = 1.0 / (x * np.log(self.upper / self.lower))
            else:
                y = (
                    (self.exponent + 1.0) 
                    * x ** self.exponent 
                    / (self.upper**(self.exponent + 1.0) - self.lower**(self.exponent + 1.0))
                    )
            y[x < self.lower] = 0.0 * y[x < self.lower]
            y[x > self.upper] = 0.0 * y[x > self.upper]
            return(y)
        else:
            if np.isclose(self.exponent, -1.0):
                y = np.log(x / self.lower) / np.log(self.upper / self.lower)
            else:
                y = (
                    (x**(self.exponent + 1.0) - self.lower**(self.exponent + 1.0))
                    / (self.upper**(self.exponent + 1.0) - self.lower**(self.exponent + 1.0))
                    )
            y[x < self.lower] = 0.0 * y[x < self.lower]
            y[x > self.upper] = 1.0 * y[x > self.upper]
            return(y)      


    def calc_loglikelihood(
            self, 
            x: np.ndarray | Quantity | None = None, 
            weights: np.ndarray | None = None,
            exponent: float | None = None, 
            lower: float | Quantity | None = None, 
            upper: float | Quantity | None = None, 
            deriv: int = 0
            ) -> float | np.ndarray:
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
        exponent : float | None, optional
            power exponent, by default None. If None, the fitted value is used
        lower : float | Quantity | None, optional
            lower x-limit of fit (must have same dimensionality as x), by
            default None. If None, the fitted value is used.
        upper : float | Quantity | None, optional
            upper x-limit of fit (must have same dimensionality as x), by
            default None. If None, the fitted value is used.
        deriv : int, optional
            derivative order of loglikelihood with respect to location and scale 
            parameters, by default 0. 

        Returns
        -------
        float | np.ndarray
            loglikelihood (float), or partial derivatives of loglikelihood with
            respect to nondimensional exponent, lower and upper parameters 
            (np.ndarray)
        """
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        exponent = self.exponent if exponent is None else exponent
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper 

        x_nondimensional = nondimensionalise(x, self.x0)
        lower_nondimensional = nondimensionalise(lower, self.x0)
        upper_nondimensional = nondimensionalise(upper, self.x0)

        c1 = np.sum(weights)
        c2 = np.sum(weights * np.log(x_nondimensional))
        if np.isclose(exponent, -1.0):
            if deriv == 0:
                return(-c2 - c1 * np.log(np.log(upper_nondimensional / lower_nondimensional)))
            elif deriv == 1:
                return(c2 + 0.5 * c1 * np.log(upper_nondimensional / lower_nondimensional))
            elif deriv == 2:
                return(-c1 / 12.0 * np.log(upper_nondimensional / lower_nondimensional)**2)
            elif deriv == 3:
                return(0.0)
        else:
            lp = lower_nondimensional**(exponent + 1.0)
            up = upper_nondimensional**(exponent + 1.0)
            ll = np.log(lower_nondimensional)
            ul = np.log(upper_nondimensional)
            if deriv == 0:
                return(
                    exponent * c2
                    + c1 * np.log(exponent + 1.0)
                    - c1 * np.log(up - lp)
                    )
            elif deriv == 1:
                return(
                    c2 
                    + c1 / (exponent + 1.0) 
                    - c1 * (up * ul - lp * ll) / (up - lp)
                    )
            elif deriv == 2:
                return(
                    - c1 / (exponent + 1.0)**2
                    + c1 * (up * ul - lp * ll)**2 / (up - lp)**2
                    - c1 * (up * ul**2 - lp * ll**2) / (up - lp)
                    )
            elif deriv == 3:
                return(
                    2.0 * c1 / (exponent + 1.0)**3
                    - 2.0 * c1 * (up * ul - lp * ll)**3 / (up - lp)**3
                    + 3.0 * c1 * (up * ul - lp * ll) * (up * ul**2 - lp * ll**2) / (up - lp)**2
                    - c1 * (up * ul**3 - lp * ll**3) / (up - lp)
                    )
        

    def calc_multiplier(self) -> float | Quantity:
        """Calculate power law multiplier

        Calculate the corresponding power law multiplier that satisfies:

            p(x) = multiplier * (x / x0)**exponent

        Returns
        -------
        float | Quantity
            Power law multiplier

        """
        if np.isclose(self.exponent, -1.0):
            return(1.0 / (self.x0 * np.log(self.upper / self.lower)))
        else:
            return(
                (self.exponent + 1.0) 
                / self.x0 
                / ((self.upper / self.x0)**(self.exponent + 1.0)
                   - (self.lower / self.x0)**(self.exponent + 1.0))
                )