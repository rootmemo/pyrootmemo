import numpy as np
from pint import Quantity
from pyrootmemo import Parameter
from scipy.optimize import root_scalar
from .utils import create_quantity, create_reference_value, check_array_values, create_weights, nondimensionalise, redimensionalise
from .linear_regression import LinearRegression
from .base_distribution import _BaseDistribution


class WeibullDistribution(_BaseDistribution):
    """Weibull distribution fitting class

    Fit a Weibull distribution using (weighted) loglikehood 

    Probability density p(x):

        p(x) = shape / scale * (x / scale)**(shape - 1)
               * exp(-(x / scale)**shape)

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
    shape : float
        Weibull distribution shape parameter
    scale : float | Quantity
        Weibull distribution scale parameterr

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
            weights: np.ndarray = None,
            x0: int | float | Quantity | Parameter | None = None,
            shape_guess: int | float | None = None,
            root_method: str = 'halley'
            ):
    
        self.x0 = create_reference_value(x, x0)
        self.x = create_quantity(x)
        check_array_values(self.x, finite = True, xmin = 0.0 * self.x0, xmin_include = True)
        self.weights = create_weights(self.x, weights = weights)

        x_nondimensional = nondimensionalise(self.x, self.x0)
        if shape_guess is None:
            xn_sorted = np.sort(x_nondimensional)
            n = len(xn_sorted)
            yp = (2.0 * np.arange(n, 0, -1) - 1.0) / (2.0 * n)
            linear_fit = LinearRegression(
                np.log(xn_sorted), 
                np.log(-np.log(yp)), 
                weights = self.weights
                )
            shape_guess = linear_fit.gradient
        
        def _weibull_fit_root(
                shape: int | float,
                x: np.ndarray,
                weights: np.ndarray, 
                return_jacobian: bool = True, 
                return_hessian: bool = True
                ):
            """
            Root function to solve to find the best-fitting weibull shape 
            parameter. Uses nondimensionalised data.
            """
            c1 = np.sum(weights)
            c2 = np.sum(weights * np.log(x))
            c3 = np.sum(weights * x**shape)
            c4 = np.sum(weights * x**shape * np.log(x))
            root = c1 / shape - c1 * c4 / c3 + c2
            if return_jacobian is False and return_hessian is False:
                return(root)
            else:
                c5 = np.sum(weights * x**shape * np.log(x)**2)
                droot_dshape = (
                    -c1 / shape**2 
                    - c1 * c5 / c3 
                    + c1 * c4**2 / c3**2
                )
                if return_hessian is False:
                    return(root, droot_dshape)
                else:
                    c6 = np.sum(weights * x**shape * np.log(x)**3)
                    d2root_dshape2 = (
                        2.0 * c1 / shape**3 
                        - c1 * c6 / c3 
                        + 3.0 * c1 * c4 * c5 / c3**2 
                        - 2.0 * c1 * c4**3 / c3**3
                    )
                    return(root, droot_dshape, d2root_dshape2)

        fit = root_scalar(
            _weibull_fit_root,
            fprime = True,
            fprime2 = True,
            x0 = shape_guess,
            method = root_method,
            args = (x_nondimensional, self.weights)
            )
        self.shape = fit.root
        
        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * x_nondimensional**self.shape)
        self.scale = redimensionalise((c3 / c1) ** (1.0 / self.shape), self.x0)


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
        return(self.scale * (-np.log(1.0 - y)) ** (1.0 / self.shape))


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
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = True)
        if cumulative is False:
            return(self.shape / self.scale
                    * (x / self.scale) ** (self.shape - 1.0)
                    * np.exp(-(x / self.scale) ** self.shape)
                    )
        else:
            return(1.0 - np.exp(-(x / self.scale) ** self.shape))
    

    def calc_loglikelihood(
            self,
            x: np.ndarray | Quantity | None = None, 
            weights: np.ndarray | None = None,
            shape: float | None = None, 
            scale: Quantity | float | None = None, 
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
        shape : float | None, optional
            weibull shape parameter, by default None. If None, the fitted value 
            is used
        scale : float | Quantity | None, optional
            weibull scale parameter (must have same dimensionality as x), by
            default None. If None, the fitted value is used.
        deriv : int, optional
            derivative order of loglikelihood with respect to location and scale 
            parameters, by default 0. 

        Returns
        -------
        float | np.ndarray
            loglikelihood (float), or partial derivatives of loglikelihood with
            respect to nondimensional shape and scale parameters (np.ndarray)
        """
        x = self.x if x is None else x
        weights = self.weights if weights is None else weights
        shape = self.shape if shape is None else shape
        scale = self.scale if scale is None else scale

        x_nondimensional = nondimensionalise(x, self.x0)
        scale_nondimensional = nondimensionalise(scale, self.x0)

        c1 = np.sum(weights * np.log(x_nondimensional))
        c2 = np.sum(weights * x_nondimensional**shape)
        c3 = np.sum(weights * x_nondimensional**shape * np.log(x_nondimensional))
        c4 = np.sum(weights * x_nondimensional**shape * np.log(x_nondimensional)**2)
        c5 = np.sum(weights * x_nondimensional**shape * np.log(x_nondimensional)**3)
        if deriv == 0:
            return(
                c1 * np.log(shape) 
                - c1 * shape * np.log(scale_nondimensional) 
                + c2 * (shape - 1.0) 
                - c3 * scale_nondimensional**(-shape)
                )
        elif deriv == 1:
            dlogL_dshape = (
                c1 / shape 
                - c1*np.log(scale_nondimensional) 
                + c2 
                + scale_nondimensional**(-shape) * (c3 * np.log(scale_nondimensional) - c4)
                )
            dlogL_dscale = (
                -c1 * shape / scale_nondimensional 
                + c3 * shape * scale_nondimensional**(-shape - 1.0)
                )
            return(np.array([dlogL_dshape, dlogL_dscale]))
        elif deriv == 2:
            d2logL_dshape2 = (
                -c1 / shape**2 
                - scale_nondimensional**(-shape)
                * (c5 - 2.0 * c4 * np.log(scale_nondimensional) + c3 * np.log(scale_nondimensional)**2)
                )
            d2logL_dshapedscale = (
                -c1 / scale_nondimensional 
                + scale_nondimensional**(-shape - 1.0)
                * (c3 - c3 * shape * np.log(scale_nondimensional) + shape * c4)
                )
            d2logL_dscale2 = (
                c1 * shape / scale_nondimensional**2 
                - c3 * shape * (shape + 1.0) * scale_nondimensional**(-shape - 2.0)
                )
            return(np.array([
                [d2logL_dshape2, d2logL_dshapedscale],
                [d2logL_dshapedscale, d2logL_dscale2]
                ]))
        else:
            raise ValueError('deriv must be 0, 1 or 2')