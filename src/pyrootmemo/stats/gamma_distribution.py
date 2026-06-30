import numpy as np
from pint import Quantity
from pyrootmemo import Parameter
from scipy.optimize import root_scalar
from scipy.special import gamma, digamma, polygamma, gammainc, gammaincinv
from .utils import create_quantity, create_reference_value, check_array_values, create_weights, nondimensionalise, redimensionalise
from .base_distribution import _BaseDistribution


class GammaDistribution(_BaseDistribution):
    """Gamma distribution fitting class

    Fit a gamma distribution using (weighted) loglikehood 

    Probability density p(x):

        p(x) = x**(shape - 1) / (Gamma(shape) * scale**shape) * exp(-x / scale)

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
        best-fitting gamma distribution shape parameter
    scale : float | Quantity
        best-fitting gamma distribution scale parameterr

    Methods
    -------
    __init__(x, weights, x0, shape_guess, root_method)
        Constructor
    calc_density(..., cumulative)
        calculate probability density or cumulative probablity density
    calc_loglikelihood(..., deriv)
        calculate (partial derivatives of) fit loglikehood
    generate_random(n)
        draw `n` new values from the fitted distribution
    plot(...)
        plot histogram of data, and the fitted distribution

    """

    def __init__(
            self,
            x: np.ndarray | Quantity | Parameter,
            weights: np.ndarray | Quantity = None,
            x0: int | float | Quantity | Parameter | None = None,
            shape_guess: float | int | None = None,
            root_method: str = 'halley'
            ):
        """Initialise instance of gamma distribution fitting class

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter
            data
        weights : np.ndarray | Quantity, optional
            weight for each data point in `x`, by default None. If None, 
            all weights are assumed as one
        x0 : int | float | Quantity | Parameter | None, optional
            reference x-value used to make data nondimensional, by default None. 
            If None, is defined as one unit of x.
        shape_guess : float | int | Quantity | None, optional
            initial guess for shape parameter. Must have same dimensionality 
            as `x`. By default None. If None, a reasonable starting guess is 
            made using the method of moments
        root_method : str, optional
            root solver routine used by `scipy.optimize.root_scalar()` solver, 
            by default 'halley'. 
        """
        
        self.x0 = create_reference_value(x, x0)
        self.x = create_quantity(x)
        check_array_values(self.x, finite = True, xmin = 0.0 * self.x0, xmin_include = True)
        self.weights = create_weights(self.x, weights = weights)

        if shape_guess is None:
            x_nondimensional = nondimensionalise(self.x, self.x0)
            c1 = np.sum(self.weights)
            c2 = np.sum(self.weights * np.log(x_nondimensional))
            c3 = np.sum(self.weights * x_nondimensional)
            shape_guess = c1 / (2.0 * c1 * np.log(c3 / c1) - 2.0 * c2)
        else:
            shape_guess = nondimensionalise(shape_guess, self.x0)

        def _gamma_fit_root(
                shape: float | int,
                x: np.ndarray,
                weights: np.ndarray,
                return_jacobian: bool = True, 
                return_hessian: bool = True
                ):
            """
            Root function to solve to find best-fitting gamma shape parameter.
            Solves in terms of nondimensionalised data
            """
            c1 = np.sum(weights)
            c2 = np.sum(weights * np.log(x))
            c3 = np.sum(weights * x)
            root = (
                c1 * (np.log(shape) - np.log(c3 / c1))
                - c1 * digamma(shape)
                + c2
                )
            if return_jacobian is False and return_hessian is False:
                return(root)    
            else:
                droot_dshape = c1 / shape  - c1 * polygamma(1, shape)
                if return_hessian is False:
                    return(root, droot_dshape)        
                else:
                    droot2_dshape2 = -c1 / shape**2 - c1 * polygamma(2, shape)
                    return(root, droot_dshape, droot2_dshape2)

        fit = root_scalar(
            _gamma_fit_root,
            fprime = True,
            fprime2 = True,
            x0 = shape_guess,
            method = root_method,
            args = (x_nondimensional, self.weights, True, True)
            )
        self.shape = fit.root

        c1 = np.sum(self.weights)
        c3 = np.sum(self.weights * x_nondimensional)
        self.scale = redimensionalise(c3 / (self.shape * c1), self.x0)


    def generate_random(
            self, 
            n: int
            ) -> np.ndarray | Quantity:
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
        return(self.scale * gammaincinv(self.shape, y))
    

    def calc_density(
            self, 
            x: np.ndarray | Quantity | float | Parameter | None = None, 
            cumulative: bool = False
            ) -> np.ndarray | Quantity | float:
        """
        Calculate probability densities

        Parameters
        ----------
        x : np.ndarray | Quantity | float | Parameter | None, optional
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
            return(
                x**(self.shape - 1.0)
                / (gamma(self.shape) * self.scale**self.shape)
                * np.exp(-x / self.scale)
                )
        else:
            return(gammainc(self.shape, nondimensionalise(x, self.scale)))
       

    def calc_loglikelihood(
            self,
            x: float | np.ndarray | Quantity | Parameter | None = None, 
            weights: np.ndarray | None = None,
            shape: float | None = None, 
            scale: float | Quantity | None = None, 
            deriv: int = 0
            ) -> float | np.ndarray:
        """
        Calculate (partial derivatives of) fit loglikelihood

        Parameters
        ----------
        x : np.ndarray | Quantity | float | Parameter | None, optional
            new x-data, by default None. Must have same dimensionality as x 
            data used to generate the fit. If None, calculations are made using
            the x-data used to generate the fit. 
        weights : np.ndarray | None, optional
            weighting for each data point in x, by default None. If None, same
            individual weightings as used to generate the fit are used
        shape : float | None, optional
            gamma shape parameter, by default None. If None, the fitted 
            value is used
        scale : float | Quantity | None, optional
            gamma scale parameter (must have same dimensionality as x), by
            default None. If None, the fitted value is used. M
        deriv : int, optional
            derivative order of loglikelihood with respect to shape and scale 
            parameters, by default 0. 

        Returns
        -------
        float | np.ndarray
            loglikelihood (float), or partial derivatives of loglikelihood with
            respect to nondimensional shape and scale parameters (np.ndarray)
        """

        if x is None:
            x = self.x
        else:
            x = create_quantity(x)
            check_array_values(x, finite = True, xmin = 0.0 * self.x0, xmin_include = True)
        weights = self.weights if weights is None else weights
        shape = self.shape if shape is None else shape
        scale = self.scale if scale is None else scale

        x_nondimensional = nondimensionalise(x, self.x0)
        scale_nondimensional = nondimensionalise(scale, self.x0)

        c1 = np.sum(weights * np.log(x_nondimensional))
        c2 = np.sum(weights * np.log(x_nondimensional))
        c3 = np.sum(weights * x_nondimensional)

        if deriv == 0:
            return(
                - c1 * shape * np.log(scale_nondimensional) 
                - c1 * np.log(gamma(shape))
                + c2 * (shape - 1.0)
                - c3 / scale_nondimensional
                )
        elif deriv == 1:
            dlogL_dshape = (
                - c1 * np.log(scale_nondimensional)
                - c1 * digamma(shape)
                + c2
                )
            dlogL_dscale = (
                - c1 * shape / scale_nondimensional
                + c3 / scale_nondimensional**2
                )
            return(np.array([dlogL_dshape, dlogL_dscale]))
        elif deriv == 2:
            d2logL_dshape2 = -c1 * polygamma(1, shape)
            d2logL_dshapedscale = -c1 / scale_nondimensional
            d2logL_dscale2 = (
                c1 * shape / scale_nondimensional**2
                - 2.0 * c3 / scale_nondimensional**3
                )
            return(np.array([
                [d2logL_dshape2, d2logL_dshapedscale],
                [d2logL_dshapedscale, d2logL_dscale2]
                ]))
        else:
            raise ValueError('deriv must be 0, 1 or 2')