import numpy as np
from pint import Quantity
from pyrootmemo import Parameter
from .utils import create_quantity, create_reference_value, check_array_values, create_weights
from .base_regression import _BaseRegression


class LinearRegression(_BaseRegression):
    """Linear fitting class

    Fit a power-law function to a set of (x, y) data. The fit is defined as:

        y_fit = intercept + x * gradient

    where the intercept and the gradient are to be fitted. Parameters are
    found by solving the (weighted) linear least-squares regression problem.
    i.e. will minimise:

        SLSQ = sum(weights * (y_fit * y) ** 2)
        
    where 'weights' is the individual weighting for each (x, y) observation.

    This class is able to deal with x and/or y data that has dimensions, e.g.
    millimetres or megapascals. Fitted values will be automatically assigned
    the correct units.

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
    intercept : float | Quantity
        Fitted intercept (fitted value of y at x = 0)
    gradient : float | Quantity
        Fitted gradient
    sd : float | Quantity
        Standard deviation of the fit (biased estimation, without Bessel 
        correction)

    Methods
    -------
    __init__(x, y, weights, x0, y0)
        Constructor
    predict(x)
        Predict the average value of y (y_fit) for each given x, using the 
        linear fit

    """

    def __init__(
            self,
            x: np.ndarray | Quantity | Parameter,
            y: np.ndarray | Quantity | Parameter,
            weights: np.ndarray | Quantity = None,
            x0: float | int | Quantity | Parameter | None = None,
            y0: float | int | Quantity | Parameter | None = None,
            ):
        """Initiate linear fitting object

        Initialisation sets data and will create the best fit.

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
        weights : int | float | np.ndarray | None
            weighting for each (x, y) observation, by default None. If None,
            a weight of 1 is assumed for each observation. If defined as a 
            scalar value, this weight is assumed to each observation.
        x0 : float | int | Quantity | Parameter | None, optional
            the reference value for x, by default None
        y0 : float | int | Quantity | Parameter | None, optional
            the reference value for y, by default None

        """
        self.x0 = create_reference_value(x, x0)
        self.y0 = create_reference_value(y, y0)
        self.x = create_quantity(x)
        check_array_values(self.x, finite = True)
        self.y = create_quantity(y)
        check_array_values(self.y, finite = True)
        self.weights = create_weights(self.x, weights)

        c = np.sum(self.weights)
        cx = np.sum(self.weights * self.x)
        cx2 = np.sum(self.weights * self.x**2)
        cy = np.sum(self.weights * self.y)
        cxy = np.sum(self.weights * self.x * self.y)
        determinant = c * cx2 - cx**2

        self.intercept = (cx2 * cy - cx * cxy) / determinant
        self.gradient = (-cx * cy + c * cxy) / determinant
        self.sd = np.sqrt(
            np.sum(self.weights * (self.predict() - self.y)**2)
            / np.sum(self.weights)
            )
    

    def predict(
            self,
            x: int | float | np.ndarray | Quantity | Parameter | None = None
            ) -> np.ndarray | Quantity:
        """Predict fitted values of y, given known x-data

        Parameters
        ----------
        x : int | float | np.ndarray | Quantity | Parameter | None, optional
            x-data at which to predict y, by default None. If None, the x-data
            used to create the fit is used instead.

        Returns
        -------
        np.ndarray | Quantity
            fitted values of y, at each x

        """
        if x is None:
            x = self.x
        else:
            check_array_values(x)
        return(self.intercept + self.gradient * x)