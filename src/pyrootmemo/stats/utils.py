import numpy as np
from pint import Quantity
from pyrootmemo.helpers import units
from pyrootmemo import Parameter


def create_quantity(
        x: Parameter | np.ndarray | Quantity | list | tuple | int | float,
        scalar_output: bool = False
        ) -> np.ndarray | Quantity | int | float:
    """Convert input data

    Converts input data into a standardised form. If input data is a 
    Parameter or Quantity object, a Quantity is returned. Lists and tuples 
    are converted into numpy arrays. Scalars remain scalars.

    Parameters
    ----------
    x : Parameter | np.ndarray | Quantity | list | tuple | int | float
        input data
    scalar_output : bool, optional
        check if output is a scalar value, by default False

    Returns
    -------
    np.ndarray | Quantity | int | float
        output data

    """
    if isinstance(x, Parameter):
        if np.isscalar(x[0]):
            xout = x.value * units(x.unit)
        else:
            xout = np.array(x.value) * units(x.unit)
    elif isinstance(x, list) | isinstance(x, tuple):
        xout = np.array(x)
    elif isinstance(x, np.ndarray) | isinstance(x, Quantity) | isinstance(x, int) | isinstance(x, float):
        xout = x
    else:
        raise TypeError('x Parameter or numeric scalar or iterable type')
    if scalar_output is True:
        if isinstance(xout, np.ndarray):
            if len(xout) == 1:
                return(xout[0])
            else:
                raise ValueError('x must be scalar value')
        elif isinstance(xout, Quantity):
            if np.isscalar(xout.magnitude):
                return(xout)
            elif len(xout.magnitude == 1):
                return(xout[0])
            else:
                raise ValueError('x must be scalar value')
        else:
            return(xout)
    else:
        return(xout)


def create_reference_value(
        x: np.ndarray | Quantity | Parameter,
        x0: int | float | Quantity | Parameter | None = None
        ) -> int | float | Quantity:
    """Create a reference value

    Create a reference value, to help converting values in x between
    dimensional and nondimensional forms. 

    Parameters
    ----------
    x : np.ndarray | Quantity | Parameter
        array with observations
    x0 : int | float | Quantity | Parameter | None, optional
        user-defined scalar reference value, by default None. If None, it is 
        set to one unit of the unit of x.

    Returns
    -------
    int | float | Quantity
        Reference value (scalar)

    """
    x = create_quantity(x)
    if x0 is None:
        if isinstance(x, Quantity):
            return(1.0 * x.units)
        elif isinstance(x, np.ndarray):
            return(1.0)
    else:
        x0 = create_quantity(x0, scalar_output = True)
        if isinstance(x, Quantity):
            if isinstance(x0, float) | isinstance(x0, int):
                return(x0 * x.units)
            elif isinstance(x0, Quantity):
                if (x / x0).to_base_units().dimensionless is True:
                    return(x0)
                else:
                    raise ValueError('units of x and x0 not compatible')
        elif isinstance(x, np.ndarray):
            if isinstance(x0, float) | isinstance(x0, int):
                return(x0)
            else:
                raise TypeError('x0 must be int, float or None when x is dimensionless')
            

def check_array_values(
        x: np.ndarray | Quantity | Parameter,
        finite: bool = True,
        xmin: float | int | Quantity | Parameter | None = None,
        xmax: float | int | Quantity | Parameter | None = None,
        xmin_include: bool = True,
        xmax_include: bool = True,
        label: str = 'x'
    ) -> np.ndarray | Quantity:  
        """Check array data input

        Checks data input for
        * all data has finite values
        * values larger (or larger and equal to) minimum value
        * values smaller (or smaller or equal to) maximum value

        Parameters
        ----------
        x : np.ndarray | Quantity | Parameter
            array with x-data
        finite : bool, optional
            if True, checks if all x-data has finite values, by default True
        xmin : float | int | Quantity | Parameter | None, optional
            if set, checks whether all x-data is larger than xmin value. If 
            None, no minimum is set. By default None
        xmax : float | int | Quantity | Parameter | None, optional
            if set, checks whether all x-data is smaller than xmax value. If 
            None, no maximum is set. By default None
        xmin_include : bool, optional
            If True, include 'xmin' value as allowable value, by default True
        xmax_include : bool, optional
            If True, include 'xmax' value as allowable value, by default True
        label : str, optional
            String to idenfity data in raised errors, by default 'x'

        Returns
        -------
        np.ndarray | Quantity
            Checked data array (will be the same as input x)

        """
        x = create_quantity(x)
        if not (isinstance(x, np.ndarray) | isinstance(x, Quantity)):
            raise TypeError(f'{label} must be a np.ndarray or Quantity')
        
        if finite is True:
            if np.any(np.isinf(x)) is True:
                raise ValueError(f'all values in {label} must be finite')
        
        if xmin is not None:
            xmin = create_quantity(xmin, scalar_output = True)
            if xmin_include is True:
                if np.any(x < xmin):
                    raise ValueError(f'all {label} values must be larger or equal than {xmin}')
            else:
                if np.any(x <= xmin):
                    raise ValueError(f'all {label} values must be larger than {xmin}')

        if xmax is not None:
            xmax = create_quantity(xmax, scalar_output = True)
            if xmax_include is True:
                if np.any(x > xmax):
                    raise ValueError(f'all {label} values must be smaller or equal than {xmax}')
            else:
                if np.any(x >= xmin):
                    raise ValueError(f'all {label} values must be smaller than {xmax}')


def create_weights(
        x: np.ndarray | Quantity,
        weights: int | float | list | np.ndarray | None = None
        ) -> np.ndarray:
    """Create an array with weighting for each observation in x
    
    Parameters
    ----------
    x : np.ndarray | List | Quantity
        array or list with observations
    weights : int | float | list | np.ndarray | None, optional
        weights. By default None, which will create a unit weight (=1) for each
        observation in x. If defined as a scalar, each the weight for each
        observation x is set to this scalar.

    Returns
    -------
    np.ndarray
        weights for each observation in x

    """
    if isinstance(weights, Quantity):
        if weights.dimensionless is True:
            weights = weights.magnitude
        else:
            raise TypeError('weight cannot be a dimensional Quantity')
    elif isinstance(weights, list):
        weights = np.array(weights)
        
    if weights is None:
        return(np.ones_like(x))
    elif isinstance(weights, int) | isinstance(weights, float):
        return(weights * np.ones_like(x))
    elif isinstance(weights, np.ndarray):
        if x.shape == weights.shape:
            return(weights)
        else:
            raise ValueError('weight must be scalar or same size as x')
    else:
        raise TypeError('weight must be None, int, float or numpy array with same size as x')


def nondimensionalise(
        x: np.ndarray | float | int | Quantity, 
        x0: int | float | Quantity | None = None
        ) -> np.ndarray:
    """Convert dimensional data array to nondimensional form

    Converts an array with data to a non-dimensional form, by dividing the 
    input data by a reference value x0. This cancels out any units, so the
    function returns a nondimensional array of data.

    Parameters
    ----------
    x : np.ndarray | float | int | Quantity
        scalar or array with (potentially dimensional) data
    x0 : int | float | Quantity | None, optional
        reference value (scalar), by default None

    Returns
    -------
    np.ndarray
        nondimensionalised data array

    """
    if isinstance(x, Quantity):
        if x0 is None:
            return(x.magnitude)
        elif isinstance(x0, Quantity):
            return(x.to(x0.units).magnitude / x0.magnitude)
        else:
            raise TypeError('x0 must be Quantity or None, with given x')
    elif isinstance(x, np.ndarray) | isinstance(x, float) | isinstance(x, int):
        if x0 is None:
            return(x)
        elif isinstance(x0, int) | isinstance(x0, float):
            return(x / x0)
        else:
            raise TypeError('x0 must be int, float or None, with given x')
    else:
        raise TypeError("x must be np.ndarray, float, int or Quantity")
    

def redimensionalise(
        x: np.ndarray, 
        x0: int | float | Quantity = 1.0
        ) -> np.ndarray | Quantity:
    """Convert data to dimensional data, using the reference value

    Parameters
    ----------
    x : np.ndarray
        array with nondimensional data
    x0 : int | float | Quantity, optional
        reference value for the inverse operation, i.e. <nondimensional data = 
        dimensional data / reference value>, by default 1.0

    Returns
    -------
    np.ndarray | Quantity
        redimensionalised data

    """
    return(x * x0)