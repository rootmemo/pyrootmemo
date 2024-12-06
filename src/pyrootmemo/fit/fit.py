# import packages
import numpy as np
from pint import Quantity
from pint.errors import DimensionalityError

# Base fit class
class FitBase:

    # check input values
    def _check_input(
            self,
            x,
            finite = True,
            min = None,
            max = None,
            min_include = True,
            max_include = True,
            label = None
    ):  
        # finite values
        if finite is True:
            if np.isinf(x).any() is True:
                raise ValueError('all values must be finite')
        # minimum value
        if np.isscalar(min):
            if min_include is True:
                if (x < min).any():
                    raise ValueError(f'all {label} values must be larger or equal than {min}')
            else:
                if (x <= min).any():
                    raise ValueError(f'all {label} values must be larger than {min}')
        # maximum value
        if np.isscalar(max):
            if max_include is True:
                if (x > max).any():
                    raise ValueError(f'all {label} values must be smaller or equal than {max}')
            else:
                if (x >= min).any():
                    raise ValueError(f'all {label} values must be smaller than {max}')

    # check dimensionality - check whether input x matches referece x0
    def check_dimensionality(
            self,
            x,
            x0
    ):
        if isinstance(x0, Quantity):
            if isinstance(x, Quantity):
                if x0.dimensionality != x.dimensionality:
                    raise DimensionalityError('units of x not compatible with fit result')
            else:
                raise DimensionalityError(f'x must be defined in units of {x0.units}')
        else:
            if isinstance(x, Quantity):
                raise DimensionalityError('units of x not compatible with fit result')

    # get reference value for an array of data
    def get_reference(
            self,
            x,
            x0 = None
            ):
        if isinstance(x, Quantity):
            # x defined with units
            if x0 is None:
                return(1.0 * x.units)
            elif isinstance(x0, Quantity):
                if x0.dimensionality == x.dimensionality:
                    return(x0)
                else:
                    raise DimensionalityError('units of x and x0 are not compatible')
            elif np.isscalar(x0):
                UserWarning(f'units of reference value not given - value assumed in {x.units}')
                return(x0 * x.units)
        else:
            # x defined without units
            if x0 is None:
                return(1.0)
            elif isinstance(x0, Quantity):
                raise DimensionalityError('units of x and x0 are not compatible')
            elif np.isscalar(x0):
                return(x0)
                    
    # generate non-dimensional data (+apply scaling)
    def nondimensionalise(
            self,
            x,
            x0 = 1.0
            ):
        # convert list (or tuple) to numpy array
        if isinstance(x, list) or isinstance(x, tuple):
            if all([isinstance(xi, Quantity) for xi in x]) is True:
                # convert list of pint.Quantity elements to single array (using unit from first item)
                unit = x[0].units
                vals = np.array([xi.magnitude for xi in x])
                x = vals * unit
            else:
                # convert list to array
                x = np.array(x)
        # make nondimensional
        if isinstance(x, Quantity):
            if isinstance(x0, Quantity):
                # convert x to units of reference value, and return magnitude
                return((x.to(x0.units)).magnitude / x0.magnitude)
            else:
                return(x.magnitude / x0)
        else:
            return(x / x0)

    # add units (+ scaling) to nondimensionalised values
    def redimensionalise(
            self,
            x,
            x0 = 1.0
    ):
        return(x * x0)
    