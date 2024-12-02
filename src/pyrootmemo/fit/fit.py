# import packages
import numpy as np
from pint import Quantity

# Base fit class
class FitBase:

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