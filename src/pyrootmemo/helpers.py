import numpy as np
from pint import UnitRegistry, Quantity
from pyrootmemo.tools.checks import is_namedtuple
from collections import namedtuple
from enum import StrEnum
from dataclasses import dataclass

units = UnitRegistry()

def limit_check(value: float | int, key:str, limit_type:str):
    match limit_type:
        case "non-negative":
            if value < 0:
                raise ValueError(f"{value} is not allowed. {key} must be non-negative")
        case "positive_only":
            if value <= 0:
                raise ValueError(f"{value} is not allowed. {key} must be positive")

#: Parameter is a named tuple that holds a value and its unit
#: It is used to store physical quantities with their respective units
#: This allows for better organization and readability of code
#: Example usage: Parameter(value=1, unit='m')
Parameter = namedtuple("parameter", "value unit")


def secant(degree) -> float:
    """
    secant _summary_

    Args:
        degree (_type_): _description_

    Raises:
        TypeError: _description_
        ValueError: _description_

    Returns:
        float: _description_
    """
    try:
        secant = 1 / np.cos(np.deg2rad(degree))
    except TypeError as te:
        print(f"TypeError: Wrong input type ({te})")
        raise TypeError
    except ValueError as ve:
        print(f"ValueError: Wrong input value ({ve})")
        raise ValueError
    else:
        return secant


def calc_shear_strain(shear_displacement, shear_zone_thickness):
    try:
        if (shear_zone_thickness < 0) or (shear_displacement < 0):
            raise ValueError("Inputs must be non-negative")
        else:
            shear_strain = shear_displacement / shear_zone_thickness
    except ZeroDivisionError:
        print("ZeroDivisionError: Shear zone thickness cannot be zero")
    except TypeError as te:
        print(f"TypeError: Wrong input type ({te})")
        raise TypeError
    else:
        return shear_strain


def create_quantity(
        x: Quantity | Parameter, 
        check_unit: None | str = None, 
        scalar: bool = False
        ) -> Quantity:
    """Check and return the input as a Quantity object

    Take input values (with units), either defined as a Quantity object or
    as a tuple with values and a unit, and create a `pint.Quantity' object.

    In addition, perform some checks:

    * x contains only numeric input
    * the unit of 'x' must be compatible with input 'check_unit' (if 
      'check_unit' is not None)
    * if scalar is True, 'x' must be a scalar value (and not an array)

    Parameters
    ----------
    x : Quantity | Parameter(value: int | float | np.ndarray, unit: chr)
        _description_
    check_unit : None | str, optional
        _description_, by default None
    scalar : bool, optional
        If True, additionally checks whether the value of x is a scalar. 
        By default False, meaning x can either be a scalar or an array.

    Returns
    -------
    Quantity
        Input x as a pint.Quantity
    """
    if isinstance(x, Quantity):
        if scalar is True:
            if not np.isscalar(x.magnitude):
                raise ValueError('Magnitude of Quantity x must be a scalar')
        if check_unit is not None:
            if isinstance(check_unit, str):
                if x.dimensionality != units(check_unit).dimensionality:
                    raise ValueError('units of x not compatible with unit')
            else:
                raise TypeError('unit must be None or a string')
        return(x)
    elif is_namedtuple(x):
        if len(x) == 2:
            if not isinstance(x[1], str):
                raise TypeError('second element of x (unit) must be str')
            if check_unit is not None:
                if isinstance(check_unit, str):
                    if units(x[1]).dimensionality != units(check_unit).dimensionality:
                        raise ValueError('unit not compatible with unit in x')
                else:
                    raise TypeError('unit must be None or a string')
            if scalar is True:
                if not np.isscalar(x[0]):
                    raise TypeError('value in x must be scalar')
            if np.isscalar(x[0]):
                if not (isinstance(x[0], int) or isinstance(x[0], float)):
                    raise TypeError('first element of x (value) must be int or float')
                else:
                    return(x[0] * units(x[1]))
            elif isinstance(x[0], list):
                if any([not (isinstance(i, int) or isinstance(i, float)) for i in x[0]]):
                    raise TypeError('all values in list must be int or float')
            elif isinstance(x[0], np.ndarray):
                if not (np.issubdtype(x[0].dtype, np.integer) or np.issubdtype(x[0].dtype, np.floating)):
                    raise TypeError('all values in array must be int or float')
            else:
                raise TypeError('values in x must not recognised')
        else:
            raise TypeError('x must be Quantity or Parameter(value: int | float | np.ndarray, unit: str)')
        return(x[0] * units(x[1]))
    else:
        raise TypeError('x must be Quantity or Parameter(value: int | float, unit: str)')
    

class ResultsType(StrEnum):
    """Options for results type."""

    ATTRIBUTE = "attribute"
    RETURN = "return"
    BOTH = "both"

@dataclass    
class Results:
    how: ResultsType | int = "attribute"
    
    def __post_init__(self):
        if not isinstance(self.how, int | str):
            raise TypeError("Results type must be int or str.")
        if isinstance(self.how, str):
            try:
                self.how = ResultsType(self.how)
            except ValueError:
                raise ValueError(f"Invalid results type: {self.how}. Must be one of {list(ResultsType)}.")
        elif isinstance(self.how, int):
            try:
                self.how = list(ResultsType)[self.how]
            except IndexError:
                raise ValueError(f"Invalid results type: {self.how}. Must be one of {list(ResultsType)}.")

def outer_ufunc(
            a: Quantity | np.ndarray, 
            b: Quantity | np.ndarray,
            ufunc: str = 'multiply'
        ) -> Quantity | np.ndarray:
    """Outer operator of two vectors that preserves dimensionality of Quantities

    Calculates C_ij = a_i <ufunc> b_j

    Outer operators are not (currently) supported by the pint package, therefore
    this function is needed

    Parameters
    ----------
    a : Quantity | np.ndarray | float | int
        first array
    b : Quantity | np.ndarray | float | int
        second array
    ufunc : str
        Numpy ufunc name. By default 'multiply'. Currently supported are
        'multiply', 'divide', 'add' and 'subtract'

    Returns
    -------
    Quantity | np.ndarray
        outer product of a and b, i.e. matrix M_ij = a_i b_j
    """
    match ufunc:
        case 'multiply' | 'divide':
            if ufunc == 'divide':
                b = 1.0 / b
            if isinstance(a, Quantity):
                a_magnitude = a.magnitude
                a_units = a.units
            else:
                a_magnitude = a
                a_units = 1.0
            if isinstance(b, Quantity):
                b_magnitude = b.magnitude
                b_units = b.units
            else:
                b_magnitude = b
                b_units = 1.0
            return(np.multiply.outer(a_magnitude, b_magnitude) * a_units * b_units)
        case 'add' | 'subtract':
            if ufunc == 'subtract':
                b = -b
            if isinstance(a, Quantity):
                a_magnitude = a.magnitude
                a_units = a.units
                b_magnitude = b.to(a_units).magnitude
                return(np.add.outer(a_magnitude, b_magnitude) * a_units)
            else:
                return(np.add.outer(a, b))
        case _:
            raise ValueError("ufunc must be one of ['multiply', 'divide', 'add', 'subtract']")