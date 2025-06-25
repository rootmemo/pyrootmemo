import numpy as np
from pint import UnitRegistry, Quantity
from pyrootmemo.tools.checks import is_namedtuple
from collections import namedtuple
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
    

def solve_quadratic(
        a: Quantity, 
        b: Quantity, 
        c: Quantity,
        ) -> Quantity:
    """Calculate largest root of a quadratic equation

    Calculate the largest root of a quadratic equation in the form:
    a * x**2 + b * x + c == 0 

    Parameters
    ----------
    a : Quantity
        second-order polynomial coefficient(s)
    b : Quantity
        first-order polynomial coefficient(s)
    c : Quantity
        zero-order polynomial coefficient(s)

    Returns
    -------
    Quantity | np.ndarray | float
        Largest root of the quadratic equation
    """
    discriminant = b**2 - 4.0 * a * c
    x = (-b + np.sign(a) * np.sqrt(discriminant)) / (2.0 * a)
    return(x)


def solve_cubic(
        a: Quantity, 
        b: Quantity, 
        c: Quantity, 
        d: Quantity
        ) -> Quantity:
    """Calculate largest real root of a cubic equation

    Calculate the largest root of a cubic equation in the form:
    a * x**3 + b * x**2 + c * x + d == 0 

    The function assumes all values of the third-order coefficient <a> are
    not equal to zero. If so, a quadratic solver is more appropriate.

    The function follows the methodology detailed on Wikipedia
    (https://en.wikipedia.org/wiki/Cubic_equation):
  
    Parameters
    ----------
    a : Quantity
        third-order polynomial coefficient(s). All values must not be equal 
        to zero for the function to work.
    b : Quantity
        second-order polynomial coefficient(s)
    c : Quantity
        first-order polynomial coefficient(s)
    d : Quantity
        zero-order polynomial coefficient(s)

    Returns
    -------
    Quantity
        Largest real root of the cubic equation
    """
    x = np.zeros(a.shape) * d.units / c.units
    e = b / a
    f = c / a
    g = d / a
    Q = (e**2 - 3.0 * f) / 9.0
    R = (2.0 * e**3 - 9.0 * e * f + 27.0 * g) / 54.0
    flag_3roots = (R**2) < (Q**3) # if true, 3 real roots exist, if false, only one real root exists
    if any(flag_3roots):
        theta = np.arccos(R[flag_3roots] / np.sqrt(Q[flag_3roots]**3))
        x[flag_3roots] = (
            -2.0 
            * np.sqrt(Q[flag_3roots]) 
            * np.cos((theta + 2.0 * np.pi) / 3.0) 
            - e[flag_3roots] 
            / 3.0
            )
    flag_1root = ~flag_3roots
    if any(flag_1root):
        A = (
            -np.sign(R[flag_1root]) 
            * (
                np.abs(R[flag_1root]) 
                + np.sqrt(R[flag_1root]**2 - Q[flag_1root]**3)
                ) ** (1.0 / 3.0)
            )
        B = Q[flag_1root] / A
        x[flag_1root] = (A + B) - e[flag_1root] / 3.0
    flag_zero = np.isclose(d.magnitude, 0.0)
    x[flag_zero] = 0.0 * d.units / c.units
    return(x)