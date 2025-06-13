import numpy as np
from pint import UnitRegistry
from collections import namedtuple
units = UnitRegistry()

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
