import numpy as np
from pyrootmemo.tools.helpers import secant


def __get_k(
    max_tangential_stress: int | float,
    shear_band_thickness: int | float,
    elastic_modulus: int | float,
    diameter: int | float,
):
    try:
        k = np.sqrt(
            4
            * max_tangential_stress
            * shear_band_thickness
            * elastic_modulus
            / diameter
        )
    except TypeError as te:
        print(f"TypeError: Wrong input type ({te})")
        raise TypeError
    except ValueError as ve:
        print(f"ValueError: Wrong input value ({ve})")
        raise ValueError
    except ZeroDivisionError as ze:
        print(f"ZeroDivisionError ({ze})")
    else:
        return k


def calc_strength_increase(
    root_area_ratio,
    max_tangential_stress,
    shear_band_thickness,
    elastic_modulus,
    diameter,
    root_inclination,
    friction_angle,
    normal_stress,
):
    k = __get_k(max_tangential_stress, shear_band_thickness, elastic_modulus, diameter)
    part1 = root_area_ratio * k
    part2 = np.sqrt(secant(root_inclination) - 1)
    part3 = np.sin(np.deg2rad(root_inclination)) + np.cos(
        np.deg2rad(root_inclination)
    ) * np.tan(np.deg2rad(friction_angle))

    return part1 * part2 * part3
