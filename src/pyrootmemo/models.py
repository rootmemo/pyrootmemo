import numpy as np
import pyrootmemo.materials
from pyrootmemo.tools.helpers import secant


class Waldron1977:
    def __init__(
        self,
        soil: pyrootmemo.materials.Soil,
        roots: pyrootmemo.materials.SingleRoot | pyrootmemo.materials.MultipleRoots,
    ):
        if soil.friction_angle is not None:
            self.soil = soil
        else:
            raise ValueError("Set a friction angle for soil")
        if (roots.diameter is not None) & (roots.elastic_modulus is not None):
            self.roots = roots
        else:
            raise ValueError("Set a diameter and elastic modulus for the root")

    def __get_k(
        self,
        max_tangential_stress: int | float,
        shear_band_thickness: int | float,
    ):
        try:
            k = np.sqrt(
                4
                * max_tangential_stress
                * shear_band_thickness
                * self.roots.elastic_modulus
                / self.roots.diameter
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

    def __calc_pullout_force(self, shear_displacement):
        test = None

    def __calc_inclination(self, shear_displacement, shear_band_thickness):
        return np.rad2deg(np.arctan(shear_displacement / shear_band_thickness))

    def calc_strength_increase(
        self,
        root_area_ratio,
        max_tangential_stress,
        shear_band_thickness,
        shear_displacement: int | float | np.ndarray,
    ):
        try:
            if isinstance(shear_displacement, np.ndarray):
                shear_band_thickness = (
                    np.ones_like(shear_displacement) * shear_band_thickness
                )
            elif isinstance(shear_displacement, (float or int)):
                shear_band_thickness = shear_band_thickness
            else:
                raise TypeError(
                    "shear_displacement can be of type int, float or np.ndarray"
                )
            self.roots.inclination = self.__calc_inclination(
                shear_displacement, shear_band_thickness
            )
            k = self.__get_k(max_tangential_stress, shear_band_thickness)
            part1 = root_area_ratio * k
            part2 = np.sqrt(secant(self.roots.inclination) - 1)
            part3 = np.sin(np.deg2rad(self.roots.inclination)) + np.cos(
                np.deg2rad(self.roots.inclination)
            ) * np.tan(np.deg2rad(self.soil.friction_angle))
        except AttributeError as ae:
            print(f"AttributeError: Missing attributes ({ae})")
        else:
            return part1 * part2 * part3
