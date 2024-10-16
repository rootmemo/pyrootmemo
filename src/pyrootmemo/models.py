import numpy as np
import pyrootmemo.materials
from pyrootmemo.tools.helpers import secant


class Waldron1977:
    def __init__(
        self, soil: pyrootmemo.materials.Soil, roots: pyrootmemo.materials.SingleRoot
    ):
        if soil.friction_angle is not None:
            self.soil = soil
        else:
            raise ValueError("Set a friction angle for soil")
        if (
            (roots.diameter is not None)
            & (roots.elastic_modulus is not None)
            & (roots.inclination is not None)
        ):
            self.roots = roots
        else:
            raise ValueError(
                "Set a diameter, elastic modulus and inclination for the root"
            )

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

    def calc_strength_increase(
        self,
        root_area_ratio,
        max_tangential_stress,
        shear_band_thickness,
        normal_stress,
    ):
        try:
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
