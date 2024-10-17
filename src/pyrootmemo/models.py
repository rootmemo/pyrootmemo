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





class Waldron1977_gjm:
    def __init__(
        self,
        soil: pyrootmemo.materials.Soil,
        roots: pyrootmemo.materials.SingleRoot | pyrootmemo.materials.MultipleRoots,
    ):
        self.soil = soil
        self.roots = roots
    
    # Pullout displacement for one half of the root
    def pullout_displacement(self, shear_displacement, shear_band_thickness):
        # initial length of root within shear band
        L0 = shear_band_thickness
        # displaced length of root within shear band
        L1 = np.sqrt(shear_displacement**2 + shear_band_thickness**2)
        # return displacement - for one side only
        return(0.5 * (L1 - L0))
    
    # Root tensile force, as function of pullout displacement
    def pullout_force(self, pullout_displacement, interface_resistance):
        # calculate max root tensile force, assuming no failure
        force = np.sqrt(
            2.
            * self.roots.elastic_modulus,
            * self.roots.xsection
            * self.roots.circumference
            * interface_resistance
            * pullout_displacement
            )
        # account for breakage
        force_max = self.roots.tensile_strength * self.roots.xsection
        force[force > force_max] = 0.
        # return
        return(force)
    
    # ratio between reinforcing force and root tensile force
    def orientation_factor(self, shear_displacement, shear_band_thickness):
        sin_beta = shear_displacement / np.sqrt(
            shear_displacement**2 + shear_band_thickness**2)
        cos_beta = shear_band_thickness / np.sqrt(
            shear_displacement**2 + shear_band_thickness**2)
        return(sin_beta 
               + cos_beta*np.tan(np.deg2rad(self.soil.friction_angle)))

    # calculate reinforcement as function of known shear displacement
    def reinforcement(
            self, 
            shear_displacement, 
            root_area_ratio,
            interface_resistance,
            shear_band_thickness,
            ):
        pullout_displacement = self.pullout_displacement(shear_displacement)
        orientation_factor = self.orientation_factor(shear_displacement, shear_band_thickness)
        pullout_force = self.pullout_force(pullout_displacement, interface_resistance)
        reinforcement_force = pullout_force * orientation_factor
        total_root_area = np.sum(self.roots.xsection)
        return(np.sum(reinforcement_force) / total_root_area * root_area_ratio)
    
    # calculate pull-out displacement at known root tensile force
    def force2pullout_displacement(self, pullout_force, interface_resistance):
        return(0.5
               * pullout_force**2
               / self.roots.elastic_modulus
               / self.roots.xsection
               / self.roots.circumference
               / interface_resistance
               )
    
    # calculate shear displacement based on known pullout displacement
    def pullout2shear_displacement(self, pullout_displacement, shear_band_thickness):
        return(2. * np.sqrt(pullout_displacement 
                            * (pullout_displacement + shear_band_thickness)))
    
    # calculate shear displacement at moment of root breakage
    def sheardisplacement_breakage(self, shear_band_thickness, interface_resistance):
        tensile_force = self.roots.tensile_strength * self.roots.xsection
        pullout_displacement = self.force2pullout_displacement(tensile_force, interface_resistance)
        return(self.pullout2shear_displacement(pullout_displacement, shear_band_thickness))
    
    # calculate peak reinforcement
    def peak_reinforcement(
            self, 
            shear_band_thickness, 
            interface_resistance,
            root_area_ratio
            ):
        # get shear displacements at root tensile failures
        shear_displacement = self.sheardisplacement_breakage(
            shear_band_thickness, interface_resistance)
        # at each shear displacement, get force in each root
        pullout_displacement = self.pullout_displacement(shear_displacement, shear_band_thickness)
        pullout_force = np.array([self.pullout_force(u_s, interface_resistance) for u_s in pullout_displacement])
        # k factor
        orientation_factor = self.orientation_factor(shear_displacement, shear_band_thickness)
        # reinforcement per root, per step
        reinforcement_root_step = pullout_force * orientation_factor
        # reinforcement per displacement step
        reinforcement_step = np.sum(reinforcement_root_step, axis = 0)
        # return shear displacement and peak reinforcement
        return(
            shear_displacement[np.argmax(reinforcement_step)],
            np.max(reinforcement_step)
        )
    