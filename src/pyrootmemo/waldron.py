# import packages and functions
import numpy as np
from pyrootmemo.tools.helpers import units
from pyrootmemo.pullout import Pullout_embedded_elastic
from pyrootmemo.pullout import Pullout_embedded_elastic_slipping
from pyrootmemo.pullout import Pullout_embedded_elastic_breakage
from pyrootmemo.pullout import Pullout_embedded_elastic_breakage_slipping
from pyrootmemo.pullout import Pullout_embedded_elastoplastic
from pyrootmemo.pullout import Pullout_embedded_elastoplastic_slipping
from pyrootmemo.pullout import Pullout_embedded_elastoplastic_breakage
from pyrootmemo.pullout import Pullout_embedded_elastoplastic_breakage_slipping
from pint import Quantity


# Waldron class
class Waldron():

    # initialise class
    def __init__(
            self, 
            shearzone,
            roots,
            soil,
            interface,
            slipping = True,
            breakage = True,
            plastic = False,
            weibull_shape = None
            ) -> None:
        # set parameters
        self.shearzone = shearzone
        self.roots = roots
        self.soil = soil
        self.interface = interface
        # set root orientation if not defined
        if not hasattr(roots, 'azimuth_angle'):
            roots.azimuth_angle = np.zeros_like(roots.diameter) * units('degrees')
        if not hasattr(roots, 'elevation_angle'):
            roots.elevation_angle = np.zeros_like(roots.diameter) * units('degrees')
        # get initial orientation vector for all roots, relative to shearzone
        roots.orientation = roots.initial_orientation_vector(
            axis_angle = shearzone.orientation if hasattr(shearzone, 'orientation') else None
        )
        # generate pullout object
        if plastic is True:
            if slipping is True:
                if breakage is True:
                    self.pullout = Pullout_embedded_elastoplastic_breakage_slipping(
                        roots, 
                        self.interface.shear_strength,
                        weibull_shape = weibull_shape
                    )
                else:
                    self.pullout = Pullout_embedded_elastoplastic_slipping(
                        roots, 
                        self.interface.shear_strength
                        )
            else:
                if breakage is True:
                    self.pullout = Pullout_embedded_elastoplastic_breakage(
                        roots, 
                        self.interface.shear_strength,
                        weibull_shape = weibull_shape
                    )
                else:
                    self.pullout = Pullout_embedded_elastoplastic(
                        roots, 
                        self.interface.shear_strength
                        )
        else:
            if slipping is True:
                if breakage is True:
                    self.pullout = Pullout_embedded_elastic_breakage_slipping(
                        roots, 
                        self.interface.shear_strength,
                        weibull_shape = weibull_shape
                    )
                else:
                    self.pullout = Pullout_embedded_elastic_slipping(
                        roots, 
                        self.interface.shear_strength
                        )
            else:
                if breakage is True:
                    self.pullout = Pullout_embedded_elastic_breakage(
                        roots, 
                        self.interface.shear_strength,
                        weibull_shape = weibull_shape
                    )
                else:
                    self.pullout = Pullout_embedded_elastic(
                        roots, 
                        self.interface.shear_strength
                        )

    # x,y,z components of root length within shearzone
    def _root_length_in_shearzone(
            self,
            shear_displacement = 0.0
    ):
        # initial length of vector components within shearzone
        vx = (
            self.shearzone.thickness
            * self.roots.orientation[..., 0]
            / self.roots.orientation[..., 2]
            )
        vy = (
            self.shearzone.thickness
            * self.roots.orientation[..., 1]
            / self.roots.orientation[..., 2]
        )
        vz = self.shearzone.thickness * np.ones_like(self.roots.orientation[..., 2])
        # return
        return(np.stack([vx + shear_displacement, vy, vz], axis = 1))

    # calculate elongation in roots based on current shear displacement
    def _root_elongation(
            self, 
            shear_displacement,
            distribution = 0.5
            ):
        length_initial = np.linalg.norm(
            self._root_length_in_shearzone(shear_displacement = 0.0),
            axis = 1
        )
        length_displaced = np.linalg.norm(
            self._root_length_in_shearzone(shear_displacement = shear_displacement),
            axis = 1
        )
        return(distribution * (length_displaced - length_initial))

    # calculate reinforcing force
    def _shear_force(
            self,
            shear_displacement,
            force,
    ):
        length_components = self._root_length_in_shearzone(shear_displacement = shear_displacement)
        length_displaced = np.linalg.norm(length_components, axis = -1)
        force_components = length_components * (force / length_displaced)[..., np.newaxis]
        return(
            force_components[..., 0] + 
            force_components[..., 1] * np.tan(self.soil.friction_angle)
        )        

    # reinforcement at current level of shear displacement
    def reinforcement(displacement):
        None

    # peak reinforcement
    def peak_reinforcement():
        None
