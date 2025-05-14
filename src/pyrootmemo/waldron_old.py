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
            shear_displacement = 0.0,
            jac = False
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
        L = np.stack([vx + shear_displacement, vy, vz], axis = 1)
        if jac is False:
            return(L)
        else:
            dL_dvxyz = np.stack([
                np.ones_like(vx),
                np.zeros_like(vy),
                np.zeros_like(vz)
            ], axis = -1)
            return(L, dL_dvxyz)


    # calculate elongation in roots based on current shear displacement
    def _pullout_displacement(
            self, 
            shear_displacement,
            distribution = 0.5,
            jac = False
            ):
        vxyz0 = self._root_length_in_shearzone(shear_displacement = 0.0)
        length0 = np.linalg.norm(vxyz0, axis = -1)
        if jac is False:
            vxyz1 = self._root_length_in_shearzone(shear_displacement = shear_displacement)
        else:
            vxyz1, dvxyz1_shear = self._root_length_in_shearzone(
                shear_displacement = shear_displacement, jac = True
                )
        length1 = np.linalg.norm(vxyz1, axis = -1)
        pullout_displacement = distribution * (length1 - length0)
        if jac is False:
            return(pullout_displacement)
        else:
            dlength1_dvxyz1 = vxyz1 / length1[..., np.newaxis]
            dlength1_dshear = np.tensordot(dlength1_dvxyz1, dvxyz1_shear, axes = (-1, -1))
            dpullout_dlength1 = distribution
            dpullout_dshear = dpullout_dlength1 * dlength1_dshear
            return(
                pullout_displacement,
                dpullout_dshear
            )        

    # calculate reinforcing force
    def _shear_reinforcement(
            self,
            shear_displacement,
            force,
            jac = False
    ):  
        if jac is False:
            vxyz = self._root_length_in_shearzone(shear_displacement = shear_displacement)
        else:
            vxyz, dvxyz_dshear = self._root_length_in_shearzone(
                shear_displacement = shear_displacement, jac = True
                )
        length1 = np.linalg.norm(vxyz, axis = -1)
        Fxyz = vxyz * (force / length1)[..., np.newaxis]
        shearforce = (
            Fxyz[..., 0]
            + Fxyz[..., 2] * np.tan(self.soil.friction_angle)
        )
        if jac is False:
            return(shearforce / self.soil.area)
        else:
            dlength1_dvxyz = vxyz / length1[..., np.newaxis]
            dlength1_dshear = np.tensordot(dlength1_dvxyz, dvxyz_dshear, axes = (-1, -1))
            dFxyz_dlength1 = -vxyz * (force / length1**2)[..., np.newaxis]
            dFxyz_dforce = vxyz / length1[..., np.newaxis]
            dshearforce_force = (
                dFxyz_dforce[..., 0]
                + dFxyz_dforce[..., 2] * np.tan(self.soil.friction_angle)
            )
            dshearforce_dlength1 = (
                dFxyz_dlength1[..., 0]
                + dFxyz_dlength1[..., 2] * np.tan(self.soil.friction_angle)
            )
            dshearforce_dshear = dshearforce_dlength1 * dlength1_dshear
            return(
                shearforce / self.soil.area,
                dshearforce_dshear / self.soil.area,
                dshearforce_force / self.soil.area,
            )


    # reinforcement at current level of shear displacement
    def reinforcement(
            self,
            shear_displacement, 
            jac = False   
            ):
        if jac is False:
            pullout_displacement = self._pullout_displacement(shear_displacement)
            force, survival, behaviour_type = self.pullout.force(pullout_displacement)
            reinforcement = self._shear_reinforcement(shear_displacement, force)
            return(np.sum(reinforcement))
        else:
            pullout_displacement, dpullout_dshear = self._pullout_displacement(
                shear_displacement, jac = True
                )
            force, survival, behaviour_type, dforce_dpullout = self.pullout.force(
                pullout_displacement, jac = True
                )
            reinforcement, dreinforcement_dshear, dreinforcement_dforce = self._shear_reinforcement(
                shear_displacement, force, jac = True
                )
            dreinforcement_dshear = (
                dreinforcement_dshear + 
                dreinforcement_dforce * dforce_dpullout * dpullout_dshear
            )
            return(
                np.sum(reinforcement),
                np.sum(dreinforcement_dshear)
            )            

    # peak reinforcement
    def peak_reinforcement():
        # if no breakage and slippage -> infinite
        # else if weibullshape = None --> discrete breakages
        # get displacement at each root breakage event
        # -> get forces at these events
        # -> find max (but does not have to be max because of orientation effect)
        # -> find 'real' max

        None
