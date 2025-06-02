# import packages and functions
import numpy as np
from pyrootmemo.tools.helpers import units
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.pullout import PulloutEmbeddedElastic
from pyrootmemo.pullout import PulloutEmbeddedElasticSlipping
from pyrootmemo.pullout import PulloutEmbeddedElasticBreakage
from pyrootmemo.pullout import PulloutEmbeddedElasticBreakageSlipping
from pyrootmemo.pullout import PulloutEmbeddedElastoplastic
from pyrootmemo.pullout import PulloutEmbeddedElastoplasticSlipping
from pyrootmemo.pullout import PulloutEmbeddedElastoplasticBreakage
from pyrootmemo.pullout import PulloutEmbeddedElastoplasticBreakageSlipping
from pyrootmemo.utils_rotation import axisangle_rotate
from pint import Quantity
import warnings


class DirectShear():

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            distribution_factor: float | int = 0.5,
            **kwargs
    ) -> None:
        # check instances are of the correct class
        if not isinstance(roots, MultipleRoots):
            TypeError('roots must be instance of class MultipleRoots')
        if not isinstance(interface, Interface):
            TypeError('interface must be instance of class Interface')
        if not isinstance(soil_profile, SoilProfile):
            TypeError('soil_profile must be instance of class SoilProfile')
        if not isinstance(failure_surface, FailureSurface):
            TypeError('failure_surface must be instance of class FailureSurface')
        # assign input
        self.roots = roots
        self.interface = interface
        self.soil_profile = soil_profile
        self.failure_surface = failure_surface
        # distribution parameter
        self.distribution_factor = distribution_factor
        # set root orientations (relative to failure surface)
        self._set_root_orientations()
        # set friction angle at failure surface
        self.failure_surface.tanphi = np.tan(
            soil_profile.get_soil(failure_surface.depth).friction_angle.to('rad')
            )
        # kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    
    def _set_root_orientations(
            self
            ) -> None:
        # function set 3-D root orientations **relative to** failure surface so that:
        # * local x = direction of shearing
        # * local y = perpendicular to x on shear plane
        # * local z = pointing downwards into the soil
        # orientations are defined in terms of 3-dimensional unit vectors
        #         
        # global coordinate system
        # * right-handed Cartesian coordinate system, with z-axis pointing down into the ground
        # * azimuth angle = angle from x-axis to projection of root vector on the x-y plane
        # * elevation angle = angle from z-axis to root vector
        #
        # failure surface
        # * assume angle is defined as angle in x-z plane, defined (positive) from x to z
        #
        # 
        # shape of root vector ('number of roots')
        roots_shape = self.roots.diameter.magnitude.shape
        # root orientations not defined - assume all perpendicular to failure surface
        if (not hasattr(self.roots, 'azimuth_angle')) & (not hasattr(self.roots, 'elevation_angle')):
            self.roots.orientation = np.stack((
                np.zeros(*roots_shape),
                np.zeros(*roots_shape),
                np.ones(*roots_shape)                    
                ), axis = -1)
        # (partial) angles provided -> rotate to local coordinate system
        else:
            if not hasattr(self.roots, 'azimuth_angle'):
                self.roots.azimuth_angle = np.zeros(*roots_shape) * units('deg')
            if not hasattr(self.roots, 'elevation_angle'):
                self.roots.elevation_angle = np.zeros(*roots_shape) * units('deg')
            # get global root orientations
            root_orientation_global = np.stack((
                np.cos(self.roots.azimuth_angle.magnitude) 
                * np.sin(self.roots.elevation_angle.magnitude),
                np.sin(self.roots.azimuth_angle.magnitude) 
                * np.sin(self.roots.elevation_angle.magnitude),
                np.cos(self.roots.elevation_angle.magnitude)
            ), axis = -1)
            # rotate to local coordinate system and set unit vectors
            if hasattr(self.failure_surface, 'orientation'):
                axisangle = np.array([0.0, -self.failure_surface.orientation.to('rad'), 0.0])
            else:
                axisangle = np.array([0.0, 0.0, 0.0])
            self.roots.orientation = axisangle_rotate(root_orientation_global, axisangle)

    def _get_orientation_parameters(
            self,
            displacement: Quantity,
            shear_zone_thickness: Quantity,
            jac: bool = False
    ):
        # vector components of initial root orientation in shear zone
        v0x = (
            shear_zone_thickness
            * self.roots.orientation[..., 0]
            / self.roots.orientation[..., 2]
            )
        v0y = (
            shear_zone_thickness
            * self.roots.orientation[..., 1]
            / self.roots.orientation[..., 2]
        )
        v0z = shear_zone_thickness * np.ones_like(v0x)
        # length in shear zone
        if shear_zone_thickness.magnitude >= 0.0:
            L0 = shear_zone_thickness / self.roots.orientation[..., 2]
            L = np.sqrt((v0x + displacement)**2 + v0y**2 + v0z**2)
        else:
            L0 = 0.0 * shear_zone_thickness * self.roots.orientation[..., 2]
            L = displacement * np.ones_like(v0x)
        # pullout displacement
        up = self.distribution_factor * (L - L0)
        # orientation factor
        k = ((v0x + displacement) + (v0z * self.failure_surface.tanphi)) / L
        # derivatives
        if jac is False:
            dup_ddisplacement = None
            dup_dshearzonethickness = None
            dk_ddisplacement = None
            dk_dshearzonethickness = None
        else:
            dv0x_dshearzonethickness = self.roots.orientation[..., 0] / self.roots.orientation[..., 2]
            dv0y_dshearzonethickness = self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            dv0z_dshearzonethickness = np.ones_like(v0z)
            if shear_zone_thickness.magnitude >= 0.0:
                dL0_dshearzonethickness = 1.0 / self.roots.orientation[..., 2]
                dL_ddisplacement = (v0x + displacement) / L
                dL_dv0x = (v0x + displacement) / L
                dL_dv0y = v0y / L
                dL_dv0z = v0z / L
            else:
                dL0_dshearzonethickness = 0.0 * self.roots.orientation[..., 2]
                dL_ddisplacement = np.ones_like(v0z)
                dL_dv0x = np.ones_like(v0z)
                dL_dv0y = np.ones_like(v0z)
                dL_dv0z = np.ones_like(v0z)
            dup_ddisplacement = self.distribution_factor * dL_ddisplacement
            dL_dshearzonethickness = (
                dL_dv0x * dv0x_dshearzonethickness
                + dL_dv0y * dv0y_dshearzonethickness
                + dL_dv0z * dv0z_dshearzonethickness
                )
            dup_dshearzonethickness = self.distribution_factor * (dL_dshearzonethickness - dL0_dshearzonethickness)
            dk_ddisplacement = (
                1.0 / L
                - k / L * dL_ddisplacement
            )
            dk_dshearzonethickness = (
                (dv0x_dshearzonethickness + dv0z_dshearzonethickness * self.failure_surface.tanphi) / L
                - k / L * dL_dshearzonethickness
            )
        return(
            up,
            k,
            dup_ddisplacement, 
            dup_dshearzonethickness,
            dk_ddisplacement,
            dk_dshearzonethickness
        )


# Waldron class
class Waldron(DirectShear):

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            slipping: bool = True,
            breakage: bool = True,
            elastoplastic: bool = False,
            weibull_shape: float | int | None = None,
            **kwargs
    ) -> None: 
        # call __init__ from parent class
        super().__init__(roots, interface, soil_profile, failure_surface, **kwargs)
        # set analysis settings as part of class
        self.slipping = slipping
        self.breakage = breakage
        self.elastoplastic = elastoplastic
        # set correct pullout object, depending on cases (slipping, breakage etc)
        if slipping is True:
            if breakage is True:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplasticBreakageSlipping(
                        roots, interface, weibull_shape = weibull_shape)
                else:
                    self.pullout = PulloutEmbeddedElasticBreakageSlipping(
                        roots, interface, weibull_shape = weibull_shape)
            else:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplasticSlipping(
                        roots, interface)
                else:
                    self.pullout = PulloutEmbeddedElasticSlipping(
                        roots, interface)
        else:
            if breakage is True:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplasticBreakage(
                        roots, interface, weibull_shape = weibull_shape)
                else:
                    self.pullout = PulloutEmbeddedElasticBreakage(
                        roots, interface, weibull_shape = weibull_shape)
            else:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplastic(
                        roots, interface)
                else:
                    self.pullout = PulloutEmbeddedElastic(
                        roots, interface)

    def reinforcement(
            self,
            displacement: Quantity,
            displacement_unit: str = 'm',
            jac: bool = False,
            ) -> Quantity:
        # if displacement has no unit, add one
        if not isinstance(displacement, Quantity):
            displacement = displacement * units(displacement_unit)
        # pullout displacement (up) and orientation factors (k)
        up, k, dup_dus, dup_dh, dk_dus, dk_dh = self._get_orientation_parameters(
            displacement,
            self.failure_surface.shear_zone_thickness
        )
        # pullout force (Tp), survival fraction (S) and behaviour type index (b)
        Tp, dTp_dup, S, b = self.pullout.force(up, jac = jac)
        # reinforcement
        cr = np.sum(k * Tp) / self.failure_surface.cross_sectional_area
        # fraction of root cross-sectional area for each behaviour types
        xsection_rel = self.roots.xsection.magnitude / np.sum(self.roots.xsection.magnitude)
        xsection_fractions = np.bincount(
            b, 
            weights = S * xsection_rel,
            minlength = len(self.pullout.behaviour_types)
            )
        # return jacobian if requested
        if jac is True:
            dcr_dus = (
                np.sum(dk_dus * Tp + k * dTp_dup * dup_dus) 
                / self.failure_surface.cross_sectional_area
                )
        else:
            dcr_dus = None
        # return
        return(cr, dcr_dus, xsection_fractions)


    def _get_displacement_root_peak(self):
        # get force at failure in each root
        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
        if self.slipping is True:
            force_slipping = self.roots.length * self.roots.circumference * self.interface.shear_strength
        # select limiting case
        if self.breakage is True:
            if self.slipping is True:
                force = np.minimum(force_breakage, force_slipping)
            else:
                force = force_breakage
        else:
            if self.slipping is True:
                force = force_slipping
            else:
                return(None)
        # get index of correct behaviour
        if self.elastoplastic is False:
            i = np.flatnonzero(self.pullout.behaviour_types == 'Anchored, elastic')[0]
            pullout_displacement = (
                self.pullout.coefficients[0][..., i] * force**2
                + self.pullout.coefficients[1][..., i] * force
                + self.pullout.coefficients[2][..., i]
                )
        else:
            i = np.flatnonzero(self.pullout.behaviour_types == 'Anchored, plastic')[0]
            print('Not yet implemented')
        # calculate shear displacement at failure
        elongation = pullout_displacement / self.distribution_factor
        vx = self.roots.orientation[..., 0]
        vy = self.roots.orientation[..., 1]
        vz = self.roots.orientation[..., 2]
        h = self.failure_surface.shear_zone_thickness
        shear_displacement = (
            np.sqrt((elongation + h / vz)**2 - h**2 * (1.0 + vy**2 / vz**2))
            - h * vx / vz
        )
        # return
        return(shear_displacement)


    def peak_reinforcement(
            self
            ):
        # no root breakage or slipping -> infinite reinforcement
        if self.breakage is False and self.slipping is False:
            warnings.warn('No breakage or slipping - peak reinforcement is infinite!')
            return(np.inf * units('kPa'))
