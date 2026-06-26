import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.special import gamma
from scipy.optimize import minimize, differential_evolution
from pyrootmemo.helpers import units, Parameter, create_quantity, solve_quadratic, solve_cubic
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.tools.utils_rotation import axisangle_rotate
from pyrootmemo.tools.utils_plot import round_range
from pint import Quantity

class _DirectShear():
    """Base class for direct shear displacement-driven models
    
    Serves as a base clas for models in which reinforcement is mobilised
    as function of direct shear displacement, such as the different iterations
    of Waldron's models, or DRAM.

    An addition to the following attributes, also sets some useful attributes
    to the input arguments:

    roots.orientation
        unit vector describing the initial orientation of each roots in 
        'roots'. numpy array with size (number of roots, 3)
    failure_surface.tanphi
        the value of tan(friction angle) for the soil that is present at the
        failure surface

    Attributes
    ----------
    roots
        MultipleRoots object
    interface
        Interface object 
    soil_profile
        SoilProfile object
    failure_surface
        FailureSurface object
    distribution_factor
        Distribution factor for assigning root elongation to pullout 
        displacement

    Methods
    -------
    TODO: update methods
    __init__(roots, interface, soil_profile, failure_surface, distribution_factor, **kwargs)
        Constructor
    get_initial_root_orientations()
        Defined initial orientations of all roots relative to the shear zone
    get_orientation_parameters(displacement, shear_zone_thickness, jac)
        Calculate root elongations in shear zone and k-factors 
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            distribution_factor: float | int = 0.5
    ):
        """Initialiser for direct shear models.

        Parameters
        ----------
        roots : MultipleRoots
            MultipleRoots object, containing root properties
        interface : Interface
            Interface object, containing properties of root--soil interface
        soil_profile : SoilProfile
            SoilProfile object
        failure_surface : FailureSurface
            FailureSurface object
        distribution_factor : float | int, optional
            distribution factor determining how much of the root elongation in
            the shear zone to assign to each side, by default 0.5. 0.5 
            corresponds with symmetry, i.e. root segments on either side of
            the shear zone behave identically in terms of mobilising forces

        Raises
        ------
        TypeError
            _description_
        """
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be instance of class MultipleRoots')
        self.roots = roots
        if not isinstance(interface, Interface):
            raise TypeError('interface must be instance of class Interface')
        self.interface = interface
        if not isinstance(soil_profile, SoilProfile):
            raise TypeError('soil_profile must be instance of class SoilProfile')
        self.soil_profile = soil_profile
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be instance of class FailureSurface')
        self.failure_surface = failure_surface
        if not(isinstance(distribution_factor, int) | isinstance(distribution_factor, float)):
            raise TypeError('distribution factor must be int or float')
        self.distribution_factor = distribution_factor
        self.roots.orientation = self.calc_initial_root_orientations()
        self.failure_surface.tanphi = np.tan(
            soil_profile
            .get_soil(failure_surface.depth)
            .friction_angle
            .to('rad')
            )
    
    def calc_initial_root_orientations(
            self
            ) -> np.ndarray:
        """Calculate initial root orientations relative to the shear direction

        Orientations are defined as 3-dimensional orientations **relative to** 
        the failure surface so that:

        * local x = direction of shearing
        * local y = perpendicular to x on shear plane
        * local z = pointing downwards into the soil
        
        Orientations are defined in terms of 3-dimensional unit vectors.
                
        The object 'roots' may contain some information about the **global**
        initial orientation of the roots. This is defined in a **global** 
        right-handed Cartesian coordinate system, with z-axis pointing down 
        into the ground. Orientations are assumed to be defined in a spherical
        coordinate system where:
        
        * azimuth angle = angle from x-axis to projection of root vector on 
          the x-y plane
        * elevation angle = angle from z-axis to root vector
        
        If the initial root orientations are not defined, it is assume they 
        are all **perpedicular** to the shear zone.

        The direction of failure failure surface (taken from FailureSurface 
        object) is assumed to be defined as the angle of the surface in x-z 
        plane, defined (positive) from x to z, i.e. the 'dip angle'.

        Returns
        -------
        np.ndarray
            Numpy array with size (nroots, 3) with the relative 3-D root 
            orientations defined as unit vectors
        """
        # shape of root vector ('number of roots')
        roots_shape = self.roots.diameter.magnitude.shape
        # root orientations not defined - assume all perpendicular to failure surface
        if (not hasattr(self.roots, 'azimuth_angle')) & (not hasattr(self.roots, 'elevation_angle')):
            return(np.stack((
                np.zeros(*roots_shape),
                np.zeros(*roots_shape),
                np.ones(*roots_shape)                    
                ), axis = -1))
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
            # rotate and return
            return(axisangle_rotate(root_orientation_global, axisangle))


    def calc_pullout_displacement(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5,
            jacobian: bool = False
            ) -> dict:
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            ones = np.ones(*self.roots.xsection.shape)
            dict_out = {'pullout_displacement': distribution_factor * shear_displacement * ones}
        else:
            length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            dict_out = {'pullout_displacement': distribution_factor * (length - length_initial)}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                dict_out['dpullout_displacement_dshear_displacement'] = distribution_factor * ones * units('mm/mm')
                dict_out['dpullout_displacement_dshear_zone_thickness'] = 0.0 * ones * units('mm/mm')
            else:
                dict_out['dpullout_displacement_dshear_displacement'] = distribution_factor * length_x / length
                dict_out['dpullout_displacement_dshear_zone_thickness'] = distribution_factor * length / shear_zone_thickness
        return(dict_out)


    def calc_orientation_factor(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jacobian: bool = False
            ) -> dict:
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            ones = np.ones(*self.roots.xsection.shape)
            dict_out = {'k': ones}
        else:
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            dict_out = {'k': (length_x + length_z * self.failure_surface.tanphi) / length}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                dict_out['dk_dshear_displacement'] = 0.0 * ones / shear_displacement.units
                if np.isclose(shear_displacement.magnitude, 0.0):
                    dict_out['dk_dshear_zone_thickness'] = 0.0 * ones / shear_zone_thickness.units
                else:
                    dict_out['dk_dshear_zone_thickness'] = -np.inf * ones / shear_zone_thickness.units
            else:
                dict_out['dk_dshear_displacement'] = 1.0 / length - dict_out['k'] * length_x / length**2
                dict_out['dk_dshear_zone_thickness'] = -shear_displacement / (shear_zone_thickness * length)
        return(dict_out)


    def calc_shear_from_pullout_displacement(
            self,
            pullout_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5
            ) -> Quantity:
        elongation = pullout_displacement / distribution_factor
        length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
        length = length_initial + elongation                        
        length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
        length_z = shear_zone_thickness
        length_x = np.sqrt(length**2 - length_y**2 - length_z**2)
        return(length_x - shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2])


    def calc_orientation_parameters(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5,
            jac: bool = False
            ) -> dict:
        """Calculate root pullout displacement and k-factor

        Calculates the pull-out displacement and the WWM orientation factor k
        for each root. 

        The pull-out displacement is defined as the axial movement of a 
        (segment of) root on one side of the shear zone. 

        The WWM orientation factor k is defined as the ratio between the amount 
        of root reinforcement each root generates (in terms of force) and the 
        current tensile force in that root.

        This function requires the root orientation - relative to the shear
        direction - to be known (in terms of a unit vector).
        
        Parameters
        ----------
        shear_displacement : Quantity
            Current level of shear displacement (scalar)
        shear_zone_thickness : Quantity
            shear zone thickness (scalar)
        distribution_factor : int | float, optional
            assumed ratio between root pull-out displacement and root 
            elongation within the shear zone, by default 0.5. When 0.5 means
            root segments on either side of the shear zone pull out by the
            same amount
        jac : bool, optional
            If True, return derivatives of pull-out displacement and k-factors
            with respect to shear displacement and shear zone thickness. By 
            default False

        Returns
        -------
        dict
            dictionary with fields:

            * 'pullout_displacement': level of pull-out displacement for each 
              root
            * 'k': WWM orientation factor for each root
            * 'dup_dus': derivative of pull-out displacement with respect to
              the shear displacement. Only returned when jac = True.
            * 'dup_dh': derivative of pull-out displacement with respect to
              the shear zone thickness. Only returned when jac = True. 
            * 'dk_dus': derivative of orientation factor k with respect to
              the shear displacement. Only returned when jac = True.
            * 'dk_dh': derivative of orientation factor k with respect to
              the shear zone thickness. Only returned when jac = True.

        """
        init_vector_x = (
            shear_zone_thickness
            * self.roots.orientation[..., 0]
            / self.roots.orientation[..., 2]
            )
        init_vector_y = (
            shear_zone_thickness
            * self.roots.orientation[..., 1]
            / self.roots.orientation[..., 2]
        )
        init_vector_z = shear_zone_thickness * np.ones_like(init_vector_x)
        if shear_zone_thickness.magnitude >= 0.0:
            init_length = shear_zone_thickness / self.roots.orientation[..., 2]
            displaced_length = np.sqrt(
                (init_vector_x + shear_displacement)**2 
                + init_vector_y**2 
                + init_vector_z**2
                )
        else:
            init_length = 0.0 * shear_zone_thickness * self.roots.orientation[..., 2]
            displaced_length = shear_displacement * np.ones_like(init_vector_x)
        pullout_displacement = distribution_factor * (displaced_length - init_length)
        k = (
            (init_vector_x + shear_displacement) 
            + (init_vector_z * self.failure_surface.tanphi)
            ) / displaced_length
        if jac is False:
            return({
                'pullout_displacement': pullout_displacement,
                'k': k
                })
        else:
            # calculate derivatives with respect to:
            # * shear displacement: us
            # * shear zone thickness: h
            divx_dh = self.roots.orientation[..., 0] / self.roots.orientation[..., 2]
            divy_dh = self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            divz_dh = np.ones_like(init_vector_z)
            if shear_zone_thickness.magnitude >= 0.0:
                dL0_dh = 1.0 / self.roots.orientation[..., 2]
                dL_dus = (init_vector_x + shear_displacement) / displaced_length
                dL_dv0x = (init_vector_x + shear_displacement) / displaced_length
                dL_dv0y = init_vector_y / displaced_length
                dL_dv0z = init_vector_z / displaced_length
            else:
                dL0_dh = 0.0 * self.roots.orientation[..., 2]
                dL_dus = np.ones_like(init_vector_z)
                dL_dv0x = np.ones_like(init_vector_z)
                dL_dv0y = np.ones_like(init_vector_z)
                dL_dv0z = np.ones_like(init_vector_z)
            dup_dus = distribution_factor * dL_dus
            dL_dh = (
                dL_dv0x * divx_dh
                + dL_dv0y * divy_dh
                + dL_dv0z * divz_dh
                )
            dup_dh = distribution_factor * (dL_dh - dL0_dh)
            dk_dus = (
                1.0 / displaced_length
                - k / displaced_length * dL_dus
            )
            dk_dh = (
                (divx_dh + divz_dh * self.failure_surface.tanphi) 
                / displaced_length
                - k / displaced_length * dL_dh
            )
            return({
                'pullout_displacement': pullout_displacement,
                'k': k,
                'dup_dus': dup_dus,
                'dup_dh': dup_dh,
                'dk_dus': dk_dus,
                'dk_dup': dk_dh
                })