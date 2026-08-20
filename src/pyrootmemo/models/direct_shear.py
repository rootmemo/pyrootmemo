import numpy as np
from pyrootmemo import Parameter
from pyrootmemo.helpers import units
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.tools.utils_rotation import axisangle_rotate
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
    output : dict
        Dictionary with calculation output

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
        self.output = {}
    
    def calc_initial_root_orientations(
            self
            ) -> np.ndarray:
        """Calculate initial root orientations relative to the shear direction

        Orientations are defined as 3-dimensional orientations **relative to** 
        the failure surface so that:

        * local x = direction of shearing
        * local y = perpendicular to x on shear plane
        * local z = pointing towards the sliding block (upwards)
        
        Orientations are defined in terms of 3-dimensional unit vectors.
                
        The object 'roots' may contain some information about the **global**
        initial orientation of the roots. This is defined in a **global** 
        right-handed Cartesian coordinate system, with z-axis pointing up, 
        towards the ground surface. Orientations are assumed to be defined 
        in a spherical coordinate system where:
        
        * azimuth angle = angle from x-axis to projection of root vector on 
          the x-y plane
        * elevation angle = angle from z-axis to root vector
        
        If the initial root orientations are not defined, it is assume they 
        are all purely vertical.

        The orientation of the failure surface is calculated using the
        `calc_orientation_matrix()` method in the FailureSurface class. This 
        returns a matrix with the local cartesian axes, defined in the global
        system.

        Returns
        -------
        np.ndarray
            Numpy array with size (nroots, 3) with the relative 3-D root 
            orientations defined as unit vectors
        """
        root_global_orientation = self.roots.calc_initial_orientation_vector()
        surface_global_orientation = self.failure_surface.calc_orientation_matrix()
        root_local_orientation = root_global_orientation @ surface_global_orientation
        return(root_local_orientation)

    def calc_pullout_displacement(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5,
            jacobian: bool = False,
            ) -> dict:
        """Calculate pullout displacement from shear displacement

        Calculates how much a root pulls out of the soil on either side of the 
        shearzone, based on the known shear displacement, shear zone thickness
        and initial root orientation.

        Parameters
        ----------
        shear_displacement : Quantity
            Shear displacement (scalar or array)
        shear_zone_thickness : Quantity
            Shear zone thickness (scalar)
        distribution_factor : int | float, optional
            The fraction of the total root length that should be assigned
            to each side of the shearzone, by default 0.5
        jacobian : bool, optional
            if True, also return the derivative of the pullout displacement with
            respect to the shear displacement and shear zone thickness, by 
            default False

        Returns
        -------
        dict
            Dictionary with keys:
            
            * 'pullout_displacement': root pullout displacement, at one side
                of the shearzone
            * 'dpullout_displacement_dshear_displacement': derivative of 
                pullout displacement with respect to the shear displacement
            * 'dpullout_displacement_dshear_zone_thickness': derivative of 
                pullout displacement with respect to the shear zone thickness
        """
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            ones = np.ones(*self.roots.xsection.shape)
            output = {'pullout_displacement': distribution_factor * shear_displacement * ones}
        else:
            length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            output = {'pullout_displacement': distribution_factor * (length - length_initial)}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                output['dpullout_displacement_dshear_displacement'] = distribution_factor * ones * units('mm/mm')
                output['dpullout_displacement_dshear_zone_thickness'] = 0.0 * ones * units('mm/mm')
            else:
                output['dpullout_displacement_dshear_displacement'] = distribution_factor * length_x / length
                output['dpullout_displacement_dshear_zone_thickness'] = distribution_factor * length / shear_zone_thickness
        return(output)


    def calc_orientation_factor(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jacobian: bool = False
            ) -> dict:
        """Calculate root orientation factors k

        Calculates the root orientation factor, based on the known shear 
        displacement, shear zone thickness, initial root orientation and soil
        angle of internal friction.

        The orientation factor 'k' describes the relationship between the
        root tensile force and its reinforcing effect, i.e.:

        k = (reinforcing force) / (tensile force)

        Parameters
        ----------
        shear_displacement : Quantity
            Shear displacement (scalar or array)
        shear_zone_thickness : Quantity
            Shear zone thickness (scalar)
        jacobian : bool, optional
            if True, also return the derivative of the orientatoin factor with
            respect to the shear displacement and shear zone thickness, by 
            default False

        Returns
        -------
        dict
            Dictionary with keys:
            
            * 'k': orientation factor
            * 'dk_dshear_displacement': derivative of orientation factor
                with respect to the shear displacement
            * 'dk_dshear_zone_thickness': derivative of orientation factor
               with respect to the shear zone thickness
        """
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            ones = np.ones(*self.roots.xsection.shape)
            output = {'k': ones}
        else:
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            output = {'k': (length_x + length_z * self.failure_surface.tanphi) / length}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                output['dk_dshear_displacement'] = 0.0 * ones / shear_displacement.units
                if np.isclose(shear_displacement.magnitude, 0.0):
                    output['dk_dshear_zone_thickness'] = 0.0 * ones / shear_zone_thickness.units
                else:
                    output['dk_dshear_zone_thickness'] = -np.inf * ones / shear_zone_thickness.units
            else:
                output['dk_dshear_displacement'] = 1.0 / length - output['k'] * length_x / length**2
                output['dk_dshear_zone_thickness'] = -shear_displacement / (shear_zone_thickness * length)
        return(output)


    def calc_shear_from_pullout_displacement(
            self,
            pullout_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5
            ) -> Quantity:
        """Calculate shear displacement from root pullout displacement

        Calculates how much the soil should move to make a root pull out by
        a certain amount from the soil surrounding the shear zone.

        This function is the inverse of the `calc_pullout_displacement()` 
        method.

        Parameters
        ----------
        pullout_displacement : Quantity
            Root pullout displacement
        shear_zone_thickness : Quantity
            Shear zone thickness
        distribution_factor : int | float, optional
            The fraction of the total root length that should be assigned
            to each side of the shearzone, by default 0.5

        Returns
        -------
        Quantity
            Shear displacement
        """
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
        
    def calc_displaced_orientation(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jacobian: bool = False
            ) -> dict:
        """Calculate the orientation vector of each roots, given known shear 
        displacements and shear zone thicknesses.

        Parameters
        ----------
        shear_displacement : Quantity
            (Current) shear displacement, as a scalar Quantity
        shear_zone_thickness : Quantity
            (Current) shear zone thickness, as a scalar Quantity
        jacobian : bool, optional
            If True, function also returns the partial derivatives of the 
            calculated fractionswith respect to both the shear zone displacement
            and the shear zone thickness. By default False

        Returns
        -------
        dict
            Results dictionary, with keys:
            * orientation: Quantity
              Array (size nroot * 3) with force decomposition fractions for
              every root
            * dorientation_dshear_displacement: Quantity
              Array (size nroot * 3). Only returned if jacobian = True
            * dorientation_dshear_zone_thickness: Quantity
              Array (size nroot * 3). Only returned if jacobian = True
        """
        ones = np.ones_like(self.roots.diameter)
        zeros = np.zeros_like(self.roots.diameter)
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            output = {'orientation': np.stack((ones, zeros, zeros), axis = -1)}
        else:
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x ** 2 + length_y ** 2 + length_z ** 2)
            output = {'orientation': np.stack((
                            length_x / length, 
                            length_y / length, 
                            length_z / length
                            ), axis = -1)}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                output['dorientation_dshear_zone_thickness'] = np.stack((
                    zeros / shear_zone_thickness.units,
                    zeros / shear_zone_thickness.units,
                    zeros / shear_zone_thickness.units
                    ), axis = -1)
                output['dorientation_dshear_displacement'] = np.stack((
                    ones / shear_displacement.units,
                    zeros / shear_displacement.units,
                    zeros / shear_displacement.units
                    ), axis = -1)
            else:
                tmp1 = length_x * shear_displacement / (length**3 * shear_zone_thickness)
                output['dorientation_dshear_zone_thickness'] = np.stack((
                    tmp1 * length_x - shear_displacement / (length * shear_zone_thickness),
                    tmp1 * length_y,
                    tmp1 * length_z
                    ), axis = -1)
                output['dorientation_dshear_displacement'] = np.stack((
                    1.0 / length * (1.0 - length_x**2 / length**2),
                    -length_y * length_x / length**3,
                    -length_z * length_x / length**3
                    ), axis = -1)
        return(output)

    def calc_displacement_to_rootpeak(
            self,
            shear_zone_thickness: Quantity
            ) -> Quantity:
        """Calculate shear displacement at peak reinforcements of individual roots

        Calculate the shear displacement associated with each root reaching its
        maximum tensile force, i.e. at the point of breakage or at the onset
        of slippage. The associated pull-out displacement is calculated, and 
        this is then converted back to shear displacements and returned.
    
        Parameters
        ----------
        shear_zone_thickness : Quantity
            Shear zone thickness

        Returns
        -------
        Quantity
            Array with shear displacements
        """
        pullout_displacement = self.pullout.calc_displacement_to_peak(results = 'return')
        elongation = pullout_displacement['peak_displacement_per_root'] / self.distribution_factor
        if (shear_zone_thickness.magnitude <= 0.0):
            return(elongation)
        else:
            length_initial =  shear_zone_thickness / self.roots.orientation[..., 2]
            length_x0 = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2]
            length_y0 = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z0 = shear_zone_thickness
            length = length_initial + elongation
            length_x = np.sqrt(length**2 - length_y0**2 - length_z0**2)
            return(length_x - length_x0)