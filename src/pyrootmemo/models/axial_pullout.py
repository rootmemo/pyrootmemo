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

class AxialPullout():
    """Class for axial pull-out of roots

    Predict pull-out forces for bundles of roots. This class follows models
    developed by Waldron (1977), Waldron & Dakessian (1981) and the DRAM model
    developed by Meijer et al. (2022), in the sense that displacements and 
    forces are mobilised due to the interaction between root-soil interface 
    resistance and root stiffness.

    This model class can take a wide variety of different root behaviours, 
    and therefore incorporates the models by a wide variety of authors. for 
    example, this class can take into account:
    - root breakage.
    - root slippage.
    - elastic or elasto-plastic behaviour.
    - 'embedded' behaviour (e.g. roots remain fully surrounded by soil) or 
      'surface' behaviour, in which the root length in contact with the soil
      gradually reduces.
    - root survival functions, which determined whether roots break 'suddenly'
      or gradually as an 'average' root, as govered by the weibull shape 
      parameter. This is implemented by looking at the ratio between the force
      at failure and the currently mobilised force in a root, assuming no 
      breakage.

    These different types of behaviours are fully described in Yildiz & 
    Meijer's "Root Reinforcement: Measurement and Modelling".

    Attributes
    ----------
    roots : pyrootmemo.materials.MultipleRoots
        MultipleRoots object containing properties of all roots in bundle. This
        instance must contain the attributes
        - 'circumference'
          root circumference
        - 'xsection'
          root cross-sectional area
        - 'elastic_modulus
          root elastic or Young's modulus
        - 'tensile_strength' (when breakage = True)
          root tensile strength
        - 'length' (when slipping = True)
          root length, i.e. the distance between point of pulling and the root
          tip
        - 'length_surface' (when slipping = True and surface = True)
          root length already sticking out of the soil surface at start of 
          pull-out
        - 'yield_strength' (when elastoplastic = True)
          root yield strength, i.e. the stress level at the start of plastic
          behaviour
        - 'plastic_modulus' (when elastoplastic = True)
          root stiffness during plastic behaviour phase
        - 'unload_modulus' (when elastoplastic = True)
          root stiffness when unloading, for plastically behaving roots
    interface : pyrootmemo.materials.Interface
        Interface object containing properties of the root-soil interface. This
        instance must contain the attribute 'shear_strength'        
    surface : bool
        Flag indicating whether the root behaviour type is 'surface', i.e. the
        root gets gradually pulled out of a soil surface and the root length 
        that remains in contact with the surrounding soil decreases gradually.
        If False, the root behaviour is assumed to be 'embedded', i.e. the
        root remains surrounded by soil at all times despite increasing
        pull-out displacements.
    breakage : bool
        Flag indicating whether roots are allowed to break. If False, roots 
        will never break in tension
    slipping : bool
        Flag indicating whether roots are allowed to slip. If False, roots 
        will never show slipping behaviour
    elastoplastic : bool
        Flag indicating whether roots behave elasto-plastically, as implemented
        as a bi-linear stress-strain response. If False, roots are assumed to
        behave fully elastic according to the root elastic stiffness.
    weibull_shape : None | float | int
        Weibull shape for root survival function. If roots break suddenly, 
        weibull_shape = None. This corresponds with a weibull shape factor 
        equal to +infinity
    behaviour_types : np.ndarray
        A list with character strings indicating the different types of 
        behaviour each root could show (e.g. elastic, plastic, slipping,
        anchored etc.)
    displacement_limits : Quantity
        A two-dimensional array of pull-out displacement levels for each root
        (columns, axis 1) at which the behaviour type changes from one type 
        to the next (rows, axis 0)
    force_limits : Quantity
        A two-dimensional array of pull-out forces for each root
        (columns, axis 1) at which the behaviour type changes from one type 
        to the next (rows, axis 0)
    coefficients : list
        A list with polynomial cubic coefficients to each root descibing the 
        polynomial relationship between force (independent variable) and 
        displacement (depedent variable). Elements in the list are ordered from
        higher-order parameters (3rd order) to lower order (0th order). Each 
        of these elements is a two-dimensional array given the coefficient for
        each root (columns, axis 1) at each of the different behaviour types
        (rows, axis 0). Note that for some behaviour types this relationship
        may not be uniquely defined.
                
    Methods
    -------
    __init__(roots, interface, surface, breakage, slipping, elastoplastic, weibull_shape)
        Constructor
    calc_force(displacement)
        Calculate force in each root given the displacement
    calc_displacement_to_peak()
        Calculate displacement until breakage or the start of slippage, for 
        each root
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            surface: bool = False,
            breakage: bool = True,
            slipping: bool = True,
            elastoplastic: bool = False,
            weibull_shape: None | int | float = None
            ):
        """Initialise a Pullout model object

        Parameters
        ----------
        roots : pyrootmemo.materials.MultipleRoots
            MultipleRoots object containing properties of all roots in bundle. 
            This instance must contain the attributes
            - 'circumference'
                root circumference
            - 'xsection'
                root cross-sectional area
            - 'elastic_modulus
                root elastic or Young's modulus
            - 'tensile_strength' (when breakage = True)
                root tensile strength
            - 'length' (when slipping = True)
                root length, i.e. the distance between point of pulling and the 
                root tip
            - 'length_surface' (when slipping = True and surface = True)
                root length already sticking out of the soil surface at start of 
                pull-out
            - 'yield_strength' (when elastoplastic = True)
                root yield strength, i.e. the stress level at the start of plastic
                behaviour
            - 'plastic_modulus' (when elastoplastic = True)
                root stiffness during plastic behaviour phase
            - 'unload_modulus' (when elastoplastic = True)
                root stiffness when unloading, for plastically behaving roots
        interface : pyrootmemo.materials.Interface
            Interface object containing properties of the root-soil interface. 
            This instance must contain the attribute 'shear_strength'   
        surface : bool, optional
            Flag indicating whether the root behaviour type is 'surface' 
            (True), i.e. the root gets gradually pulled out of a soil surface 
            and the root length that remains in contact with the surrounding 
            soil decreases gradually. If False, the root behaviour is assumed 
            to be 'embedded', i.e. the root remains surrounded by soil at all 
            times despite increasing pull-out displacements. By default False.
        breakage : bool, optional
            Flag indicating whether roots are allowed to break (True). If 
            False, roots will never break in tension. By default True
        slipping : bool, optional
            Flag indicating whether roots are allowed to slip (True). If False, 
            roots will never show slipping behaviour. By default True.
        elastoplastic : bool, optional
            Flag indicating whether roots behave elasto-plastically, as 
            implemented as a bi-linear stress-strain response (True). If False, 
            roots are assumed to behave fully elastic according to the root 
            elastic stiffness. By default False.
        weibull_shape : None | float | int, optional
            Weibull shape for root survival function. If roots break suddenly, 
            weibull_shape = None. This corresponds with a weibull shape factor 
            equal to +infinity. By default None
        """
        roots_attributes_required = ['circumference', 'xsection', 'elastic_modulus']
        if surface is True:
            roots_attributes_required += ['length_surface']
        if breakage is True:
            roots_attributes_required += ['tensile_strength']
        if slipping is True:
            roots_attributes_required += ['length']
        if elastoplastic is True:
            roots_attributes_required += ['yield_strength', 'plastic_modulus']
        for i in roots_attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots must contain ' + str(i) + ' attribute')
        if surface is True:
            if elastoplastic is True:
                if not hasattr(roots, 'unload_modulus'):
                    roots.unload_modulus = roots.elastic_modulus
        self.roots = roots
        interface_attributes_required = ['shear_strength']
        for i in interface_attributes_required:
            if not hasattr(interface, i):
                raise AttributeError('interface must contain ' + str(i) + ' attribute')
        self.interface = interface
        self.surface = surface
        self.breakage = breakage
        self.slipping = slipping
        self.elastoplastic = elastoplastic
        if isinstance(weibull_shape, int) | isinstance(weibull_shape, float):
            if weibull_shape <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            elif np.isinf(weibull_shape):
                raise ValueError('weibull_shape must have finite value. Set to None for instant breakages')
        else:
            if weibull_shape is not None:
                raise TypeError('weibull_shape must be an int, float or None')
        self.weibull_shape = weibull_shape

        if np.isscalar(roots.xsection.magnitude):
            nroots = (1)
        else:
            nroots = roots.xsection.shape
        behaviour_types = np.array([
            'Not in tension',
            'Anchored, elastic',
            'Slipping, elastic',
            'Full pullout',      # (when behaviour is elastic)
            'Anchored, plastic',   
            'Slipping, plastic', # (stress above yield stress)
            'Slipping, plastic', # (stress below yield stress)
            'Full pullout'       # (when behaviour is plastic)
        ])
        coefficients = [
            np.zeros((len(behaviour_types), *nroots)) * units('mm/N^3'),
            np.zeros((len(behaviour_types), *nroots)) * units('mm/N^2'),
            np.zeros((len(behaviour_types), *nroots)) * units('mm/N'),
            np.zeros((len(behaviour_types), *nroots)) * units('mm')
            ]
        displacement_limits = np.zeros((len(behaviour_types) - 1, *nroots)) * units('mm')
        force_limits = np.zeros((len(behaviour_types) - 1, *nroots)) * units('N')
        
        if surface is True:
            ## SURFACE ROOTS
            # anchored, elastic [type 1]
            coefficients[0][1, ...] = (
                1.0 / (2.0 * (roots.elastic_modulus * roots.xsection)**2
                       * roots.circumference * interface.shear_strength)
                )
            coefficients[1][1, ...] = (
                1.0 / (2.0 * roots.elastic_modulus * roots.xsection
                       * roots.circumference * interface.shear_strength)
                )
            coefficients[2][1, ...] = roots.length_surface / (roots.elastic_modulus * roots.xsection)
            if slipping is True:
                # slipping, elastic [type 2]
                coefficients[1][2, ...] = (
                    -1.0 / (roots.elastic_modulus * roots.xsection
                            * roots.circumference * interface.shear_strength)
                    )
                coefficients[2][2, ...] = (
                     roots.length / (roots.elastic_modulus * roots.xsection)
                     - 1.0 / (roots.circumference * interface.shear_strength)
                     )
                coefficients[3][2, ...] = roots.length - roots.length_surface
                # displacement at start of slippage, elastic <limit 1>
                force_limits[1, :] = solve_quadratic(
                    1.0 / (2.0 * roots.elastic_modulus * roots.xsection * roots.circumference * interface.shear_strength),
                    1.0 / (roots.circumference * interface.shear_strength),
                    roots.length_surface - roots.length
                    )
                displacement_limits[1, :] = (
                    coefficients[0][1, ...] * force_limits[1, :]**3
                    + coefficients[1][1, ...] * force_limits[1, :]**2
                    + coefficients[2][1, ...] * force_limits[1, :]
                    + coefficients[3][1, ...]
                    )
                # displacement until full pullout, elastic <limit 2>
                displacement_limits[2, :] = roots.length - roots.length_surface
            if elastoplastic is True:
                # force and displacement at yield <limit 3>
                force_limits[3, :] = roots.xsection * roots.yield_strength
                displacement_limits[3, :] = (
                    coefficients[0][1, ...] * force_limits[3, :]**3
                    + coefficients[1][1, ...] * force_limits[3, :]**2
                    + coefficients[2][1, ...] * force_limits[3, :]
                    + coefficients[3][1, ...]
                    )
                # anchored, plastic [type 4]
                coefficients[0][4, ...] = (
                    1.0 / (2.0 * (roots.plastic_modulus * roots.xsection)**2 
                           * roots.circumference * interface.shear_strength)
                    )
                coefficients[1][4, ...] = (
                    1.0 
                    / (2.0 * roots.plastic_modulus * roots.xsection 
                       * roots.circumference * interface.shear_strength)
                    * (1.0
                       + 3.0 * roots.yield_strength / roots.elastic_modulus
                       - 3.0 * roots.yield_strength / roots.plastic_modulus) 
                    )
                coefficients[2][4, ...] = (
                    roots.yield_strength
                    / (2.0 * roots.elastic_modulus * roots.plastic_modulus
                        * roots.circumference * interface.shear_strength)
                    * (
                        roots.yield_strength 
                        * (3.0 * roots.elastic_modulus / roots.plastic_modulus
                            + 2.0 * roots.plastic_modulus / roots.elastic_modulus
                            - 5.0)
                        - 2.0 * roots.elastic_modulus
                        + 2.0 * roots.plastic_modulus
                    )
                    + roots.length_surface / (roots.plastic_modulus * roots.xsection)
                    )
                coefficients[3][4, ...] = (
                    roots.yield_strength 
                    * (roots.elastic_modulus - roots.plastic_modulus)
                    / (2.0 * roots.elastic_modulus * roots.plastic_modulus
                       * roots.circumference * interface.shear_strength)
                    * (
                        roots.yield_strength * roots.xsection
                        - roots.yield_strength**2 * roots.xsection 
                        * (1.0 / roots.plastic_modulus - 1.0 / roots.elastic_modulus)
                        - 2.0 * roots.circumference * interface.shear_strength * roots.length_surface
                        )
                    )
                if slipping is True:
                    # force and displacement at start of slipping, plastic <limit 4>
                    force_limits[4, :] = solve_quadratic(
                        1.0 / (2.0 * roots.plastic_modulus * roots.xsection 
                            * roots.circumference * interface.shear_strength),
                        (1.0 
                        / (roots.circumference * interface.shear_strength)
                        * (1.0
                            + force_limits[3, :] / (roots.elastic_modulus * roots.xsection)
                            - force_limits[3, :] / (roots.plastic_modulus * roots.xsection))),
                        (roots.length_surface
                        + (force_limits[3, :]**2 
                            / (2.0 * roots.xsection * roots.circumference * interface.shear_strength)
                            * (1.0 / roots.plastic_modulus - 1.0 / roots.elastic_modulus))
                        - roots.length)
                        )
                    displacement_limits[4, :] = (
                        coefficients[0][4, ...] * force_limits[4, :]**3
                        + coefficients[1][4, ...] * force_limits[4, :]**2
                        + coefficients[2][4, ...] * force_limits[4, :]
                        + coefficients[3][4, ...]
                    )
                    # slipping, plastic, above yield (type 5)
                    coefficients[1][5, ...] = (
                        -1.0
                        / (2.0 * roots.xsection * roots.circumference * interface.shear_strength)
                        * (1.0 / roots.plastic_modulus + 1.0 / roots.unload_modulus)
                        )
                    coefficients[2][5, ...] = (
                        roots.length / (roots.unload_modulus * roots.xsection)
                        - 1.0 
                        / (roots.circumference * interface.shear_strength)
                        * (
                            1.0
                            + roots.yield_strength / roots.elastic_modulus
                            - roots.yield_strength / roots.plastic_modulus
                            )
                        )
                    coefficients[3][5, ...] = (
                        roots.length 
                        - roots.length_surface
                        + roots.yield_strength * roots.length
                        * (1.0 / roots.elastic_modulus - 1.0 / roots.plastic_modulus)
                        + force_limits[4, :] / roots.xsection
                        * (roots.length - force_limits[4, :] / (2.0 * roots.circumference * interface.shear_strength))
                        * (1.0 / roots.plastic_modulus - 1.0 / roots.unload_modulus)
                        )
                    # force and displacement to yield during plastic unloading <limit 5>
                    force_limits[5, :] = force_limits[3, :]
                    displacement_limits[5, :] = (
                        coefficients[0][5, ...] * force_limits[5, :]**3
                        + coefficients[1][5, ...] * force_limits[5, :]**2
                        + coefficients[2][5, ...] * force_limits[5, :]
                        + coefficients[3][5, ...]
                    )                    
                    # slipping, plastic, below yield (type 6)
                    coefficients[1][6, ...] = (
                        -1.0
                        / (roots.elastic_modulus * roots.xsection
                        * roots.circumference * interface.shear_strength)
                        )
                    coefficients[2][6, ...] = (
                        roots.length / (roots.unload_modulus * roots.xsection)
                        - 1.0 / (roots.circumference * interface.shear_strength)
                        * (1.0 
                        - roots.yield_strength / roots.unload_modulus 
                        + roots.yield_strength / roots.elastic_modulus)
                        )
                    coefficients[3][6, ...] = (
                        roots.length 
                        - roots.length_surface
                        + 1.0 / (2.0 * roots.xsection * roots.circumference * interface.shear_strength)
                        * (
                            force_limits[4, :]**2 
                            * (1.0 / roots.unload_modulus - 1.0 / roots.plastic_modulus)
                            + (roots.yield_strength * roots.xsection)**2
                            * (1.0 / roots.unload_modulus + 1.0 / roots.plastic_modulus - 2.0 / roots.elastic_modulus)
                            )
                        - roots.length / roots.xsection
                        * (
                            force_limits[4, :]
                            * (1.0 / roots.unload_modulus - 1.0 / roots.plastic_modulus)
                            + (roots.yield_strength * roots.xsection)
                            * (1.0 / roots.plastic_modulus - 1.0 / roots.elastic_modulus)
                            )
                        )
                    # displacement until full pull-out, plastic <limit 6>
                    displacement_limits[6, :] = coefficients[3][6, ...]
                    # adjust limits: slippage before yielding --> never plasticity
                    slip_before_yield = (displacement_limits[1, ...] <= displacement_limits[3, ...])
                    displacement_limits[3:7, slip_before_yield] = np.inf * units('mm')
                    force_limits[3:7, slip_before_yield] = 0.0 * units('N')
                    # adjust limits: slippage after yielding --> never elastic slippage
                    yield_before_slip = ~slip_before_yield
                    displacement_limits[1:3, yield_before_slip] = displacement_limits[3, yield_before_slip]
                    force_limits[1:3, yield_before_slip] = force_limits[3, yield_before_slip]
        else:
            ## EMBEDDED ROOTS
            # anchored, elastic [type 1]
            coefficients[1][1, ...] = (
                1.0 / (2.0 * roots.elastic_modulus * roots.xsection 
                       * roots.circumference * interface.shear_strength)
                )
            if slipping is True:
                # slipping, elastic [type 2]
                force_limits[1, :] = roots.length * roots.circumference * interface.shear_strength
                # displacement at start of slippage, elastic <limit 1>
                displacement_limits[1, :] = coefficients[1][1, ...] * force_limits[1, :]**2
            if elastoplastic is True:
                # displacement at yield <limit 3>
                force_limits[3, :] = roots.xsection * roots.yield_strength
                displacement_limits[3, :] = coefficients[1][1, ...] * force_limits[3, :]**2
                # anchored, plastic [type 4]
                coefficients[1][4, ...] = (
                    1.0 / (2.0 * roots.plastic_modulus * roots.xsection 
                           * roots.circumference * interface.shear_strength)
                    )
                coefficients[2][4, ...] = (
                    roots.yield_strength 
                    / (roots.elastic_modulus * roots.circumference * interface.shear_strength)
                    - roots.yield_strength 
                    / (roots.plastic_modulus * roots.circumference * interface.shear_strength)
                    )
                coefficients[3][4, ...] = (
                    -roots.yield_strength**2 * roots.xsection 
                    / (2.0 * roots.elastic_modulus * roots.circumference * interface.shear_strength)
                    + roots.yield_strength**2 * roots.xsection 
                    / (2.0 * roots.plastic_modulus * roots.circumference * interface.shear_strength)
                    )
                if slipping is True:
                    # displacement at start of slippage, plastic <limit 4>
                    displacement_limits[4, :] = (
                        coefficients[1][4, ...] * force_limits[1, :]**2
                        + coefficients[2][4, ...] * force_limits[1, :]
                        + coefficients[3][4, ...]
                        )
                    force_limits[4, :] = force_limits[1, :]
                    # adjust limits: slippage before yielding --> never plasticity
                    slip_before_yield = displacement_limits[1, ...] <= displacement_limits[3, ...]
                    displacement_limits[2:7, slip_before_yield] = np.inf * units('mm')
                    force_limits[2:7, slip_before_yield] = force_limits[1, slip_before_yield]
                    # adjust limits: slippage after yielding --> never elastic slippage
                    yield_before_slip = ~slip_before_yield
                    displacement_limits[1:3, yield_before_slip] = displacement_limits[3, yield_before_slip]
                    force_limits[1:3, yield_before_slip] = force_limits[3, yield_before_slip]

        # for displacement limits that are not needed, add dummy values based on 'next' displacement limit
        mask = np.isclose(displacement_limits[-1, ...].magnitude, 0.0)
        displacement_limits[-1, mask] = np.inf * units('mm')
        for i in np.flip(np.arange(1, 6)):
            mask = np.isclose(displacement_limits[i, ...].magnitude, 0.0)
            displacement_limits[i, mask] = displacement_limits[i + 1, mask]
            force_limits[i, mask] = force_limits[i + 1, mask]

        self.coefficients = coefficients
        self.behaviour_types = behaviour_types
        self.displacement_limits = displacement_limits
        self.force_limits = force_limits


    def calc_force(
            self,
            displacement: Quantity,
            jacobian: bool = False
            ) -> dict:
        """Calculate force in each root, as function of given displacement

        Parameters
        ----------
        displacement : Quantity | Parameter(value: int | float | np.ndarray, unit: str)
            Displacement level. Must be a scalar, in which case the this
            is the applied displacement to each root. If inputted as an 
            array, this is the array of displacements applied to individual
            roots (must have same length as number of roots).
        jacobian : bool
            If True, also calculate and return the derivative of pull-out force(s) with
            respect to the applied pull-out displacement. By default False.

        Returns
        -------
        dict
            results dictionary with fields:
            - 'force': array with forces in each root
            - 'behaviour_index': array with the index of the behaviour type
              of each roots. see class attribute 'behaviour_type' for a full
              list of behaviour type names
            - 'survival_fraction': array with survival fraction for each root
            - 'dforce_ddisplacement': derivative of pullout forces with respect 
              to displacement. Only returned when 'jacobian = True'. 
        """
        displacement = create_quantity(displacement, check_unit = 'mm')
        nroots = self.roots.xsection.shape
        if not np.isscalar(displacement.magnitude):
            if not displacement.shape == nroots:
                raise ValueError('displacement must be a scalar or an array with seperate displacements for each individual root')
        behaviour_index = np.sum(displacement > self.displacement_limits, axis = 0).astype(int)
       
        force_unbroken = np.zeros(*nroots) * units('N')
        if jacobian is True:
            dforceunbroken_ddisplacement = np.zeros(*nroots) * units('N/mm')

        if self.surface is True:
            mask_el_anch = (behaviour_index == 1)
            if any(mask_el_anch):
                force_unbroken[mask_el_anch] = solve_cubic(
                    self.coefficients[0][1, mask_el_anch],
                    self.coefficients[1][1, mask_el_anch],
                    self.coefficients[2][1, mask_el_anch],
                    (self.coefficients[3][1, ...] - displacement)[mask_el_anch]
                    )
                if jacobian is True:
                    dforceunbroken_ddisplacement[mask_el_anch] = (1.0 / (
                        3.0 * self.coefficients[0][1, mask_el_anch] * force_unbroken[mask_el_anch]**2
                        + 2.0 * self.coefficients[1][1, mask_el_anch] * force_unbroken[mask_el_anch]
                        + self.coefficients[2][1, mask_el_anch]
                    ))
            if self.slipping is True:
                mask_el_slip = (behaviour_index == 2)
                if any(mask_el_slip):
                    force_unbroken[mask_el_slip] = solve_quadratic(
                        self.coefficients[1][2, mask_el_slip],
                        self.coefficients[2][2, mask_el_slip],
                        (self.coefficients[3][2, ...] - displacement)[mask_el_slip]
                    )
                    if jacobian is True:
                        dforceunbroken_ddisplacement[mask_el_slip] = (1.0 / (
                            2.0 * self.coefficients[1][2, mask_el_slip] * force_unbroken[mask_el_slip]
                            + self.coefficients[2][2, mask_el_slip]
                        ))
            if self.elastoplastic is True:
                mask_pl_anch = (behaviour_index == 4)
                if any(mask_pl_anch):
                    force_unbroken[mask_pl_anch] = solve_cubic(
                        self.coefficients[0][4, mask_pl_anch],
                        self.coefficients[1][4, mask_pl_anch],
                        self.coefficients[2][4, mask_pl_anch],
                        (self.coefficients[3][4, ...] - displacement)[mask_pl_anch]
                        )
                    if jacobian is True:
                        dforceunbroken_ddisplacement[mask_pl_anch] = (1.0 / (
                            3.0 * self.coefficients[0][4, mask_pl_anch] * force_unbroken[mask_pl_anch]**2
                            + 2.0 * self.coefficients[1][4, mask_pl_anch] * force_unbroken[mask_pl_anch]
                            + self.coefficients[2][4, mask_pl_anch]
                        ))
                if self.slipping is True:
                    mask_pl_slip_aboveyield = (behaviour_index == 5)
                    if any(mask_pl_slip_aboveyield):
                        force_unbroken[mask_pl_slip_aboveyield] = solve_quadratic(
                            self.coefficients[1][5, mask_pl_slip_aboveyield],
                            self.coefficients[2][5, mask_pl_slip_aboveyield],
                            (self.coefficients[3][5, ...] - displacement)[mask_pl_slip_aboveyield]
                            )
                        if jacobian is True:
                            dforceunbroken_ddisplacement[mask_pl_slip_aboveyield] = (1.0 / (
                                2.0 * self.coefficients[1][5, mask_pl_slip_aboveyield] * force_unbroken[mask_pl_slip_aboveyield]
                                + self.coefficients[2][5, mask_pl_slip_aboveyield]
                            ))
                    mask_pl_slip_belowyield = (behaviour_index == 6)
                    if any(mask_pl_slip_belowyield):
                        force_unbroken[mask_pl_slip_belowyield] = solve_quadratic(
                            self.coefficients[1][6, mask_pl_slip_belowyield],
                            self.coefficients[2][6, mask_pl_slip_belowyield],
                            (self.coefficients[3][6, ...] - displacement)[mask_pl_slip_belowyield]
                            )
                        if jacobian is True:
                            dforceunbroken_ddisplacement[mask_pl_slip_belowyield] = (1.0 / (
                                2.0 * self.coefficients[1][6, mask_pl_slip_belowyield] * force_unbroken[mask_pl_slip_belowyield]
                                + self.coefficients[2][6, mask_pl_slip_belowyield]
                            ))
            force_unbroken_cummax = force_unbroken.copy()
            mask_el_reducing = np.isin(behaviour_index, [2, 3])
            force_unbroken_cummax[mask_el_reducing] = self.force_limits[1, mask_el_reducing]
            mask_pl_reducing = np.isin(behaviour_index, [5, 6, 7])
            force_unbroken_cummax[mask_pl_reducing] = self.force_limits[4, mask_pl_reducing]
            if jacobian is True:
                dforceunbrokencummax_ddisplacement = dforceunbroken_ddisplacement.copy()
                dforceunbrokencummax_ddisplacement[mask_el_reducing] = 0.0 * units('N/mm')
                dforceunbrokencummax_ddisplacement[mask_pl_reducing] = 0.0 * units('N/mm')
        else:
            mask_el_anch = (behaviour_index == 1)
            if any(mask_el_anch):
                force_unbroken[mask_el_anch] = np.sqrt(
                    (displacement / self.coefficients[1][1, ...])[mask_el_anch]
                    )
                if jacobian is True:
                    dforceunbroken_ddisplacement[mask_el_anch] = (1.0 / (
                        2.0 * self.coefficients[1][1, mask_el_anch] * force_unbroken[mask_el_anch]
                        ))
            if self.slipping is True:
                mask_el_slip = (behaviour_index == 2)
                if any(mask_el_slip):
                    force_unbroken[mask_el_slip] = (
                        self.roots.length[mask_el_slip]
                        * self.roots.circumference[mask_el_slip]
                        * self.interface.shear_strength
                        )
            if self.elastoplastic is True:
                mask_pl_anch = (behaviour_index == 4)
                if any(mask_pl_anch):
                    force_unbroken[mask_pl_anch] = solve_quadratic(
                        self.coefficients[1][4, mask_pl_anch],
                        self.coefficients[2][4, mask_pl_anch],
                        (self.coefficients[3][4, ...] - displacement)[mask_pl_anch]
                        )
                    if jacobian is True:
                        dforceunbroken_ddisplacement[mask_pl_anch] = (1.0 / (
                            2.0 * self.coefficients[1][4, mask_pl_anch] * force_unbroken[mask_pl_anch]
                            + self.coefficients[2][4, mask_pl_anch]
                            ))
                if self.slipping is True:
                    mask_pl_slip = (behaviour_index == 5)
                    if any(mask_pl_slip):
                        force_unbroken[mask_pl_slip] = (
                            self.roots.length[mask_pl_slip]
                            * self.roots.circumference[mask_pl_slip]
                            * self.interface.shear_strength
                            )
            force_unbroken_cummax = force_unbroken.copy()
            if jacobian is True:
                dforceunbrokencummax_ddisplacement = dforceunbroken_ddisplacement.copy()

        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
            if self.weibull_shape is None:
                survival = (force_unbroken_cummax <= force_breakage).astype(float)
                if jacobian is True:
                    dsurvival_ddisplacement = np.zeros(*nroots) * units('1/mm')
            else:
                y = (gamma(1.0 + 1.0 / self.weibull_shape) * force_unbroken_cummax / force_breakage).magnitude                   
                survival = np.exp(-(y**self.weibull_shape))
                if jacobian is True:
                    dy_dforceunbrokencummax = gamma(1.0 + 1.0 / self.weibull_shape) / force_breakage
                    dsurvival_dy = -self.weibull_shape * y**(self.weibull_shape - 1.0) * survival
                    dsurvival_ddisplacement = dsurvival_dy * dy_dforceunbrokencummax * dforceunbrokencummax_ddisplacement
        else:
            survival = np.ones(*nroots).astype(float)
            if jacobian is True:
                dsurvival_ddisplacement = np.zeros(*nroots) * units('1/mm')         

        dict_return = {
            'force': force_unbroken * survival,
            'behaviour_index': behaviour_index,
            'survival_fraction': survival
        }
        if jacobian is True:
            dict_return['dforce_ddisplacement'] = (
                dforceunbroken_ddisplacement * survival
                + force_unbroken * dsurvival_ddisplacement
            )
        return(dict_return)
    

    def calc_displacement_to_peak(self) -> Quantity:
        """Calculate the displacement to peak, for in each root

        Calculates the displacement required for the each each in the 
        MultipleRoots object to reach the largest force it will even reach
        as function of *any* displacement level.

        The function does (currently) not take the survival function into 
        account, i.e. it looks at the *average* root for each root in the 
        MultipleRoots object. 

        Returns
        -------
        Quantity
            Displacement to peak for each root
        """
        if self.slipping is True:
            if self.elastoplastic is True:
                displacement_slipping = self.displacement_limits[4, ...]
                slip_before_yield = np.isinf(self.displacement_limits[4, ...].magnitude)
                displacement_slipping[slip_before_yield] = self.displacement_limits[1, slip_before_yield]
            else:
                displacement_slipping = self.displacement_limits[1, ...]
        else:
            displacement_slipping = np.full(self.roots.xsection.shape, np.inf) * units('mm')
        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
            if self.elastoplastic is True:
                behaviour_index = 4
            else:
                behaviour_index = 1
            if self.surface is True:
                displacement_breakage = (
                    self.coefficients[0][behaviour_index, ...] * force_breakage**3
                    + self.coefficients[1][behaviour_index, ...] * force_breakage**2
                    + self.coefficients[2][behaviour_index, ...] * force_breakage
                    + self.coefficients[3][behaviour_index, ...]
                )
            else:
                displacement_breakage = (
                    self.coefficients[1][behaviour_index, ...] * force_breakage**2
                    + self.coefficients[2][behaviour_index, ...] * force_breakage
                    + self.coefficients[3][behaviour_index, ...]
                )
        else:
            displacement_breakage = np.full(self.roots.xsection.shape, np.inf) * units('mm')
        return(np.minimum(displacement_slipping, displacement_breakage))