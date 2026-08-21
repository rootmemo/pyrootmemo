import numpy as np
from scipy.special import gamma
from pyrootmemo import Parameter
from pyrootmemo.helpers import units, create_quantity, Results, ResultsType
from pyrootmemo.materials import MultipleRoots, Interface
from pint import Quantity

def _solve_quadratic(
        a: Quantity, 
        b: Quantity, 
        c: Quantity,
        ) -> Quantity:
    """Calculate largest root of a quadratic equation

    Calculate the largest root of a quadratic equation in the form:
    a * x**2 + b * x + c == 0 

    Parameters
    ----------
    a : Quantity
        second-order polynomial coefficient(s)
    b : Quantity
        first-order polynomial coefficient(s)
    c : Quantity
        zero-order polynomial coefficient(s)

    Returns
    -------
    Quantity | np.ndarray | float
        Largest root of the quadratic equation
    """
    discriminant = b**2 - 4.0 * a * c
    x = (-b + np.sign(a) * np.sqrt(discriminant)) / (2.0 * a)
    return(x)

def _solve_cubic(
        a: Quantity, 
        b: Quantity, 
        c: Quantity, 
        d: Quantity
        ) -> Quantity:
    """Calculate largest real root of a cubic equation

    Calculate the largest root of a cubic equation in the form:
    a * x**3 + b * x**2 + c * x + d == 0 

    The function assumes all values of the third-order coefficient <a> are
    not equal to zero. If so, a quadratic solver is more appropriate.

    The function follows the methodology detailed on Wikipedia
    (https://en.wikipedia.org/wiki/Cubic_equation):
  
    Parameters
    ----------
    a : Quantity
        third-order polynomial coefficient(s). All values must not be equal 
        to zero for the function to work.
    b : Quantity
        second-order polynomial coefficient(s)
    c : Quantity
        first-order polynomial coefficient(s)
    d : Quantity
        zero-order polynomial coefficient(s)

    Returns
    -------
    Quantity
        Largest real root of the cubic equation
    """
    x = np.zeros(a.shape) * d.units / c.units
    e = b / a
    f = c / a
    g = d / a
    Q = (e**2 - 3.0 * f) / 9.0
    R = (2.0 * e**3 - 9.0 * e * f + 27.0 * g) / 54.0
    mask_3realroots = (R**2) < (Q**3) # if true, 3 real roots exist, if false, only one real root exists
    if any(mask_3realroots):
        theta = np.arccos(R[mask_3realroots] / np.sqrt(Q[mask_3realroots]**3))
        x[mask_3realroots] = (
            -2.0 
            * np.sqrt(Q[mask_3realroots]) 
            * np.cos((theta + 2.0 * np.pi) / 3.0) 
            - e[mask_3realroots] 
            / 3.0
            )
    mask_1real1root = ~mask_3realroots
    if any(mask_1real1root):
        A = (
            -np.sign(R[mask_1real1root]) 
            * (
                np.abs(R[mask_1real1root]) 
                + np.sqrt(R[mask_1real1root]**2 - Q[mask_1real1root]**3)
                ) ** (1.0 / 3.0)
            )
        B = Q[mask_1real1root] / A
        x[mask_1real1root] = (A + B) - e[mask_1real1root] / 3.0
    mask_zerodiscriminant = np.isclose(d.magnitude, 0.0)
    x[mask_zerodiscriminant] = 0.0 * d.units / c.units
    return(x)

def _solve_cubic_polynomial(
        a: Quantity, 
        b: Quantity, 
        c: Quantity, 
        d: Quantity
        ) -> Quantity:
    is_zero_a = np.isclose(a.magnitude, 0.0)
    is_zero_b = np.isclose(b.magnitude, 0.0)
    is_zero_c = np.isclose(c.magnitude, 0.0)
    is_cubic = ~is_zero_a
    is_quadratic = np.bitwise_and(~is_cubic, ~is_zero_b)
    is_linear = np.bitwise_and(~is_quadratic, ~is_zero_c)
    root = np.zeros_like(a) * d.units / c.units
    if np.any(is_linear):
        root[is_linear] = -d[is_linear] / c[is_linear]
    if np.any(is_quadratic):
        root[is_quadratic] = _solve_quadratic(b[is_quadratic], c[is_quadratic], d[is_quadratic])
    if np.any(is_cubic):
        root[is_cubic] = _solve_cubic(a[is_cubic], b[is_cubic], c[is_cubic], d[is_cubic])
    return(root)

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
    output : dict
        Dictionary with all calculation results. By default includes the key
        `behaviour_types` containing a list with character strings indicating 
        the different different types of behaviour each root could show
        depending on the level of displacement (e.g. elastic, plastic, slipping,
        anchored etc.).
                
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
            'Full pullout, elastic',      # (when behaviour is elastic)
            'Anchored, plastic',   
            'Slipping, plastic',          # (stress above yield stress)
            'Slipping, plastic',          # (stress below yield stress)
            'Full pullout, plastic'       # (when behaviour is plastic)
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
                force_limits[1, :] = _solve_quadratic(
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
                    force_limits[4, :] = _solve_quadratic(
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

        self.output = {'behaviour_types': behaviour_types}

    def calc_force(
            self,
            displacement: Quantity | Parameter,
            jacobian: bool = False,
            results: str = "attribute"
            ):
        """Calculate force in each root, as function of given displacement

        Function creates a dictionary with the keys:

            * `displacement_per_root`: array or scalar with specified pullout 
              displacements.
            * `force_per_root`: array with forces in each root
            * `behaviour_index`: array with the index of the behaviour type
              of each roots. see class attribute 'behaviour_type' for a full
              list of behaviour type names
            * `survival_fraction`: array with survival fraction for each root
            * `dforce_per_root_ddisplacement_per_root`: derivative of pullout 
              forces in each root with respect to displacement in each root. 
              Only returned when `jacobian = True`.

        All returned fields are two-dimensional arrays, with the specified 
        'time' steps on the first axis (rows) and the result per individual
        root on the second axis (columns).
                      
        How this dictionary returned depends on the value of the `results` 
        argument.
        
        Parameters
        ----------
        displacement : Quantity | Parameter(value: int | float | np.ndarray, unit: str)
            Displacement level. If a scalar, this displacement is applied to 
            each root. If a one-dimensional array, each value is applied to 
            all roots consecutively. If a two-dimensional array, the first axis
            signifies 'time' and the second axis is the displacement applied to 
            each individual root at that time. In this case, the length of the
            second axis must match the number of roots.
        jacobian : bool
            If True, also calculate and return the derivative of pull-out force(s) with
            respect to the applied pull-out displacement. By default False.
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.
        """
        nroots = len(self.roots.xsection)
        displacement = create_quantity(displacement, check_unit = 'mm')
        if np.isscalar(displacement.magnitude):
            is_scalar_displacement = True
            ndisplacements = 1
            displacement_per_root = np.ones((ndisplacements, nroots)) * displacement
        else:
            is_scalar_displacement = False
            if np.ndim(displacement) == 1:
                ndisplacements = len(displacement)
                displacement_per_root = np.broadcast_to(displacement[:, np.newaxis], (ndisplacements, nroots))
            elif np.ndim(displacement) == 2:
                if not displacement.shape[1] == nroots:
                    raise ValueError('displacement must be a scalar or an array with seperate displacements for each individual root')
                ndisplacements = displacement.shape[0]
                displacement_per_root = displacement
            else:
                raise ValueError('displacement must be a scalar, 1-D array or 2-D array with 2nd axis matching the number of roots')

        behaviour_mask = displacement_per_root[:, np.newaxis, :] > self.displacement_limits[np.newaxis, :, :]
        behaviour_index = np.sum(behaviour_mask, axis = 1, dtype = int)
        coefficients_all = [
            np.sum(np.array([(behaviour_index == i) * c.magnitude[np.newaxis, i, :] for i in np.unique(behaviour_index)]), axis = 0) * c.units
            for c in self.coefficients
            ]
        coefficients_all[3] = coefficients_all[3] - displacement_per_root

        force_unbroken = _solve_cubic_polynomial(*coefficients_all)
        if self.surface is False:
            mask_el_slip = (behaviour_index == 2)
            if np.any(mask_el_slip):
                _, root_index = np.where(mask_el_slip)
                force_unbroken[mask_el_slip] = self.force_limits[1, root_index]
            mask_pl_slip = (behaviour_index == 5)
            if np.any(mask_pl_slip):
                _, root_index = np.where(mask_pl_slip)
                force_unbroken[mask_pl_slip] = self.force_limits[4, root_index]
        if jacobian is True:
            dforceunbroken_ddisplacement = np.zeros((ndisplacements, nroots)) * units('N/mm')
            if self.surface is True:
                mask = (behaviour_index >= 0)
            else:
                mask = ~np.isin(behaviour_index, [0, 2, 5])
            dforceunbroken_ddisplacement[mask] = 1.0 / (
                3.0 * coefficients_all[0][mask] * force_unbroken[mask]**2
                + 2.0 * coefficients_all[1][mask] * force_unbroken[mask]
                + coefficients_all[2][mask]
                )

        if self.breakage is True:
            force_unbroken_cummax = force_unbroken.copy()
            if self.surface is True:
                mask_el_reducing = np.isin(behaviour_index, [2, 3])
                _, root_index = np.where(mask_el_reducing)
                force_unbroken_cummax[mask_el_reducing] = self.force_limits[1, root_index]
                mask_pl_reducing = np.isin(behaviour_index, [5, 6, 7])
                _, root_index = np.where(mask_pl_reducing)
                force_unbroken_cummax[mask_pl_reducing] = self.force_limits[4, root_index]
            force_breakage = self.roots.xsection * self.roots.tensile_strength
            if self.weibull_shape is None:
                survival = (force_unbroken_cummax <= force_breakage).astype(float)
            else:
                y = (gamma(1.0 + 1.0 / self.weibull_shape) 
                     * force_unbroken_cummax 
                     / force_breakage
                     ).magnitude                   
                survival = np.exp(-(y**self.weibull_shape))
            if jacobian is True:
                dforceunbrokencummax_ddisplacement = dforceunbroken_ddisplacement.copy()
                if self.surface is True:
                    dforceunbrokencummax_ddisplacement[mask_el_reducing] = 0.0 * units('N/mm')
                    dforceunbrokencummax_ddisplacement[mask_pl_reducing] = 0.0 * units('N/mm')
                if self.weibull_shape is None:
                    dsurvival_ddisplacement = np.zeros((ndisplacements, nroots)) * units('1/mm')
                else:
                    dy_dforceunbrokencummax = gamma(1.0 + 1.0 / self.weibull_shape) / force_breakage
                    dsurvival_dy = -self.weibull_shape * y**(self.weibull_shape - 1.0) * survival
                    dsurvival_ddisplacement = dsurvival_dy * dy_dforceunbrokencummax * dforceunbrokencummax_ddisplacement
        else:
            survival = np.zeros_like(displacement_per_root, dtype = float)
            if jacobian is True:
                dsurvival_ddisplacement = np.zeros((ndisplacements, nroots)) * units('1/mm')

        output = {
            'displacement_per_root': displacement_per_root,
            'force_per_root': force_unbroken * survival,
            'behaviour_index': behaviour_index,
            'survival_fraction': survival
            }
        if jacobian is True:
            output['dforce_per_root_ddisplacement_per_root'] = (
                dforceunbroken_ddisplacement * survival
                + force_unbroken * dsurvival_ddisplacement
                )
        if is_scalar_displacement is True:
            output = {k: v[0, ...] for k, v in output.items()}
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output

    def calc_displacement_to_peak(
            self,
            results: str = "attribute"
            ):
        """Calculate the displacement to peak, for in each root

        Calculates the displacement required for the each each in the 
        MultipleRoots object to reach the largest force it will even reach
        as function of *any* displacement level.

        The function does (currently) not take the survival function into 
        account, i.e. it looks at the *average* root for each root in the 
        MultipleRoots object. 

        Function creates a dictionary with the key `peak_displacement_per_root`/
        How this dictionary is returned depends on the value of the `results` 
        argument.

        Parameters
        ----------
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.
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
        displacement_peak = np.minimum(displacement_slipping, displacement_breakage)
        output = {'peak_displacement_per_root': displacement_peak}
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output