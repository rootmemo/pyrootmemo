# import packages
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from pyrootmemo.helpers import units
from pyrootmemo.materials import MultipleRoots, Interface
from pint import Quantity


def solve_quadratic(
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


def solve_cubic(
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
    flag_3roots = (R**2) < (Q**3) # if true, 3 real roots exist, if false, only one real root exists
    theta = np.arccos(R[flag_3roots] / np.sqrt(Q[flag_3roots]**3))
    x[flag_3roots] = (
        -2.0 
        * np.sqrt(Q[flag_3roots]) 
        * np.cos((theta + 2.0 * np.pi) / 3.0) 
        - e[flag_3roots] 
        / 3.0
        )
    flag_1root = not flag_3roots
    A = (
        -np.sign(R[flag_1root]) 
        * (
            np.abs(R[flag_1root]) 
            + np.sqrt(R[flag_3roots]**2 - Q[flag_1root]**3)
            ) ** (1.0 / 3.0)
        )
    B = Q[flag_1root] / A
    x[flag_1root] = (A + B) - e[flag_1root] / 3.0
    flag_zero = np.isclose(d.magnitude, 0.0)
    x[flag_zero] = 0.0 *d.units / c.units
    return(x)
   

class Pullout():

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
        # set parameters
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
        self.weibull_shape = weibull_shape

        # get polynomial coefficients, displacement limits and behaviour types
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
            # anchored, elastic [1]
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
                # slipping, elastic [2]
                coefficients[1][2, ...] = (
                    -1.0 / (roots.elastic_modulus * roots.xsection
                            * roots.circumference * interface.shear_strength)
                    )
                coefficients[2][2, ...] = (
                     roots.length / (roots.elastic_modulus * roots.xsection)
                     - 1.0 / (roots.circumference * interface.shear_strength)
                     )
                coefficients[3][2, ...] = roots.length - roots.length_surface
                # displacement at start of slippage, elastic <1>
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
                # displacement until full pullout, elastic <2>
                displacement_limits[2, :] = roots.length - roots.length_surface
            if elastoplastic is True:
                # force and displacement at yield <3>
                force_limits[3, :] = roots.xsection * roots.yield_strength
                displacement_limits[3, :] = (
                    coefficients[0][1, ...] * force_limits[3, :]**3
                    + coefficients[1][1, ...] * force_limits[3, :]**2
                    + coefficients[2][1, ...] * force_limits[3, :]
                    + coefficients[3][1, ...]
                    )
                # anchored, plastic [4]
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
                    # force and displacement at start of slipping, plastic <4>
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
                    # slipping, plastic, above yield (5)
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
                    # force and displacement to yield during plastic unloading <5>
                    force_limits[5, :] = force_limits[3, :]
                    displacement_limits[5, :] = (
                        coefficients[0][5, ...] * force_limits[5, :]**3
                        + coefficients[1][5, ...] * force_limits[5, :]**2
                        + coefficients[2][5, ...] * force_limits[5, :]
                        + coefficients[3][5, ...]
                    )                    
                    # slipping, plastic, below yield (6)
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
                    # displacement until full pull-out, plastic <6>
                    displacement_limits[6, :] = coefficients[3][6, ...]
                    # adjust displacement limits: slippage before yielding --> never plasticity
                    slip_before_yield = (displacement_limits[1, ...] <= displacement_limits[3, ...])
                    displacement_limits[3:7, slip_before_yield] = np.inf * units('mm')
                    # adjust slippage after yielding --> never elastic slippage
                    yield_before_slip = ~slip_before_yield
                    displacement_limits[1:3, yield_before_slip] = displacement_limits[3, yield_before_slip]
                    force_limits[1:3, yield_before_slip] = force_limits[3, yield_before_slip]
        else:
            ## EMBEDDED ROOTS
            # anchored, elastic [1]
            coefficients[1][1, ...] = (
                1.0 / (2.0 * roots.elastic_modulus * roots.xsection 
                       * roots.circumference * interface.shear_strength)
                )
            if slipping is True:
                # slipping, elastic [2]
                force_limits[1, :] = roots.length * roots.circumference * interface.shear_strength
                # displacement at start of slippage, elastic <1>
                displacement_limits[1, :] = coefficients[1][1, ...] * force_limits[1, :]**2
            if elastoplastic is True:
                # displacement at yield <3>
                force_limits[3, :] = roots.xsection * roots.yield_strength
                displacement_limits[3, :] = coefficients[1][1, ...] * force_limits[3, :]**2
                # anchored, plastic [4]
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
                    # displacement at start of slippage, plastic <4>
                    displacement_limits[4, :] = (
                        coefficients[1][4, ...] * force_limits[1, :]**2
                        + coefficients[2][4, ...] * force_limits[1, :]
                        + coefficients[3][4, ...]
                        )
                    force_limits[4, :] = force_limits[1, :]
                    # adjust displacement limits to ensure correct order of behaviours
                    slip_before_yield = displacement_limits[1, ...] <= displacement_limits[3, ...]
                    displacement_limits[3:5, slip_before_yield] = np.inf * units('mm')
                    force_limits[3:5, slip_before_yield] = force_limits[1, slip_before_yield]
                    yield_before_slip = ~slip_before_yield
                    displacement_limits[1, yield_before_slip] = displacement_limits[3, yield_before_slip]
                    force_limits[1, yield_before_slip] = force_limits[3, yield_before_slip]

        # for displacement limits that are not needed, add dummy values based on 'next' displacement limit
        mask = np.isclose(displacement_limits[-1, ...].magnitude, 0.0)
        displacement_limits[-1, mask] = np.inf * units('mm')
        for i in np.flip(np.arange(1, 6)):
            mask = np.isclose(displacement_limits[i, ...].magnitude, 0.0)
            displacement_limits[i, mask] = displacement_limits[i + 1, mask]
            force_limits[i, mask] = force_limits[i + 1, mask]
        # assign calculated values
        self.coefficients = coefficients
        self.behaviour_types = behaviour_types
        self.displacement_limits = displacement_limits
        self.force_limits = force_limits

    def calc_force(
            self,
            displacement: Quantity
            ):
        nroots = self.roots.xsection.shape
        behaviour_index = np.sum(displacement > self.displacement_limits, axis = 0).astype(int)
        force_unbroken = np.zeros(*nroots) * units('N')
        if self.surface is True:
            ## SURFACE ROOTS
            mask_el_anch = (behaviour_index == 1)
            force_unbroken[mask_el_anch] = solve_cubic(
                self.coefficients[0][1, mask_el_anch],
                self.coefficients[1][1, mask_el_anch],
                self.coefficients[2][1, mask_el_anch],
                (self.coefficients[3][1, ...] - displacement)[mask_el_anch]
                )
            if self.slipping is True:
                mask_el_slip = (behaviour_index == 2)
                force_unbroken[mask_el_slip] = solve_quadratic(
                    self.coefficients[1][2, mask_el_slip],
                    self.coefficients[2][2, mask_el_slip],
                    (self.coefficients[3][2, ...] - displacement)[mask_el_slip]
                )
            if self.elastoplastic is True:
                mask_pl_anch = (behaviour_index == 4)
                force_unbroken[mask_pl_anch] = solve_cubic(
                    self.coefficients[0][4, mask_pl_anch],
                    self.coefficients[1][4, mask_pl_anch],
                    self.coefficients[2][4, mask_pl_anch],
                    (self.coefficients[3][4, ...] - displacement)[mask_pl_anch]
                    )
                if self.slipping is True:
                    mask_pl_slip_aboveyield = (behaviour_index == 5)
                    force_unbroken[mask_pl_slip_aboveyield] = solve_quadratic(
                        self.coefficients[1][5, mask_pl_slip_aboveyield],
                        self.coefficients[2][5, mask_pl_slip_aboveyield],
                        (self.coefficients[3][5, ...] - displacement)[mask_pl_slip_aboveyield]
                    )
                    mask_pl_slip_belowyield = (behaviour_index == 6)
                    force_unbroken[mask_pl_slip_belowyield] = solve_quadratic(
                        self.coefficients[1][5, mask_pl_slip_belowyield],
                        self.coefficients[2][5, mask_pl_slip_belowyield],
                        (self.coefficients[3][5, ...] - displacement)[mask_pl_slip_belowyield]
                    )
            force_unbroken_cummax = force_unbroken
            mask_el_reducing = behaviour_index in [2, 3]
            force_unbroken_cummax[mask_el_reducing] = self.force_limits[1, mask_el_reducing]
            mask_pl_reducing = behaviour_index in [5, 6, 7]
            force_unbroken_cummax[mask_pl_reducing] = self.force_limits[4, mask_pl_reducing]
        else:
            ## EMBEDDED ROOTS
            mask_el_anch = (behaviour_index == 1)
            force_unbroken[mask_el_anch] = np.sqrt(
                (displacement / self.coefficients[1][1, ...])[mask_el_anch]
                )
            if self.slipping is True:
                mask_el_slip = (behaviour_index == 2)
                force_unbroken[mask_el_slip] = (
                    self.roots.length[mask_el_slip]
                    * self.roots.circumference[mask_el_slip]
                    * self.interface.shear_strength
                    )
            if self.elastoplastic is True:
                mask_pl_anch = (behaviour_index == 4)
                force_unbroken[mask_pl_anch] = solve_quadratic(
                    self.coefficients[1][4, mask_pl_anch],
                    self.coefficients[2][4, mask_pl_anch],
                    (self.coefficients[3][4, ...] - displacement)[mask_pl_anch]
                    )
                if self.slipping is True:
                    mask_pl_slip = (behaviour_index == 5)
                    force_unbroken[mask_pl_slip] = (
                        self.roots.length[mask_pl_slip]
                        * self.roots.circumference[mask_pl_slip]
                        * self.interface.shear_strength
                        )
            force_unbroken_cummax = force_unbroken

        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
            if self.weibull_shape is None:
                survival = (force_unbroken_cummax <= force_breakage).astype(float)
            else:
                y = (force_unbroken_cummax / force_breakage * gamma(1.0 + 1.0 / self.weibull_shape)).magnitude                   
                survival = np.exp(-(y**self.weibull_shape))
        else:
            survival = np.ones(nroots).astype(float)            

        return({
            'force': force_unbroken * survival,
            'behaviour_index': behaviour_index,
            'survival_fraction': survival
        })  