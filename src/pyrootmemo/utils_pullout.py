## FUNCTIONS FOR CALCULATING PULLOUT FORCES
## 11/11/2024 - GJM


# load packages
import numpy as np
from scipy.special import gamma


# Weibull survival function (if weibull shape not defined, then Heaviside, i.e. sudden breakage)
def survival(
        load: int | float | np.ndarray, 
        capacity: int | float | np.ndarray, 
        weibull_shape = None
        ) -> float | np.ndarray:
    if weibull_shape is None:
        return(1.0 * (load <= capacity))
    else:
        weibull_scale = capacity / gamma(1. + 1. / weibull_shape)
        return(np.exp(-(load / weibull_scale) ** weibull_shape))


# Calculate force in root pulled out of the ground surface
def pulloutforce_surface(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        elastic_modulus,
        tensile_strength = None,
        yield_strength = None,
        plastic_modulus = None,
        root_length_soil = None,
        root_length_surface = 0.0,
        weibull_shape_parameter = None,
        survival_fraction_previous = None
        ):
    
    ## deal with roots in compression - set pullout displacement to zero to model zero tensile force
    # set boolean
    compression_boolean = (pullout_displacement < 0.)
    # set pullout displacement to zero, to avoid numerical issues with pullout equations
    pullout_displacement = np.maximum(pullout_displacement, 0.)
    
    ## calculate pull-out force for the different behaviour 'modes'
    # anchored, elastic
    force0 = pulloutforce_surface_anchored_elastic(
        pullout_displacement, interface_resistance, root_xsection, 
        root_circumference, elastic_modulus,
        root_length_surface = root_length_surface
        )
    # anchored, elasto-plastic
    if (plastic_modulus is not None) & (yield_strength is not None):
        force1 = pulloutforce_surface_anchored_elastoplastic(
            pullout_displacement, interface_resistance, root_xsection, 
            root_circumference, yield_strength, 
            elastic_modulus, plastic_modulus,
            root_length_surface = root_length_surface
            )
    else:
        force1 = None
    # slipping, elastic
    if root_length_soil is not None:
        force2 = pulloutforce_surface_slipping_elastic(
            pullout_displacement, interface_resistance, root_xsection, 
            root_circumference, elastic_modulus, root_length_soil,
            root_length_surface = root_length_surface
            )
    else:
        force2 = None
    # slipping, elasto-plastic
    if (plastic_modulus is not None) & (yield_strength is not None) & (root_length_soil is not None):
        force3 = pulloutforce_surface_slipping_elastoplastic(
            pullout_displacement, interface_resistance, root_xsection, 
            root_circumference, yield_strength, 
            elastic_modulus, plastic_modulus, root_length_soil
            root_length_surface = root_length_surface
            )
    else:
        force3 = None
    # combine all 'modes' into a single array
    forces_unbroken = np.stack(np.array([i for i in (force0, force1, force2, force3) if i is not None]))
    # find correct 'modes' - this will always be the one resulting in the lowest forces
    mode = 1 + np.argmin(forces_unbroken, axis = 0)    
    mode[compression_boolean] = 0
    force_unbroken = np.min(forces_unbroken, axis = 0) 
    # calculate fraction of root(s) that are still intact under the current force
    if tensile_strength is not None:
        tensile_capacity = tensile_strength * root_xsection
        survival_fraction = survival(
            force_unbroken,
            tensile_capacity,
            weibull_shape_parameter
            )
    else:
        # root breakage not accounted for
        survival_fraction = np.ones_like(force_unbroken)
    # account for user-defined survival fraction, resulting from previous steps
    if survival_fraction_previous is not None:
        survival_fraction = np.minimum(survival_fraction, survival_fraction_previous)
    # final pullout force in each root
    force = force_unbroken * survival_fraction
    # generate array with fractions of root behaving in each 'mode'
    mode_fraction = np.zeros((*force.shape, 6))
    mode_fraction[np.arange(len(force)), mode] = survival_fraction
    mode_fraction[..., 5] = 1. - survival_fraction
    # return
    return(force, mode_fraction)


# 'Surface' roots - Anchored, elastic behaviour
def pulloutforce_surface_anchored_elastic(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        elastic_modulus,
        root_length_surface = 0.0
        ):
    # cubic polynomial terms
    xi3 = 1./(2. * elastic_modulus**2 * root_xsection**2 * root_circumference * interface_resistance)
    xi2 = 1./(2. * elastic_modulus * root_xsection * root_circumference * interface_resistance)
    xi1 = root_length_surface / (elastic_modulus * root_xsection)
    xi0 = -pullout_displacement * np.ones_like(xi1)
    # solve cubic equation
    xi = np.stack((xi3, xi2, xi1, xi0))
    force_anchored_elastic = _solve_cubic(xi.squeeze())
    # return forces
    return(force_anchored_elastic)

    
# 'Surface' roots - Slipping, elastic behaviour
def pulloutforce_surface_slipping_elastic(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        elastic_modulus,
        root_length_soil,
        root_length_surface = 0.0
        ):
    # quadratic polynomial terms
    xi2 = -1. / (elastic_modulus * root_xsection * root_circumference * interface_resistance)
    xi1 = ((root_length_soil + root_length_surface) / (elastic_modulus * root_xsection)
           - 1. / (root_circumference * interface_resistance))
    xi0 = root_length_soil - pullout_displacement
    # initiate output vector
    force_slippage_elastic = np.zeros_like(xi0 + xi1 + xi2)
    # check if root not yet fully pulled out
    mask = (pullout_displacement < root_length_soil)
    # solve cubic equation
    if np.any(mask) == True:
        xi = np.vstack((xi2, xi1, xi0))
        if np.all(mask) == True:
            force_slippage_elastic = _solve_quadratic(xi.squeeze())
        else:
            force_slippage_elastic[mask] = _solve_quadratic(xi[mask,...].squeeze())
    # return forces    
    return(force_slippage_elastic)

    
# 'Surface' roots - Anchored, elasto-plastic behaviour
def pulloutforce_surface_anchored_elastoplastic(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        yield_strength,
        elastic_modulus,
        plastic_modulus,
        root_length_surface = 0.0
        ):
    # calculate root force at yielding
    yield_force = yield_strength * root_xsection
    # cubic polynomial terms
    xi3 = 1. / (2. * plastic_modulus**2 * root_xsection**2 * root_circumference * interface_resistance)
    xi2 = ((1. + 3. * yield_force / (elastic_modulus * root_xsection) 
            - 3. * yield_force / (plastic_modulus * root_xsection))
           / (2. * plastic_modulus * root_xsection * root_circumference * interface_resistance))
    xi1 = (((2. * root_circumference * interface_resistance * root_length_surface)
           + 2. * yield_force * (plastic_modulus / elastic_modulus - 1.)
           + yield_force**2 / (plastic_modulus * root_xsection)
           * (2. * plastic_modulus**2 / elastic_modulus**2 - 5. * plastic_modulus / elastic_modulus + 3.)
           ) / (2. * plastic_modulus * root_xsection * root_circumference * interface_resistance))
    xi0 = (((yield_force - 2. * root_circumference * interface_resistance * root_length_surface)
            * (elastic_modulus - plastic_modulus)
            - yield_force**2 * (elastic_modulus - plastic_modulus)**2 
            / (elastic_modulus * plastic_modulus * root_xsection)
            ) / (2. * elastic_modulus * plastic_modulus * root_xsection * root_circumference * interface_resistance)
           - pullout_displacement)
    # solve cubic equation
    xi = np.vstack((xi3, xi2, xi1, xi0))
    force_anchored_elastoplastic = _solve_cubic(xi.squeeze())
    # return forces
    return(force_anchored_elastoplastic)
    

# 'Surface' roots - Slipping, elasto-plastic behaviour
def pulloutforce_surface_slipping_elastoplastic(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        yield_strength,
        elastic_modulus,
        plastic_modulus,
        root_length_soil,
        root_length_surface = 0.0
        ):
    # calculate root force at yielding
    yield_force = yield_strength * root_xsection
    # quadratic polynomial terms
    xi2 = -1. / (plastic_modulus * root_xsection * root_circumference * interface_resistance)
    xi1 = ((root_length_soil + root_length_surface) / (plastic_modulus * root_xsection)
           - 1. / (root_circumference * interface_resistance)
           + yield_force / (plastic_modulus * root_xsection * root_circumference * interface_resistance)
           - yield_force / (elastic_modulus * root_xsection * root_circumference * interface_resistance))
    xi0 = (root_length_soil 
           + (root_length_soil + root_length_surface) * yield_force / (elastic_modulus * root_xsection)
           - (root_length_soil + root_length_surface) * yield_force / (plastic_modulus * root_xsection)
           - pullout_displacement)
    # initiate output vector
    force_slippage_elastoplastic = np.zeros_like(xi0 + xi1 + xi2)
    # check if root not yet fully pulled out
    mask = (pullout_displacement < root_length_soil)
    # solve quadratic equation
    if np.any(mask) == True:
        xi = np.vstack((xi2, xi1, xi0))
        if np.all(mask) == True:
            force_slippage_elastoplastic = _solve_quadratic(xi.squeeze())
        else:
            force_slippage_elastoplastic[mask] = _solve_quadratic(xi[mask,...].squeeze())
    # return forces
    return(force_slippage_elastoplastic)


# 'Embedded' roots - Anchored, elastic behaviour
def pulloutforce_embedded_anchored_elastic(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        elastic_modulus,
):
    # polynomial coefficients
    xi2 = 1. / (2. * elastic_modulus * root_xsection * root_circumference * interface_resistance)
    # solve and return
    return(np.sqrt(pullout_displacement / xi2))


# 'Embedded' roots - Anchored, elasto-plastic behaviour
def pulloutforce_embedded_anchored_elastoplastic(
        pullout_displacement,
        interface_resistance,
        root_xsection,
        root_circumference,
        yield_strength,
        elastic_modulus,
        plastic_modulus
):
    # polynomial coefficients
    xi2 = 1. / (2. * plastic_modulus * root_xsection * root_circumference * interface_resistance)
    xi1 = (yield_strength / (elastic_modulus * root_circumference * interface_resistance)
           - yield_strength / (plastic_modulus * root_circumference * interface_resistance))
    xi0 = (-yield_strength**2 * root_xsection / (2. * elastic_modulus * root_circumference * interface_resistance)
           + yield_strength**2 * root_xsection / (2. * plastic_modulus * root_circumference * interface_resistance)
           - pullout_displacement)
    # solve for displacements
    xi = np.vstack((xi2, xi1, xi0))
    force_anchored_elastoplastic = _solve_quadratic(xi)
    # return
    return(force_anchored_elastoplastic)


# 'Embedded' roots - Slipping, behaviour
def pulloutforce_embedded_slipping(
        interface_resistance,
        root_circumference,
        root_length,
):
    # calculate force
    force_slippage = root_circumference * root_length * interface_resistance
    # return
    return(force_slippage)


# solve quadratic equations for pullout equations
def _solve_quadratic(
        coef: np.ndarray
        ) -> float | np.ndarray:
    ## solve equations a*x^2 + b*x^2 + c = 0 in DRAM
    ## * a = always negative in DRAM equations
    ## function returns largest root
    a, b, c = coef
    discriminant = b**2 - 4.*a*c
    x = (-b - np.sqrt(discriminant)) / (2.*a)
    return(x)
    

# solve cubic equations for DRAM
def _solve_cubic(
        coef: np.ndarray
        ) -> float | np.ndarray:
    ## solve equations a*x^2 + b*x^2 + c = 0
    ## * a = always positive in DRAM equations
    # unpack parameters
    a, b, c, d = coef
    # construct empty output vector
    if coef.ndim > 1:
        x = np.empty_like(a)
    else:
        x = np.array([0.])
    # new parameters - so that: x^3 + e*x^2 + f*x + g = 0
    e = b/a
    f = c/a
    g = d/a
    # temporary values
    Q = (e**2 - 3.*f) / 9.
    R = (2.*e**3 - 9.*e*f + 27.*g)/54.
    # check if one (False) or three (True) real roots exist
    mask = (R**2 < Q**3)
    # three real roots - calculate largest value
    if np.any(mask == True):
        theta = np.arccos(R[mask] / np.sqrt(Q[mask]**3))
        x[mask] = -2.*np.sqrt(Q[mask])*np.cos((theta + 2.*np.pi)/3.) - e[mask]/3.
    # one real root
    if np.any(mask == False):
        A = (-np.sign(R[~mask])
             * (np.abs(R[~mask]) + np.sqrt(R[~mask]**2 - Q[~mask]**3))**(1/3))
        B = Q[~mask]/A
        x[~mask] = (A + B) - e[~mask]/3.
    # x=0 solution (when d==0)
    x[d == 0.] = 0.
    # return solution(s)
    if coef.ndim > 1:
        return(x)
    else:
        return(x[0])
    