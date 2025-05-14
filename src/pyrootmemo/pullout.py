# import packages
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from pyrootmemo.tools.helpers import units
from pint import Quantity


# DATA
BEHAVIOUR_NAMES = [
    'Not in tension',
    'Anchored, elastic',
    'Slipping, elastic',
    'Full pullout', # (elastic)
    'Anchored, plastic',   
    'Slipping, plastic', # (above yield above)
    'Slipping, plastic', # (below yield force)
    'Full pullout' # (plastic)
]


########################
### EMBEDDED ELASTIC ###
########################


# EMBEDDED - ELASTIC
class PulloutEmbeddedElastic():

    def __init__(
            self,
            roots,
            interface,
            **kwargs
            ):
        # check input, then assign
        self._check_input(roots, interface)
        self.roots = roots
        self.interface = interface
        # set behaviour types, coefficients and limits
        self._set_behaviour_types()
        self._set_coefficients()
        self._set_limits()
        # set keyword arguments
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _check_contains_attributes(
            self,
            object, 
            attributes
            ):
        missing = [a for a in attributes if not hasattr(object, a)]
        if len(missing) > 0:
            error_message = "Object of type {0} is missing attributes {1}".format(type(object), missing)
            raise(TypeError(error_message))

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _nroots(self):
        return(len(self.roots.xsection.magnitude))

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1]]

    def _get_coefficients_notintension(self):
        nroots = self._nroots()
        c2 = np.zeros(nroots) * units['m/N^2']
        c1 = np.zeros(nroots) * units['m/N']
        c0 = np.zeros(nroots) * units['m']
        return(c2, c1, c0)

    def _get_coefficients_anchored_elastic(self):
        nroots = self._nroots()
        c2 = 1.0 / (2.0 * self.roots.elastic_modulus * self.roots.xsection 
                    * self.roots.circumference * self.interface.shear_strength)
        c1 = np.zeros(nroots) * units['m/N']
        c0 = np.zeros(nroots) * units['m']
        return(c2, c1, c0)

    def _combine_coefficients(
            self,
            list
            ):
        return([np.column_stack(x) for x in zip(*list)])

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic()
            ])

    def _get_limits_notintension(self):
        nroots = self._nroots()
        force = np.zeros(nroots) * units['N']
        displacement = np.zeros(nroots) * units['m']
        return(displacement, force)

    def _set_limits(self):
        self.limits = self._combine_coefficients([
            self._get_limits_notintension()
            ])
        
    def _get_behaviour_index(
            self, 
            displacement
            ):
        return((displacement > self.limits[0]).sum(axis = 1))

    def _get_force_unbroken(
            self, 
            displacement, 
            jac = False
        ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # derivative of force with respect to pullout displacement
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros_like(force_unbroken) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (
                1.0 
                / (2.0 * force_unbroken[index1] * self.coefficients[0][index1, 1])
                )
        # return
        return(
            force_unbroken, 
            dforceunbroken_ddisplacement, 
            behaviour_index
            )

    def _get_survival(
            self, 
            force, 
            behaviour_index,
            jac = False,
            ):
        survival = np.ones_like(force)
        if jac is False:
            dsurvival_dforce = None
        else:
            dsurvival_dforce = np.zeros_like(survival) * units('1/N')
        return(survival, dsurvival_dforce)
    
    def force(
            self,
            displacement,
            jac = False
    ):
        force_unbroken, dforceunbroken_ddisplacement, behaviour_index = self._get_force_unbroken(
            displacement, 
            jac = jac
            )
        survival, dsurvival_dforceunbroken = self._get_survival(
            force_unbroken, 
            behaviour_index,
            jac = jac
            )
        force = force_unbroken * survival
        if jac is False:
            dforce_ddisplacement = None
        else:
            dforce_ddisplacement = (
                dforceunbroken_ddisplacement * survival
                + force * dsurvival_dforceunbroken * dforceunbroken_ddisplacement
            )
        return(
            force,
            dforce_ddisplacement,
            survival, 
            behaviour_index
        )

    def _solve_quadratic(
            self, 
            a, 
            b, 
            c
            ):
        # solve equations a*x^2 + b*x^2 + c = 0 in DRAM/Pullout
        # function assumes that a > 0
        # function always returns the largest root
        discriminant = b**2 - 4.0 * a * c
        x = (-b + np.sqrt(discriminant)) / (2.0 * a)
        return(x)
    

# EMBEDDED - ELASTIC, SLIPPING
class PulloutEmbeddedElasticSlipping(PulloutEmbeddedElastic):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'length']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 2]]

    def _get_coefficients_slipping(self):
        nroots = self._nroots()
        c2 = np.zeros(nroots) * units['m/N^2']
        c1 = np.zeros(nroots) * units['m/N']
        c0 = np.inf * np.ones(nroots) * units['m']
        return(c2, c1, c0)

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic(),
            self._get_coefficients_slipping()
            ])

    def _get_limits_elastic_slipping(self):
        force = self.roots.length * self.roots.circumference * self.interface.shear_strength
        c2, _, _ = self._get_coefficients_anchored_elastic()
        displacement = c2 * force**2
        return(displacement, force)

    def _set_limits(self):
        self.limits = self._combine_coefficients([
            self._get_limits_notintension(),
            self._get_limits_elastic_slipping()
            ])
        
    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # slipping behaviour
        index2 = (behaviour_index == 2)
        force_unbroken[index2] = self.limits[1][index2, 1]
        # derivatives
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros_like(force_unbroken) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (
                1.0 
                / (2.0 * force_unbroken[index1] * self.coefficients[0][index1, 1])
                )
        # return
        return(
            force_unbroken, 
            dforceunbroken_ddisplacement,
            behaviour_index,             
            )
    

# EMBEDDED - ELASTIC, BREAKAGE
class PulloutEmbeddedElasticBreakage(PulloutEmbeddedElastic):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'tensile_strength']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _get_survival(
            self, 
            force, 
            behaviour_index,
            jac = False
            ):
        stress = force / self.roots.xsection
        if hasattr(self, 'weibull_shape'):
            weibull_shape = self.weibull_shape
            weibull_scale = self.roots.tensile_strength / gamma(1.0 + 1.0 / weibull_shape)
            survival = np.exp(-(stress / weibull_scale)**weibull_shape)
            if jac is False:
                dsurvival_dforce = None
            else:
                dstress_dforce = 1.0 / self.roots.xsection
                dsurvival_dstress = (
                    -weibull_shape / weibull_scale
                    * (stress / weibull_scale) ** (weibull_shape - 1.0)
                    * survival
                )
                dsurvival_dforce = dsurvival_dstress * dstress_dforce
        else:
            survival = (stress <= self.roots.tensile_strength).astype('float')
            if jac is False:
                dsurvival_dforce = None
            else:
                dsurvival_dforce = np.zeros(self._nroots()) * units('1/N')
        return(survival, dsurvival_dforce)


# EMBEDDED - ELASTIC, BREAKAGE, SLIPPING
class PulloutEmbeddedElasticBreakageSlipping(
    PulloutEmbeddedElasticSlipping,
    PulloutEmbeddedElasticBreakage
):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'length', 'tensile_strength']
            )
        self._check_contains_attributes(interface, ['shear_strength'])


###############################
### EMBEDDED ELASTO_PLASTIC ###
###############################

# EMBEDDED, ELASTOPLASTIC
class PulloutEmbeddedElastoplastic(PulloutEmbeddedElastic):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'plastic_modulus']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 3]]

    def _get_coefficients_anchored_plastic(self):
        c2 = (
            1.0 
            / (2.0 * self.roots.plastic_modulus * self.roots.xsection 
               * self.roots.circumference * self.interface.shear_strength)
            )
        c1 = (
            self.roots.yield_strength 
            / (self.roots.elastic_modulus * self.roots.circumference * self.interface.shear_strength)
            - self.roots.yield_strength 
            / (self.roots.plastic_modulus * self.roots.circumference * self.interface.shear_strength)
            )
        c0 = (
            -self.roots.yield_strength**2 * self.roots.xsection 
            / (2.0 * self.roots.elastic_modulus * self.roots.circumference * self.interface.shear_strength)
            + self.roots.yield_strength**2 * self.roots.xsection 
            / (2.0 * self.roots.plastic_modulus * self.roots.circumference * self.interface.shear_strength)
            )
        return(c2, c1, c0)
    
    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic(),
            self._get_coefficients_anchored_plastic()
            ])

    def _get_limits_yield_anchored(self):
        force = self.roots.yield_strength * self.roots.xsection
        c2, _, _ = self._get_coefficients_anchored_elastic()
        displacement = c2 * force**2
        return(displacement, force)
    
    def _set_limits(self):
        self.limits = self._combine_coefficients([
            self._get_limits_notintension(),
            self._get_limits_yield_anchored()
            ])

    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # plastic behaviour
        index2 = (behaviour_index == 2)
        force_unbroken[index2] = self._solve_quadratic(
            self.coefficients[0][index2, 2],
            self.coefficients[1][index2, 2],
            self.coefficients[2][index2, 2] - displacement
        )
        # derivative of force with respect to displacement
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros(nroots) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (
                1.0
                / (2.0 * self.coefficients[0][index1, 1] * force_unbroken[index1])                
            )
            dforceunbroken_ddisplacement[index2] = (
                1.0
                / (2.0 * self.coefficients[0][index2, 2] * force_unbroken[index2]
                   + self.coefficients[1][index2, 2])
                )
        # return
        return(
            force_unbroken,
            dforceunbroken_ddisplacement,
            behaviour_index
        )
    

# EMBEDDED, ELASTOPLASTIC, SLIPPING
class PulloutEmbeddedElastoplasticSlipping(
    PulloutEmbeddedElastoplastic,
    PulloutEmbeddedElasticSlipping
    ):

    def _check_input(
        self,
        roots,
        interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'plastic_modulus',
             'length']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 2, 4, 5]]

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic(),
            self._get_coefficients_slipping(),
            self._get_coefficients_anchored_plastic(),
            self._get_coefficients_slipping()
            ])

    def _get_limits_plastic_slipping(self):
        force = self.roots.length * self.roots.circumference * self.interface.shear_strength
        c2, c1, c0 = self._get_coefficients_anchored_plastic()
        displacement = c2 * force**2 + c1 * force + c0
        return(displacement, force)
    
    def _set_limits(self):
        # calculate limits
        displacement_limits, force_limits = self._combine_coefficients([
            self._get_limits_notintension(),
            self._get_limits_elastic_slipping(),
            self._get_limits_yield_anchored(),
            self._get_limits_plastic_slipping()
            ])
        # adjust: slippage before yielding --> never plasticity
        index = displacement_limits[:, 1] <= displacement_limits[:, 2]
        displacement_limits[index, 2] = np.inf * units('mm')
        displacement_limits[index, 3] = np.inf * units('mm')
        # adjust slippage after yielding --> never elastic slippage
        displacement_limits[~index, 1] = displacement_limits[~index, 2]
        # set limits
        self.limits = [displacement_limits, force_limits]
    
    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic anchored behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # plastic anchored behaviour
        index3 = (behaviour_index == 3)
        force_unbroken[index3] = self._solve_quadratic(
            self.coefficients[0][index3, 3],
            self.coefficients[1][index3, 3],
            self.coefficients[2][index3, 3] - displacement
        )
        # slipping behaviour
        index24 = (behaviour_index == 2) | (behaviour_index == 4)
        force_unbroken[index24] = self.limits[1][index24, 3]
        # return
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros(nroots) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (
                1.0
                / (2.0 * self.coefficients[0][index1, 1] * force_unbroken[index1])                
            )
            dforceunbroken_ddisplacement[index3] = (
                1.0
                / (2.0 * self.coefficients[0][index3, 3] * force_unbroken[index3]
                   + self.coefficients[1][index3, 3])
                )
        # return
        return(
            force_unbroken,
            dforceunbroken_ddisplacement,
            behaviour_index            
        )
    

# EMBEDDED, ELASTOPLASTIC, BREAKAGE
class PulloutEmbeddedElastoplasticBreakage(
    PulloutEmbeddedElastoplastic,
    PulloutEmbeddedElasticBreakage
    ):

    def _check_input(
        self,
        roots,
        interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'plastic_modulus',
             'tensile_strength']
            )
        self._check_contains_attributes(interface, ['shear_strength'])


# EMBEDDED, ELASTOPLASTIC, BREAKAGE, SLIPPING
class PulloutEmbeddedElastoplasticBreakageSlipping(
    PulloutEmbeddedElastoplasticSlipping,
    PulloutEmbeddedElastoplasticBreakage,
    ):

    def _check_input(
        self,
        roots,
        interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'plastic_modulus',
             'tensile_strength', 'length']
            )
        self._check_contains_attributes(interface, ['shear_strength'])



#####################
### SURFACE ROOTS ###
#####################

# SURFACE - ELASTIC
class PulloutSurfaceElastic(PulloutEmbeddedElastic):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'length_surface']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1]]

    def _get_coefficients_notintension(self):
        nroots = self._nroots()
        c3 = np.zeros(nroots) * units['m/N^3']
        c2 = np.zeros(nroots) * units['m/N^2']
        c1 = np.zeros(nroots) * units['m/N']
        c0 = np.zeros(nroots) * units['m']
        return(c3, c2, c1, c0)

    def _get_coefficients_anchored_elastic(self):
        nroots = self._nroots()
        c3 = 1.0 / (2.0 * (self.roots.elastic_modulus * self.roots.xsection)**2
                    * self.roots.circumference * self.interface.shear_strength)
        c2 = 1.0 / (2.0 * self.roots.elastic_modulus * self.roots.xsection
                    * self.roots.circumference * self.interface.shear_strength)
        c1 = self.roots.length_surface / (self.roots.elastic_modulus * self.roots.xsection)
        c0 = np.zeros(nroots) * units['m']
        return(c3, c2, c1, c0)

    def _solve_cubic(
            self, 
            a, 
            b, 
            c,
            d
            ):
        # solve equations a*x^3 + b*x^2 + c*x + d = 0 in DRAM/Pullout
        # a = always positive in DRAM/Pullout equations
        # function returns largest positive root
        # 
        # construct empty output vector
        x = np.zeros(a.shape) * units('N')
        # new parameters - so that: x^3 + e*x^2 + f*x + g = 0
        e = b / a
        f = c / a
        g = d / a
        # temporary values
        Q = (e**2 - 3.0 * f) / 9.0
        R = (2.0 * e**3 - 9.0 * e * f + 27.0 * g) / 54.0
        mask = (R**2) < (Q**3) # if true, 3 real roots exist, if false, only one real root exists
        # three real roots - calculate largest value
        theta = np.arccos(R[mask] / np.sqrt(Q[mask]**3))
        x[mask] = -2.0 * np.sqrt(Q[mask]) * np.cos((theta + 2.0*np.pi) / 3.0) - e[mask] / 3.0
        # one real root
        A = -np.sign(R[~mask]) * (np.abs(R[~mask]) + np.sqrt(R[~mask]**2 - Q[~mask]**3)) ** (1.0 / 3.0)
        B = Q[~mask] / A
        x[~mask] = (A + B) - e[~mask] / 3.0
        # x = 0 solution (when d == 0)
        x[np.isclose(d.magnitude, 0.0)] = 0.0 * units('N')
        # return array of solutions
        return(x)

    def _get_force_unbroken(
            self, 
            displacement, 
            jac = False
        ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = self._solve_cubic(
            self.coefficients[0][index1, 1],
            self.coefficients[1][index1, 1],
            self.coefficients[2][index1, 1],
            self.coefficients[3][index1, 1] - displacement
        )
        # derivative of force with respect to pullout displacement
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros_like(force_unbroken) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (1.0 / (
                3.0 * self.coefficients[0][index1, 1] * force_unbroken[index1]**2
                + 2.0 * self.coefficients[1][index1, 1] * force_unbroken[index1]
                + self.coefficients[2][index1, 1]
                ))
        # return
        return(
            force_unbroken, 
            dforceunbroken_ddisplacement, 
            behaviour_index
            )
   
class PulloutSurfaceElasticSlipping(PulloutSurfaceElastic):

    def _check_input(
        self,
        roots,
        interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'length',
             'length_surface']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 2, 3]]

    def _get_coefficients_slipping_elastic(self):
        nroots = self._nroots()
        c3 = np.zeros(nroots) * units['m/N^3']
        c2 = -1.0 / (self.roots.elastic_modulus * self.roots.xsection
                     * self.roots.circumference * self.interface.shear_strength)
        c1 = (self.roots.length / (self.roots.elastic_modulus * self.roots.xsection)
              - 1.0 / (self.roots.circumference * self.interface.shear_strength))
        c0 = self.roots.length - self.roots.length_surface
        return(c3, c2, c1, c0)

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic(),
            self._get_coefficients_slipping_elastic(),
            self._get_coefficients_notintension()
            ])

    def _get_limits_elastic_slipping(self):
        force = self._solve_quadratic(
            -1.0 / (2.0 * self.roots.elastic_modulus * self.roots.xsection * self.roots.circumference * self.interface.shear_strength),
            -1.0 / (self.roots.circumference * self.interface.shear_strength),
            self.roots.length - self.roots.length_surface
            )            
        c3, c2, c1, c0 = self._get_coefficients_anchored_elastic()
        displacement = c3 * force**3 + c2 * force**2 + c1 * force + c0
        return(displacement, force)
    
    def _get_limits_elastic_fullpullout(self):
        nroots = self._nroots()
        force = np.zeros(nroots) * units('N')
        displacement = self.roots.length - self.roots.length_surface
        return(displacement, force)

    def _set_limits(self):
        self.limits = self._combine_coefficients([
            self._get_limits_notintension(),
            self._get_limits_elastic_slipping(),
            self._get_limits_elastic_fullpullout()
            ])
        
    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = self._solve_cubic(
            self.coefficients[0][index1, 1],
            self.coefficients[1][index1, 1],
            self.coefficients[2][index1, 1],
            self.coefficients[3][index1, 1] - displacement
        )
        # slipping behaviour
        index2 = (behaviour_index == 2)
        force_unbroken[index2] = self._solve_quadratic(
            self.coefficients[1][index2, 2],
            self.coefficients[2][index2, 2],
            self.coefficients[3][index2, 2] - displacement
            )
        # derivatives
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros_like(force_unbroken) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (1.0 / (
                3.0 * self.coefficients[0][index1, 1] * force_unbroken[index1]**2
                + 2.0 * self.coefficients[1][index1, 1] * force_unbroken[index1]
                + self.coefficients[2][index1, 1]
                ))
            dforceunbroken_ddisplacement[index2] = (1.0 / (
                + 2.0 * self.coefficients[1][index2, 2] * force_unbroken[index2]
                + self.coefficients[2][index2, 2]
                ))
        # return
        return(
            force_unbroken, 
            dforceunbroken_ddisplacement,
            behaviour_index,             
            )
    
# SURFACE - ELASTIC, BREAKAGE
class PulloutSurfaceElasticBreakage(PulloutSurfaceElastic):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'tensile_strength',
             'length_surface']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _get_survival(
            self, 
            force, 
            behaviour_index,
            jac = False
            ):
        # generate mask - for when force has been higher during any previous displacement (e.g. during slipping)
        behaviour_names_reducing = [BEHAVIOUR_NAMES[i] for i in [2, 4, 5]]
        behaviour_index_reducing = np.array([b in behaviour_names_reducing for b in self.behaviour_types])
        mask = behaviour_index_reducing[behaviour_index]
        # set force for reducing points to maximum experienced so far
        force[mask] = np.max(self.limits[1][mask, :], axis = -1)
        # calculate survival
        stress = force / self.roots.xsection
        if hasattr(self, 'weibull_shape'):
            weibull_shape = self.weibull_shape
            weibull_scale = self.roots.tensile_strength / gamma(1.0 + 1.0 / weibull_shape)
            survival = np.exp(-(stress / weibull_scale)**weibull_shape)
            if jac is False:
                dsurvival_dforce = None
            else:
                dstress_dforce = 1.0 / self.roots.xsection
                dsurvival_dstress = (
                    -weibull_shape / weibull_scale
                    * (stress / weibull_scale) ** (weibull_shape - 1.0)
                    * survival
                )
                dsurvival_dforce = dsurvival_dstress * dstress_dforce
                # adjust for reducing points (survival function does not change once reducing force)
                dsurvival_dforce[mask] = np.zeros(np.sum(mask)) * units('1/N')
        else:
            survival = (stress <= self.roots.tensile_strength).astype('float')
            if jac is False:
                dsurvival_dforce = None
            else:
                dsurvival_dforce = np.zeros(self._nroots()) * units('1/N')
        return(survival, dsurvival_dforce)
    
# SURFACE - ELASTIC, BREAKAGE SLIPPAGE
class PulloutSurfaceElasticBreakageSlipping(
    PulloutSurfaceElasticBreakage,
    PulloutSurfaceElasticSlipping  
    ):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 'elastic_modulus', 'length', 
             'tensile_strength', 'length_surface']
            )
        self._check_contains_attributes(interface, ['shear_strength'])


###########################
# SURFACE - ELASTOPLASTIC #
###########################

class PulloutSurfaceElastoplastic(PulloutSurfaceElastic):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 
             'elastic_modulus', 'plastic_modulus',
             'length_surface']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 4]]

    def _get_coefficients_anchored_plastic(self):
        c3 = (
            1.0 
            / (2.0 * (self.roots.plastic_modulus * self.roots.xsection)**2 
               * self.roots.circumference * self.interface.shear_strength)
            )
        c2 = (
            1.0 
            / (2.0 * self.roots.plastic_modulus * self.roots.xsection 
               * self.roots.circumference * self.interface.shear_strength)
            * (1.0
               + 3.0 * self.roots.yield_strength / self.roots.elastic_modulus
               - 3.0 * self.roots.yield_strength / self.roots.plastic_modulus) 
            )
        c1 = (
            self.roots.yield_strength
            / (2.0 * self.roots.elastic_modulus * self.roots.plastic_modulus
               * self.roots.circumference * self.interface.shear_strength)
            * (
                self.roots.yield_strength 
                * (3.0 * self.roots.elastic_modulus / self.roots.plastic_modulus
                   + 2.0 * self.roots.plastic_modulus / self.roots.elastic_modulus
                   - 5.0)
                - 2.0 * self.roots.elastic_modulus
                + 2.0 * self.roots.plastic_modulus
            )
            + self.roots.length_surface / (self.roots.plastic_modulus * self.roots.xsection)
            )
        c0 = (
            self.roots.yield_strength 
            * (self.roots.elastic_modulus - self.roots.plastic_modulus)
            / (2.0 * self.roots.elastic_modulus * self.roots.plastic_modulus
               * self.roots.circumference * self.interface.shear_strength)
            * (
                self.roots.yield_strength * self.roots.xsection
                - self.roots.yield_strength**2 * self.roots.xsection 
                * (1.0 / self.roots.plastic_modulus - 1.0 / self.roots.elastic_modulus)
                - 2.0 * self.roots.circumference * self.interface.shear_strength * self.roots.length_surface
                )
            )
        return(c3, c2, c1, c0)
    
    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic(),
            self._get_coefficients_anchored_plastic()
            ])

    def _get_limits_yield_anchored(self):
        force = self.roots.yield_strength * self.roots.xsection
        c3, c2, c1, c0 = self._get_coefficients_anchored_elastic()
        displacement = c3 * force**3 + c2 * force**2 + c1 * force + c0
        return(displacement, force)
    
    def _set_limits(self):
        self.limits = self._combine_coefficients([
            self._get_limits_notintension(),
            self._get_limits_yield_anchored()
            ])

    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = self._solve_cubic(
            self.coefficients[0][index1, 1],
            self.coefficients[1][index1, 1],
            self.coefficients[2][index1, 1],
            self.coefficients[3][index1, 1] - displacement
        )
        # plastic behaviour
        index2 = (behaviour_index == 2)
        force_unbroken[index2] = self._solve_cubic(
            self.coefficients[0][index2, 2],
            self.coefficients[1][index2, 2],
            self.coefficients[2][index2, 2],
            self.coefficients[3][index2, 2] - displacement
            )
        # derivative of force with respect to displacement
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros(nroots) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (1.0 / (
                3.0 * self.coefficients[0][index1, 1] * force_unbroken[index1]**2
                + 2.0 * self.coefficients[1][index1, 1] * force_unbroken[index1]
                + self.coefficients[2][index1, 1]
                ))
            dforceunbroken_ddisplacement[index2] = (1.0 / (
                3.0 * self.coefficients[0][index2, 2] * force_unbroken[index2]**2
                + 2.0 * self.coefficients[1][index2, 2] * force_unbroken[index2]
                + self.coefficients[2][index2, 2]
                ))
        # return
        return(
            force_unbroken,
            dforceunbroken_ddisplacement,
            behaviour_index
        )   
    
# SURFACE - ELASTOPLASTIC
class PulloutSurfaceElastoplasticSlipping(
    PulloutSurfaceElastoplastic,
    PulloutSurfaceElasticSlipping
    ):

    def _check_input(
            self,
            roots,
            interface
    ):
        self._check_contains_attributes(
            roots, 
            ['xsection', 'circumference', 
             'elastic_modulus', 'plastic_modulus', 'unload_modulus', 
             'length', 'length_surface']
            )
        self._check_contains_attributes(interface, ['shear_strength'])

    def _set_behaviour_types(self):
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]

    def _get_coefficients_slipping_plastic_aboveyield(self):
        _, Tps = self._get_limits_plastic_slipping_start()
        nroots = self._nroots()
        c3 = np.zeros(nroots) * units('m/N^3')
        c2 = (
            -1.0
            / (2.0 * self.roots.xsection * self.roots.circumference * self.interface.shear_strength)
            * (1.0 / self.roots.plastic_modulus + 1.0 / self.roots.unload_modulus)
            )
        c1 = (
            self.roots.length / (self.roots.unload_modulus * self.roots.xsection)
            + 1.0 
            / (self.roots.circumference * self.interface.shear_strength)
            * (
                1.0
                - self.roots.yield_strength / self.roots.elastic_modulus
                + self.roots.yield_strength / self.roots.plastic_modulus
                )
            )
        c0 = (
            self.roots.length 
            - self.roots.length_surface
            + self.roots.yield_strength * self.roots.length
            * (1.0 / self.roots.elastic_modulus - 1.0 / self.roots.plastic_modulus)
            + Tps / self.roots.xsection
            * (self.roots.length - Tps / (2.0 * self.roots.circumference * self.interface.shear_strength))
            * (1.0 / self.roots.plastic_modulus - 1.0 / self.roots.unload_modulus)
            )
        return(c3, c2, c1, c0)

    def _get_coefficients_slipping_plastic_belowyield(self):
        _, Tps = self._get_limits_plastic_slipping_start()
        nroots = self._nroots()
        c3 = np.zeros(nroots) * units('m/N^3')
        c2 = (
            -1.0
            / (self.roots.elastic_modulus * self.roots.xsection
               * self.roots.circumference * self.interface.shear_strength)
            )
        c1 = (
            self.roots.length / (self.roots.unload_modulus * self.roots.xsection)
            - 1.0 / (self.roots.circumference * self.interface.shear_strength)
            * (1.0 
               - self.roots.yield_strength / self.roots.unload_modulus 
               + self.roots.yield_strength / self.roots.elastic_modulus)
            )
        c0 = (
            self.roots.length 
            - self.roots.length_surface
            + 1.0 / (2.0 * self.roots.xsection * self.roots.circumference * self.interface.shear_strength)
            * (
                Tps**2 
                * (1.0 / self.roots.unload_modulus - 1.0 / self.roots.plastic_modulus)
                + (self.roots.yield_strength * self.roots.xsection)**2
                * (1.0 / self.roots.unload_modulus + 1.0 / self.roots.plastic_modulus - 2.0 / self.roots.elastic_modulus)
                )
            - self.roots.length / self.roots.xsection
            * (
                Tps
                * (1.0 / self.roots.unload_modulus - 1.0 / self.roots.plastic_modulus)
                + (self.roots.yield_strength * self.roots.xsection)
                * (1.0 / self.roots.plastic_modulus - 1.0 / self.roots.elastic_modulus)
                )
            )
        return(c3, c2, c1, c0)

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_elastic(),
            self._get_coefficients_slipping_elastic(),
            self._get_coefficients_notintension(),
            self._get_coefficients_anchored_plastic(),
            self._get_coefficients_slipping_plastic_aboveyield(),
            self._get_coefficients_slipping_plastic_belowyield(),
            self._get_coefficients_notintension()
            ])
        
    def _get_limits_plastic_slipping_start(self):
        Try = self.roots.yield_strength * self.roots.xsection
        force = self._solve_quadratic(
            1.0 / (self.roots.plastic_modulus * self.roots.xsection),
            (
                1.0
                + 2.0 * Try / (self.roots.elastic_modulus * self.roots.xsection)
                - 2.0 * Try / (self.roots.plastic_modulus * self.roots.xsection)
            ),
            (
                Try**2 / (self.roots.plastic_modulus * self.roots.xsection)
                - Try**2 / (self.roots.elastic_modulus * self.roots.xsection)
                - 2.0 * (self.roots.length - self.roots.length_surface)
                * (self.roots.circumference * self.interface.shear_strength)
            )
        )
        c3a, c2a, c1a, c0a = self._get_coefficients_anchored_plastic()
        displacement = c3a * force**3 + c2a * force**2 + c1a*force + c0a
        return(displacement, force)

    def _get_limits_plastic_slipping_yield(self):
        force = self.roots.yield_strength * self.roots.xsection
        c3, c2, c1, c0 = self._get_coefficients_slipping_plastic_aboveyield()
        displacement = c3 * force**3 + c2 * force**2 + c1 * force + c0
        return(displacement, force)

    def _get_limits_plastic_fullpullout(self):
        nroots = self._nroots()
        force = np.zeros(nroots) * units['N']
        _, _, _, displacement = self._get_coefficients_slipping_plastic_belowyield()
        return(displacement, force)

    def _set_limits(self):
        displacement_limits, force_limits = self._combine_coefficients([
            self._get_limits_notintension(),
            self._get_limits_elastic_slipping(),
            self._get_limits_elastic_fullpullout(),
            self._get_limits_yield_anchored(),
            self._get_limits_plastic_slipping_start(),
            self._get_limits_plastic_slipping_yield(),
            self._get_limits_plastic_fullpullout()
            ])
        # adjust: slippage before yielding --> never plasticity
        index = (displacement_limits[:, 1] <= displacement_limits[:, 3])
        displacement_limits[index, 3] = np.inf * units('mm')
        displacement_limits[index, 4] = np.inf * units('mm')
        displacement_limits[index, 5] = np.inf * units('mm')
        displacement_limits[index, 6] = np.inf * units('mm')
        # adjust slippage after yielding --> never elastic slippage
        displacement_limits[~index, 1] = displacement_limits[~index, 3]
        displacement_limits[~index, 2] = displacement_limits[~index, 3]
        # set limits
        self.limits = [displacement_limits, force_limits]

    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_index = self._get_behaviour_index(displacement)
        # elastic anchored behaviour
        index1 = (behaviour_index == 1)
        force_unbroken[index1] = self._solve_cubic(
            self.coefficients[0][index1, 1],
            self.coefficients[1][index1, 1],
            self.coefficients[2][index1, 1],
            self.coefficients[3][index1, 1] - displacement
        )
        # elastic slipping behaviour
        index2 = (behaviour_index == 2)
        force_unbroken[index2] = self._solve_quadratic(
            self.coefficients[1][index2, 2],
            self.coefficients[2][index2, 2],
            self.coefficients[3][index2, 2] - displacement
            )
        # plastic anchored behaviour
        index4 = (behaviour_index == 4)
        force_unbroken[index4] = self._solve_cubic(
            self.coefficients[0][index4, 4],
            self.coefficients[1][index4, 4],
            self.coefficients[2][index4, 4],
            self.coefficients[3][index4, 4] - displacement
            )
        # plastic slipping behaviour - above yield
        index5 = (behaviour_index == 5)
        force_unbroken[index5] = self._solve_quadratic(
            self.coefficients[1][index5, 5],
            self.coefficients[2][index5, 5],
            self.coefficients[3][index5, 5] - displacement
            )
        # plastic slipping behaviour - below yield
        index6 = (behaviour_index == 6)
        force_unbroken[index6] = self._solve_quadratic(
            self.coefficients[1][index6, 6],
            self.coefficients[2][index6, 6],
            self.coefficients[3][index6, 6] - displacement
            )
        # derivative of force with respect to displacement
        if jac is False:
            dforceunbroken_ddisplacement = None
        else:
            dforceunbroken_ddisplacement = np.zeros(nroots) * units['N/m']
            dforceunbroken_ddisplacement[index1] = (1.0 / (
                3.0 * self.coefficients[0][index1, 1] * force_unbroken[index1]**2
                + 2.0 * self.coefficients[1][index1, 1] * force_unbroken[index1]
                + self.coefficients[2][index1, 1]
                ))
            dforceunbroken_ddisplacement[index2] = (1.0 / (
                + 2.0 * self.coefficients[1][index2, 2] * force_unbroken[index2]
                + self.coefficients[2][index2, 2]
                ))
            dforceunbroken_ddisplacement[index4] = (1.0 / (
                3.0 * self.coefficients[0][index4, 4] * force_unbroken[index4]**2
                + 2.0 * self.coefficients[1][index4, 4] * force_unbroken[index4]
                + self.coefficients[2][index4, 4]
                ))
            dforceunbroken_ddisplacement[index5] = (1.0 / (
                2.0 * self.coefficients[1][index5, 5] * force_unbroken[index5]
                + self.coefficients[2][index5, 5]
                ))
            dforceunbroken_ddisplacement[index6] = (1.0 / (
                2.0 * self.coefficients[1][index6, 6] * force_unbroken[index6]
                + self.coefficients[2][index6, 6]
                ))
        # return
        return(
            force_unbroken,
            dforceunbroken_ddisplacement,
            behaviour_index
        )       
