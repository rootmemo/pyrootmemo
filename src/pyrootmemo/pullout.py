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
    'Anchored, plastic',   
    'Slipping, plastic'
]


########################
### EMBEDDED ELASTIC ###
########################


# EMBEDDED - ELASTIC
class Pullout_embedded_elastic():

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
        
    def _get_behaviour_type(
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
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic behaviour
        index1 = (behaviour_type == 1)
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
            behaviour_type
            )

    def _get_survival(
            self, 
            force, 
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
        force_unbroken, dforceunbroken_ddisplacement, behaviour_type = self._get_force_unbroken(
            displacement, 
            jac = jac
            )
        survival, dsurvival_dforceunbroken = self._get_survival(
            force_unbroken, 
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
            behaviour_type
        )


# EMBEDDED - ELASTIC, SLIPPING
class Pullout_embedded_elastic_slipping(Pullout_embedded_elastic):

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
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic behaviour
        index1 = (behaviour_type == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # slipping behaviour
        index2 = (behaviour_type == 2)
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
            behaviour_type,             
            )
    

# EMBEDDED - ELASTIC, BREAKAGE
class Pullout_embedded_elastic_breakage(Pullout_embedded_elastic):

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
class Pullout_embedded_elastic_breakage_slipping(
    Pullout_embedded_elastic_slipping,
    Pullout_embedded_elastic_breakage
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
class Pullout_embedded_elastoplastic(Pullout_embedded_elastic):

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
    
    def _solve_quadratic(self, a, b, c):
        # solve equations a*x^2 + b*x^2 + c = 0 in DRAM/Pullout
        # a = always positive in DRAM/Pullout equations
        # function returns largest root
        discriminant = b**2 - 4.0 * a * c
        x = (-b + np.sqrt(discriminant)) / (2.0 * a)
        return(x)

    def _get_force_unbroken(
            self, 
            displacement,
            jac = False
            ):
        # initialise force vector and find behaviour indices
        nroots = self._nroots()
        force_unbroken = np.zeros(nroots) * units('N')
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic behaviour
        index1 = (behaviour_type == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # plastic behaviour
        index2 = (behaviour_type == 2)
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
            behaviour_type
        )
    

# EMBEDDED, ELASTOPLASTIC, SLIPPING
class Pullout_embedded_elastoplastic_slipping(
    Pullout_embedded_elastoplastic,
    Pullout_embedded_elastic_slipping
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
        self.behaviour_types = [BEHAVIOUR_NAMES[i] for i in [0, 1, 2, 3, 4]]

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
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic anchored behaviour
        index1 = (behaviour_type == 1)
        force_unbroken[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # plastic anchored behaviour
        index3 = (behaviour_type == 3)
        force_unbroken[index3] = self._solve_quadratic(
            self.coefficients[0][index3, 3],
            self.coefficients[1][index3, 3],
            self.coefficients[2][index3, 3] - displacement
        )
        # slipping behaviour
        index24 = (behaviour_type == 2) | (behaviour_type == 4)
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
            behaviour_type            
        )
    

# EMBEDDED, ELASTOPLASTIC, BREAKAGE
class Pullout_embedded_elastoplastic_breakage(
    Pullout_embedded_elastoplastic,
    Pullout_embedded_elastic_breakage
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
class Pullout_embedded_elastoplastic_breakage_slipping(
    Pullout_embedded_elastoplastic_slipping,
    Pullout_embedded_elastoplastic_breakage,
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