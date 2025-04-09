# import packages
import numpy as np
from scipy.special import gamma
from pyrootmemo.tools.helpers import units
from pint import Quantity
import matplotlib.pyplot as plt


# each function
# - behaviour_types
# - behaviour_limits (displacement/force tuples)
# - force(displacement): calculate force, survival function and type

# COMMON FUNCTIONS

def _number_of_roots(roots):
    return(len(roots.xsection.magnitude))

def _solve_quadratic(a, b, c):
    ## solve equations a*x^2 + b*x^2 + c = 0 in DRAM
    ## a = always positive in DRAM equations
    ## function returns largest root
    discriminant = b**2 - 4.0 * a * c
    x = (-b + np.sqrt(discriminant)) / (2.0 * a)
    return(x)


# POLYNOMIAL COEFFICIENTS

def embedded_coefficients_notintension(roots):
    nroots = _number_of_roots(roots)
    c2 = np.zeros(nroots) * units['m/N^2']
    c1 = np.zeros(nroots) * units['m/N']
    c0 = np.zeros(nroots) * units['m']
    return(c2, c1, c0)

def embedded_coefficients_anchored_elastic(roots, interface_resistance):
    nroots = _number_of_roots(roots)
    c2 = 1.0 / (2.0 * roots.elastic_modulus * roots.xsection * roots.circumference * interface_resistance)
    c1 = np.zeros(nroots) * units['m/N']
    c0 = np.zeros(nroots) * units['m']
    return(c2, c1, c0)

def embedded_coefficients_slipping(roots):
    nroots = _number_of_roots(roots)
    c2 = np.zeros(nroots) * units['m/N^2']
    c1 = np.zeros(nroots) * units['m/N']
    c0 = np.inf * np.ones(nroots) * units['m']
    return(c2, c1, c0)

def embedded_coefficients_anchored_plastic(roots, interface_resistance):
    c2 = (
        1.0 
        / (2.0 * roots.plastic_modulus * roots.xsection * roots.circumference * interface_resistance)
        )
    c1 = (
        roots.yield_strength 
        / (roots.elastic_modulus * roots.circumference * interface_resistance)
        - roots.yield_strength 
        / (roots.plastic_modulus * roots.circumference * interface_resistance)
        )
    c0 = (
        -roots.yield_strength**2 * roots.xsection 
        / (2.0 * roots.elastic_modulus * roots.circumference * interface_resistance)
        + roots.yield_strength**2 * roots.xsection 
        / (2.0 * roots.plastic_modulus * roots.circumference * interface_resistance)
        )
    return(c2, c1, c0)


# LIMITS 

def embedded_limits_notintension(roots):
    nroots = _number_of_roots(roots)
    force = np.zeros(nroots) * units['N']
    displacement = np.zeros(nroots) * units['m']
    return(displacement, force)

def embedded_limits_elastic_slipping(roots, interface_resistance):
    force = roots.length * roots.circumference * interface_resistance
    c2,_,_ = embedded_coefficients_anchored_elastic(roots, interface_resistance)
    displacement = c2 * force**2
    return(displacement, force)

def embedded_limits_plastic_slipping(roots, interface_resistance):
    force = roots.length * roots.circumference * interface_resistance
    c2,c1,c0 = embedded_coefficients_anchored_plastic(roots, interface_resistance)
    displacement = c2 * force**2 + c1 * force + c0
    return(displacement, force)

def embedded_limits_yield_anchored(roots, interface_resistance):
    force = roots.yield_strength * roots.xsection
    c2,_,_ = embedded_coefficients_anchored_elastic(roots, interface_resistance)
    displacement = c2 * force**2
    return(displacement, force)


# BASE CLASS

class Pullout_base():

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = ['xsection', 'circumference', 'elastic_modulus'],
            ):
        # set multiple roots input
        self._check_fields(roots, roots_required_attr)
        self.roots = roots
        # set number of roots 
        self.nroots = _number_of_roots(roots)
        # set interface resistance
        if not isinstance(interface_resistance, Quantity):
            raise TypeError('soil resistance must be defined as a Quantity')
        else:
            unit_input = interface_resistance.to_base_units().units
            unit_required = units('kPa').to_base_units().units
            if unit_input != unit_required:
                raise ValueError('soil resitance must be specified as a stress')
            if not np.isscalar(interface_resistance.magnitude):
                if len(interface_resistance.magnitude) == self.nroots:
                    raise ValueError('soil resistance must be scalar or defined per root')
        self.interface_resistance = interface_resistance
        # set behaviour types, displacement/force limits and polynomial coefficients
        self._set_behaviour_types()
        self._set_limits()
        self._set_coefficients()

    def _check_fields(self, object, fields):
        missing = [f for f in fields if not hasattr(object, f)]
        if len(missing) > 0:
            error_message = "Object is missing attributes {0}".format(missing)
            raise(TypeError(error_message))

    def _combine_coefficients(self, list):
        return([np.column_stack(x) for x in zip(*list)])

    def _get_behaviour_type(self, displacement):
        return((displacement > self.limits[0]).sum(axis = 1))

    def _get_survival_weibull(self, force, weibull_shape = None):
        stress = force / self.roots.xsection
        if weibull_shape is None:
            return((stress <= self.roots.tensile_strength).astype('float'))
        else:
            weibull_scale = self.roots.tensile_strength / gamma(1.0 + 1.0 / weibull_shape)
            return(np.exp(-(stress / weibull_scale)**weibull_shape))

    def calculate(self, displacements):
        # make vector of displacements
        if np.isscalar(displacements.magnitude):
            displacements = Quantity.from_list([displacements])
        # calculate results for each displacement step
        tmp = [self.force(i) for i in displacements]
        force, survival, behaviour = zip(*tmp)
        # generate results arrays (axis 1 = roots, axis 0 = displacements)
        force = np.stack(force, axis = 0)
        survival = np.array(survival)
        behaviour = np.array(behaviour)
        # return
        return(force, survival, behaviour)

    def plot(self, displacement_max, n = 251):
        displacement = np.linspace(0.0, displacement_max, n)
        force,survival,behaviour = self.calculate(displacement)
        fig, ax = plt.subplots(1, 1)
        ax.stackplot(displacement.magnitude, force.magnitude.transpose())
        return(fig, ax)


# embedded - elastic
class Pullout_embedded_elastic(Pullout_base):
    
    def _set_behaviour_types(self):
        self.behaviour_types = ['not in tension', 'anchored, elastic']

    def _set_limits(self):
        self.limits = self._combine_coefficients([
            embedded_limits_notintension(self.roots)
            ])

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            embedded_coefficients_notintension(self.roots),
            embedded_coefficients_anchored_elastic(self.roots, self.interface_resistance)
            ])

    def _force_unbroken(self, displacement):
        # initialise force vector and find behaviour indices
        force = np.zeros(self.nroots) * units('N')
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic behaviout
        index1 = (behaviour_type == 1)
        force[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # return
        return(force, behaviour_type)

    def force(self, displacement):
        force, behaviour_type = self._force_unbroken(displacement)
        survival = np.ones(self.nroots)
        return(force, survival, behaviour_type)


# embedded - elastic, slipping
class Pullout_embedded_elastic_slipping(Pullout_embedded_elastic):

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = ['xsection', 'circumference', 'elastic_modulus', 'length']
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastic_slipping, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr
            )
        
    def _set_behaviour_types(self):
        self.behaviour_types = ['not in tension', 'anchored, elastic', 'slipping_elastic']

    def _set_limits(self):
        self.limits = self._combine_coefficients([
            embedded_limits_notintension(self.roots),
            embedded_limits_elastic_slipping(self.roots, self.interface_resistance)
            ])

    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            embedded_coefficients_notintension(self.roots),
            embedded_coefficients_anchored_elastic(self.roots, self.interface_resistance),
            embedded_coefficients_slipping(self.roots)
            ])

    def _force_unbroken(self, displacement):
        # initialise force vector and find behaviour indices
        force = np.zeros(self.nroots) * units('N')
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic behaviour
        index1 = (behaviour_type == 1)
        force[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # slipping behaviour
        index2 = (behaviour_type == 2)
        force[index2] = self.limits[1][index2, 1]
        # return
        return(force, behaviour_type)


# embedded - elastic, breakage
class Pullout_embedded_elastic_breakage(Pullout_embedded_elastic):

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = ['xsection', 'circumference', 'elastic_modulus', 'tensile_strength'],
            weibull_shape = None
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastic_breakage, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr
            )
        # set weibull shape parameter
        self.weibull_shape = weibull_shape

    def force(self, displacement):
        force_unbroken, behaviour_type = self._force_unbroken(displacement)
        survival = self._get_survival_weibull(force_unbroken, self.weibull_shape)
        return(
            force_unbroken * survival, 
            survival, 
            behaviour_type
        )


# embedded - elastic, breakage, slipping
class Pullout_embedded_elastic_breakage_slipping(Pullout_embedded_elastic_slipping):

    def __init__(
            self, 
            roots, 
            interface_resistance, 
            roots_required_attr = [
                'xsection', 'circumference', 'elastic_modulus', 
                'tensile_strength', 'length'
                ],
            weibull_shape = None
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastic_breakage_slipping, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr,
            )
        # set weibull shape parameter
        self.weibull_shape = weibull_shape

    def force(self, displacement):
        force_unbroken, behaviour_type = self._force_unbroken(displacement)
        survival = self._get_survival_weibull(force_unbroken, self.weibull_shape)
        return(
            force_unbroken * survival, 
            survival, 
            behaviour_type
        )
    

# embedded - elasto-plastic
class Pullout_embedded_elastoplastic(Pullout_base):

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = [
                'xsection', 'circumference', 'elastic_modulus',
                'yield_strength', 'plastic_modulus'
                ]
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastoplastic, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr
            )
        
    def _set_behaviour_types(self):
        self.behaviour_types = [
            'not in tension', 
            'anchored, elastic', 
            'anchored, plastic'
            ]

    def _set_limits(self):
        self.limits = self._combine_coefficients([
            embedded_limits_notintension(self.roots),
            embedded_limits_yield_anchored(self.roots, self.interface_resistance)
            ])
    
    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            embedded_coefficients_notintension(self.roots),
            embedded_coefficients_anchored_elastic(self.roots, self.interface_resistance),
            embedded_coefficients_anchored_plastic(self.roots, self.interface_resistance)
            ])

    def _force_unbroken(self, displacement):
        # initialise force vector and find behaviour indices
        force = np.zeros(self.nroots) * units('N')
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic behaviour
        index1 = (behaviour_type == 1)
        force[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # plastic behaviour
        index2 = (behaviour_type == 2)
        force[index2] = _solve_quadratic(
            self.coefficients[0][index2, 2],
            self.coefficients[1][index2, 2],
            self.coefficients[2][index2, 2] - displacement
        )
        # return
        return(force, behaviour_type)

    def force(self, displacement):
        force, behaviour_type = self._force_unbroken(displacement)
        survival = np.ones(self.nroots)
        return(force, survival, behaviour_type)


# embedded - elasto-plastic, slipping
class Pullout_embedded_elastoplastic_slipping(Pullout_embedded_elastoplastic):

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = [
                'xsection', 'circumference', 'elastic_modulus',
                'yield_strength', 'plastic_modulus',
                'length'
                ]
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastoplastic_slipping, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr
            )
        
    def _set_behaviour_types(self):
        self.behaviour_types = [
            'not in tension', 
            'anchored, elastic', 
            'slipping, elastic',
            'anchored, plastic',            
            'slipping, plastic'
            ]

    def _set_limits(self):
        # calculate limits
        displacement_limits, force_limits = self._combine_coefficients([
            embedded_limits_notintension(self.roots),
            embedded_limits_elastic_slipping(self.roots, self.interface_resistance),
            embedded_limits_yield_anchored(self.roots, self.interface_resistance),            
            embedded_limits_plastic_slipping(self.roots, self.interface_resistance)
            ])
        # adjust: slippage before yielding --> never plasticity
        index = displacement_limits[:, 1] <= displacement_limits[:, 2]
        displacement_limits[index, 2] = np.inf * displacement_limits[index, 2]
        displacement_limits[index, 3] = np.inf * displacement_limits[index, 3]
        # adjust slippage after yielding --> never elastic slippage
        displacement_limits[~index, 1] = displacement_limits[~index, 2]
        # return
        self.limits = [displacement_limits, force_limits]
    
    def _set_coefficients(self):
        self.coefficients = self._combine_coefficients([
            embedded_coefficients_notintension(self.roots),
            embedded_coefficients_anchored_elastic(self.roots, self.interface_resistance),
            embedded_coefficients_slipping(self.roots),
            embedded_coefficients_anchored_plastic(self.roots, self.interface_resistance),
            embedded_coefficients_slipping(self.roots)
            ])

    def _force_unbroken(self, displacement):
        # initialise force vector and find behaviour indices
        force = np.zeros(self.nroots) * units('N')
        behaviour_type = self._get_behaviour_type(displacement)
        # elastic anchored behaviour
        index1 = (behaviour_type == 1)
        force[index1] = np.sqrt(displacement / self.coefficients[0][index1, 1])
        # plastic anchored behaviour
        index3 = (behaviour_type == 3)
        force[index3] = _solve_quadratic(
            self.coefficients[0][index3, 3],
            self.coefficients[1][index3, 3],
            self.coefficients[2][index3, 3] - displacement
        )
        # slipping behaviour
        index24 = (behaviour_type == 2) | (behaviour_type == 4)
        force[index24] = self.limits[1][index24, 3]
        # return
        return(force, behaviour_type)


# embedded - elastic, breakage
class Pullout_embedded_elastoplastic_breakage(Pullout_embedded_elastoplastic):

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = [
                'xsection', 'circumference', 'elastic_modulus',
                'yield_strength', 'plastic_modulus',
                'tensile_strength'
                ],
            weibull_shape = None
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastoplastic_breakage, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr
            )
        # set weibull shape parameter
        self.weibull_shape = weibull_shape

    def force(self, displacement):
        force_unbroken, behaviour_type = self._force_unbroken(displacement)
        survival = self._get_survival_weibull(force_unbroken, self.weibull_shape)
        return(
            force_unbroken * survival, 
            survival, 
            behaviour_type
        )

# embedded - elasto-plastic, breakage, slipping
class Pullout_embedded_elastoplastic_breakage_slipping(Pullout_embedded_elastoplastic_slipping):

    def __init__(
            self, 
            roots, 
            interface_resistance,
            roots_required_attr = [
                'xsection', 'circumference', 'elastic_modulus',
                'yield_strength', 'plastic_modulus',
                'tensile_strength',
                'length'
                ],
            weibull_shape = None
            ):
        # call initialiser of parent class
        super(Pullout_embedded_elastoplastic_breakage_slipping, self).__init__(
            roots, 
            interface_resistance, 
            roots_required_attr = roots_required_attr
            )
        # set weibull shape parameter
        self.weibull_shape = weibull_shape

    def force(self, displacement):
        force_unbroken, behaviour_type = self._force_unbroken(displacement)
        survival = self._get_survival_weibull(force_unbroken, self.weibull_shape)
        return(
            force_unbroken * survival, 
            survival, 
            behaviour_type
        )

