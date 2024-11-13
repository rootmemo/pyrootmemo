import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from utils_rotation import axisangle_rotate, vector_normalise
from utils_pullout import pulloutforce_surface
from utils_plot import round_range


# DRAM class
class Dram():
    
    # initialise class
    def __init__(
            self,
            roots,
            soil,
            soil_area = 1.0,
            shearplane_rotation = np.zeros(3)
            ):
        
        # assign root parameters
        self.roots = roots
        # check required fields
        # * xsection
        # * circumference
        # * elastic_modulus
        # number of roots
        n_root = len(self.roots.tensile_strength)
        # * 3D root orientation - in local coordinate system to shearband
        self.shearplane_rotation = shearplane_rotation
        if not hasattr(self.roots, 'orientation'):
            self.roots.orientation_local = np.tile(np.array([0., 0., 1.]), (n_root, 1))
            UserWarning('Root orientations not defined - assumed perpendicular to shear plane')
        else:
            self.roots.orientation_local = axisangle_rotate(
                self.roots.orientation,
                shearplane_rotation
                )
        
        # assign soil parameters
        self.soil = soil
        # check required fields
        # * effective_stress
        if not hasattr(self.soil, 'cohesion'):
            ValueError('Soil effective stress level at shear plane not defined')
        # * cohesion
        if not hasattr(self.soil, 'cohesion'):
            self.soil.cohesion = 0.0
            UserWarning('Soil cohesion not defined - assumed zero')
        # * friction_angle
        if not hasattr(self.soil, 'friction_angle'):
            self.soil.friction_angle = 0.0
            UserWarning('Soil friction angle not defined - assumed zero')
        
        # assign soil cross-sectional area
        self.soil_area = soil_area


    # calculate reinforcements at predefined shear displacement steps
    def calculate(
            self,
            shear_displacement,
            interface_resistance,
            shearband_thickness_initial = 0.0,
            shearband_thickness_max = None,
            weibull_shape_parameter = None
            ):
        
        ## process properties
        # number of roots
        n_root = self.roots.orientation_local.shape[0]
        # number of soil displacement steps
        n_displacement = len(shear_displacement)
        # maximum shear band thickness - create finite, but realistic max value
        if shearband_thickness_max is None:
            # roots break
            if hasattr(self.roots, 'tensile_strength'):
                # assume:
                # * all roots break at the same time
                # * at maximum shear displacement
                # * all roots are parallel at this point
                # then shearband stability distances:
                #   coh + sigmav * tan(phi) + root * cos(beta) * tanphi = root * (sinbeta)
                # let's write this in the following form, and solve
                #   1 = a * sin(beta) - b * cos(beta)
                tensile_capacity = np.sum(self.roots.tensile_strength * self.roots.xsection)
                stress_root = tensile_capacity / self.soil_area
                stress_soil = self.soil.cohesion + self.soil.effective_stress * np.tan(np.deg2rad(self.soil.friction_angle))    
                if stress_soil > stress_root:
                    # shear band will never grow
                    shearband_thickness_max = shearband_thickness_initial
                else:
                    # shear band may grow
                    a = stress_root / stress_soil
                    b = stress_root * np.tan(np.deg2rad(self.soil.friction_angle)) / stress_soil
                    beta = 2.*np.arctan2(np.sqrt(a**2 + b**2 - 1.) - a, b - 1.) + 2. * np.pi
                    shearband_thickness_max = np.max(shear_displacement) / np.tan(beta)
            else:
                # no breakage
                shearband_thickness_max = 1e6
                
        ## initiate output vectors
        # shearband thickness at each displacement step
        shearband_thickness = shearband_thickness_initial * np.ones_like(shear_displacement)
        # root reinforcement, per root
        root_reinforcement = np.zeros((n_displacement, n_root))
        # root behaviour 'mode', for each root at each displacement step
        root_mode = np.zeros((n_displacement, n_root, 6))
            
        ## loop through all soil displacement steps
        # loop
        for i in np.arange(n_displacement):
            # get fraction of roots currently intact
            if i == 0:
                survival_fraction_previous = np.ones(n_root)
                shearband_thickness_step = shearband_thickness_initial
            else:
                survival_fraction_previous = 1. - root_mode[i-1, ..., 5]
                shearband_thickness_step = shearband_thickness[i-1]
            # initial calculation
            yield_criterion, root_orientation, pullout_force, mode = _dram_calculate_singlestep(
                shear_displacement[i],
                shearband_thickness_step,
                interface_resistance,
                self.roots,
                self.soil,
                soil_area = self.soil_area,
                weibull_shape_parameter = weibull_shape_parameter,
                survival_fraction_previous = survival_fraction_previous
                )
            # check shear stability of shear zone - unstable and shearband can still increase in thickness
            if (yield_criterion >= 0.0) & (shearband_thickness_step < shearband_thickness_max): 
                # increase shear band thickness, using root finding (bisection)
                fun = lambda x: _dram_calculate_singlestep(
                    shear_displacement[i],
                    x,
                    interface_resistance,
                    self.roots,
                    self.soil,
                    soil_area = self.soil_area,
                    weibull_shape_parameter = weibull_shape_parameter,
                    survival_fraction_previous = survival_fraction_previous
                    )[0]
                shearband_thickness_step = root_scalar(
                    fun,
                    bracket = (shearband_thickness_step, shearband_thickness_max)
                    ).root
                # use new, stable shear band thickness to calculate new forces, reinforcements etc
                yield_criterion, root_orientation, pullout_force, mode = _dram_calculate_singlestep(
                    shear_displacement[i],                    
                    shearband_thickness_step,                    
                    interface_resistance,
                    self.roots,
                    self.soil,
                    soil_area = self.soil_area,
                    weibull_shape_parameter = weibull_shape_parameter,
                    survival_fraction_previous = survival_fraction_previous
                    )
            # assign solution of steps to 
            root_mode[i, ...] = mode
            shearband_thickness[i] = shearband_thickness_step
            root_reinforcement[i, ...] = (
                pullout_force  
                * (root_orientation[..., 0] + root_orientation[..., 2] 
                   * np.deg2rad(np.tan(self.soil.friction_angle)))
                / self.soil_area
                )
            
        ## after loop - assign output to class
        self.shear_displacement = shear_displacement
        self.reinforcement_per_root = root_reinforcement
        self.mode = root_mode
        self.shearband_thickness = shearband_thickness
        self.reinforcement = np.sum(root_reinforcement, axis = -1)
        self.peak_reinforcement = np.max(self.reinforcement)
        
    
    # plot shear displacement versus reinforcement
    def plot_reinforcement(
            self,
            xlabel: chr = 'Shear displacement',
            ylabel: chr = 'Root reinforcement'
            ) -> tuple:
        # initiate figure
        fig, ax = plt.subplots()
        # plot shear displacement vs reinforcement
        ax.plot(self.shear_displacement, self.reinforcement)
        # set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # axis limits
        ax.set_xlim(round_range(self.shear_displacement, limits = [0., None])['limits'])
        ax.set_ylim(round_range(self.reinforcement, limits = [0., None])['limits'])
        # return
        return(fig, ax)
    
    # plot shear displacement versys shear band thickness
    def plot_shearband(
            self,
            xlabel: chr = 'Shear displacement',
            ylabel: chr = 'Shearband thickness'
            ) -> tuple:
        # initiate figure
        fig, ax = plt.subplots()
        # plot shear displacement vs reinforcement
        ax.plot(self.shear_displacement, self.shearband_thickness)
        # set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # axis limits
        ax.set_xlim(round_range(self.shear_displacement, limits = [0., None])['limits'])
        ax.set_ylim(round_range(self.shearband_thickness, limits = [0., None])['limits'])
        # return
        return(fig, ax)
    
    # plot shear displacement versus root behaviour type
    def plot_mode(
            self,
            xlabel: chr = 'Shear displacement',
            ylabel: chr = 'Fraction of root cross-sectional area',
            loc: chr = 'outside right upper'
            ) -> tuple:
        # initiate figure
        fig, ax = plt.subplots()
        # calculate fraction of root cross-sectional area for each behaviour
        A_root = np.sum(self.roots.xsection)
        fraction = np.sum(
            self.mode * self.roots.xsection[np.newaxis, ..., np.newaxis],
            axis = 1) / A_root
        # plot
        ax.stackplot(
            self.shear_displacement,
            fraction.T,
            labels = [
                'Compression',
                'Anchored, elastic',
                'Anchored, elasto-plastic',
                'Slipping, elastic',
                'Slipping, elasto-plastic',
                'Broken'
                ]
            )
        # add legend
        fig.legend(loc = loc)
        # set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # axis limits
        ax.set_xlim(round_range(self.shear_displacement, limits = [0., None])['limits'])
        ax.set_ylim([0., 1.])
        # return
        return(fig, ax)
        
    
# key DRAM calculation function - calculate root orientation, root forces and
# shearband stability at given shear displacement step and shearband thickness
def _dram_calculate_singlestep(
        shear_displacement,
        shearband_thickness,
        interface_resistance,
        roots,
        soil,
        soil_area = 1.0,
        weibull_shape_parameter = None,
        survival_fraction_previous = None,
        ):
    # displaced root orientations - 3D unit vectors
    root_orientation = _dram_rotated_unit_vector(
            roots.orientation_local, 
            shear_displacement,
            shearband_thickness
            )
    # pullout displacement - one side of the (symmetric) root
    pullout_disp = _dram_pullout_displacement(
            roots.orientation_local, 
            shear_displacement, 
            shearband_thickness
            )
    # calculations for pullout root lenghts
    root_length_surface = 0.5 * shearband_thickness / roots.orientation_local[..., 2]
    if not hasattr(roots, 'length'):
        root_length_soil = 0.5 * roots.length - root_length_surface
    else:
        root_length_soil = None
    # pullout force
    pullout_force, mode = pulloutforce_surface(
        pullout_disp,
        interface_resistance,
        roots.xsection,
        roots.circumference,
        roots.elastic_modulus,
        tensile_strength = roots.tensile_strength,
        yield_strength = roots.yield_strength,
        plastic_modulus = roots.plastic_modulus,
        root_length_soil = root_length_soil,
        root_length_surface = root_length_surface,
        weibull_shape_parameter = weibull_shape_parameter,
        survival_fraction_previous = survival_fraction_previous
        )
    # calculate stability in shear plane
    yield_criterion = _dram_yield_shearband(
        pullout_force,
        root_orientation,
        soil.effective_stress,
        soil.cohesion,
        soil.friction_angle,
        soil_area
        )
    # return
    return(yield_criterion, root_orientation, pullout_force, mode)


# unit vector describing the rotated root orientation in the shear band
def _dram_rotated_unit_vector(
        root_unit_vector, 
        shear_displacement: float,
        shearband_thickness: float
        ):
    
    ## INPUT
    ## root_unit_vector - numpy array (size 3, or m*3)
    
    if np.isclose(shearband_thickness, 0.):
        # zero shear band thickness - all roots horizontal
        root_vector_rotated = np.zeros_like(root_unit_vector)
        root_vector_rotated[..., 0] = 1.
        return(root_vector_rotated)
    else:
        # finite shear band thickness
        vector_add = np.zeros_like(root_unit_vector)
        vector_add[..., 0] = shear_displacement/shearband_thickness
        return(vector_normalise(root_unit_vector + vector_add))


# calculate length of root located within the shear band
def _dram_root_length_in_shearband(
        root_unit_vector, 
        shear_displacement, 
        shearband_thickness
        ):
    
    if np.isclose(shearband_thickness, 0.):
        # zero shear band thickness - all roots align with shear displacement
        length = shear_displacement * np.ones(root_unit_vector.shape[:-1])
    else:
        # finite shear band thickness - rotated roots
        L1 = root_unit_vector * shearband_thickness / root_unit_vector[..., [-1]]
        L1[..., 0] = L1[..., 0] + shear_displacement
        length = np.linalg.norm(L1, axis = -1)
    # return
    return(length)
    
    
# calculate pullout displacement in DRAM: equal to 0.5 the elongation in root length in shearband
def _dram_pullout_displacement(
        root_unit_vector, 
        shear_displacement, 
        shearband_thickness
        ):
    
    # initial length of root in shearband
    L0 = _dram_root_length_in_shearband(root_unit_vector, 0.0, shearband_thickness)
    # displaced length of root in shearband
    L1 = _dram_root_length_in_shearband(root_unit_vector, shear_displacement, shearband_thickness)
    # elongation on one side of the root
    dL = 0.5*(L1 - L0)
    # return pullout displacement for one side
    return(dL)


# yield on edge of shearband (if >=0, then yielding)
def _dram_yield_shearband(
        root_force,
        root_orientation,
        soil_stress,
        soil_cohesion,
        soil_friction_angle,  # in deg
        soil_area = 1.0
    ):
    # force decomposition
    root_force_parallel = np.sum(root_force * root_orientation[..., 0])
    root_force_perpendicular = np.sum(root_force * root_orientation[..., 2])
    # root stresses
    root_stress_parallel = root_force_parallel / soil_area
    root_stress_perpendicular = root_force_perpendicular / soil_area
    # soil strength (Mohr-Coulomb)
    soil_shear_strength = soil_cohesion + soil_stress * np.tan(np.deg2rad(soil_friction_angle))
    # return stability criterion
    return(root_stress_parallel 
           + root_stress_perpendicular * np.tan(np.deg2rad(soil_friction_angle))
           - soil_shear_strength)



    

from class_test import MultipleRoots, Soil
soil = Soil(
    friction_angle = 35,
    cohesion = 1.e3,
    effective_stress = 50.e3
    )
roots = MultipleRoots(
    diameter = np.array([1, 1])*1e-3,
    yield_strength = np.array([20, 21])*1e6,
    tensile_strength = np.array([30, 25])*1e6,
    elastic_modulus = np.array([200, 200])*1e6,
    plastic_modulus = np.array([50, 50])*1e6,
    length = np.array([1, 1])
    )
dram = Dram(roots, soil, soil_area = 1)

interface_resistance = 5e3
shear_displacement = np.linspace(0, 0.4, 100)
dram.calculate(shear_displacement, interface_resistance, weibull_shape_parameter = 40.)
dram.plot_reinforcement()

