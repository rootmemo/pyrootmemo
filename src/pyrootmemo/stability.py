import numpy as np
from pint import Quantity
from pyrootmemo.tools.checks import is_namedtuple
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.helpers import Parameter, units


class InfiniteSlope():

    def __init__(
            self,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            root_reinforcement: None | Parameter | Quantity = None
            ):
        if not isinstance(soil_profile, SoilProfile):
            raise TypeError('soil_profile must be an object of class SoilProfile')
        profile_attributes_required = ['soils', 'depth', 'groundwater_table']
        for i in profile_attributes_required:
            if not hasattr(soil_profile, i):
                raise AttributeError('soil_profile must contain ' + str(i) + ' attribute')
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be an object of class FailureSurface')
        surface_attributes_required = ['depth', 'orientation']
        for i in surface_attributes_required:
            if not hasattr(failure_surface, i):
                raise AttributeError('failure_surface must contain ' + str(i) + ' attribute')
        if root_reinforcement is None:
            root_reinforcement = 0.0 * units('kPa')
        elif isinstance(root_reinforcement, Quantity):
            if root_reinforcement.dimensionality != units('kPa').dimensionality:
                raise ValueError('unit of root_reinforcement must be compatible with kPa')
        elif is_namedtuple(root_reinforcement):
            root_reinforcement = root_reinforcement[0] * units(root_reinforcement[1])
        else:
            raise TypeError('root_reinforcement must be defined as a Parameter tuple or a Quantity')
        self.pore_pressure = soil_profile.calc_pore_pressure(
            failure_surface.depth,
            direction = failure_surface.orientation
        )        
        self.total_vertical_stress = soil_profile.calc_vertical_stress(failure_surface.depth)
        soil_cohesion = soil_profile.get_soil(failure_surface.depth).cohesion
        friction_angle = soil_profile.get_soil(failure_surface.depth).friction_angle
        self.destabilising_stress = (
            self.total_vertical_stress
            * np.sin(failure_surface.orientation)
            * np.cos(failure_surface.orientation)
            )
        self.resisting_stress = (
            soil_cohesion
            + root_reinforcement
            + np.tan(friction_angle) * (
                self.total_vertical_stress * np.cos(failure_surface.orientation)**2 
                - self.pore_pressure
                )
            )
        self.fos = (self.resisting_stress / self.destabilising_stress).magnitude
