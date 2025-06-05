import numpy as np
from pyrootmemo.tools.checks import is_namedtuple
from pyrootmemo.materials import Soil
from collections import namedtuple
from pyrootmemo.tools.helpers import units
from pint import DimensionalityError


Parameter = namedtuple("parameter", "value unit")

SOIL_PROFILE_PARAMETERS = {
    "depth": {"type": (float | int), "unit": units("m")},
    "groundwater_table": {"type": (float | int), "unit": units("m")},
}

FAILURE_SURFACE_PARAMETERS = {
    "depth": {"type": (float | int), "unit": units("m")},
    "orientation": {"type": (float | int), "unit": units("deg")},
    "shear_zone_thickness": {"type": (float | int), "unit": units("m")},
    "cross_sectional_area": {"type": (float | int), "unit": units("m^2")},
}

class SoilProfile:
    def __init__(self, soils, **kwargs):
        """
        Initialize a SoilProfile object with a list of Soil objects and optional parameters.
        This class represents a profile of soils, allowing for the calculation of vertical stress

        Parameters
        ----------
        soils : pyrootmemo.materials.Soil
            A list of Soil objects representing the different soil layers in the profile.
        **kwargs : dict
            Optional parameters for the soil profile, such as depth and groundwater table.
        """
        if not isinstance(soils, list):
            raise TypeError("Soils should be a list of Soil objects")
        if not all([isinstance(s, Soil) for s in soils]):
            raise TypeError("Soils should be a list of Soil objects")
        self.soils = soils
            
        for k, v in kwargs.items():
            if k not in SOIL_PROFILE_PARAMETERS.keys():
                raise ValueError(
                    f"Undefined parameter. Choose one of the following: {SOIL_PROFILE_PARAMETERS.keys()}"
                )
            if not is_namedtuple(v):
                raise TypeError("Parameter should be of type Parameter(value, unit)")
            if not isinstance(v.value, (SOIL_PROFILE_PARAMETERS[k]["type"] | list)):
                raise TypeError(
                    f"Value should be of type {SOIL_PROFILE_PARAMETERS[k]["type"]} or a list"
                )
            if not isinstance(v.unit, str):
                raise TypeError("Unit should be entered as a string")
            if not units(v.unit).check(SOIL_PROFILE_PARAMETERS[k]["unit"].dimensionality):
                raise DimensionalityError(
                    units1=v.unit, units2=SOIL_PROFILE_PARAMETERS[k]["unit"]
                )
            if isinstance(v.value, list):
                if not all(
                    [isinstance(entry, SOIL_PROFILE_PARAMETERS[k]["type"]) for entry in v.value]
                ):
                    raise TypeError(
                        f"{k} should only be of type {SOIL_PROFILE_PARAMETERS[k]["type"]} in a list"
                    )
            if k == "depth":
                if len(v.value) != len(soils):
                    raise ValueError(
                        f"Length of depth ({len(v.value)}) should be equal to the number of soils ({len(soils)})"
                    )
                if any([entry < 0 for entry in v.value]):
                    raise ValueError("Depth should be positive")
                if v.value[0] == 0:
                    raise ValueError("Depth should start with a positive value")
                if any([v.value[i] <= v.value[i - 1] for i in range(1, len(v.value))]):
                    raise ValueError("Depth should be monotonically increasing")
            if k == "groundwater_table":
                if not isinstance(v.value, (float, int)):
                    raise TypeError("Groundwater table should be a single value")
                if v.value < 0:
                    raise ValueError("Groundwater table cannot be negative")

            setattr(self, k, v.value * units(v.unit))

    # get soil object at specified depth
    def get_soil(
            self,
            depth            
            ):
        """
        Get the soil object at a specified depth.

        Parameters
        ----------
        depth : float or int
            The depth at which to retrieve the soil object.
        Returns
        ------- 
        Soil
            The Soil object corresponding to the specified depth.
        """
        soils_deeper = [s for s, d in zip(self.soils, self.depth) if d >= depth]
        return(soils_deeper[0])
    
    # calculate vertical stress at specific depth
    def vertical_stress(
            self,
            depth
            ):
        """
        Calculate the vertical stress at a specific depth in the soil profile.

        Parameters
        ----------
        depth : float or int
            The depth at which to calculate the vertical stress.
        Returns
        -------
        float
            The vertical stress at the specified depth, in kPa.
        """
        # depth at top of each layer
        depth_top = np.append(0.0 * units('m'), self.depth[:-1])
        # thickness of (part of) each layer, above the requested depth
        thickness = (
            np.minimum(self.depth, depth)
            - np.minimum(depth_top, depth)
        )
        # get thickness contribution of (part of) each layer, but above and below water table
        if depth > self.groundwater_table:
            tmp_above = np.minimum(depth, self.groundwater_table)
            thickness_above = (
                np.minimum(self.depth, tmp_above)
                - np.minimum(depth_top, tmp_above)
                )
            thickness_below = thickness - thickness_above
        else:
            # all above water table
            thickness_above = thickness
            thickness_below = 0.0 * thickness
        # calculate total stress 
        unit = 'kN/m^3'
        unit_weight_above = np.array([s.unit_weight_bulk.to(unit).magnitude for s in self.soils]) * units(unit)
        unit_weight_below = np.array([s.unit_weight_saturated.to(unit).magnitude for s in self.soils]) * units(unit)
        return(np.sum(
            unit_weight_above * thickness_above
            + unit_weight_below * thickness_below
            ))
    
    # calculate pore pressure at specific depth
    def pore_pressure(
            self,
            depth,
            unit_weight_water = 9.81 * units('kN/m^3'),
            direction = 0.0 * units('deg')    # flow direction, relative to the horizontal
            ):
        if depth <= self.groundwater_table:
            # depth above water table -> retun zero (no suction assumed)
            return(0.0 * units('kPa'))
        else:
            # below water table -> calculate
            pore_pressure = unit_weight_water * (depth - self.groundwater_table)
            # adjust for flow direction, and return
            return(pore_pressure * np.cos(direction)**2)

        
      
#TODO: Add a method to calculate the total depth of the profile.
#TODO: Add a method to plot the profile.
#TODO: FailureSurface(depth, orientation, shear_zone_thickness, cross_sectional_area) # AY

# GJM: quick placeholder for FailureSurface class, so I can test it with models
class FailureSurface:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in FAILURE_SURFACE_PARAMETERS.keys():
                raise ValueError(
                    f"Undefined parameter. Choose one of the following: {FAILURE_SURFACE_PARAMETERS.keys()}"
                )
            if not is_namedtuple(v):
                raise TypeError("Parameter should be of type Parameter(value, unit)")
            if not isinstance(v.value, (FAILURE_SURFACE_PARAMETERS[k]["type"] | list)):
                raise TypeError(
                    f"Value should be of type {FAILURE_SURFACE_PARAMETERS[k]["type"]} or a list"
                )
            setattr(self, k, v.value * units(v.unit))
