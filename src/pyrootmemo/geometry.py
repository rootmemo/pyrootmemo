import numpy as np
from pyrootmemo.tools.checks import is_namedtuple
from pyrootmemo.materials import Soil
from pint import DimensionalityError
from collections import namedtuple
from pyrootmemo.tools.helpers import units

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
            setattr(self, k, v.value * units(v.unit))


#TODO: Groundwater table cannot be entered as a list. It should be a single value.
#TODO: Groundwater table cannot be negative, but can be zero.
#TODO: Add a method to calculate the total depth of the profile.
#TODO: Add a method to plot the profile.
#TODO: FailureSurface(depth, orientation, shear_zone_thickness, cross_sectional_area) # AY