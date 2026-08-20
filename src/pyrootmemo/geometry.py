import numpy as np
from pyrootmemo.tools.checks import is_namedtuple
from pyrootmemo.materials import Soil
from pyrootmemo.helpers import units
from pint import Quantity, DimensionalityError
from pyrootmemo.constants import SOIL_PROFILE_PARAMETERS, FAILURE_SURFACE_PARAMETERS, UNIT_WEIGHT_WATER


class SoilProfile:
    """
       Initialize a SoilProfile object with a list of Soil objects and optional parameters.
       This class represents a profile of soils, allowing for the calculation of vertical stress

       Attributes
       ----------
       soils : pyrootmemo.materials.Soil
           A list of Soil objects representing the different soil layers in the profile.
       depth : pyrootmemo.tools.helpers.Parameter
           The depth of each soil layer in the profile, specified as a Parameter object.
       groundwater_table : pyrootmemo.tools.helpers.Parameter
           The depth of the groundwater table in the profile, specified as a Parameter object.
        
       Methods 
       ------
       get_soil
           Returns the Soil object at the specified depth.
       calc_vertical_stress
           Calculates the vertical stress at a specific depth in the soil profile.
       calc_pore_pressure
           Calculates the pore pressure at a specific depth in the soil profile.
    """
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
            if k == "groundwater_table":
                if not isinstance(v.value, (float, int)):
                    raise TypeError("Groundwater table should be a single value")
                if v.value < 0:
                    raise ValueError("Groundwater table cannot be negative")

            setattr(self, k, v.value * units(v.unit))

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
        soils_deeper : pyrootmemo.materials.Soil
            Returns the Soil object at the specified depth.
        """
        soils_deeper = [s for s, d in zip(self.soils, self.depth) if d >= depth]
        return(soils_deeper[0])
    
    def calc_total_vertical_stress(
            self,
            depth: Quantity
            ) -> Quantity:
        """
        Calculate the vertical total stress at a specific depth in the soil
        profile.

        Uses `unit_weight_dry` as unit weight in soils above the water table, 
        but if not defined, uses `unit_weight_bulk` instead.

        Uses `unit_weight_saturated` as unit weight in soils below the water 
        table, but if not defined, uses `unit_weight_bulk` instead.

        Parameters
        ----------
        depth : Quantity
            The depth at which to calculate the vertical total stress.

        Returns
        -------
        Quantity
            The vertical stress at the specified depth
        """
        depth_top = np.append(0.0 * units('m'), self.depth[:-1])
        thickness = np.minimum(self.depth, depth) - np.minimum(depth_top, depth)
        if depth > self.groundwater_table:
            tmp_above = np.minimum(depth, self.groundwater_table)
            thickness_above_wt = (
                np.minimum(self.depth, tmp_above)
                - np.minimum(depth_top, tmp_above)
                )
            thickness_below_wt = thickness - thickness_above_wt
        else:
            thickness_above_wt = thickness
            thickness_below_wt = 0.0 * thickness
        unit = 'kN/m^3'
        unit_weight_above_wt = np.array([
            s.unit_weight_dry.to(unit).magnitude if hasattr(s, 'unit_weight_dry') else s.unit_weight_bulk.to(unit).magnitude
            for s in self.soils]) * units(unit)
        unit_weight_below_wt = np.array([
            s.unit_weight_saturated.to(unit).magnitude if hasattr(s, 'unit_weight_saturated') else s.unit_weight_bulk.to(unit).magnitude
            for s in self.soils]) * units(unit)
        return(np.sum(
            unit_weight_above_wt * thickness_above_wt
            + unit_weight_below_wt * thickness_below_wt
            ))
    
    def calc_pore_pressure(
            self,
            depth: Quantity,
            orientation: float | int = 0.0  
            ) -> Quantity:
        """Calculate pore pressure at specific depth

        A flow direction can be defined if needed. All flow is assumed to be
        parallel to this direction. 

        Parameters
        ----------
        depth : Quantity
            depth below the soil surface, measured vertically
        orientation : float | int, optional
            direction of flow, in radians

        Returns
        -------
        Quantity
            The pore pressure at the specified depth
        """
        if depth <= self.groundwater_table:
            return(0.0 * units('kPa'))
        else:
            pore_pressure = UNIT_WEIGHT_WATER * (depth - self.groundwater_table)
            return(pore_pressure * np.cos(orientation)**2)
    
    def calc_shear_strength(
            self,
            depth: Quantity,
            orientation: float | int = 0.0
            ) -> Quantity:
        """Calculate the shear strength

        Can deal with inclined failure planes, assumed perpendicular to the soil
        surface. 

        Parameters
        ----------
        depth : Quantity
            Depth to calculate strengt at
        orientation : float | int, optional
            dip direction of the failure surface, in radians. By default 0.0

        Returns
        -------
        Quantity
            Soil shear strength
        """
        total_vertical_stress = self.calc_total_vertical_stress(depth)
        total_normal_stress = total_vertical_stress * np.cos(orientation)**2
        pore_pressure = self.calc_pore_pressure(depth, orientation = orientation)
        effective_normal_stress = total_normal_stress - pore_pressure
        soil = self.get_soil(depth)
        return(soil.cohesion + effective_normal_stress * np.tan(soil.friction_angle))

        
      
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

    def calc_orientation_matrix(self) -> np.ndarray:
        """
        Calculate the matrix that describes the Cartesian axes of the failure
        surface, in global coordinates.

        Matrix consist of three horizontally concatenated unit column vectors, 
        describing the x'-axis (direction of shearing), z'-axis (normal to the
        shear plane, pointing towards the sliding block) and the y'-axis. All 
        are defined in the global coordinate system.

        These rotations are described by two subsequent rotations, starting off
        with the global system ([[1,0,0],[0,1,0],[0,0,1]]), where z is poiting
        upwards, i.e. the global z is analogues with the surface elevation.

        * azimuth: rotation around z-axis, positive from x to y
        * elevation: rotation around (now rotated) y-axis, positive from z to x

        Returns
        -------
        np.ndarray
            3*3 matrix with unit column vectors of each axes
        """
        if hasattr(self, 'azimuth_angle'):
            sin_azimuth = np.sin(self.azimuth_angle).magnitude
            cos_azimuth = np.cos(self.azimuth_angle).magnitude
        else:
            ## GJM: raise a warning - assumed x
            sin_azimuth = 0.0
            cos_azimuth = 1.0
        if hasattr(self, 'elevation_angle'):
            sin_elevation = np.sin(self.elevation_angle).magnitude
            cos_elevation = np.cos(self.elevation_angle).magnitude
        else:
            sin_elevation = 0.0
            cos_elevation = 1.0
        R_azimuth = np.array([
            [cos_azimuth, -sin_azimuth, 0.0],
            [sin_azimuth, cos_azimuth, 0.0],
            [0.0, 0.0, 1.0]
            ])
        R_elevation = np.array([
            [cos_elevation, 0.0, sin_elevation],
            [0.0, 1.0, 0.0],
            [-sin_elevation, 0.0, cos_elevation]
            ])
        return(R_elevation @ R_azimuth)
