from pyrootmemo.helpers import units

#: A dictionary that maps root parameter names to their types and units.
ROOT_PARAMETERS = {
    "elastic_modulus": {"type": (float | int), "unit": units("MPa"), "limit_check": "positive_only"},
    "diameter": {"type": (float | int), "unit": units("m"), "limit_check": "positive_only"},
    "tensile_strength": {"type": (float | int), "unit": units("MPa"), "limit_check": "positive_only"},
    "yield_strength": {"type": (float | int), "unit": units("MPa"), "limit_check": "positive_only"},
    "plastic_modulus": {"type": (float | int), "unit": units("MPa"), "limit_check": "positive_only"},
    "unload_modulus": {"type": (float | int), "unit": units("MPa"), "limit_check": "positive_only"},
    "length": {"type": (float | int), "unit": units("m"), "limit_check": "positive_only"},
    "length_surface": {"type": (float | int), "unit": units("m"), "limit_check": "non-negative"},
    "azimuth_angle": {"type": (float | int), "unit": units("degrees"), "limit_check": "any"},
    "elevation_angle": {"type": (float | int), "unit": units("degrees"), "limit_check": "any"},
}
#TODO: Check for angles (azimuth, elevation)

#: A dictionary that maps soil parameter names to their types and units.
SOIL_PARAMETERS = {
    "cohesion": {"type": (float | int), "unit": units("kPa"), "limit_check": "non-negative"},
    "friction_angle": {"type": (float | int), "unit": units("degrees"), "limit_check": "non-negative"},
    "unit_weight_bulk": {"type": (float | int), "unit": units("kN/m^3"), "limit_check": "positive_only"},
    "unit_weight_dry": {"type": (float | int), "unit": units("kN/m^3"), "limit_check": "positive_only"},
    "unit_weight_saturated": {"type": (float | int), "unit": units("kN/m^3"), "limit_check": "positive_only"},
    "water_content": {"type": (float | int), "unit": units("").to("percent"), "limit_check": "non-negative"},
}
#TODO: Implement unit_weight_saturated > unit_weight_bulk > unit_weight_dry

#: A dictionary that maps root-soil interface parameter names to their types and units.
ROOT_SOIL_INTERFACE_PARAMETERS = {
    "shear_strength": {"type": (float | int), "unit": units("kPa"), "limit_check": "positive_only"},
    "adhesion": {"type": (float | int), "unit": units("kPa"), "limit_check": "non-negative"},
    "friction_angle": {"type": (float | int), "unit": units("degrees"), "limit_check": "positive_only"},
    "effective_stress": {"type": (float | int), "unit": units("kPa"), "limit_check": "positive_only"},
}

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

UNIT_WEIGHT_WATER = 9.81 * units('kN/m^3')