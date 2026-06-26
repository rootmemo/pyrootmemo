import pytest
from pint import Quantity
from pyrootmemo.constants import (
    ROOT_PARAMETERS,
    SOIL_PARAMETERS,
    ROOT_SOIL_INTERFACE_PARAMETERS,
    SOIL_PROFILE_PARAMETERS,
    FAILURE_SURFACE_PARAMETERS,
    UNIT_WEIGHT_WATER,
)
from pyrootmemo.helpers import units

VALID_LIMIT_CHECKS = {"positive_only", "non-negative", "any"}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _has_required_fields(param_dict, key, expect_limit_check=True):
    assert "type" in param_dict[key]
    assert "unit" in param_dict[key]
    if expect_limit_check:
        assert "limit_check" in param_dict[key]


def _unit_dimensionality_matches(param_dict, key, reference_unit_str):
    assert param_dict[key]["unit"].dimensionality == units(reference_unit_str).dimensionality


# --------------------------------------------------------------------------- #
# ROOT_PARAMETERS
# --------------------------------------------------------------------------- #

ROOT_KEYS = [
    "elastic_modulus", "diameter", "tensile_strength", "yield_strength",
    "plastic_modulus", "unload_modulus", "length", "length_surface",
    "azimuth_angle", "elevation_angle",
]


class TestRootParameters:
    def test_has_expected_keys(self):
        assert set(ROOT_PARAMETERS.keys()) == set(ROOT_KEYS)

    @pytest.mark.parametrize("key", ROOT_KEYS)
    def test_has_required_fields(self, key):
        _has_required_fields(ROOT_PARAMETERS, key)

    @pytest.mark.parametrize("key", ROOT_KEYS)
    def test_unit_is_quantity(self, key):
        assert isinstance(ROOT_PARAMETERS[key]["unit"], Quantity)

    @pytest.mark.parametrize("key", ROOT_KEYS)
    def test_type_is_float_or_int(self, key):
        assert ROOT_PARAMETERS[key]["type"] == (float | int)

    @pytest.mark.parametrize("key", ROOT_KEYS)
    def test_limit_check_is_valid(self, key):
        assert ROOT_PARAMETERS[key]["limit_check"] in VALID_LIMIT_CHECKS

    @pytest.mark.parametrize("key", [
        "elastic_modulus", "tensile_strength", "yield_strength",
        "plastic_modulus", "unload_modulus",
    ])
    def test_pressure_units(self, key):
        _unit_dimensionality_matches(ROOT_PARAMETERS, key, "MPa")

    @pytest.mark.parametrize("key", ["diameter", "length", "length_surface"])
    def test_length_units(self, key):
        _unit_dimensionality_matches(ROOT_PARAMETERS, key, "m")

    @pytest.mark.parametrize("key", ["azimuth_angle", "elevation_angle"])
    def test_angle_units(self, key):
        _unit_dimensionality_matches(ROOT_PARAMETERS, key, "degrees")

    @pytest.mark.parametrize("key", [
        "elastic_modulus", "diameter", "tensile_strength", "yield_strength",
        "plastic_modulus", "unload_modulus", "length",
    ])
    def test_positive_only_limit(self, key):
        assert ROOT_PARAMETERS[key]["limit_check"] == "positive_only"

    def test_length_surface_non_negative(self):
        assert ROOT_PARAMETERS["length_surface"]["limit_check"] == "non-negative"

    @pytest.mark.parametrize("key", ["azimuth_angle", "elevation_angle"])
    def test_angles_any_limit(self, key):
        assert ROOT_PARAMETERS[key]["limit_check"] == "any"


# --------------------------------------------------------------------------- #
# SOIL_PARAMETERS
# --------------------------------------------------------------------------- #

SOIL_KEYS = [
    "cohesion", "friction_angle", "unit_weight_bulk",
    "unit_weight_dry", "unit_weight_saturated", "water_content",
]


class TestSoilParameters:
    def test_has_expected_keys(self):
        assert set(SOIL_PARAMETERS.keys()) == set(SOIL_KEYS)

    @pytest.mark.parametrize("key", SOIL_KEYS)
    def test_has_required_fields(self, key):
        _has_required_fields(SOIL_PARAMETERS, key)

    @pytest.mark.parametrize("key", SOIL_KEYS)
    def test_unit_is_quantity(self, key):
        assert isinstance(SOIL_PARAMETERS[key]["unit"], Quantity)

    @pytest.mark.parametrize("key", SOIL_KEYS)
    def test_type_is_float_or_int(self, key):
        assert SOIL_PARAMETERS[key]["type"] == (float | int)

    @pytest.mark.parametrize("key", SOIL_KEYS)
    def test_limit_check_is_valid(self, key):
        assert SOIL_PARAMETERS[key]["limit_check"] in VALID_LIMIT_CHECKS

    def test_cohesion_units(self):
        _unit_dimensionality_matches(SOIL_PARAMETERS, "cohesion", "kPa")

    def test_friction_angle_units(self):
        _unit_dimensionality_matches(SOIL_PARAMETERS, "friction_angle", "degrees")

    @pytest.mark.parametrize("key", ["unit_weight_bulk", "unit_weight_dry", "unit_weight_saturated"])
    def test_unit_weight_units(self, key):
        _unit_dimensionality_matches(SOIL_PARAMETERS, key, "kN/m^3")

    def test_water_content_is_dimensionless(self):
        assert SOIL_PARAMETERS["water_content"]["unit"].dimensionality == {}

    @pytest.mark.parametrize("key", ["unit_weight_bulk", "unit_weight_dry", "unit_weight_saturated"])
    def test_unit_weights_positive_only(self, key):
        assert SOIL_PARAMETERS[key]["limit_check"] == "positive_only"

    @pytest.mark.parametrize("key", ["cohesion", "friction_angle", "water_content"])
    def test_non_negative_limit(self, key):
        assert SOIL_PARAMETERS[key]["limit_check"] == "non-negative"


# --------------------------------------------------------------------------- #
# ROOT_SOIL_INTERFACE_PARAMETERS
# --------------------------------------------------------------------------- #

INTERFACE_KEYS = ["shear_strength", "adhesion", "friction_angle", "effective_stress"]


class TestRootSoilInterfaceParameters:
    def test_has_expected_keys(self):
        assert set(ROOT_SOIL_INTERFACE_PARAMETERS.keys()) == set(INTERFACE_KEYS)

    @pytest.mark.parametrize("key", INTERFACE_KEYS)
    def test_has_required_fields(self, key):
        _has_required_fields(ROOT_SOIL_INTERFACE_PARAMETERS, key)

    @pytest.mark.parametrize("key", INTERFACE_KEYS)
    def test_unit_is_quantity(self, key):
        assert isinstance(ROOT_SOIL_INTERFACE_PARAMETERS[key]["unit"], Quantity)

    @pytest.mark.parametrize("key", INTERFACE_KEYS)
    def test_type_is_float_or_int(self, key):
        assert ROOT_SOIL_INTERFACE_PARAMETERS[key]["type"] == (float | int)

    @pytest.mark.parametrize("key", ["shear_strength", "adhesion", "effective_stress"])
    def test_pressure_units(self, key):
        _unit_dimensionality_matches(ROOT_SOIL_INTERFACE_PARAMETERS, key, "kPa")

    def test_friction_angle_units(self):
        _unit_dimensionality_matches(ROOT_SOIL_INTERFACE_PARAMETERS, "friction_angle", "degrees")

    def test_adhesion_non_negative(self):
        assert ROOT_SOIL_INTERFACE_PARAMETERS["adhesion"]["limit_check"] == "non-negative"

    @pytest.mark.parametrize("key", ["shear_strength", "friction_angle", "effective_stress"])
    def test_positive_only_limit(self, key):
        assert ROOT_SOIL_INTERFACE_PARAMETERS[key]["limit_check"] == "positive_only"


# --------------------------------------------------------------------------- #
# SOIL_PROFILE_PARAMETERS
# --------------------------------------------------------------------------- #

SOIL_PROFILE_KEYS = ["depth", "groundwater_table"]


class TestSoilProfileParameters:
    def test_has_expected_keys(self):
        assert set(SOIL_PROFILE_PARAMETERS.keys()) == set(SOIL_PROFILE_KEYS)

    @pytest.mark.parametrize("key", SOIL_PROFILE_KEYS)
    def test_has_type_and_unit_fields(self, key):
        _has_required_fields(SOIL_PROFILE_PARAMETERS, key, expect_limit_check=False)

    @pytest.mark.parametrize("key", SOIL_PROFILE_KEYS)
    def test_unit_is_quantity(self, key):
        assert isinstance(SOIL_PROFILE_PARAMETERS[key]["unit"], Quantity)

    @pytest.mark.parametrize("key", SOIL_PROFILE_KEYS)
    def test_length_units(self, key):
        _unit_dimensionality_matches(SOIL_PROFILE_PARAMETERS, key, "m")

    @pytest.mark.parametrize("key", SOIL_PROFILE_KEYS)
    def test_type_is_float_or_int(self, key):
        assert SOIL_PROFILE_PARAMETERS[key]["type"] == (float | int)


# --------------------------------------------------------------------------- #
# FAILURE_SURFACE_PARAMETERS
# --------------------------------------------------------------------------- #

FAILURE_SURFACE_KEYS = ["depth", "orientation", "shear_zone_thickness", "cross_sectional_area"]


class TestFailureSurfaceParameters:
    def test_has_expected_keys(self):
        assert set(FAILURE_SURFACE_PARAMETERS.keys()) == set(FAILURE_SURFACE_KEYS)

    @pytest.mark.parametrize("key", FAILURE_SURFACE_KEYS)
    def test_has_type_and_unit_fields(self, key):
        _has_required_fields(FAILURE_SURFACE_PARAMETERS, key, expect_limit_check=False)

    @pytest.mark.parametrize("key", FAILURE_SURFACE_KEYS)
    def test_unit_is_quantity(self, key):
        assert isinstance(FAILURE_SURFACE_PARAMETERS[key]["unit"], Quantity)

    @pytest.mark.parametrize("key", FAILURE_SURFACE_KEYS)
    def test_type_is_float_or_int(self, key):
        assert FAILURE_SURFACE_PARAMETERS[key]["type"] == (float | int)

    @pytest.mark.parametrize("key", ["depth", "shear_zone_thickness"])
    def test_length_units(self, key):
        _unit_dimensionality_matches(FAILURE_SURFACE_PARAMETERS, key, "m")

    def test_orientation_units(self):
        _unit_dimensionality_matches(FAILURE_SURFACE_PARAMETERS, "orientation", "deg")

    def test_cross_sectional_area_units(self):
        _unit_dimensionality_matches(FAILURE_SURFACE_PARAMETERS, "cross_sectional_area", "m^2")


# --------------------------------------------------------------------------- #
# UNIT_WEIGHT_WATER
# --------------------------------------------------------------------------- #

class TestUnitWeightWater:
    def test_is_quantity(self):
        assert isinstance(UNIT_WEIGHT_WATER, Quantity)

    def test_value(self):
        assert UNIT_WEIGHT_WATER.magnitude == pytest.approx(9.81)

    def test_units(self):
        assert UNIT_WEIGHT_WATER.dimensionality == units("kN/m^3").dimensionality
