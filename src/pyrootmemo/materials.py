import numpy as np
from pyrootmemo.tools.checks import is_namedtuple
from pint import DimensionalityError
from collections import namedtuple
from pyrootmemo.tools.helpers import units

Parameter = namedtuple("parameter", "value unit")

ROOT_PARAMETERS = {
    "elastic_modulus": {"type": (float | int), "unit": units("MPa")},
    "diameter": {"type": (float | int), "unit": units("m")},
    "tensile_strength": {"type": (float | int), "unit": units("MPa")},
    "yield_strength": {"type": (float | int), "unit": units("MPa")},
    "plastic_modulus": {"type": (float | int), "unit": units("MPa")},
    "length": {"type": (float | int), "unit": units("m")}
}

SOIL_PARAMETERS = {
    "cohesion": {"type": (float | int), "unit": units("kPa")},
    "friction_angle": {"type": (float | int), "unit": units("degrees")},
    "unit_weight_bulk": {"type": (float | int), "unit": units("kN/m^3")},
    "unit_weight_dry": {"type": (float | int), "unit": units("kN/m^3")},
    "unit_weight_saturated": {"type": (float | int), "unit": units("kN/m^3")},
    "water_content": {"type": (float | int), "unit": units("").to("percent")},
}


class Roots:
    def __init__(self, species: str, **kwargs):
        if not isinstance(species, str):
            raise TypeError("Species should be entered as a string, e.g. alnus_incana")
        if "_" not in species:
            raise ValueError(
                "Species name should be separated with an underscore, e.g. alnus_incana"
            )
        if len(species.split("_")) > 2:
            raise ValueError(
                "It is suggested follow botanical nomenclature with genus and species name,e.g. alnus_incana"
            )
        self.species = species
        for k, v in kwargs.items():
            if k not in ROOT_PARAMETERS.keys():
                raise ValueError(
                    f"Undefined parameter. Choose one of the following: {ROOT_PARAMETERS.keys()}"
                )
            if not is_namedtuple(v):
                raise TypeError("Parameter should be of type Parameter(value, unit)")
            if not isinstance(v.value, (ROOT_PARAMETERS[k]["type"] | list)):
                raise TypeError(
                    f"Value should be of type {ROOT_PARAMETERS[k]["type"]} or a list"
                )
            if not isinstance(v.unit, str):
                raise TypeError("Unit should be entered as a string")
            if not units(v.unit).check(ROOT_PARAMETERS[k]["unit"].dimensionality):
                raise DimensionalityError(
                    units1=v.unit, units2=ROOT_PARAMETERS[k]["unit"]
                )
            if isinstance(v.value, list):
                if not all(
                    [isinstance(entry, ROOT_PARAMETERS[k]["type"]) for entry in v.value]
                ):
                    raise TypeError(
                        f"{k} should only be of type {ROOT_PARAMETERS[k]["type"]} in a list"
                    )

            setattr(self, k, v.value * units(v.unit))

            if hasattr(self, "diameter"):
                self.xsection = np.pi * self.diameter**2 / 4
            else:
                raise AttributeError(
                    "Diameter is needed to calculate cross-sectional area"
                )

            if hasattr(self, "diameter"):
                self.circumference = np.pi * self.diameter
            else:
                raise AttributeError("Diameter is needed to calculate circumference")


class SingleRoot(Roots):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k != "species":
                if isinstance(v.value, list):
                    if len(v.value) > 1:
                        raise TypeError(
                            f"SingleRoot class cannot have multiple values for {k}. Use MultipleRoots class"
                        )

        super().__init__(**kwargs)


class MultipleRoots(Roots):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Soil:
    def __init__(
        self,
        name: str,
        uscs_classification: str = None,
        usda_classification: str = None,
        **kwargs,
    ):
        if not isinstance(name, str):
            raise TypeError("Soil name should be entered as a string")
        for k, v in kwargs.items():
            if k not in SOIL_PARAMETERS.keys():
                raise ValueError(
                    f"Undefined parameter. Choose one of the following: {SOIL_PARAMETERS.keys()}"
                )
            if not is_namedtuple(v):
                raise TypeError("Parameter should be of type Parameter(value, unit)")
            if not isinstance(v.value, (SOIL_PARAMETERS[k]["type"] | list)):
                raise TypeError(
                    f"Value should be of type {SOIL_PARAMETERS[k]["type"]} or a list"
                )
            if not isinstance(v.unit, str):
                raise TypeError("Unit should be entered as a string")
            if not units(v.unit).check(SOIL_PARAMETERS[k]["unit"].dimensionality):
                raise DimensionalityError(
                    units1=v.unit, units2=SOIL_PARAMETERS[k]["unit"]
                )
            if isinstance(v.value, list):
                if not all(
                    [isinstance(entry, SOIL_PARAMETERS[k]["type"]) for entry in v.value]
                ):
                    raise TypeError(
                        f"{k} should only be of type {SOIL_PARAMETERS[k]["type"]} in a list"
                    )

            setattr(self, k, v.value * units(v.unit))
