import numpy as np
from pyrootmemo.tools.checks import check_kwargs, is_namedtuple
from pint import UnitRegistry, DimensionalityError
from collections import namedtuple

ureg = UnitRegistry()
Parameter = namedtuple("parameter", "value unit")

ROOT_PARAMETERS = {
    "elastic_modulus": {"type": (float | int), "unit": ureg("MPa")},
    "diameter": {"type": (float | int), "unit": ureg("m")},
    "tensile_strength": {"type": (float | int), "unit": ureg("MPa")},
}

SOIL_PARAMETERS = {
    "cohesion": {"type": (float | int), "unit": ureg("kPa")},
    "friction_angle": {"type": (float | int), "unit": ureg("degrees")},
    "unit_weight_bulk": {"type": (float | int), "unit": ureg("kN/m^3")},
    "unit_weight_dry": {"type": (float | int), "unit": ureg("kN/m^3")},
    "unit_weight_saturated": {"type": (float | int), "unit": ureg("kN/m^3")},
    "water_content": {"type": (float | int), "unit": ureg("").to("percent")},
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
            if not ureg(v.unit).check(ROOT_PARAMETERS[k]["unit"].dimensionality):
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

            setattr(self, k, v.value * ureg(v.unit))

    def calc_xsection(self) -> float:
        return np.pi * np.array(self.diameter.magnitude) ** 2 / 4

    def calc_circumference(self) -> float:
        return np.pi * np.array(self.diameter.magnitude)


class SingleRoot(Roots):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xsection = self.calc_xsection()
        self.circumference = self.calc_circumference()


class MultipleRoots(Roots):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xsection = self.calc_xsection()
        self.circumference = self.calc_circumference()


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
            if not ureg(v.unit).check(SOIL_PARAMETERS[k]["unit"].dimensionality):
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

            setattr(self, k, v.value * ureg(v.unit))
