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
    "name": str,
    "uscs_class": str,
    "usda_class": str,
    "cohesion": (float | int),
    "friction_angle": (float | int),
    "unit_weight_bulk": (float | int),
    "unit_weight_dry": (float | int),
    "unit_weight_saturated": (float | int),
    "water_content": (float | int),
}


class Roots:
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
        if check_kwargs(arguments=kwargs, parameters=SOIL_PARAMETERS):
            for k, v in kwargs.items():
                setattr(self, k, v)
