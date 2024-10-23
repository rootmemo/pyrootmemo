import numpy as np
from pyrootmemo.tools.checks import check_kwargs

ROOT_PARAMETERS = {
    "species": str,
    "elastic_modulus": (float | int),
    "diameter": (float | int),
    "tensile_strength": (float | int),
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
        if check_kwargs(arguments=kwargs, parameters=ROOT_PARAMETERS):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def calc_xsection(self) -> float:
        return np.pi * np.array(self.diameter) ** 2 / 4

    def calc_circumference(self) -> float:
        return np.pi * np.array(self.diameter)


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
