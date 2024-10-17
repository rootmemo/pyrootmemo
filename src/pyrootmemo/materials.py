import numpy as np
from pyrootmemo.tools.checks import check_kwargs

ROOT_PARAMETERS = {
    "species": str,
    "elastic_modulus": (float | int),
    "diameter": (float | int),
    "tensile_strength": (float | int),
}
SOIL_PARAMETERS = {
    "type": str,
    "cohesion": (float | int),
    "friction_angle": (float | int),
    "unit_weight": (float | int),
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
    def __init__(
        self, type: str, friction_angle=None, unit_weight: dict = None, cohesion=None
    ):
        """
        Initialises Soil class

        Args:
            type (str): _description_
            friction_angle (int or float, optional): Friction angle of the soil in degrees. Defaults to None.
            unit_weight (dict, optional): Unit weight of soil in kN/m^3. Allowed keys are saturated, dry, bulk. Defaults to None.
            cohesion (int or float, optional): Cohesion of soil in kPa. Defaults to None.
        """
        self.type = type
        self.friction_angle = friction_angle
        self.cohesion = cohesion
        unit_weight_keys = ["saturated", "dry", "bulk"]

        try:
            for k, v in unit_weight.items():
                if k in unit_weight_keys:
                    setattr(self, f"{k}_unit_weight", v)
                    # TODO: check against int or float
                    # TODO: Users should not be able to multiple entries with same key
                else:
                    raise KeyError(
                        f"{k} should be one of the following: {unit_weight_keys}"
                    )
        except TypeError:
            print("TypeError: Unit weight should be of type dict")
        except AttributeError:
            print("AttributeError: Unit weight should be of type dict")
