class Roots:
    def __init__(self, species, elastic_modulus: int | float = None):
        self.species = species
        if elastic_modulus is not None:
            self.elastic_modulus = elastic_modulus


class SingleRoot(Roots):
    def __init__(
        self,
        species,
        elastic_modulus: int | float = None,
        diameter: int | float = None,
        **kwargs,
    ):
        super().__init__(species, elastic_modulus)
        self.diameter = diameter
        for k, v in kwargs.items():
            setattr(self, k, v)


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
                else:
                    raise KeyError(
                        f"{k} should be one of the following: {unit_weight_keys}"
                    )
        except TypeError:
            print("TypeError: Unit weight should be of type dict")
        except AttributeError:
            print("AttributeError: Unit weight should be of type dict")
