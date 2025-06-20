import numpy as np
from pyrootmemo.tools.checks import is_namedtuple
from pint import DimensionalityError
from collections import namedtuple
from pyrootmemo.helpers import units, limit_check

Parameter = namedtuple("parameter", "value unit")


#TODO: Check for non-negativeness and positivity of parameters

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

#TODO: Implement unit_weight_saturated > unit_weight_bulk > unit_weight_dry
#: A dictionary that maps soil parameter names to their types and units.
SOIL_PARAMETERS = {
    "cohesion": {"type": (float | int), "unit": units("kPa"), "limit_check": "non-negative"},
    "friction_angle": {"type": (float | int), "unit": units("degrees"), "limit_check": "non-negative"},
    "unit_weight_bulk": {"type": (float | int), "unit": units("kN/m^3"), "limit_check": "positive_only"},
    "unit_weight_dry": {"type": (float | int), "unit": units("kN/m^3"), "limit_check": "positive_only"},
    "unit_weight_saturated": {"type": (float | int), "unit": units("kN/m^3"), "limit_check": "positive_only"},
    "water_content": {"type": (float | int), "unit": units("").to("percent"), "limit_check": "non-negative"},
}

#: A dictionary that maps root-soil interface parameter names to their types and units.
ROOT_SOIL_INTERFACE_PARAMETERS = {
    "shear_strength": {"type": (float | int), "unit": units("kPa"), "limit_check": "positive_only"},
    "adhesion": {"type": (float | int), "unit": units("kPa"), "limit_check": "non-negative"},
    "friction_angle": {"type": (float | int), "unit": units("degrees"), "limit_check": "positive_only"},
    "effective_stress": {"type": (float | int), "unit": units("kPa"), "limit_check": "positive_only"},    
}

class Roots:
    """
    Roots class is used to create a root object with specified parameters.
    The parameters are defined in the ROOT_PARAMETERS dictionary: elastic_modulus, diameter, tensile_strength, yield_strength, plastic_modulus, unload_modulus, length, length_surface, azimuth_angle, elevation_angle.
    It allows for a single root or multiple roots to be created, depending on the class used (SingleRoot or MultipleRoots).

    Attributes
    ----------
    species : str
        Name of the species in the format 'genus_species', e.g. 'alnus_incana'.
    elastic_modulus : Parameter(float | int, str)
        Elastic modulus of the root (MPa).
    diameter : Parameter(float | int, str)
        Diameter of the root (m).
    tensile_strength : Parameter(float | int, str)
        Tensile strength of the root (MPa).
    yield_strength : Parameter(float | int, str)
        Yield strength of the root (MPa).
    plastic_modulus : Parameter(float | int, str)
        Plastic modulus of the root (MPa).
    unload_modulus : Parameter(float | int, str)
        Unload modulus of the root (MPa).
    length : Parameter(float | int, str)
        Length of the root (m).
    length_surface : Parameter(float | int, str)
        Length of the root surface (m).
    azimuth_angle : Parameter(float | int, str)
        Azimuth angle of the root (degrees).
    elevation_angle : Parameter(float | int, str)
        Elevation angle of the root (degrees).
    xsection : Parameter(float | int, str)
        Cross-sectional area of the root (m^2), calculated as π * (diameter / 2)^2.
    circumference : Parameter(float | int, str)
        Circumference of the root (m), calculated as π * diameter.
    initial_orientation_vector : np.ndarray
        A 3D vector representing the initial orientation of the root in space, calculated based on the azimuth and elevation angles.
    axis_angle : Parameter(float | int, str), optional
        Angle in radians to rotate the initial orientation vector around the z-axis, by default None.
        If None, the initial orientation vector is returned without rotation.

    Methods
    -------
    initial_orientation_vector(axis_angle: float | int = None) -> np.ndarray
        Returns the initial orientation vector of the root in 3D space.
        The vector is calculated based on the azimuth and elevation angles of the root.
        If axis_angle is provided, the vector is rotated around the z-axis by that angle.
    """
    def __init__(self, species: str, **kwargs):
        """
        Creates a root object with specified parameters.
        The parameters are defined in the ROOT_PARAMETERS dictionary.

        Parameters
        ----------
        species : str
            Name of the species in the format 'genus_species', e.g. 'alnus_incana'.
        **kwargs : dict
            Keyword arguments for root parameters. Each key should be a valid parameter name
            from ROOT_PARAMETERS, and the value should be a namedtuple of type Parameter(value, unit). Parameters include:
            - elastic_modulus: Elastic modulus of the root (MPa)
            - diameter: Diameter of the root (m)
            - tensile_strength: Tensile strength of the root (MPa)
            - yield_strength: Yield strength of the root (MPa)
            - plastic_modulus: Plastic modulus of the root (MPa)
            - unload_modulus: Unload modulus of the root (MPa)
            - length: Length of the root (m)
            - length_surface: Length of the root surface (m)
            - azimuth_angle: Azimuth angle of the root (degrees)
            - elevation_angle: Elevation angle of the root (degrees)

        Raises
        ------
        TypeError
            Species should be entered as a string, e.g. alnus_incana
        ValueError
            Species name should be separated with an underscore, e.g. alnus_incana
        ValueError
            It is suggested to follow botanical nomenclature with genus and species name, e.g. alnus_incana
        ValueError
            Undefined parameter. Choose one of the following: elastic_modulus, diameter, tensile_strength, yield_strength, plastic_modulus, unload_modulus, length, length_surface, azimuth_angle, elevation_angle
        TypeError
            Parameter should be of type Parameter(value, unit)
        TypeError
            Value should be of type float or int or a list
        TypeError
            Unit should be entered as a string
        DimensionalityError
            Unit dimensionality does not match the expected dimensionality for the parameter
        TypeError
            {parameter_name} should only be of type {expected_type} in a list
        AttributeError
            Diameter is needed to calculate cross-sectional area
        AttributeError
            Diameter is needed to calculate circumference
        """
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
            if not isinstance(v.value, (ROOT_PARAMETERS[k]['type'] | list)):
                raise TypeError(
                    f"Value should be of type {ROOT_PARAMETERS[k]['type']} or a list"
                )
            if not isinstance(v.unit, str):
                raise TypeError("Unit should be entered as a string")
            if not units(v.unit).check(ROOT_PARAMETERS[k]['unit'].dimensionality):
                raise DimensionalityError(
                    units1=v.unit, units2=ROOT_PARAMETERS[k]['unit']
                )
            if isinstance(v.value, list):
                if not all(
                    [isinstance(entry, ROOT_PARAMETERS[k]['type']) for entry in v.value]
                ):
                    raise TypeError(
                        f"{k} should only be of type {ROOT_PARAMETERS[k]['type']} in a list"
                    )
                [limit_check(entry, k, ROOT_PARAMETERS[k]['limit_check']) for entry in v.value]
            else:
                limit_check(v.value, k, ROOT_PARAMETERS[k]['limit_check'])
            #TODO: check if all values in the list are positive or non-negative
            
            setattr(self, k, v.value * units(v.unit))

        if hasattr(self, "diameter"):
            self.xsection = np.pi * (self.diameter / 2)**2
            self.circumference = np.pi * self.diameter
            #TODO: total circumference and total cross-sectional area for multiple roots
        else:
            raise AttributeError("Diameter is needed to calculate cross-sectional area and circumference")
            
    def initial_orientation_vector(
            self,
            axis_angle: (float | int) = None
            ):
        """
        Returns the initial orientation vector of the root in 3D space.
        The vector is calculated based on the azimuth and elevation angles of the root. 

        Parameters
        ----------
        axis_angle : float  |  int, optional
            Angle in radians to rotate the initial orientation vector around the z-axis, by default None.
            If None, the initial orientation vector is returned without rotation.
        Returns
        -------
        np.ndarray
            A 3D vector representing the initial orientation of the root in space.
            The vector is of shape (1, 3) and contains the x, y, and z components.
        """

        if hasattr(self, 'elevation_angle'):
            if hasattr(self, 'azimuth_angle'):
                v = np.stack([
                    np.cos(self.azimuth_angle.magnitude) * np.sin(self.elevation_angle.magnitude),
                    np.sin(self.azimuth_angle.magnitude) * np.sin(self.elevation_angle.magnitude),
                    np.cos(self.elevation_angle.magnitude)
                ], axis = 1)
            else:
                v = np.stack([
                    np.sin(self.elevation_angle.magnitude),
                    np.zeros_like(self.diameter.magnitude),
                    np.cos(self.elevation_angle.magnitude)
                ], axis = 1)
        else:
            v = np.stack([
                np.zeros_like(self.diameter.magnitude),
                np.zeros_like(self.diameter.magnitude),
                np.zeros_like(self.diameter.magnitude)
            ], axis = 1)
        # rotate using axis-angle vector (Rodriguez equation)
        if axis_angle is None:
            return v
        else:
            if np.isscalar(axis_angle):
                # scalar input --> assume input equals angle between z-axis and rotated x-axis
                axis_angle = np.array([
                    np.sin(axis_angle),
                    0.0,
                    np.cos(axis_angle)
                ])
            elif len(axis_angle) == 2:
                # 2-D input --> assume rotation vector in x-z space
                axis_angle = np.array([
                    axis_angle[0],
                    0.0,
                    axis_angle[1]
                ])
            # magnitude of rotation, in rad
            theta = np.linalg.norm(axis_angle)  
            # rotation axis - unit vector
            if np.isclose(theta, 0.0):
                return v
            else:
                k = axis_angle / theta 
                # apply Rodriguez equation
                root_vector_rotated = (
                    v * np.cos(theta)
                    + np.cross(k, v) * np.sin(theta)
                    + np.outer(np.dot(v, k), k) * (1.0 - np.cos(theta))
                )
                return root_vector_rotated


class SingleRoot(Roots):
    """
    SingleRoot class inherits from Roots and is used to create a single root object with specified parameters.
    """
    def __init__(self, **kwargs):
        """
        Creates a single root object with specified parameters.
        The parameters are defined in the ROOT_PARAMETERS dictionary.

        Raises
        ------
        TypeError
            SingleRoot class cannot have multiple values for parameters other than species. Use MultipleRoots class instead.
        """
        for k, v in kwargs.items():
            if k != "species":
                if isinstance(v.value, list):
                    if len(v.value) > 1:
                        raise TypeError(
                            f"SingleRoot class cannot have multiple values for {k}. Use MultipleRoots class"
                        )

        super().__init__(**kwargs)


class MultipleRoots(Roots):
    """
    MultipleRoots class inherits from Roots and is used to create a collection of root objects with specified parameters.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Soil:
    """
    Soil class is used to create a soil object with specified parameters.
    The parameters are defined in the SOIL_PARAMETERS dictionary: cohesion, friction_angle, unit_weight_bulk, unit_weight_dry, unit_weight_saturated, water_content.
    It also allows for USCS and USDA classifications of the soil.

    Attributes
    ----------
    name : str
        Name of the soil, e.g. 'silty_sand'.
    uscs_classification : str, optional
        USCS classification of the soil, by default None
    usda_classification : str, optional
        USDA classification of the soil, by default None
    cohesion : float | int
        Cohesion of the soil (kPa).
    friction_angle : float | int
        Friction angle of the soil (degrees).
    unit_weight_bulk : float | int
        Bulk unit weight of the soil (kN/m^3).
    unit_weight_dry : float | int
        Dry unit weight of the soil (kN/m^3).
    unit_weight_saturated : float | int
        Saturated unit weight of the soil (kN/m^3).
    water_content : float | int
        Water content of the soil (percentage).
    """
    def __init__(
        self,
        name: str,
        uscs_classification: str = None,
        usda_classification: str = None,
        **kwargs,
    ):
        """
        Creates a soil object with specified parameters.
        The parameters are defined in the SOIL_PARAMETERS dictionary: cohesion, friction_angle, unit_weight_bulk, unit_weight_dry, unit_weight_saturated, water_content.

        Parameters
        ----------
        name : str
            Name of the soil, e.g. 'silty_sand'.
        uscs_classification : str, optional
            uscs classification of the soil, by default None
        usda_classification : str, optional
            usda classification of the soil, by default None

        Raises
        ------
        TypeError
            Soil name should be entered as a string
        ValueError
            Undefined parameter. Choose one of the following: cohesion, friction_angle, unit_weight_bulk, unit_weight_dry, unit_weight_saturated, water_content
        TypeError
            Parameter should be of type Parameter(value, unit)
        TypeError
            Value should be of type float or int or a list
        TypeError
            Unit should be entered as a string
        DimensionalityError
            Unit dimensionality does not match the expected dimensionality for the parameter
        TypeError
            Parameter chosen is not of the expected type in a list
        """
        if not isinstance(name, str):
            raise TypeError("Soil name should be entered as a string")
        self.name = name.lower().replace(" ", "_")
        self.uscs_classification = uscs_classification
        self.usda_classification = usda_classification
        #TODO: add checks for uscs classification
        #TODO: add checks for usda classification
        for k, v in kwargs.items():
            if k not in SOIL_PARAMETERS.keys():
                raise ValueError(
                    f"Undefined parameter. Choose one of the following: {SOIL_PARAMETERS.keys()}"
                )
            if not is_namedtuple(v):
                raise TypeError("Parameter should be of type Parameter(value, unit)")
            if not isinstance(v.value, (SOIL_PARAMETERS[k]['type'] | list)):
                raise TypeError(
                    f"Value should be of type {SOIL_PARAMETERS[k]['type']} or a list"
                )
            if not isinstance(v.unit, str):
                raise TypeError("Unit should be entered as a string")
            if not units(v.unit).check(SOIL_PARAMETERS[k]['unit'].dimensionality):
                raise DimensionalityError(
                    units1=v.unit, units2=SOIL_PARAMETERS[k]['unit']
                )
            if isinstance(v.value, list):
                if not all(
                    [isinstance(entry, SOIL_PARAMETERS[k]['type']) for entry in v.value]
                ):
                    raise TypeError(
                        f"{k} should only be of type {SOIL_PARAMETERS[k]['type']} in a list"
                    )
                [limit_check(entry, k, SOIL_PARAMETERS[k]['limit_check']) for entry in v.value]
            else:
                limit_check(v.value, k, SOIL_PARAMETERS[k]['limit_check'])

            setattr(self, k, v.value * units(v.unit))

class Interface:
    def __init__(
        self,
        **kwargs,
    ):
        """
        Creates an soil-root interface object with specified parameters.

        Attributes
        ----------
        shear_strength : float | int
            Shear strength of the interface (kPa).
        adhesion : float | int
            Adhesion of the interface (kPa).
        friction_angle : float | int
            Friction angle of the interface (degrees).
        effective_stress : float | int
            Effective stress of the interface (kPa).
       """
        for k, v in kwargs.items():
            if k not in ROOT_SOIL_INTERFACE_PARAMETERS.keys():
                raise ValueError(
                    f"Undefined parameter. Choose one of the following: {ROOT_SOIL_INTERFACE_PARAMETERS.keys()}"
                )
            if not is_namedtuple(v):
                raise TypeError("Parameter should be of type Parameter(value, unit)")
            if not isinstance(v.value, (ROOT_SOIL_INTERFACE_PARAMETERS[k]['type'] | list)):
                raise TypeError(
                    f"Value should be of type {ROOT_SOIL_INTERFACE_PARAMETERS[k]['type']} or a list"
                )
            if not isinstance(v.unit, str):
                raise TypeError("Unit should be entered as a string")
            if not units(v.unit).check(ROOT_SOIL_INTERFACE_PARAMETERS[k]['unit'].dimensionality):
                raise DimensionalityError(
                    units1=v.unit, units2=ROOT_SOIL_INTERFACE_PARAMETERS[k]['unit']
                )
            if isinstance(v.value, list):
                if not all(
                    [isinstance(entry, ROOT_SOIL_INTERFACE_PARAMETERS[k]['type']) for entry in v.value]
                ):
                    raise TypeError(
                        f"{k} should only be of type {ROOT_SOIL_INTERFACE_PARAMETERS[k]['type']} in a list"
                    )
                [limit_check(entry, k, ROOT_SOIL_INTERFACE_PARAMETERS[k]['limit_check']) for entry in v.value]
            else:
                limit_check(v.value, k, ROOT_SOIL_INTERFACE_PARAMETERS[k]['limit_check'])

            setattr(self, k, v.value * units(v.unit))