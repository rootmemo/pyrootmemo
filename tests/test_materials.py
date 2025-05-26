import pytest
import numpy as np
from collections import namedtuple

from pyrootmemo.materials import Roots, SingleRoot, MultipleRoots

Parameter = namedtuple("parameter", "value unit")


def test_roots_species():
    with pytest.raises(TypeError, match="Species should be entered as a string, e.g. alnus_incana"):
        Roots(species=190123)

def test_roots_diameter():
    with pytest.raises(ValueError, match="Undefined parameter. Choose one of the following"):
        Roots(species="alnus_incana", dmeter=Parameter(value=0.1, unit="m"))

def test_roots_unit():
    with pytest.raises(TypeError, match="Unit should be entered as a string"):
        MultipleRoots(species="alnus_incana", diameter=Parameter(value=[0.1,0.2,0.3,0.4], unit=190123))