*********
Materials
*********

This module provides classes for representing various materials used in the root reinforcement modelling, including roots, soils, and interfaces between them. 

Currently implemented classes are:

* :class:`.Roots`: Creates a root object with specified parameters

* :class:`.Soil`: Creates a soil object with specified parameters

* :class:`.Interface`: Creates an interface object with specified parameters

.. automodule:: pyrootmemo.materials
   :members: Roots, Soil, Interface, ROOT_PARAMETERS, SOIL_PARAMETERS, INTERFACE_PARAMETERS
   :undoc-members:
   :show-inheritance:
   :exclude-members: init
   :noindex: true

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Materials

    Roots <roots>
    Soil <soil>
    Interface <interface>