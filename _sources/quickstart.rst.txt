Getting Started
===============

.. note::
   This documentation is a work in progress. If you have any suggestions or find any issues, please feel free to :ref:`contribute <sec-contribute>` !

Overview
--------

`pyrootmemo` is a Python package designed to unify and homogenise the models implemented to estimate the contribution of root systems to the strength of soil. It provides a consistent interface to various models, allowing users to easily switch between them and compare their results. `pyrootmemo` relies on object-oriented programming principles, making it easy to extend and modify.

Installation
----------------

We strongly recommend to work in a virtual environment, and we suggest to use a `Conda <https://docs.conda.io/en/latest/>`_ environment to manage dependencies and avoid conflicts with other Python packages. Follow the instructions below to install `pyrootmemo` in a Conda environment. You can follow the steps below to create a new Conda environment and install `pyrootmemo`:

1. If you have not already installed, download your preferred installation of Conda. You can try `Miniforge <https://conda-forge.org/download/>`_, which is a minimal installer for Conda that includes only the packages necessary to get started with Conda and the conda-forge channel.

2. Create a virtual environment with Python

.. code-block:: bash

   conda create -n rrmm python

3. Once the installation is complete, activate the environment

.. code-block:: bash

   conda activate rrmm

**Alternative 1**

4. To install `pyrootmemo`, you can use pip:

.. code-block:: bash

    pip install pyrootmemo

**Alternative 2**

4. To install `pyrootmemo` from the source code, you can clone the repository and install it:

.. code-block:: bash

    git clone git@github.com:rootmemo/pyrootmemo.git

5. Navigate to the cloned repository and install `pyrootmemo` using the following command:

.. code-block:: bash

    cd pyrootmemo
    poetry install

Usage
--------

Check out Jupyter Notebooks in `tests/*.ipynb` for a comprehensive tutorials on how to use `pyrootmemo`!

