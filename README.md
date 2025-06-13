
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/rootmemo/pyrootmemo)
![GitHub Release Date](https://img.shields.io/github/release-date/rootmemo/pyrootmemo)
![GitHub License](https://img.shields.io/github/license/rootmemo/pyrootmemo)
![GitHub Repo stars](https://img.shields.io/github/stars/rootmemo/pyrootmemo)
![Read the Docs](https://img.shields.io/readthedocs/pyrootmemo)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rootmemo/pyrootmemo.git/sbee2025)

# pyrootmemo

A Python package to modelling and measuring the effects of roots on slope stability

## Installation

We strongly recommend to work in a virtual environment, and we suggest to use a [Conda](https://docs.conda.io/en/latest/) environment. Follow the instructions below to install `pyrootmemo` in a Conda environment.

1. Clone the repository hosted on Github [https://github.com/rootmemo/pyrootmemo](https://github.com/rootmemo/pyrootmemo)
2. If you have not already installed, download your preferred installation of Conda. You can try [Miniforge](https://conda-forge.org/download/)
3. Create a virtual environment with Python

    ```bash
    conda create -n rrmm python
    ```

4. Once the installation is complete, activate the environment

    ```bash
    conda activate rrmm
    ```

### Alternative 1

5. Navigate to the cloned repository and install `pyrootmemo` using the following command:

    ```bash
    pip install -e .
    ```

### Alternative 2

5. Navigate to the cloned repository and install `pyrootmemo` using the following command:

    ```bash
    poetry install
    ```

## Usage

Check out Jupyter Notebooks in `tests/*.ipynb` for a comprehensive tutorials on how to use `pyrootmemo`!

## Documentation

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pyrootmemo` was created by Anil Yildiz (RWTH Aachen University, Germany) and Gerrit Meijer (University of Bath, United Kingdom). It is licensed under the terms of the MIT license.

## Credits

`pyrootmemo` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
