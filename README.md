
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/rootmemo/pyrootmemo)
![GitHub License](https://img.shields.io/github/license/rootmemo/pyrootmemo)
![GitHub Repo stars](https://img.shields.io/github/stars/rootmemo/pyrootmemo)
![Read the Docs](https://img.shields.io/readthedocs/pyrootmemo)

# pyrootmemo

`pyrootmemo` is a Python package for modelling the contribution of root systems to the shear strength of soil. It provides a consistent, object-oriented interface to several root reinforcement models, making it easy to compare results across models, fit parameters to measurements, and extend the library with new models.

## Models

- Models currently implemented are below. Modular structure of `pyrootmemo` makes it very easy to extend the further models. Please consult our Contribution Guidelines.

| Model | Class | Description |
| ----- | ----- | ----------- |
| Wu/Waldron Model | `Wwm` | Peak reinforcement assuming simultaneous mobilisation of all roots |
| Fibre Bundle Model | `Fbm` | Progressive breakage with load sharing between roots (Pollen & Simon 2005) |
| Root Bundle Model Weibull | `Rbmw` | Displacement-driven reinforcement with Weibull survival function (Schwarz et al. 2013) |
| Axial Pullout | `AxialPullout` | Force–displacement response of a single root pulled out of soil |
| Waldron | `Waldron` | Direct shear reinforcement model (Waldron 1977) |

## Installation

We recommend working in a virtual environment, e.g. [Miniforge](https://conda-forge.org/download/)

```bash
conda create -n rrmm python
conda activate rrmm
```

### From source

```bash
git clone https://github.com/rootmemo/pyrootmemo.git
cd pyrootmemo
poetry install
```

## Usage

Jupyter Notebooks in `tests/*.ipynb` provide comprehensive tutorials covering each model.

## Documentation

Full documentation is available at [pyrootmemo.readthedocs.io](https://pyrootmemo.readthedocs.io), including a quickstart guide, theoretical background, and API reference.

## Contributing

Contributions are welcome! Check out the contributing guidelines. By contributing to this project you agree to abide by its Code of Conduct.

## License

`pyrootmemo` was created by Anil Yildiz (RWTH Aachen University, Germany) and Gerrit Meijer (University of Bath, United Kingdom). It is licensed under the terms of the MIT license.

## Credits

`pyrootmemo` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
