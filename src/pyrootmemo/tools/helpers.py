import numpy as np


def secant(degree) -> float:
    """
    secant _summary_

    Args:
        degree (_type_): _description_

    Raises:
        TypeError: _description_
        ValueError: _description_

    Returns:
        float: _description_
    """
    try:
        secant = 1 / np.cos(np.deg2rad(degree))
    except TypeError as te:
        print(f"TypeError: Wrong input type ({te})")
        raise TypeError
    except ValueError as ve:
        print(f"ValueError: Wrong input value ({ve})")
        raise ValueError
    else:
        return secant
