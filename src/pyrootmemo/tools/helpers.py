import numpy as np


def secant(degree):
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
