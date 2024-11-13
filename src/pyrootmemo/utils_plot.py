# packages
import numpy as np


# create 'nice looking' axis range and tick positions
# GJM, 12/11/2024
def round_range(
        data: np.ndarray,
        limits: list = [None, None],
        step: np.ndarray = np.array([0.1, 0.2, 0.5, 1, 2, 5]),  
        ticks: int = 8,
        margin: float = 0.0,
        offset: float = 0.001
        ):
    """
    Create 'nice looking' axis limits and tick spacings, based on input data

    Parameters
    ----------
    data : np.ndarray
        Numpy array with data to plot on axis.
    limits : list (size 2), optional
        User-defined, fixed lower and upper limits. If any of these are None,
        they will be determined by the function. 
        The default is [None, None].
    step : np.ndarray, optional
        Optional tick spacings, after data is corrected for order of magnitude. 
        The default is np.array([0.1, 0.2, 0.5, 1, 2, 5]).
    ticks : integer, optional
        the maximum number of individual ticks on the axis. The default is 8.
    margin : list (size 2), or float, optional
        Extra margin to apply on upper and/or lower domain. Defined as a 
        fraction of the range described by data limits. If defined as a scalar,
        the same limit is used for the lower and upper limit. 
        The default is 0.0.
    offset : float, optional
        If all data has the same value, axis are defined as value +/- offset.
        The default is 0.001  # offset for plotting.

    Returns
    -------
    Dictionary with fields 'limit' contaning a list (length 2) with 'nice'
    lower and upper axis limits, and a field 'breaks' containing an array
    (np.ndarray) with 'nice' tick positions.

    """
    # margin - seperate margin for both sides
    if np.isscalar(margin):
        margin = [margin, margin]
    # get all limits
    lower = limits[0] if limits[0] is not None else np.nanmin(data)
    upper = limits[1] if limits[1] is not None else np.nanmax(data)
    # cases
    if (upper <= lower):
        return({
            'limits': [lower - offset, upper + offset],
            'breaks': np.array([lower - offset, lower, upper + offset])
            })    
    else:
        limits_margin = [
            lower if limits[0] is not None else lower - margin[0]*(upper - lower),
            upper if limits[1] is not None else upper + margin[1]*(upper - lower)
            ]
        # order of magnitude of difference
        oom = np.floor(np.log10(limits_margin[1] - limits_margin[0]))
        # round scaled upper and lower values using potential step sizes (step)
        z0 = np.floor((limits_margin[0] / 10**oom) / step) * step
        z1 = np.ceil((limits_margin[1] / 10**oom) / step) * step
        # calculate number of ticks with distance <step> required for each option
        ntick = np.ceil((z1 - z0)/step).astype(int) + 1
        ntick[ntick > ticks] = 0
        # pick option that has the most ticks but number still smaller than <n>
        i = np.argmax(ntick)
        # set range
        limits_new = [z0[i] * 10**oom, z1[i] * 10**oom]
        # substitute fixed values
        if limits[0] is not None:
            limits_new[0] = limits[0]
        if limits[1] is not None:
            limits_new[1] = limits[1]
        # return disctionary with results
        return({
            'limits': limits_new,
            'breaks': np.linspace(*limits_new, ntick[i])
            })
