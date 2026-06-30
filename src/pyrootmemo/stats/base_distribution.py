import numpy as np
import matplotlib.pyplot as plt
from pint import Quantity
from pyrootmemo.helpers import units
from pyrootmemo.tools.utils_plot import round_range
from .utils import nondimensionalise, redimensionalise


class _BaseDistribution:
    """
    Base class for distribution fitting classes
    """

    def plot(
            self,
            x_unit: str | None = None,
            bins: int | np.ndarray | str = None,
            x_label: str = 'x',
            y_label: str = 'Probability density',
            show_data: bool = True,
            show_fit: bool = True,
            show_legend: bool = False,
            legend_location: str = 'best',
            n: int = 251,
            axis_expand = 0.05,
            x_limits = [1.0e-12, None]
            ) -> tuple:
        """Plot histogram of data, and fitted distribution

        Parameters
        ----------
        x_unit : str | None, optional
            unit of x-data to show on axis, by default None
        bins : int | np.ndarray | str, optional
            `bins` argument for histogram, by default None
        x_label : str, optional
            x-axis label, by default 'x'
        y_label : str, optional
            y-axis label_, by default 'Probability density'
        show_data : bool, optional
            if `True`, display data as histogram, by default True
        show_fit : bool, optional
            if `True`, display fit using a line, by default True
        show_legend : bool, optional
            if `True`, show the legend, by default False
        legend_location : str, optional
            matplotlib legend location string, by default 'best'
        n : int, optional
            number of points to use for plotting the fitted distribution, 
            by default 251
        axis_expand : float, optional
            Scale all axis by this margin in order to show all points nicely, 
            by default 0.05
        x_limits : list, optional
            x-axis limits, by default [1.0e-12, None]. If None, a sensible 
            axis limit is automatically determined

        Returns
        -------
        tuple
            tuple with matplotlib figure and axis objects
        """

        if isinstance(x_unit, str):
            x_multiplier = (1.0 * self.x0.units).to(x_unit).magnitude
            x_unit_label = format(units(x_unit).units, '~')
            y_unit_label = format((1.0 / units(x_unit_label)).units, '~')
        else:
            x_multiplier = 1.0
            if isinstance(self.x0, Quantity):
                x_unit_label = format(self.x0.units, '~')
                y_unit_label = format((1.0 / units(x_unit_label)).units, '~')
            else:
                x_unit_label = None
                y_unit_label = None
        y_multiplier = 1.0 / x_multiplier
        x_data_nondimensional = nondimensionalise(self.x, self.x0)
        x_limits = round_range(
            x_multiplier * x_data_nondimensional * (1.0 + axis_expand), 
            limits = x_limits
            )['limits']        
        x_fit = np.linspace(
            redimensionalise(x_limits[0] / x_multiplier, self.x0), 
            redimensionalise(x_limits[1] / x_multiplier, self.x0), 
            n)
        x_fit_nondimensional = nondimensionalise(x_fit, self.x0)

        fig, ax = plt.subplots()

        if show_data is True:
            ax.hist(
                x_multiplier * x_data_nondimensional, 
                bins = bins,
                weights = self.weights,
                label = 'Data',
                density = True
                )

        if show_fit is True:
            if hasattr(self, 'calc_density'):
                y_fit = self.calc_density(x_fit)
                y_fit_nondimensional = nondimensionalise(y_fit, 1.0 / self.x0)
                ax.plot(
                    x_multiplier * x_fit_nondimensional, 
                    y_multiplier * y_fit_nondimensional, 
                    '-', 
                    label = 'Fit'
                    )
                
        if isinstance(x_unit_label, str):
            x_label = x_label + ' [' + x_unit_label + ']'
        ax.set_xlabel(x_label)
        if isinstance(y_unit_label, str):
            y_label = y_label + ' [' + y_unit_label + ']'
        ax.set_ylabel(y_label)
        ax.set_xlim(x_limits)
        
        if show_legend is True:
            ax.legend(loc = legend_location)

        return(fig, ax)