import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pint import Quantity
from pyrootmemo.helpers import units
from pyrootmemo.tools.utils_plot import round_range
from .utils import nondimensionalise


class _BaseRegression:

    def plot(
            self,
            x_unit: str | None = None,
            y_unit: str | None = None,
            x_label: str | None = 'x',
            y_label: str | None = 'y',
            show_data: bool = True,
            show_fit: bool = True,
            show_confidence: bool = False,
            confidence_level = 0.95,
            covariance_method = 'fisher',
            show_prediction: bool = False,
            prediction_level = 0.95,
            show_legend: bool = False,
            legend_location: str = 'best',
            n: int = 101,
            axis_expand = 0.05,
            x_limits = [0.0, None],
            y_limits = [0.0, None],
            ax: Axes | None = None
            ) -> tuple | None:
        """Plot powerlaw fitting results

        This method generates a matplotlib plot that can show:
        1) data used to generate the fit
        2) the best power law fit
        3) confidence interval of the fit
        4) prediction interval of the fit

        Parameters
        ----------
        x_unit : str | None, optional
            unit for x-axis, by default None. If None, the unit of the x-data
            used to generate the fit is used.
        y_unit : str | None, optional
            unit for y-axis, by default None. If None, the unit of the y-data
            used to generate the fit is used.
        x_label : str | None, optional
            label for x-axis, by default 'x'. Units are automatically added (in
            square brackets)
        y_label : str | None, optional
            label for y-axis, by default 'y'. Units are automatically added (in
            square brackets)
        show_data : bool, optional
            plot the data used to generate the fit, by default True
        show_fit : bool, optional
            plot the powerlaw fit, by default True
        show_confidence : bool, optional
            show confidence intervals of the fit, by default False
        confidence_level : float, optional
            confidence level used to calculate the confidence interval, 
            by default 0.95
        covariance_method : str, optional
            method used to calculate the covariance matrix of fitting 
            parameters, required to calculate confidence interals. Can be
            'fisher' (based in Fisher information criterion) or 'bootstrap' 
            (numerical approximation), by default 'fisher'. See class method
            `calc_covariance_matrix()` for more info
        show_prediction : bool, optional
            show prediction interval, by default False
        prediction_level : float, optional
            confidence level used to calculate the prediction interval, 
            by default 0.95
        show_legend : bool, optional
            show a plot legend, by default False
        legend_location : str, optional
            matplotlib string to determine legend location, by default 'best'
        n : int, optional
            number of equally-spaced x-points to use for plotting the fit, the
            confidence interval and/or the prediction interval, by default 101
        axis_expand : float, optional
            scalar multiplier to add a little extra space to each axis, 
            by default 0.05. Axis limits are automatically rounded to 'nice'
            values regardless
        x_limits : list, optional
            user defined lower and upper limit for the x-axis, by default 
            [0.0, None]. If any of these are None, they are determined
            automatically. The units of these values are assuemd as 'x_unit',
            or in the absence of this, the units of x-data used to generate
            the plot
        y_limits : list, optional
            user defined lower and upper limit for the y-axis, by default 
            [0.0, None]. If any of these are None, they are determined
            automatically. The units of these values are assuemd as 'y_unit',
            or in the absence of this, the units of y-data used to generate
            the plot
        ax : matplotlib.axes.Axes | None, optional
            matplotlib axis on which to plot the results, by default None.
            if None, a new axis object is created and returned by
            the function. If the axis is defined, results are added to the 
            existing axis.

        Returns
        -------
        matplotlib.axes.Axes
            Plot axis object

        """        
        if isinstance(x_unit, str):
            x_multiplier = (1.0 * self.x0.units).to(x_unit).magnitude
            x_unit_label = format(units(x_unit).units, '~')
        else:
            x_multiplier = 1.0
            if isinstance(self.x0, Quantity):
                x_unit_label = format(self.x0.units, '~')
            else:
                x_unit_label = None
        if isinstance(y_unit, str):
            y_multiplier = (1.0 * self.y0.units).to(y_unit).magnitude
            y_unit_label = format(units(y_unit).units, '~')
        else:
            y_multiplier = 1.0
            if isinstance(self.y0, Quantity):
                y_unit_label = format(self.y0.units, '~')
            else:
                y_unit_label = None
        x_data_nondimensional = nondimensionalise(self.x, self.x0)
        y_data_nondimensional = nondimensionalise(self.y, self.y0)
        x_fit = np.linspace(self.x.min(), self.x.max(), n)
        x_fit_nondimensional = nondimensionalise(x_fit, self.x0)
        
        if not isinstance(ax, Axes):
            ax = plt.gca()

        if show_data is True:
            ax.plot(
                x_multiplier * x_data_nondimensional, 
                y_multiplier * y_data_nondimensional, 
                'x', 
                label = 'Data'
                )

        if show_fit is True:
            if hasattr(self, 'predict'):
                y_fit = self.predict(x_fit)
                y_fit_nondimensional = nondimensionalise(y_fit, self.y0)
                ax.plot(
                    x_multiplier * x_fit_nondimensional, 
                    y_multiplier * y_fit_nondimensional, 
                    '-', 
                    label = 'Fit'
                    )
                
        if show_prediction is True:
            if hasattr(self, 'calc_prediction_interval'):
                _, y_lower, y_upper = self.calc_prediction_interval(
                    x_fit, 
                    level = prediction_level, 
                    n = n,
                    )
                y_lower_nondimensional = nondimensionalise(y_lower, self.y0)
                y_upper_nondimensional = nondimensionalise(y_upper, self.y0)
                label_pred = str(round(prediction_level * 100)) + '% Prediction interval' 
                ax.fill_between(
                    x_multiplier * x_fit_nondimensional,
                    y_multiplier * y_lower_nondimensional,
                    y_multiplier * y_upper_nondimensional,
                    label = label_pred, 
                    alpha = 0.25
                    )           

        if show_confidence is True:
            if hasattr(self, 'calc_confidence_interval'):
                _, y_lower, y_upper = self.calc_confidence_interval(
                    x_fit, 
                    level = confidence_level, 
                    n = n,
                    method = covariance_method
                    )
                y_lower_nondimensional = nondimensionalise(y_lower, self.y0)
                y_upper_nondimensional = nondimensionalise(y_upper, self.y0)
                label_conf = str(round(confidence_level * 100)) + '% Confidence interval' 
                ax.fill_between(
                    x_multiplier * x_fit_nondimensional,
                    y_multiplier * y_lower_nondimensional,
                    y_multiplier * y_upper_nondimensional,
                    label = label_conf, 
                    alpha = 0.25
                    )
        if x_label is not None:
            if isinstance(x_unit_label, str):
                x_label = x_label + ' [' + x_unit_label + ']'
            ax.set_xlabel(x_label)
        if y_label is not None:
            if isinstance(y_unit_label, str):
                y_label = y_label + ' [' + y_unit_label + ']'
            ax.set_ylabel(y_label)

        ax.set_xlim(round_range(
            x_multiplier * x_data_nondimensional * (1.0 + axis_expand), 
            limits = x_limits
            )['limits'])
        ax.set_ylim(round_range(
            y_multiplier * y_data_nondimensional * (1.0 + axis_expand), 
            limits = y_limits
            )['limits'])
        
        if show_legend is True:
            ax.legend(loc = legend_location)
        
        return(ax)
         