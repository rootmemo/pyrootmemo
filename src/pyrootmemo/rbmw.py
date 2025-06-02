# import packages and functions
from scipy.special import gamma
from scipy.optimize import minimize
import numpy as np
from pyrootmemo.materials import MultipleRoots
from pyrootmemo.geometry import FailureSurface
from pyrootmemo.utils_plot import round_range
from pyrootmemo.tools.helpers import units
from pint import Quantity
import matplotlib.pyplot as plt


# RBMw class
class Rbmw():
    """
    Class for Root Bundle Model Weibull (RBMw).
           
    The class constains some additional methods:
    * 'force()': calculate the force in the root bundle at the current level of displacement
    * 'peak_force()': calculate peak force in root bundle
    * 'peak_reinforcement()': calculate peak root reinforcement, given a known
      soil area and Wu/Waldron orientation factor 'k'
    * 'plot()': plot showing how force changes as function of displacement
    
    """

    # initialise class
    def __init__(
            self, 
            roots: MultipleRoots,
            weibull_shape: float | int, 
            weibull_scale: float | int | None = None
            ) -> None:
        """
        Initiate RBMw bundle model class

        Parameters
        ----------
        roots : instance of MultipleRoots class. 
            Must contain fields 'xsection', 'tensile_strength', 
            'length', 'elastic_modulus'
        weibull_shape : float
            Weibull shape parameter (dimensionless)
        weibull_scale : float, optional
            Weibull scale parameter describing the (dimensionless) ratio 
            tensile stress/average strength (or tensile force/average force at 
            failure). Default is None, in which case it is calculated from the 
            Weibull shape parameter assuming an average ratio of 1.

        Returns
        -------
        None.

        """
        # check if roots of correct class
        if not isinstance(roots, MultipleRoots):
            TypeError('roots must be instance of class MultipleRoots')
        # check if roots contains all required instances
        attributes_required = ['xsection', 'tensile_strength', 'length', 'elastic_modulus']
        for i in attributes_required:
            if not hasattr(roots, i):
                AttributeError('roots must contain ' + i + ' values')
        # set roots
        self.roots = roots
        # check and set weibull shape parameter
        if not np.isscalar(weibull_shape) or weibull_shape is None:
            ValueError('weibull_shape must be a scalar value')
        if weibull_shape <= 0.0:
            ValueError('weibull_shape must exceed zero ')
        if np.isinf(weibull_shape):
            ValueError('weibull_shape must have a finite value')
        self.weibull_shape = weibull_shape
        # calculate weibull scale parameter
        if weibull_scale is None: 
            self.weibull_scale = 1. / gamma(1. + 1. / weibull_shape)


    # forces in roots at current level of axial displacement
    def force(
            self,
            displacement: Quantity | int | float | np.ndarray,
            displacement_unit = 'm',
            total: bool = True,
            deriv: int = 0,
            sign: int | float = 1.0
            ) -> Quantity:
        """
        Calculate RBMw force at given displacement

        Parameters
        ----------
        displacement : Quantity
            Quantity object with displacements. Should have unit with dimension 
            length.
        total : bool, optional
            If True, return the total force of all roots. Otherwise, return
            results per displacement step (rows) for each root (columns)
        deriv : int, optional
            Differentiation order for displacement. If deriv == 1, the first
            derivatives of force with respect to displacement are returned. 
            If deriv == 2, return second-order derivative. Default is 0.
        sign : int, float, optional
            Multiplication factor for all result returned by the function. 
            This is used to be able to use minimisation algorithms in order
            to find the global maximum force, see function self.peak_force(). 
            Default = 1.0

        Returns
        -------
        Quantity object with (derivative of) forces.
        """
        # if displacement is not defined with a unit, make into a Quantity
        if not isinstance(displacement, Quantity):
            displacement = displacement * units(displacement_unit)
        # if displacement is a scalar -> make a numpy array
        if np.isscalar(displacement.magnitude):
            displacement_scalar_input = True
            displacement = np.array([displacement.magnitude]) * displacement.units
        else:
            displacement_scalar_input = False
        # write force mobilisation curve in form: 
        #
        #   y = a*x*exp(-(x/b)^k)
        # 
        # using 2D arrays: First axis = displacement, second axis = roots
        a = (self.roots.elastic_modulus 
             * self.roots.xsection
             / self.roots.length)[np.newaxis, :]
        b = (self.roots.length
             * self.roots.tensile_strength
             / self.roots.elastic_modulus 
             * self.weibull_scale
            )[np.newaxis, :]
        x = displacement[:, np.newaxis]
        k = self.weibull_shape
        # force
        if deriv == 0:
            # calculate force per displacement step, per root
            y = a * x * np.exp(-(x / b) ** k)
            # return
            if total is True:
                if displacement_scalar_input is True:
                    return(sign * np.sum(y))
                else:
                    return(sign * np.sum(y, axis = 1))
            else:
                return(sign * y.squeeze())
        # derivative of force with respect to displacement
        elif deriv == 1:
            # calculate derivative of force with respect to dispalcement
            # per displacement step, per root
            dy_dx = (
                a * (1.0 - k * (x / b) ** k)
                * np.exp(-(x / b) ** k)
            )
            # return
            if total is True:
                if displacement_scalar_input is True:
                    return(sign * np.sum(dy_dx))
                else:
                    return(sign * np.sum(dy_dx, axis = 1))
            else:
                return(sign * dy_dx.squeeze())
        # second derivative of force with respect to displacement
        elif deriv == 2:
            # calculate 2nd derivative of force with respect to displacement
            # per displacement step, per root
            dy2_dx2 = (
                a * k / x
                * np.exp(-(x / b) ** k)
                * (x / b) ** k
                * (k * (x / b) ** k - k - 1.0)
                )
            # return
            if total is True:
                if displacement_scalar_input is True:
                    return(sign * np.sum(dy2_dx2))
                else:
                    return(sign * np.sum(dy2_dx2, axis = 1))
            else:
                return(sign * dy2_dx2.squeeze())


    # Calculate peak force 
    def peak_force(
            self,
            full_output: bool = False,
            method: str = 'Newton-CG'
            ) -> float | dict:
        """
        Calculate RBMw peak force

        The RBMw force--displacement trace may have multiple local maxima,
        making finding real maximum challenging. This function uses a root 
        solve method to find peaks, using multiple initial guesses.

        Initial guesses for peaks are determined by first finding the 
        displacements <u> where each single root reaches its maximum force. 
        Total bundle forces <F> at these locations are then calculated. 
        Subsequently, a guess is made of how large the force could become 
        in the interval to the next peak, using the force gradient at each 
        location <u>. Only locations where this potential force is larger 
        than the largest of forces in <F> are considered as initial guesses.
        This trims down the need for initial guesses and still ensures the
        global maximum is found.        

        Parameters
        ----------
        full_output : boolean, optional
            If False, returns the peak force only. If 'True', returns a 
            dictionary with fields 'force' and 'displacement' indicating
            the position of peak reinforcement in the force--displacement
            plot.
        method : str, optional
            Method to use in the scipy.optimize.minimize algorithm.
            Default is 'Newton-CG'. Analytical jacobian and hessian are 
            analytically known (see function self.force()).

        Returns
        -------
        Peak force (scalar), or dictionary with full results.
        """
        # displacements until peak reinforcement of each root
        displacements = (
            self.roots.tensile_strength / self.roots.elastic_modulus * self.roots.length
            / self.weibull_shape ** (1. / self.weibull_shape)
            * self.weibull_scale
        )
        displacements = np.sort(np.unique(displacements.magnitude)) * displacements.units
        # force and gradients at each peak
        forces = self.force(displacements, total = True).magnitude
        gradients = self.force(displacements, total = True, deriv = 1).magnitude
        # forwards predictions and max
        disp_diff = np.diff(displacements.magnitude)
        forces_forwards = np.append(forces[:-1] + disp_diff * gradients[:-1], 0.0)
        forces_max = np.maximum(forces, forces_forwards)
        # select which starting guesses to keep (max force that can be achieved from starting point must be larger than max from <forces> array)
        keep = (forces_max >= np.max(forces))
        # starting guesses for displacement at peak
        displacement_guesses = displacements[keep]
        # find displacement at peaks, for each of the starting guesses, using root solving
        unit = displacement_guesses.units
        fun = lambda x: self.force(x * unit, total = True, deriv = 0, sign = -1.0).magnitude
        jac = lambda x: self.force(x * unit, total = True, deriv = 1, sign = -1.0).magnitude
        hes = lambda x: self.force(x * unit, total = True, deriv = 2, sign = -1.0).magnitude
        displacement_options = np.concatenate([
            minimize(
                fun = fun,
                x0 = i.magnitude,
                jac = jac,
                hess = hes,
                method = method
            ).x
            for i in displacement_guesses
        ]) * unit
        # calculate forces at each displacement option (local peaks)
        peak_force_options = self.force(displacement_options, total = True)
        index = np.argmax(peak_force_options.magnitude)
        # return 
        if full_output is True:
            return({
                'force': peak_force_options[index],
                'displacement': displacement_options[index]    
            })
        else:
            return(peak_force_options[index])


    # reinforcement
    def peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: float = 1.0
            ) -> Quantity:
        """
        Calculate peak reinforcement (largest soil reinforcement at any point)
        generated by the fibre bundle

        Parameters
        ----------
        failure_surface : FailureSurface
            Instance of "FailureSurface" class. Must contain the attribute 
            "cross_sectional_area" that contains the cross-sectinonal area of the
            failure surface
        k : float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.0.

        Returns
        -------
        float
            peak root reinforcement.

        """
        # check cross-sectional area correctly defined
        if not isinstance(failure_surface, FailureSurface):
            TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            AttributeError('failure_surface must contain attribute "cross_sectional_area"')
        # return
        return(k * self.peak_force() / failure_surface.cross_sectional_area)
    
    
    # reduction factor
    def reduction_factor(
            self
            ) -> float:
        """
        Calculate the ratio between bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0 and 1. '1' indicates all roots break simultaneously.

        Returns
        -------
        float
            reduction factor.

        """
        force_rbmw = self.peak_force()
        force_root = np.sum(self.roots.xsection * self.roots.tensile_strength)
        return(force_rbmw / force_root)
    
    def plot(
            self,
            n: int = 251,
            stack: bool = False,
            peak: bool = True,
            fraction: int | float = 0.9,  # minimum fraction of roots broken in each diameter,
            labels: list | bool = False, 
            xlabel: chr = 'Displacement', 
            ylabel: chr = 'Total force in root bundle',
            xunit: chr = 'mm',
            yunit: chr = 'N'
            ): 
        """
        Show force-displacement plot for the RBMw model

        Parameters
        ----------
        n : int, optional
            Number of equally-spaced points to use to plot. 
        fraction : float, optional
            Plot until for every root, <fraction> of roots has broken
        xlabel, ylabel : str, optional
            x- and y-axis labels. Units will be added to this automatically
        xunit, yunit : str, optional
            displacement and force units to use on the axes. Results are 
            automatically converted before plotting
        peak : bool, optional
            plot the location of peak reinforcement using a scatter plot 
            marker

        Returns
        -------
        tuple
            tuple containing Matplotlib figure and axis objects

        """
        # calculate peak force and displacement
        peak_results = self.peak_force(full_output = True)
        # displacement to average root failure
        displacement_average = self.roots.tensile_strength / self.roots.elastic_modulus * self.roots.length
        displacement_max = (
            np.max(displacement_average) 
            * self.weibull_scale
            * (-np.log(1. - fraction)) ** (1. / self.weibull_shape)
        )
        # displacement range
        displacement = np.linspace(0, displacement_max, n)
        # calculate total forces
        force = self.force(displacement, total = True)
        # generate plot object
        fig, ax = plt.subplots()
        # convert values to magnitudes, for plotting
        peak_displacement_magnitude = peak_results['displacement'].to(xunit).magnitude
        peak_force_magnitude = peak_results['force'].to(yunit).magnitude
        displacement_magnitude = displacement.to(xunit).magnitude
        force_magnitude = force.to(yunit).magnitude
        # stack plot
        if stack is True:
            force_each = self.force(displacement, total = False)
            force_each_magnitude = force_each.to(yunit).magnitude
            ax.stackplot(displacement_magnitude, force_each_magnitude.transpose())
        # plot line
        ax.plot(displacement_magnitude, force_magnitude, '-', c = 'black')
        # plot peak reinforcement
        if peak is True:
            plt.scatter(peak_displacement_magnitude, peak_force_magnitude, c = 'black')
        # label text
        nroots = len(self.roots.diameter)
        if labels is True:
            labels = list(range(1, nroots + 1))
            plot_labels = True
        elif isinstance(labels, list):
            if len(labels) == nroots:
                plot_labels = True
            else:
                plot_labels = False
        else:
            plot_labels = False
        # add labels to plot
        if plot_labels is True:
            # label x-positions - at peaks of each individual root
            x_factor = self.weibull_scale * (1.0 / self.weibull_shape)**(1.0 / self.weibull_shape)
            labels_x_dimensional = (
                x_factor
                    * self.roots.tensile_strength
                    / self.roots.elastic_modulus
                    * self.roots.length
                )
            labels_x = labels_x_dimensional.to(xunit).magnitude
            # labels y-positions - halfway up each stacking instance
            labels_y_dimensional_all = self.force(labels_x_dimensional, total = False)
            labels_y_all = np.tril(labels_y_dimensional_all.to(yunit).magnitude)
            labels_y = np.sum(labels_y_all, axis = 1) -  0.5 * np.diag(labels_y_all)
            # add labels to plot
            for xi, yi, li in zip(labels_x, labels_y, labels):
                ax.annotate(
                    li, xy = (xi, yi), 
                    ha = 'center', 
                    va = 'center', 
                    bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                    fontsize = 'small'
                    )
        # axis limits
        ax.set_xlim(round_range(displacement_max.to(xunit).magnitude, limits = [0, None])['limits'])
        ax.set_ylim(round_range(peak_results['force'].to(yunit).magnitude, limits = [0., None])['limits'])
        # axis labels
        ax.set_xlabel(xlabel + ' [' + xunit + ']')
        ax.set_ylabel(ylabel + ' [' + yunit + ']')
        # return
        return(fig, ax)
