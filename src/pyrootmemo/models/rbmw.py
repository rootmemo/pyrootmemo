import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.special import gamma
from scipy.optimize import minimize, differential_evolution
from pyrootmemo.helpers import units, Parameter, create_quantity, solve_quadratic, solve_cubic
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.tools.utils_rotation import axisangle_rotate
from pyrootmemo.tools.utils_plot import round_range
from pint import Quantity

class Rbmw():
    """
    Class for Root Bundle Model Weibull (RBMw).
    
    The Root Bundle Model (Weibull) is a displacement-driven reinforcement 
    model developed by Schwarz et al (2013). Breakage of roots is taken 
    into account through a Weibull survival function, calculating the 
    likelihood that a root is still intact given the current loading.
    
    While Schwarz et al. set both the Weibull shape and scale parameter 
    independently, the implementation in this code automatically infers the 
    correct scale parameter from the root biomechanical properties instead to
    avoid overfitting. It is still possible to manually set the weibull scale
    parameter but realise that in this case the average strength of a root
    no longer matches the average strength defined in the 'roots' object.

    Attributes
    ----------
    roots
        MultipleRoots object containing properties of all roots in bundle
    weibull_shape
        Weibull shape parameter used in survival function
    weibull_scale
        Weibull scale parameter used in survival function
    
    Methods
    -------
    __init__(roots, load_sharing)
        Constructor
    calc_force(displacement)
        Calculate force in bundle at given displacement
    calc_peak_force()
        Calculate peak force in bundle
    calc_peak_reinforcement(failure_surface, k)
        Calculate peak reinforcement by bundle
    calc_reduction_factor()
        RBMw reinforcement relative to WWM reinforcement
    plot(...)
        Generate plot showing how reinforcements mobilises with displacement
    """

    def __init__(
            self, 
            roots: MultipleRoots,
            weibull_shape: float | int, 
            weibull_scale: float | int | None = None
            ):
        """Construct a RBMw bundle model class object

        Parameters
        ----------
        roots : MultipleRoots
            MultipleRoots object containing root properties.
            Must contain fields 'xsection', 'tensile_strength', 
            'length', 'elastic_modulus'.
        weibull_shape : float | int
            Weibull shape parameter (dimensionless). Must be finite and 
            larger than zero.
        weibull_scale : float | int | None, optional
            Weibull scale parameter describing the (dimensionless) ratio 
            tensile stress/average strength (or tensile force/average force, 
            or displacement/average displacement at failure). 
            Default is None, in which case it is calculated from the 
            Weibull shape parameter assuming an average ratio of 1, i.e.
            scale = 1 / gamma(1 + 1 / shape)
        """
        if weibull_scale is None: 
            self.weibull_scale = 1.0 / gamma(1.0 + 1.0 / weibull_shape)
            root_attributes_required = ['xsection', 'tensile_strength', 'length', 'elastic_modulus']
        else:
            if not (isinstance(weibull_scale, float) | isinstance(weibull_scale, int)):
                raise ValueError('weibull_scale must be a scalar value or None')
            if weibull_scale <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            if np.isinf(weibull_scale):
                raise ValueError('weibull_shape must have a finite value')
            self.weibull_scale = weibull_scale
            root_attributes_required = ['xsection', 'length', 'elastic_modulus']

        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be instance of class MultipleRoots')
        for i in root_attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots must contain ' + i + ' values')
        self.roots = roots

        if not (isinstance(weibull_shape, float) | isinstance(weibull_shape, int)):
            raise ValueError('weibull_shape must be a scalar value')
        if weibull_shape <= 0.0:
            raise ValueError('weibull_shape must exceed zero ')
        if np.isinf(weibull_shape):
            raise ValueError('weibull_shape must have a finite value')
        self.weibull_shape = weibull_shape

    def calc_force(
            self,
            displacement: Quantity | Parameter,
            total: bool = True,
            deriv: int = 0,
            sign: int | float = 1.0
            ) -> Quantity:
        """Calculate RBMw force at given level of displacement

        Parameters
        ----------
        displacement : Quantity | Parameter(int | float | np.ndarray, str)
            Current displacement. Can contain a single as well as multiple
            displacement levels simultanously
        total : bool, optional
            If True, return the total force of all roots. Otherwise, return
            results per root (axis 0, rows) at each displacement step 
            (axis 1, columns)
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
        Quantity
            If 'total' is True, returns the total force in the root bundle at
            given level(s) of displacement. If False, returns the force in
            each root (last axis) at each level of displacement (first axis)
        """
        displacement = create_quantity(displacement, 'mm')
        if np.isscalar(displacement.magnitude):
            displacement = np.array([displacement.magnitude]) * displacement.units
            displacement_scalar_input = True
        else:
            displacement_scalar_input = False
        # write force mobilisation curve in form: 
        #
        #   y = a*x*exp(-(x/b)^k)
        # 
        # using 2D arrays: First axis = displacement, second axis = roots
        a = (self.roots.elastic_modulus 
             * self.roots.xsection
             / self.roots.length)[:, np.newaxis]
        b = (self.roots.length
             * self.roots.tensile_strength
             / self.roots.elastic_modulus 
             * self.weibull_scale
            )[:, np.newaxis]
        x = displacement[np.newaxis, :]
        k = self.weibull_shape
        if deriv == 0:
            force = a * x * np.exp(-(x / b) ** k)
            if total is True:
                if displacement_scalar_input is True:
                    return(sign * np.sum(force))
                else:
                    return(sign * np.sum(force, axis = 0))
            else:
                return(sign * force.squeeze())
        elif deriv == 1:
            dforce_dx = (
                a * (1.0 - k * (x / b) ** k)
                * np.exp(-(x / b) ** k)
            )
            if total is True:
                if displacement_scalar_input is True:
                    return(sign * np.sum(dforce_dx))
                else:
                    return(sign * np.sum(dforce_dx, axis = 0))
            else:
                return(sign * dforce_dx.squeeze())
        elif deriv == 2:
            dforce2_dx2 = (
                a * k / x
                * np.exp(-(x / b) ** k)
                * (x / b) ** k
                * (k * (x / b) ** k - k - 1.0)
                )
            if total is True:
                if displacement_scalar_input is True:
                    return(sign * np.sum(dforce2_dx2))
                else:
                    return(sign * np.sum(dforce2_dx2, axis = 0))
            else:
                return(sign * dforce2_dx2.squeeze())
        else:
            raise ValueError('Only deriv = 0, 1 or 2 are currently available.')

    def calc_peak_force(
            self,
            method: str = 'Newton-CG'
            ) -> dict:
        """Calculate RBMw peak force

        The RBMw force--displacement trace may have multiple local maxima,
        making finding real maximum challenging. This function uses a root 
        solve method to find peaks, using multiple initial guesses.

        Initial guesses for peaks are determined by using the following method:
        
        1. for each root in the bundle, determine the displacement level <u>
           where the peak force in the bundle is generated. 
        2. sort these displacement levels in order
        3. calculate the forces as well as the derivative of force with 
           displacementin, each root at each of these displacement levels
        4. for each displacement, predict how large the total force in the 
           bundle may become at the next level of displacement considered, 
           using the calculated forces and gradients.
        5. Only keep displacement values where this potential total force
           exceeds the largest total force calculated during step 3.
        6. Use each of these (reduced number of) displacement as a starting
           point to find the global maximum. A gradient-based method is used
           since the calc_peak_force() function can analytically calculate
           the first and second derivative of force with respect to 
           displacement.
        7. Use the solution that generates the largest total reinforcement.

        Parameters
        ----------
        method : str, optional
            Method to use in the scipy.optimize.minimize algorithm.
            Default is 'Newton-CG'. Analytical jacobian and hessian are 
            analytically known (see function self.calc_force(), which allows 
            for returning derivatives).

        Returns
        -------
        dict
            Dictionary with keys 'displacement' and 'force', 
            containing the Quantity of the displacement and force at 
            peak.   
        """
        displacement_peak_all = (
            self.roots.tensile_strength / self.roots.elastic_modulus * self.roots.length
            / self.weibull_shape ** (1. / self.weibull_shape)
            * self.weibull_scale
        )
        displacement_peak_unique = (
            np.sort(np.unique(displacement_peak_all.magnitude)) 
            * displacement_peak_all.units
            )
        forces = self.calc_force(displacement_peak_unique, total = True).magnitude
        gradients = self.calc_force(displacement_peak_unique, total = True, deriv = 1).magnitude
        displacement_interval = np.diff(displacement_peak_unique.magnitude)
        forces_next_point = np.append(forces[:-1] + displacement_interval * gradients[:-1], 0.0)
        forces_max = np.maximum(forces, forces_next_point)
        displacement_guesses = displacement_peak_unique[forces_max >= np.max(forces)]
        displacement_options = np.concatenate([
            minimize(
                fun = lambda x: self.calc_force(
                    x * displacement_guesses.units, 
                    total = True, 
                    deriv = 0, 
                    sign = -1.0
                    ).magnitude,
                x0 = i.magnitude,
                jac = lambda x: self.calc_force(
                    x * displacement_guesses.units, 
                    total = True, 
                    deriv = 1, 
                    sign = -1.0
                    ).magnitude,
                hess = lambda x: self.calc_force(
                    x * displacement_guesses.units, 
                    total = True, 
                    deriv = 2, 
                    sign = -1.0
                    ).magnitude,
                method = method
            ).x
            for i in displacement_guesses
        ]) * displacement_guesses.units
        peak_force_options = self.calc_force(displacement_options, total = True)
        guess_index = np.argmax(peak_force_options.magnitude)
        return({
            'force': peak_force_options[guess_index],
            'displacement': displacement_options[guess_index] 
        })

    def calc_peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: int | float = 1.0
            ) -> dict:
        """
        Calculate peak reinforcement (largest soil reinforcement at any point)
        generated by the fibre bundle

        Parameters
        ----------
        failure_surface : FailureSurface
            Instance of "FailureSurface" class. Must contain the attribute 
            "cross_sectional_area" that contains the cross-sectinonal area of the
            failure surface
        k : int | float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.0.

        Returns
        -------
        dict
            Dictionary with keys 'displacement' and 'reinforcement', 
            containing the Quantity of the displacement and reinforcement at 
            peak.            
        """
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('failure_surface must contain attribute "cross_sectional_area"')
        if not(isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be scalar integer or float')
        peak = self.calc_peak_force()
        return({
            'reinforcement': k * peak['force'] / failure_surface.cross_sectional_area,
            'displacement': peak['displacement']
        })    
    
    def calc_reduction_factor(
            self
            ) -> float:
        """RBMw reduction factor, compared to WWM
        
        Calculate the ratio between bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0 and 1. '1' indicates all roots break simultaneously.

        Returns
        -------
        float
            reduction factor.
        """
        force_rbmw = self.calc_peak_force()['force']
        force_root = np.sum(self.roots.xsection * self.roots.tensile_strength)
        ratio = force_rbmw / force_root
        if isinstance(ratio, Quantity):
            return(ratio.magnitude)
        else:
            return(ratio)
    
    def plot(
            self,
            fig = None,
            ax = None,
            n: int = 251,
            stack: bool = False,
            peak: bool = True,
            fraction: int | float = 0.75,
            labels: list | bool = False, 
            xlabel: chr = 'Pull-out displacement', 
            ylabel: chr = 'Total force in root bundle',
            xunit: chr = 'mm',
            yunit: chr = 'N'
            ): 
        """Plot how forces in the RBMw mobilise with displacements

        Generate a matplotlib plot showing how forces in the root bundle are 
        mobilised, as function of (axial pull-out) displacement

        All values of displacements and force are shown in terms of 
        user-defined units, controlled by input arguments 'xunit' and 
        'yunit', respectively.

        The contribution of each individual roots can be shows by means of
        a stackplot (if `stack = True'). Each root can be labelled using the 
        optional 'label' input argument. By default, `stack = False' and only
        the force in the entire bundle is shown.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            matplotlib figure object. If not defined, a new figure is created. By 
            default None
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object to plot on. If not defined, a new axis is 
            created. By default None
        n : int, optional
            number of displacement positions to plot, by default 251
        stack : bool, optional
            shows contributions of all individual roots by means of a 
            stackplot. By default False
        peak : bool, optional
            show the location of the peak using a scatter point. By default 
            True
        fraction : int | float, optional
            Used to determine the maximum displacement level, by default 0.75.
            The maximum displacement is defined as the point at which all 
            'fraction' fraction of roots have broken, for any of the roots
            defined.
        labels : bool | list, optional
            labels to plot on contribution of each root, by default False.
            If False, no labels are plotted. If True, labels are plotted using
            the index of the root in the MultipleRoots object. Custom labels 
            can be inputted using a list, which must have the same length as 
            the number of roots in the bundle. Labels are plotted at those
            displacement levels where each root reaches its maximum force.
        xlabel : chr, optional
            x-axis label, by default 'Pull-out displacement'
        ylabel : chr, optional
            y-axis label, by default 'Total force in root bundle'
        xunit : chr, optional
            x-axis unit, by default 'mm'
        yunit : chr, optional
            y-axis unit, by default 'N'

        Returns
        -------
        tuple
            tuple containing Matplotlib figure and axis objects
        """
        peak_results = self.calc_peak_force()
        failure_displacement_per_root = (
            self.roots.tensile_strength 
            / self.roots.elastic_modulus 
            * self.roots.length
            )
        displacement_max = (
            np.max(failure_displacement_per_root) 
            * self.weibull_scale
            * (-np.log(1.0 - fraction)) ** (1.0 / self.weibull_shape)
            )
        displacement = np.linspace(0, displacement_max, n)
        force = self.calc_force(displacement, total = True)
        
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        if stack is True:
            ax.stackplot(
                displacement.to(xunit).magnitude, 
                self.calc_force(displacement, total = False).to(yunit).magnitude
                )
        ax.plot(
            displacement.to(xunit).magnitude, 
            force.to(yunit).magnitude, 
            '-', 
            c = 'black'
            )
        if peak is True:
            plt.scatter(
                peak_results['displacement'].to(xunit).magnitude, 
                peak_results['force'].to(yunit).magnitude, 
                c = 'black'
                )
        
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
        if plot_labels is True:
            labels_x_dimensional = (
                self.weibull_scale * (1.0 / self.weibull_shape)**(1.0 / self.weibull_shape)
                * self.roots.tensile_strength
                / self.roots.elastic_modulus
                * self.roots.length
                )
            labels_x = labels_x_dimensional.to(xunit).magnitude
            labels_y_dimensional_all = self.calc_force(labels_x_dimensional, total = False)
            labels_y_all = np.triu(labels_y_dimensional_all.to(yunit).magnitude)
            labels_y = np.sum(labels_y_all, axis = 0) -  0.5 * np.diag(labels_y_all)
            for xi, yi, li in zip(labels_x, labels_y, labels):
                ax.annotate(
                    li, xy = (xi, yi), 
                    ha = 'center', 
                    va = 'center', 
                    bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                    fontsize = 'small'
                    )
                
        ax.set_xlim(round_range(
            displacement_max.to(xunit).magnitude, 
            limits = [0, None]
            )['limits'])
        ax.set_ylim(round_range(
            peak_results['force'].to(yunit).magnitude, 
            limits = [0., None]
            )['limits'])
        ax.set_xlabel(xlabel + ' [' + xunit + ']')
        ax.set_ylabel(ylabel + ' [' + yunit + ']')
        
        return(fig, ax)