import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import minimize
from pyrootmemo import Parameter
from pyrootmemo.helpers import create_quantity, Results, ResultsType
from pyrootmemo.geometry import FailureSurface
from pyrootmemo.materials import MultipleRoots
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
    output : dict
        Dictionary with all calculation results. By default, this contains a
        key `breakage_order` which indicates the order of breakage for all
        roots.
    
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
        output : dict
            Dictionary with all calculation results.
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

        self.output = {}

    def calc_force(
            self,
            displacement: Quantity | Parameter,
            results_per_root: bool = True,
            deriv: int = 0,
            multiplier: int | float = 1.0,
            results: str = "attribute"
            ):
        """Calculate RBMw force at given level of displacement

        Function creates a dictionary with the keys `displacement` and `force`. 
        How this is returned depends on the value of the `results` argument.
        
        If `total = False` also the key `force_per_root`, corresponding with
        a matrix with forces in each root (axis 0) at each displacement level 
        of displacement (axis 1).
        
        Parameters
        ----------
        displacement : Quantity | Parameter(int | float | np.ndarray, str)
            Current displacement. Can contain a single as well as multiple
            displacement levels simultanously
        results_per_root : bool, optional
            If True, also returns the matrix with the force in each root 
            (axis 0, rows) at each displacement step (axis 1, columns), by 
            default False
        deriv : int, optional
            Differentiation order for displacement. If deriv == 1, the first
            derivatives of force with respect to displacement are returned. 
            If deriv == 2, return second-order derivative. Default is 0.
        multiplier : int, float, optional
            Multiplication factor for all result returned by the function. 
            This is used to be able to use minimisation algorithms in order
            to find the global maximum force, see function self.peak_force(). 
            Default = 1.0
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.
        """        
        displacement = create_quantity(displacement, 'mm')
        output = {'displacement': displacement}
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

        match deriv:
            case 0:
                force_per_root = multiplier * a * x * np.exp(-(x / b) ** k)
            case 1:
                force_per_root = multiplier * (
                    a * (1.0 - k * (x / b) ** k)
                    * np.exp(-(x / b) ** k)
                    )
            case 2:
                force_per_root = multiplier * (
                a * k / x
                * np.exp(-(x / b) ** k)
                * (x / b) ** k
                * (k * (x / b) ** k - k - 1.0)
                )
            case _:
                raise ValueError('Only deriv = 0, 1 or 2 are currently available.')
            
        if results_per_root is True:
            output['force_per_root'] = force_per_root.squeeze()
        if displacement_scalar_input is True:
            output['force'] = np.sum(force_per_root)
        else:
            output['force'] = np.sum(force_per_root, axis = 0)

        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output


    def calc_peak_force(
            self,
            method: str = 'Newton-CG',
            results: str = 'attribute'
            ):
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

        Function creates a dictionary with the keys `peak_displacement` and 
        `peak_force`. How this is returned depends on the value of the 
        `results` argument.
        
        Parameters
        ----------
        method : str, optional
            Method to use in the scipy.optimize.minimize algorithm.
            Default is 'Newton-CG'. Analytical jacobian and hessian are 
            analytically known (see function self.calc_force(), which allows 
            for returning derivatives).
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.
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
        forces = self.calc_force(displacement_peak_unique, results = 'return')['force'].magnitude
        gradients = self.calc_force(displacement_peak_unique, results = 'return', deriv = 1)['force'].magnitude
        displacement_interval = np.diff(displacement_peak_unique.magnitude)
        forces_next_point = np.append(forces[:-1] + displacement_interval * gradients[:-1], 0.0)
        forces_max = np.maximum(forces, forces_next_point)
        displacement_guesses = displacement_peak_unique[forces_max >= np.max(forces)]
        displacement_options = np.concatenate([
            minimize(
                fun = lambda x: self.calc_force(
                    x * displacement_guesses.units, 
                    deriv = 0, 
                    results = 'return',
                    multiplier = -1.0
                    )['force'].magnitude,
                x0 = i.magnitude,
                jac = lambda x: self.calc_force(
                    x * displacement_guesses.units, 
                    deriv = 1, 
                    results = 'return',
                    multiplier = -1.0
                    )['force'].magnitude,
                hess = lambda x: self.calc_force(
                    x * displacement_guesses.units, 
                    deriv = 2, 
                    results = 'return',
                    multiplier = -1.0
                    )['force'].magnitude,
                method = method
            ).x
            for i in displacement_guesses
        ]) * displacement_guesses.units
        peak_force_options = self.calc_force(displacement_options, results = 'return')['force']
        best_start_index = np.argmax(peak_force_options.magnitude)
        output = {
            'peak_force': peak_force_options[best_start_index],
            'peak_displacement': displacement_options[best_start_index] 
            }
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output

    def calc_peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: int | float = 1.0,
            results: str = 'attribute'
            ):
        """
        Calculate peak reinforcement (largest soil reinforcement at any point)
        generated by the fibre bundle

        Function creates a dictionary with the keys `peak_displacement` and 
        `peak_reinforcement`. How this is returned depends on the value of the 
        `results` argument.
        
        Parameters
        ----------
        failure_surface : FailureSurface
            Instance of "FailureSurface" class. Must contain the attribute 
            "cross_sectional_area" that contains the cross-sectinonal area of the
            failure surface
        k : int | float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.0.
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.            
        """
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('failure_surface must contain attribute "cross_sectional_area"')
        if not(isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be scalar integer or float')
        output_force_peak = self.calc_peak_force(results = 'return')
        output = {
            'peak_displacement': output_force_peak['peak_displacement'],
            'peak_reinforcement': k * output_force_peak['peak_force'] / failure_surface.cross_sectional_area
            }
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output
            
    def calc_reduction_factor(
            self,
            results: str = 'attribute'
            ):
        """RBMw reduction factor, compared to WWM
        
        Calculate the ratio between bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0 and 1. '1' indicates all roots break simultaneously.

        Function creates a dictionary with the keys `reduction_factor. How this 
        is returned depends on the value of the `results` argument.
        
        Parameters
        ----------
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.            
        """
        force_rbmw = self.calc_peak_force(results = 'return')['peak_force']
        force_root = np.sum(self.roots.xsection * self.roots.tensile_strength)
        ratio = force_rbmw / force_root
        if isinstance(ratio, Quantity):
            ratio = ratio.magnitude
        output = {'reduction_factor': ratio}
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output
    
    def plot(
            self,
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
        matplotlib.axes.Axes
            Matplotlib axis object
        """
        output_peak_force = self.calc_peak_force(results = 'return')
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
        output_force = self.calc_force(
            displacement, 
            results_per_root = stack,
            results = 'return'
            )
        
        if ax is None:
            ax = plt.gca()
        if stack is True:
            ax.stackplot(
                displacement.to(xunit).magnitude, 
                output_force['force_per_root'].to(yunit).magnitude
                )
        ax.plot(
            displacement.to(xunit).magnitude, 
            output_force['force'].to(yunit).magnitude, 
            '-', 
            c = 'black'
            )
        if peak is True:
            plt.scatter(
                output_peak_force['peak_displacement'].to(xunit).magnitude, 
                output_peak_force['peak_force'].to(yunit).magnitude, 
                c = 'black'
                )
        
        nroots = len(self.roots.diameter)
        if labels is True:
            labels = list(range(nroots))
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
            labels_y_dimensional_all = self.calc_force(labels_x_dimensional, results_per_root = True, results = "return")['force_per_root']
            labels_y_all = np.triu(labels_y_dimensional_all.to(yunit).magnitude)
            labels_y = np.sum(labels_y_all, axis = 0) -  0.5 * np.diag(labels_y_all)
            print(labels_x)
            print(labels_y_dimensional_all)
            print(labels_y)
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
            output_peak_force['peak_force'].to(yunit).magnitude, 
            limits = [0., None]
            )['limits'])
        ax.set_xlabel(xlabel + ' [' + xunit + ']')
        ax.set_ylabel(ylabel + ' [' + yunit + ']')
        
        return(ax)