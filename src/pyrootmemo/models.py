import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.special import gamma
from scipy.optimize import minimize, differential_evolution, bisect, root_scalar
from pyrootmemo.helpers import units, Parameter, create_quantity, solve_quadratic, solve_cubic
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.tools.checks import is_namedtuple
from pyrootmemo.tools.utils_rotation import axisangle_rotate
from pyrootmemo.tools.utils_plot import round_range
from pint import Quantity



class Wwm():
    """
    This class implements the Wu/Waldron model for root reinforcement.

    The Wu/Waldron model is a simple model that calculates the peak force
    mobilised by a bundle of roots, and the peak reinforcement that can be
    mobilised by the bundle at a given failure surface.

    The model assumes that all roots are mobilised at the same time, and that
    the peak force is the sum of the maximum tensile forces that can be
    mobilised in all roots.

    Attributes
    ----------
    roots : pyrootmemo.materials.MultipleRoots
        MultipleRoots object containing properties of all roots in bundle. roots
        must contain attributes 'diameter' and 'tensile_strength'.
            
    Methods
    -------
    __init__(roots)
        Constructor
    calc_peak_force()
        Calculate peak force in the bundle
    calc_peak_reinforcement(failure_surface, k)
        Calculate peak reinforcement by the bundle
    """
    
    def __init__(
            self, 
            roots: MultipleRoots,
            ):
        """Initiates a Wu/Waldron model object.

        Parameters
        ----------
        roots : MultipleRoots
            Contains information about all reinforcing roots, defined using the 
            MultipleRoots class.
            Class must contain attributes 'diameter', 'xsection', 'tensile_strength'
        """
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be an object of class MultipleRoots')
        attributes_required = ['diameter', 'xsection', 'tensile_strength']
        for i in attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots object must contain ' + str(i) + ' attribute')
        self.roots = roots

    def calc_peak_force(self) -> np.ndarray:
        """
        Calculates WWM peak force.

        This is defined as the sum of the maximum tensile forces that can 
        be mobilised in all roots.

        Returns
        -------
        peak_force : np.ndarray
            Peak force mobilised by the bundle of roots, in units of force.
        """
        self.peak_force = np.sum(self.roots.xsection * self.roots.tensile_strength)
        return self.peak_force

    def calc_peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: int | float = 1.2
            ) -> Quantity:
        """
        Calculate peak reinforcement (largest soil reinforcement at any point)
        according to the WWM model

        Parameters
        ----------
        failure_surface : FailureSurface
            "FailureSurface" class object. Must contain the attribute 
            "cross_sectional_area" that contains the cross-sectinonal area 
            of the failure surface
        k : float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.2.

        Returns
        -------
        Quantity
            Peak reinforcement
        """        
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('failure_surface must contain attribute "cross_sectional_area"')
        if not (isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be an scalar integer or float')
        return(k * self.calc_peak_force() / failure_surface.cross_sectional_area)


class Fbm():
    """
    Class for fibre bundle models.
    
    Fibre bundle models assume the total force carried by the 'bundle' of
    individual roots is distributed according to a load sharing factor. The
    ratio in forces between two intact roots is equal to the ratio of their
    diameters to the power of the (dimensionless) load sharing coefficient.

    The traditional implementation by Pollen & Simon (2005) uses an iterative
    algorithm, incrementally increasing the force until all roots have broken.
    This class instead uses the "matrix" method as described in Yildiz & 
    Meijer. This method follows the following steps:
    
    1. Roots are sorted in order of breakage. The 'sorting order', i.e. the 
       list of indices to sort root properties into the correct order, is 
       stored in class attribute 'sort_order'. The 'breakage order', i.e. the 
       list of indices describing the order of breakage for each root, is 
       stored in class attribute 'breakage_order'.
     
    2. A matrix is generated that calculates the force in every root (rows), 
       at the moment of breakage of any root (columns). This matrix is stored
       as the class attribute 'matrix', and assumes roots have already been 
       sorted in order of breakage.
       
    3. Peak forces can now easily be termined by finding the column in the 
       matrix that has the largest sum of forces.
       
    Attributes
    ----------
    roots : pyrootmemo.materials.MultipleRoots
        MultipleRoots object containing properties of all roots in bundle
    load_sharing
        Load sharing value used
    sort_order
        order to sort roots in order of breakage
    breakage_order
        order of root breakage defined for each root
    matrix
        2-D matrix with forces in each root (rows, axis 0) at breakage of each 
        root (columns, axis 1)
    
    Methods
    -------
    __init__(roots, load_sharing)
        Constructor
    calc_peak_force()
        Calculate peak force in bundle
    calc_peak_reinforcement(failure_surface, k)
        Calculate peak reinforcement by bundle
    calc_reduction_factor()
        FBMw reinforcement relative to WWM reinforcement
    plot(...)
        Generate plot showing how reinforcement mobilises
    """
    
    def __init__(
            self, 
            roots: MultipleRoots, 
            load_sharing: float | int
            ): 
        """Create FBM model object

        Parameters
        ----------
        roots : pyrootmemo.materials.MultipleRoots
            MultipleRoots object containing root properties. Must contain 
            attributes 'diameter', 'xsection' and 'tensile_strength'
        load_sharing : float | int
            FBM load sharing parameter

        """
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be object of class MultipleRoots')
        roots_attributes_required = ['diameter', 'xsection', 'tensile_strength']
        for i in roots_attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots must contain ' + str(i) + ' attribute')
        self.roots = roots
        if not (isinstance(load_sharing, int) | isinstance(load_sharing, float)):
            raise ValueError('load_sharing must be a scalar integer or float')
        if np.isinf(load_sharing):
            raise ValueError('load_sharing must have finite value')
        self.load_sharing = load_sharing
        self.sort_order = np.argsort(
            roots.tensile_strength 
            * roots.xsection
            / (roots.diameter ** load_sharing)
            )
        self.breakage_order = np.argsort(self.sort_order)
        self.matrix = self.calc_matrix()

    def calc_matrix(self) -> Quantity:
        """Create matrix with forces in each root at breakage of all roots

        Generate matrix for force in each root (rows) at breakage of each 
        root (columns). In this matrix, roots are pre-sorted in order of 
        breakage using the 'breakage_order'. This makes it easy to only
        keep roots in the matrix that are still intact (lower triangle of 
        the square matrix).

        Returns
        -------
        Quantity
            matrix with forces in each roots (rows) at point of breakage of
            each root (columns). Roots are sorted in order of breakage.
        """
        force_break = self.roots.tensile_strength * self.roots.xsection
        force_unit = force_break.units
        y_sorted = (force_break.magnitude)[self.sort_order]
        x_sorted = (self.roots.diameter.magnitude)[self.sort_order]
        matrix = np.outer(
            x_sorted ** self.load_sharing,
            y_sorted / (x_sorted ** self.load_sharing)
            )
        matrix_broken = np.tril(matrix)
        return(matrix_broken * force_unit)

    def calc_peak_force(self) -> Quantity:
        """Calculate peak force in the fibre bundle

        Calculat the peak force that the entire fibre bundle can carry. This
        reinforcement is calculated using the 'matrix method' as described 
        by Yildiz and Meijer.
        
        Returns
        -------
        Quantity
            Peak force, i.e. largest force the entire bundle can carry
        """
        return(np.max(np.sum(self.matrix, axis = 0)))

    def calc_peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: int | float = 1.0
            ) -> Quantity:
        """Calculate peak reinforcement in the fibre bundle

        Calculate peak reinforcement (largest soil reinforcement at any point)
        generated by the fibre bundle

        Parameters
        ----------
        failure_surface : FailureSurface
            Instance of "FailureSurface" class. Must contain the attribute 
            "cross_sectional_area" that contains the cross-sectinonal area of the
            failure surface
        k : float | int, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.0.

        Returns
        -------
        Quantity
            Peak root reinforcement.
        """
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('Failure surface does not contain attribute cross_sectional_area')
        if not (isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be an scalar integer or float')
        return(k * self.calc_peak_force() / failure_surface.cross_sectional_area)
    
    def calc_reduction_factor(self) -> float:
        """Calculate FBM peak reinforcement relative to WWM

        Calculate the ratio between fibre bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0.0 and 1.0. '1.0' indicates all roots break simultaneously.

        Returns
        -------
        float
            ratio between FBM and WWM peak reinforcements.
        """
        force_fbm = self.peak_force()
        force_sum = np.sum(self.roots.xsection * self.roots.tensile_strength)
        factor = force_fbm / force_sum
        if isinstance(factor, Quantity):
            return(factor.magnitude)
        else:
            return(factor)
    
    def plot(
            self,
            fig: Figure = None,
            ax: Axes = None,
            unit: str = 'N',
            reference_diameter: Quantity | Parameter = Parameter(1.0, 'mm'),
            stack: bool = False,
            peak: bool = True,
            labels: list | bool = False, 
            label_margin: float = 0.05, 
            xlabel: str = 'Force in reference root', 
            ylabel: str = 'Total force in root bundle'      
            ) -> tuple:
        """Plot FBM force mobilisation

        Generate a matplotlib plot showing how forces in the fibre bundle are 
        mobilised, as function of the force in a root with a specific 
        reference diameter ('reference_diameter') that never breaks.

        All values of force are shown in terms of a user-defined unit,
        controlled by input argument 'unit'. 

        The contribution of each individual roots can be shows by means of
        a stackplot (if `stack = True'). Each root can be labelled using the 
        optional 'label' input argument. By default, `stack = False' and only
        the force in the entire bundle is shown.

        In order to show distinct breakages, forces are caculated both at the 
        moment of breakages as well as *just* after the moment of breakages.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            matplotlib figure object. If not defined, a new figure is created. By 
            default None
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object to plot on. If not defined, a new axis is 
            created. By default None
        unit : str, optional
            unit used for plotting forces, by default 'N'
        reference_diameter : Quantity | Parameter(float | int, str), optional
            value of the reference diameter, by default Pameter(1.0, 'mm')
        stack : bool, optional
            show contributions of each individual root by means of a 
            stackplot, by default False
        peak : bool, optional
            show location of peak force, by default True
        labels : list | bool, optional
            labels to plot on contribution of each root, by default False.
            If False, no labels are plotted. If True, labels are plotted using
            the index of the root in the MultipleRoots object. Custom labels 
            can be inputted using a list, which must have the same length as 
            the number of roots in the bundle.
        label_margin : float, optional
            relative x-offset for plotting labels, by default 0.05
        xlabel : str, optional
            x-axis label, by default 'Force in reference root'
        ylabel : str, optional
            y-axis label, by default 'Total force in root bundle'

        Returns
        -------
        tuple
            Tuple containing the matplotlib figure and axis object
            
        """
        reference_diameter = create_quantity(reference_diameter, 'mm', scalar = True)
        diameter = self.roots.diameter.to(reference_diameter.units).magnitude[self.sort_order]
        matrix = self.matrix.to(unit).magnitude
        force_reference_before = np.diag(matrix) * (reference_diameter.magnitude / diameter)**self.load_sharing
        force_reference_after = force_reference_before + 1.0e-12 * np.max(force_reference_before)
        force_reference_all = np.append(0.0, np.stack((force_reference_before, force_reference_after)).ravel(order = 'F'))
        force_total_before = np.sum(matrix, axis = 0)
        force_total_after = force_total_before - np.diag(matrix)
        force_total_all = np.append(0.0, np.stack((force_total_before, force_total_after)).ravel(order = 'F'))
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        if stack is True:
            force_individual_before = matrix
            force_individual_after = matrix - np.diag(np.diag(matrix))
            force_individual_all = np.flip(
                np.concatenate(
                    (
                        np.zeros((matrix.shape[0], 1)),
                        np.stack((
                            force_individual_before, 
                            force_individual_after
                            ), axis = -1)
                            .reshape(matrix.shape[0], 2 * matrix.shape[1]),
                    ), 
                    axis = -1),
                axis = 0
                )
            prop_cycle = mpl.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            n_color = len(colors)
            colors_new = np.array(
                colors 
                * int(np.ceil(matrix.shape[0] / n_color))
                )[np.flip(self.sort_order)]
            ax.stackplot(force_reference_all, force_individual_all, colors = colors_new)
            if labels is True:
                labels = list(self.sort_order + 1)
                plot_labels = True
            elif isinstance(labels, list):
                if len(labels) == matrix.shape[0]:
                    labels = np.array(labels)[self.sort_order]
                    plot_labels = True
                else:
                    plot_labels = False
            else:
                plot_labels = False
            if plot_labels is True:
                labels_x = force_reference_before - label_margin * np.max(force_reference_before)
                labels_y = (force_total_before - 0.5 * np.diag(matrix)) * labels_x / force_reference_before
                for xi, yi, li in zip(labels_x, labels_y, labels):
                    ax.annotate(
                        li, xy = (xi, yi), 
                        ha = 'center', 
                        va = 'center', 
                        bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                        fontsize = 'small'
                        )
                    
        ax.plot(force_reference_all, force_total_all, c = 'black')

        matrix_sum0 = np.sum(matrix, axis = 0)
        i_peak = np.argmax(matrix_sum0)
        force_reference_peak = force_reference_before[i_peak]
        force_total_peak = matrix_sum0[i_peak]
        if peak is True:
            plt.scatter(force_reference_peak, force_total_peak, c = 'black')

        ax.set_xlabel(xlabel + " [" + unit + "]")
        ax.set_ylabel(ylabel + " [" + unit + "]")
        ax.set_xlim(round_range(force_reference_all, limits = [0.0, None])['limits'])
        ax.set_ylim(round_range(
            force_total_peak, 
            limits = [0.0, None]
            )['limits'])
        
        return(fig, ax)


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



#####################
### AXIAL PULLOUT ###
#####################

class AxialPullout():
    """Class for axial pull-out of roots

    Predict pull-out forces for bundles of roots. This class follows models
    developed by Waldron (1977), Waldron & Dakessian (1981) and the DRAM model
    developed by Meijer et al. (2022), in the sense that displacements and 
    forces are mobilised due to the interaction between root-soil interface 
    resistance and root stiffness.

    This model class can take a wide variety of different root behaviours, 
    and therefore incorporates the models by a wide variety of authors. for 
    example, this class can take into account:
    - root breakage.
    - root slippage.
    - elastic or elasto-plastic behaviour.
    - 'embedded' behaviour (e.g. roots remain fully surrounded by soil) or 
      'surface' behaviour, in which the root length in contact with the soil
      gradually reduces.
    - root survival functions, which determined whether roots break 'suddenly'
      or gradually as an 'average' root, as govered by the weibull shape 
      parameter. This is implemented by looking at the ratio between the force
      at failure and the currently mobilised force in a root, assuming no 
      breakage.

    These different types of behaviours are fully described in Yildiz & 
    Meijer's "Root Reinforcement: Measurement and Modelling".

    Attributes
    ----------
    roots : pyrootmemo.materials.MultipleRoots
        MultipleRoots object containing properties of all roots in bundle. This
        instance must contain the attributes
        - 'circumference'
          root circumference
        - 'xsection'
          root cross-sectional area
        - 'elastic_modulus
          root elastic or Young's modulus
        - 'tensile_strength' (when breakage = True)
          root tensile strength
        - 'length' (when slipping = True)
          root length, i.e. the distance between point of pulling and the root
          tip
        - 'length_surface' (when slipping = True and surface = True)
          root length already sticking out of the soil surface at start of 
          pull-out
        - 'yield_strength' (when elastoplastic = True)
          root yield strength, i.e. the stress level at the start of plastic
          behaviour
        - 'plastic_modulus' (when elastoplastic = True)
          root stiffness during plastic behaviour phase
        - 'unload_modulus' (when elastoplastic = True)
          root stiffness when unloading, for plastically behaving roots
    interface : pyrootmemo.materials.Interface
        Interface object containing properties of the root-soil interface. This
        instance must contain the attribute 'shear_strength'        
    surface : bool
        Flag indicating whether the root behaviour type is 'surface', i.e. the
        root gets gradually pulled out of a soil surface and the root length 
        that remains in contact with the surrounding soil decreases gradually.
        If False, the root behaviour is assumed to be 'embedded', i.e. the
        root remains surrounded by soil at all times despite increasing
        pull-out displacements.
    breakage : bool
        Flag indicating whether roots are allowed to break. If False, roots 
        will never break in tension
    slipping : bool
        Flag indicating whether roots are allowed to slip. If False, roots 
        will never show slipping behaviour
    elastoplastic : bool
        Flag indicating whether roots behave elasto-plastically, as implemented
        as a bi-linear stress-strain response. If False, roots are assumed to
        behave fully elastic according to the root elastic stiffness.
    weibull_shape : None | float | int
        Weibull shape for root survival function. If roots break suddenly, 
        weibull_shape = None. This corresponds with a weibull shape factor 
        equal to +infinity
    behaviour_types : np.ndarray
        A list with character strings indicating the different types of 
        behaviour each root could show (e.g. elastic, plastic, slipping,
        anchored etc.)
    displacement_limits : Quantity
        A two-dimensional array of pull-out displacement levels for each root
        (columns, axis 1) at which the behaviour type changes from one type 
        to the next (rows, axis 0)
    force_limits : Quantity
        A two-dimensional array of pull-out forces for each root
        (columns, axis 1) at which the behaviour type changes from one type 
        to the next (rows, axis 0)
    coefficients : list
        A list with polynomial cubic coefficients to each root descibing the 
        polynomial relationship between force (independent variable) and 
        displacement (depedent variable). Elements in the list are ordered from
        higher-order parameters (3rd order) to lower order (0th order). Each 
        of these elements is a two-dimensional array given the coefficient for
        each root (columns, axis 1) at each of the different behaviour types
        (rows, axis 0). Note that for some behaviour types this relationship
        may not be uniquely defined.
                
    Methods
    -------
    __init__(roots, interface, surface, breakage, slipping, elastoplastic, weibull_shape)
        Constructor
    calc_force(displacement)
        Calculate force in each root given the displacement
    calc_displacement_to_peak()
        Calculate displacement until breakage or the start of slippage, for 
        each root
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            surface: bool = False,
            breakage: bool = True,
            slipping: bool = True,
            elastoplastic: bool = False,
            weibull_shape: None | int | float = None
            ):
        """Initialise a Pullout model object

        Parameters
        ----------
        roots : pyrootmemo.materials.MultipleRoots
            MultipleRoots object containing properties of all roots in bundle. 
            This instance must contain the attributes
            - 'circumference'
                root circumference
            - 'xsection'
                root cross-sectional area
            - 'elastic_modulus
                root elastic or Young's modulus
            - 'tensile_strength' (when breakage = True)
                root tensile strength
            - 'length' (when slipping = True)
                root length, i.e. the distance between point of pulling and the 
                root tip
            - 'length_surface' (when slipping = True and surface = True)
                root length already sticking out of the soil surface at start of 
                pull-out
            - 'yield_strength' (when elastoplastic = True)
                root yield strength, i.e. the stress level at the start of plastic
                behaviour
            - 'plastic_modulus' (when elastoplastic = True)
                root stiffness during plastic behaviour phase
            - 'unload_modulus' (when elastoplastic = True)
                root stiffness when unloading, for plastically behaving roots
        interface : pyrootmemo.materials.Interface
            Interface object containing properties of the root-soil interface. 
            This instance must contain the attribute 'shear_strength'   
        surface : bool, optional
            Flag indicating whether the root behaviour type is 'surface' 
            (True), i.e. the root gets gradually pulled out of a soil surface 
            and the root length that remains in contact with the surrounding 
            soil decreases gradually. If False, the root behaviour is assumed 
            to be 'embedded', i.e. the root remains surrounded by soil at all 
            times despite increasing pull-out displacements. By default False.
        breakage : bool, optional
            Flag indicating whether roots are allowed to break (True). If 
            False, roots will never break in tension. By default True
        slipping : bool, optional
            Flag indicating whether roots are allowed to slip (True). If False, 
            roots will never show slipping behaviour. By default True.
        elastoplastic : bool, optional
            Flag indicating whether roots behave elasto-plastically, as 
            implemented as a bi-linear stress-strain response (True). If False, 
            roots are assumed to behave fully elastic according to the root 
            elastic stiffness. By default False.
        weibull_shape : None | float | int, optional
            Weibull shape for root survival function. If roots break suddenly, 
            weibull_shape = None. This corresponds with a weibull shape factor 
            equal to +infinity. By default None
        """
        roots_attributes_required = ['circumference', 'xsection', 'elastic_modulus']
        if surface is True:
            roots_attributes_required += ['length_surface']
        if breakage is True:
            roots_attributes_required += ['tensile_strength']
        if slipping is True:
            roots_attributes_required += ['length']
        if elastoplastic is True:
            roots_attributes_required += ['yield_strength', 'plastic_modulus']
        for i in roots_attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots must contain ' + str(i) + ' attribute')
        if surface is True:
            if elastoplastic is True:
                if not hasattr(roots, 'unload_modulus'):
                    roots.unload_modulus = roots.elastic_modulus
        self.roots = roots
        interface_attributes_required = ['shear_strength']
        for i in interface_attributes_required:
            if not hasattr(interface, i):
                raise AttributeError('interface must contain ' + str(i) + ' attribute')
        self.interface = interface
        self.surface = surface
        self.breakage = breakage
        self.slipping = slipping
        self.elastoplastic = elastoplastic
        if isinstance(weibull_shape, int) | isinstance(weibull_shape, float):
            if weibull_shape <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            elif np.isinf(weibull_shape):
                raise ValueError('weibull_shape must have finite value. Set to None for instant breakages')
        else:
            if weibull_shape is not None:
                raise TypeError('weibull_shape must be an int, float or None')
        self.weibull_shape = weibull_shape

        if np.isscalar(roots.xsection.magnitude):
            nroots = (1)
        else:
            nroots = roots.xsection.shape
        behaviour_types = np.array([
            'Not in tension',
            'Anchored, elastic',
            'Slipping, elastic',
            'Full pullout',      # (when behaviour is elastic)
            'Anchored, plastic',   
            'Slipping, plastic', # (stress above yield stress)
            'Slipping, plastic', # (stress below yield stress)
            'Full pullout'       # (when behaviour is plastic)
        ])
        coefficients = [
            np.zeros((len(behaviour_types), *nroots)) * units('mm/N^3'),
            np.zeros((len(behaviour_types), *nroots)) * units('mm/N^2'),
            np.zeros((len(behaviour_types), *nroots)) * units('mm/N'),
            np.zeros((len(behaviour_types), *nroots)) * units('mm')
            ]
        displacement_limits = np.zeros((len(behaviour_types) - 1, *nroots)) * units('mm')
        force_limits = np.zeros((len(behaviour_types) - 1, *nroots)) * units('N')
        
        if surface is True:
            ## SURFACE ROOTS
            # anchored, elastic [type 1]
            coefficients[0][1, ...] = (
                1.0 / (2.0 * (roots.elastic_modulus * roots.xsection)**2
                       * roots.circumference * interface.shear_strength)
                )
            coefficients[1][1, ...] = (
                1.0 / (2.0 * roots.elastic_modulus * roots.xsection
                       * roots.circumference * interface.shear_strength)
                )
            coefficients[2][1, ...] = roots.length_surface / (roots.elastic_modulus * roots.xsection)
            if slipping is True:
                # slipping, elastic [type 2]
                coefficients[1][2, ...] = (
                    -1.0 / (roots.elastic_modulus * roots.xsection
                            * roots.circumference * interface.shear_strength)
                    )
                coefficients[2][2, ...] = (
                     roots.length / (roots.elastic_modulus * roots.xsection)
                     - 1.0 / (roots.circumference * interface.shear_strength)
                     )
                coefficients[3][2, ...] = roots.length - roots.length_surface
                # displacement at start of slippage, elastic <limit 1>
                force_limits[1, :] = solve_quadratic(
                    1.0 / (2.0 * roots.elastic_modulus * roots.xsection * roots.circumference * interface.shear_strength),
                    1.0 / (roots.circumference * interface.shear_strength),
                    roots.length_surface - roots.length
                    )
                displacement_limits[1, :] = (
                    coefficients[0][1, ...] * force_limits[1, :]**3
                    + coefficients[1][1, ...] * force_limits[1, :]**2
                    + coefficients[2][1, ...] * force_limits[1, :]
                    + coefficients[3][1, ...]
                    )
                # displacement until full pullout, elastic <limit 2>
                displacement_limits[2, :] = roots.length - roots.length_surface
            if elastoplastic is True:
                # force and displacement at yield <limit 3>
                force_limits[3, :] = roots.xsection * roots.yield_strength
                displacement_limits[3, :] = (
                    coefficients[0][1, ...] * force_limits[3, :]**3
                    + coefficients[1][1, ...] * force_limits[3, :]**2
                    + coefficients[2][1, ...] * force_limits[3, :]
                    + coefficients[3][1, ...]
                    )
                # anchored, plastic [type 4]
                coefficients[0][4, ...] = (
                    1.0 / (2.0 * (roots.plastic_modulus * roots.xsection)**2 
                           * roots.circumference * interface.shear_strength)
                    )
                coefficients[1][4, ...] = (
                    1.0 
                    / (2.0 * roots.plastic_modulus * roots.xsection 
                       * roots.circumference * interface.shear_strength)
                    * (1.0
                       + 3.0 * roots.yield_strength / roots.elastic_modulus
                       - 3.0 * roots.yield_strength / roots.plastic_modulus) 
                    )
                coefficients[2][4, ...] = (
                    roots.yield_strength
                    / (2.0 * roots.elastic_modulus * roots.plastic_modulus
                        * roots.circumference * interface.shear_strength)
                    * (
                        roots.yield_strength 
                        * (3.0 * roots.elastic_modulus / roots.plastic_modulus
                            + 2.0 * roots.plastic_modulus / roots.elastic_modulus
                            - 5.0)
                        - 2.0 * roots.elastic_modulus
                        + 2.0 * roots.plastic_modulus
                    )
                    + roots.length_surface / (roots.plastic_modulus * roots.xsection)
                    )
                coefficients[3][4, ...] = (
                    roots.yield_strength 
                    * (roots.elastic_modulus - roots.plastic_modulus)
                    / (2.0 * roots.elastic_modulus * roots.plastic_modulus
                       * roots.circumference * interface.shear_strength)
                    * (
                        roots.yield_strength * roots.xsection
                        - roots.yield_strength**2 * roots.xsection 
                        * (1.0 / roots.plastic_modulus - 1.0 / roots.elastic_modulus)
                        - 2.0 * roots.circumference * interface.shear_strength * roots.length_surface
                        )
                    )
                if slipping is True:
                    # force and displacement at start of slipping, plastic <limit 4>
                    force_limits[4, :] = solve_quadratic(
                        1.0 / (2.0 * roots.plastic_modulus * roots.xsection 
                            * roots.circumference * interface.shear_strength),
                        (1.0 
                        / (roots.circumference * interface.shear_strength)
                        * (1.0
                            + force_limits[3, :] / (roots.elastic_modulus * roots.xsection)
                            - force_limits[3, :] / (roots.plastic_modulus * roots.xsection))),
                        (roots.length_surface
                        + (force_limits[3, :]**2 
                            / (2.0 * roots.xsection * roots.circumference * interface.shear_strength)
                            * (1.0 / roots.plastic_modulus - 1.0 / roots.elastic_modulus))
                        - roots.length)
                        )
                    displacement_limits[4, :] = (
                        coefficients[0][4, ...] * force_limits[4, :]**3
                        + coefficients[1][4, ...] * force_limits[4, :]**2
                        + coefficients[2][4, ...] * force_limits[4, :]
                        + coefficients[3][4, ...]
                    )
                    # slipping, plastic, above yield (type 5)
                    coefficients[1][5, ...] = (
                        -1.0
                        / (2.0 * roots.xsection * roots.circumference * interface.shear_strength)
                        * (1.0 / roots.plastic_modulus + 1.0 / roots.unload_modulus)
                        )
                    coefficients[2][5, ...] = (
                        roots.length / (roots.unload_modulus * roots.xsection)
                        - 1.0 
                        / (roots.circumference * interface.shear_strength)
                        * (
                            1.0
                            + roots.yield_strength / roots.elastic_modulus
                            - roots.yield_strength / roots.plastic_modulus
                            )
                        )
                    coefficients[3][5, ...] = (
                        roots.length 
                        - roots.length_surface
                        + roots.yield_strength * roots.length
                        * (1.0 / roots.elastic_modulus - 1.0 / roots.plastic_modulus)
                        + force_limits[4, :] / roots.xsection
                        * (roots.length - force_limits[4, :] / (2.0 * roots.circumference * interface.shear_strength))
                        * (1.0 / roots.plastic_modulus - 1.0 / roots.unload_modulus)
                        )
                    # force and displacement to yield during plastic unloading <limit 5>
                    force_limits[5, :] = force_limits[3, :]
                    displacement_limits[5, :] = (
                        coefficients[0][5, ...] * force_limits[5, :]**3
                        + coefficients[1][5, ...] * force_limits[5, :]**2
                        + coefficients[2][5, ...] * force_limits[5, :]
                        + coefficients[3][5, ...]
                    )                    
                    # slipping, plastic, below yield (type 6)
                    coefficients[1][6, ...] = (
                        -1.0
                        / (roots.elastic_modulus * roots.xsection
                        * roots.circumference * interface.shear_strength)
                        )
                    coefficients[2][6, ...] = (
                        roots.length / (roots.unload_modulus * roots.xsection)
                        - 1.0 / (roots.circumference * interface.shear_strength)
                        * (1.0 
                        - roots.yield_strength / roots.unload_modulus 
                        + roots.yield_strength / roots.elastic_modulus)
                        )
                    coefficients[3][6, ...] = (
                        roots.length 
                        - roots.length_surface
                        + 1.0 / (2.0 * roots.xsection * roots.circumference * interface.shear_strength)
                        * (
                            force_limits[4, :]**2 
                            * (1.0 / roots.unload_modulus - 1.0 / roots.plastic_modulus)
                            + (roots.yield_strength * roots.xsection)**2
                            * (1.0 / roots.unload_modulus + 1.0 / roots.plastic_modulus - 2.0 / roots.elastic_modulus)
                            )
                        - roots.length / roots.xsection
                        * (
                            force_limits[4, :]
                            * (1.0 / roots.unload_modulus - 1.0 / roots.plastic_modulus)
                            + (roots.yield_strength * roots.xsection)
                            * (1.0 / roots.plastic_modulus - 1.0 / roots.elastic_modulus)
                            )
                        )
                    # displacement until full pull-out, plastic <limit 6>
                    displacement_limits[6, :] = coefficients[3][6, ...]
                    # adjust limits: slippage before yielding --> never plasticity
                    slip_before_yield = (displacement_limits[1, ...] <= displacement_limits[3, ...])
                    displacement_limits[3:7, slip_before_yield] = np.inf * units('mm')
                    force_limits[3:7, slip_before_yield] = 0.0 * units('N')
                    # adjust limits: slippage after yielding --> never elastic slippage
                    yield_before_slip = ~slip_before_yield
                    displacement_limits[1:3, yield_before_slip] = displacement_limits[3, yield_before_slip]
                    force_limits[1:3, yield_before_slip] = force_limits[3, yield_before_slip]
        else:
            ## EMBEDDED ROOTS
            # anchored, elastic [type 1]
            coefficients[1][1, ...] = (
                1.0 / (2.0 * roots.elastic_modulus * roots.xsection 
                       * roots.circumference * interface.shear_strength)
                )
            if slipping is True:
                # slipping, elastic [type 2]
                force_limits[1, :] = roots.length * roots.circumference * interface.shear_strength
                # displacement at start of slippage, elastic <limit 1>
                displacement_limits[1, :] = coefficients[1][1, ...] * force_limits[1, :]**2
            if elastoplastic is True:
                # displacement at yield <limit 3>
                force_limits[3, :] = roots.xsection * roots.yield_strength
                displacement_limits[3, :] = coefficients[1][1, ...] * force_limits[3, :]**2
                # anchored, plastic [type 4]
                coefficients[1][4, ...] = (
                    1.0 / (2.0 * roots.plastic_modulus * roots.xsection 
                           * roots.circumference * interface.shear_strength)
                    )
                coefficients[2][4, ...] = (
                    roots.yield_strength 
                    / (roots.elastic_modulus * roots.circumference * interface.shear_strength)
                    - roots.yield_strength 
                    / (roots.plastic_modulus * roots.circumference * interface.shear_strength)
                    )
                coefficients[3][4, ...] = (
                    -roots.yield_strength**2 * roots.xsection 
                    / (2.0 * roots.elastic_modulus * roots.circumference * interface.shear_strength)
                    + roots.yield_strength**2 * roots.xsection 
                    / (2.0 * roots.plastic_modulus * roots.circumference * interface.shear_strength)
                    )
                if slipping is True:
                    # displacement at start of slippage, plastic <limit 4>
                    displacement_limits[4, :] = (
                        coefficients[1][4, ...] * force_limits[1, :]**2
                        + coefficients[2][4, ...] * force_limits[1, :]
                        + coefficients[3][4, ...]
                        )
                    force_limits[4, :] = force_limits[1, :]
                    # adjust limits: slippage before yielding --> never plasticity
                    slip_before_yield = displacement_limits[1, ...] <= displacement_limits[3, ...]
                    displacement_limits[2:7, slip_before_yield] = np.inf * units('mm')
                    force_limits[2:7, slip_before_yield] = force_limits[1, slip_before_yield]
                    # adjust limits: slippage after yielding --> never elastic slippage
                    yield_before_slip = ~slip_before_yield
                    displacement_limits[1:3, yield_before_slip] = displacement_limits[3, yield_before_slip]
                    force_limits[1:3, yield_before_slip] = force_limits[3, yield_before_slip]

        # for displacement limits that are not needed, add dummy values based on 'next' displacement limit
        mask = np.isclose(displacement_limits[-1, ...].magnitude, 0.0)
        displacement_limits[-1, mask] = np.inf * units('mm')
        for i in np.flip(np.arange(1, 6)):
            mask = np.isclose(displacement_limits[i, ...].magnitude, 0.0)
            displacement_limits[i, mask] = displacement_limits[i + 1, mask]
            force_limits[i, mask] = force_limits[i + 1, mask]

        self.coefficients = coefficients
        self.behaviour_types = behaviour_types
        self.displacement_limits = displacement_limits
        self.force_limits = force_limits


    def calc_force(
            self,
            displacement: Quantity,
            jacobian: bool = False
            ) -> dict:
        """Calculate force in each root, as function of given displacement

        Parameters
        ----------
        displacement : Quantity | Parameter(value: int | float | np.ndarray, unit: str)
            Displacement level. Must be a scalar, in which case the this
            is the applied displacement to each root. If inputted as an 
            array, this is the array of displacements applied to individual
            roots (must have same length as number of roots).
        jacobian : bool
            If True, also calculate and return the derivative of pull-out force(s) with
            respect to the applied pull-out displacement. By default False.

        Returns
        -------
        dict
            results dictionary with fields:
            - 'force': array with forces in each root
            - 'behaviour_index': array with the index of the behaviour type
              of each roots. see class attribute 'behaviour_type' for a full
              list of behaviour type names
            - 'survival_fraction': array with survival fraction for each root
            - 'dforce_ddisplacement': derivative of pullout forces with respect 
              to displacement. Only returned when 'jacobian = True'. 
        """
        displacement = create_quantity(displacement, check_unit = 'mm')
        nroots = self.roots.xsection.shape
        if not np.isscalar(displacement.magnitude):
            if not displacement.shape == nroots:
                raise ValueError('displacement must be a scalar or an array with seperate displacements for each individual root')
        behaviour_index = np.sum(displacement > self.displacement_limits, axis = 0).astype(int)
       
        force_unbroken = np.zeros(*nroots) * units('N')
        if jacobian is True:
            dforceunbroken_ddisplacement = np.zeros(*nroots) * units('N/mm')

        if self.surface is True:
            mask_el_anch = (behaviour_index == 1)
            if any(mask_el_anch):
                force_unbroken[mask_el_anch] = solve_cubic(
                    self.coefficients[0][1, mask_el_anch],
                    self.coefficients[1][1, mask_el_anch],
                    self.coefficients[2][1, mask_el_anch],
                    (self.coefficients[3][1, ...] - displacement)[mask_el_anch]
                    )
                if jacobian is True:
                    dforceunbroken_ddisplacement[mask_el_anch] = (1.0 / (
                        3.0 * self.coefficients[0][1, mask_el_anch] * force_unbroken[mask_el_anch]**2
                        + 2.0 * self.coefficients[1][1, mask_el_anch] * force_unbroken[mask_el_anch]
                        + self.coefficients[2][1, mask_el_anch]
                    ))
            if self.slipping is True:
                mask_el_slip = (behaviour_index == 2)
                if any(mask_el_slip):
                    force_unbroken[mask_el_slip] = solve_quadratic(
                        self.coefficients[1][2, mask_el_slip],
                        self.coefficients[2][2, mask_el_slip],
                        (self.coefficients[3][2, ...] - displacement)[mask_el_slip]
                    )
                    if jacobian is True:
                        dforceunbroken_ddisplacement[mask_el_slip] = (1.0 / (
                            2.0 * self.coefficients[1][2, mask_el_slip] * force_unbroken[mask_el_slip]
                            + self.coefficients[2][2, mask_el_slip]
                        ))
            if self.elastoplastic is True:
                mask_pl_anch = (behaviour_index == 4)
                if any(mask_pl_anch):
                    force_unbroken[mask_pl_anch] = solve_cubic(
                        self.coefficients[0][4, mask_pl_anch],
                        self.coefficients[1][4, mask_pl_anch],
                        self.coefficients[2][4, mask_pl_anch],
                        (self.coefficients[3][4, ...] - displacement)[mask_pl_anch]
                        )
                    if jacobian is True:
                        dforceunbroken_ddisplacement[mask_pl_anch] = (1.0 / (
                            3.0 * self.coefficients[0][4, mask_pl_anch] * force_unbroken[mask_pl_anch]**2
                            + 2.0 * self.coefficients[1][4, mask_pl_anch] * force_unbroken[mask_pl_anch]
                            + self.coefficients[2][4, mask_pl_anch]
                        ))
                if self.slipping is True:
                    mask_pl_slip_aboveyield = (behaviour_index == 5)
                    if any(mask_pl_slip_aboveyield):
                        force_unbroken[mask_pl_slip_aboveyield] = solve_quadratic(
                            self.coefficients[1][5, mask_pl_slip_aboveyield],
                            self.coefficients[2][5, mask_pl_slip_aboveyield],
                            (self.coefficients[3][5, ...] - displacement)[mask_pl_slip_aboveyield]
                            )
                        if jacobian is True:
                            dforceunbroken_ddisplacement[mask_pl_slip_aboveyield] = (1.0 / (
                                2.0 * self.coefficients[1][5, mask_pl_slip_aboveyield] * force_unbroken[mask_pl_slip_aboveyield]
                                + self.coefficients[2][5, mask_pl_slip_aboveyield]
                            ))
                    mask_pl_slip_belowyield = (behaviour_index == 6)
                    if any(mask_pl_slip_belowyield):
                        force_unbroken[mask_pl_slip_belowyield] = solve_quadratic(
                            self.coefficients[1][6, mask_pl_slip_belowyield],
                            self.coefficients[2][6, mask_pl_slip_belowyield],
                            (self.coefficients[3][6, ...] - displacement)[mask_pl_slip_belowyield]
                            )
                        if jacobian is True:
                            dforceunbroken_ddisplacement[mask_pl_slip_belowyield] = (1.0 / (
                                2.0 * self.coefficients[1][6, mask_pl_slip_belowyield] * force_unbroken[mask_pl_slip_belowyield]
                                + self.coefficients[2][6, mask_pl_slip_belowyield]
                            ))
            force_unbroken_cummax = force_unbroken.copy()
            mask_el_reducing = np.isin(behaviour_index, [2, 3])
            force_unbroken_cummax[mask_el_reducing] = self.force_limits[1, mask_el_reducing]
            mask_pl_reducing = np.isin(behaviour_index, [5, 6, 7])
            force_unbroken_cummax[mask_pl_reducing] = self.force_limits[4, mask_pl_reducing]
            if jacobian is True:
                dforceunbrokencummax_ddisplacement = dforceunbroken_ddisplacement.copy()
                dforceunbrokencummax_ddisplacement[mask_el_reducing] = 0.0 * units('N/mm')
                dforceunbrokencummax_ddisplacement[mask_pl_reducing] = 0.0 * units('N/mm')
        else:
            mask_el_anch = (behaviour_index == 1)
            if any(mask_el_anch):
                force_unbroken[mask_el_anch] = np.sqrt(
                    (displacement / self.coefficients[1][1, ...])[mask_el_anch]
                    )
                if jacobian is True:
                    dforceunbroken_ddisplacement[mask_el_anch] = (1.0 / (
                        2.0 * self.coefficients[1][1, mask_el_anch] * force_unbroken[mask_el_anch]
                        ))
            if self.slipping is True:
                mask_el_slip = (behaviour_index == 2)
                if any(mask_el_slip):
                    force_unbroken[mask_el_slip] = (
                        self.roots.length[mask_el_slip]
                        * self.roots.circumference[mask_el_slip]
                        * self.interface.shear_strength
                        )
            if self.elastoplastic is True:
                mask_pl_anch = (behaviour_index == 4)
                if any(mask_pl_anch):
                    force_unbroken[mask_pl_anch] = solve_quadratic(
                        self.coefficients[1][4, mask_pl_anch],
                        self.coefficients[2][4, mask_pl_anch],
                        (self.coefficients[3][4, ...] - displacement)[mask_pl_anch]
                        )
                    if jacobian is True:
                        dforceunbroken_ddisplacement[mask_pl_anch] = (1.0 / (
                            2.0 * self.coefficients[1][4, mask_pl_anch] * force_unbroken[mask_pl_anch]
                            + self.coefficients[2][4, mask_pl_anch]
                            ))
                if self.slipping is True:
                    mask_pl_slip = (behaviour_index == 5)
                    if any(mask_pl_slip):
                        force_unbroken[mask_pl_slip] = (
                            self.roots.length[mask_pl_slip]
                            * self.roots.circumference[mask_pl_slip]
                            * self.interface.shear_strength
                            )
            force_unbroken_cummax = force_unbroken.copy()
            if jacobian is True:
                dforceunbrokencummax_ddisplacement = dforceunbroken_ddisplacement.copy()

        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
            if self.weibull_shape is None:
                survival = (force_unbroken_cummax <= force_breakage).astype(float)
                if jacobian is True:
                    dsurvival_ddisplacement = np.zeros(*nroots) * units('1/mm')
            else:
                y = (gamma(1.0 + 1.0 / self.weibull_shape) * force_unbroken_cummax / force_breakage).magnitude                   
                survival = np.exp(-(y**self.weibull_shape))
                if jacobian is True:
                    dy_dforceunbrokencummax = gamma(1.0 + 1.0 / self.weibull_shape) / force_breakage
                    dsurvival_dy = -self.weibull_shape * y**(self.weibull_shape - 1.0) * survival
                    dsurvival_ddisplacement = dsurvival_dy * dy_dforceunbrokencummax * dforceunbrokencummax_ddisplacement
        else:
            survival = np.ones(*nroots).astype(float)
            if jacobian is True:
                dsurvival_ddisplacement = np.zeros(*nroots) * units('1/mm')         

        dict_return = {
            'force': force_unbroken * survival,
            'behaviour_index': behaviour_index,
            'survival_fraction': survival
        }
        if jacobian is True:
            dict_return['dforce_ddisplacement'] = (
                dforceunbroken_ddisplacement * survival
                + force_unbroken * dsurvival_ddisplacement
            )
        return(dict_return)
    

    def calc_displacement_to_peak(self) -> Quantity:
        """Calculate the displacement to peak, for in each root

        Calculates the displacement required for the each each in the 
        MultipleRoots object to reach the largest force it will even reach
        as function of *any* displacement level.

        The function does (currently) not take the survival function into 
        account, i.e. it looks at the *average* root for each root in the 
        MultipleRoots object. 

        Returns
        -------
        Quantity
            Displacement to peak for each root
        """
        if self.slipping is True:
            if self.elastoplastic is True:
                displacement_slipping = self.displacement_limits[4, ...]
                slip_before_yield = np.isinf(self.displacement_limits[4, ...].magnitude)
                displacement_slipping[slip_before_yield] = self.displacement_limits[1, slip_before_yield]
            else:
                displacement_slipping = self.displacement_limits[1, ...]
        else:
            displacement_slipping = np.full(self.roots.xsection.shape, np.inf) * units('mm')
        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
            if self.elastoplastic is True:
                behaviour_index = 4
            else:
                behaviour_index = 1
            if self.surface is True:
                displacement_breakage = (
                    self.coefficients[0][behaviour_index, ...] * force_breakage**3
                    + self.coefficients[1][behaviour_index, ...] * force_breakage**2
                    + self.coefficients[2][behaviour_index, ...] * force_breakage
                    + self.coefficients[3][behaviour_index, ...]
                )
            else:
                displacement_breakage = (
                    self.coefficients[1][behaviour_index, ...] * force_breakage**2
                    + self.coefficients[2][behaviour_index, ...] * force_breakage
                    + self.coefficients[3][behaviour_index, ...]
                )
        else:
            displacement_breakage = np.full(self.roots.xsection.shape, np.inf) * units('mm')
        return(np.minimum(displacement_slipping, displacement_breakage))


##########################################
### BASE CLASS FOR DIRECT SHEAR MODELS ###
##########################################

class DirectShearBase():
    """Base class for direct shear displacement-driven models
    
    Serves as a base clas for models in which reinforcement is mobilised
    as function of direct shear displacement, such as the different iterations
    of Waldron's models, or DRAM.

    An addition to the following attributes, also calculates some useful
    attributes to the input arguments:

    roots.orientation
        unit vector describing the initial orientation of each roots in 
        'roots'. These are defined in the local coordinate system of the 
        failure surface (x-axis in direction of shearing, z-axis normal to 
        shear plane):
        
        (local root vector)                    # defined in local coordinates
        = (failure surface coord system)^T     # defined in global coordinates
        * (global root vector)                 # defined in global coordinates

        Numpy array with size (number of roots, 3)

    Attributes
    ----------
    roots
        MultipleRoots object, containing properties for all individual roots.
        All roots are assumed to cross the failure surface.
    interface
        Interface object, containing root-soil interface properties
    soil_profile
        SoilProfile object
    failure_surface
        FailureSurface object
    length_distribution_factor
        Distribution factor for assigning root elongation to pullout 
        displacement

    Methods
    -------
    TODO: update methods
    __init__(roots, interface, soil_profile, failure_surface, length_distribution_factor, **kwargs)
        Constructor
    get_orientation_parameters(displacement, shear_zone_thickness, jac)
        Calculate root elongations in shear zone and k-factors 
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            length_distribution_factor: float | int = 0.5
            ):
        """Initialiser for direct shear models.

        Parameters
        ----------
        roots : MultipleRoots
            MultipleRoots object, containing root properties
        interface : Interface
            Interface object, containing properties of root--soil interface
        soil_profile : SoilProfile
            SoilProfile object
        failure_surface : FailureSurface
            FailureSurface object
        length_distribution_factor: float | int = 0.5: float | int, optional
            distribution factor determining how much of the root elongation in
            the shear zone to assign to each side, by default 0.5. 0.5 
            corresponds with symmetry, i.e. root segments on either side of
            the shear zone behave identically in terms of mobilising forces
        """
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be instance of class MultipleRoots')
        self.roots = roots
        if not isinstance(interface, Interface):
            raise TypeError('interface must be instance of class Interface')
        self.interface = interface
        if not isinstance(soil_profile, SoilProfile):
            raise TypeError('soil_profile must be instance of class SoilProfile')
        self.soil_profile = soil_profile
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be instance of class FailureSurface')
        self.failure_surface = failure_surface
        if not(
            isinstance(length_distribution_factor, int) 
            | isinstance(length_distribution_factor, float)
            ):
            raise TypeError('distribution factor must be int or float')
        self.length_distribution_factor = length_distribution_factor
        
        self.roots.orientation = np.einsum(
            'ji,...j->...i',
            failure_surface.calc_orientation(),
            roots.calc_orientation()
            )
    

    def calc_pullout_displacement(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jacobian: bool = False
            ) -> dict:
        """Calculate pullout displacement for every root

        Parameters
        ----------
        shear_displacement : Quantity
            (Current) shear displacement, as a scalar Quantity
        shear_zone_thickness : Quantity
            (Current) shear zone thickness, as a scalar Quantity
        jacobian : bool, optional
            If True, function also returns the partial derivatives of the 
            calculated pullout displacement with respect to both the shear
            zone displacement and the shear zone thickness. By default False

        Returns
        -------
        dict
            Results dictionary, with keys:
            * pullout_displacement: Quantity
              Array (size nroot) with pullout displacement for every root
            * dpullout_displacement_dshear_displacement: Quantity
              Array (size nroot). Only returned if jacobian = True
            * dpullout_displacement_dshear_zone_thickness: Quantity
              Array (size nroot). Only returned if jacobian = True
        """
        zeros = np.zeros_like(self.roots.diameter.magnitude, dtype = np.float64)
        ones = np.ones_like(self.roots.diameter.magnitude, dtype = np.float64)
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            dict_out = {'pullout_displacement': ones * self.length_distribution_factor * shear_displacement}
        else:
            length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            dict_out = {'pullout_displacement': self.length_distribution_factor * (length - length_initial)}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                dict_out['dpullout_displacement_dshear_displacement'] = self.length_distribution_factor * ones * units('mm/mm')
                dict_out['dpullout_displacement_dshear_zone_thickness'] = zeros * units('mm/mm')
            else:
                dict_out['dpullout_displacement_dshear_displacement'] = self.length_distribution_factor * length_x / length
                dict_out['dpullout_displacement_dshear_zone_thickness'] = self.length_distribution_factor * length / shear_zone_thickness
        return(dict_out)


    def calc_displaced_orientation(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jacobian: bool = False
            ) -> dict:
        """Calculate the orientation vector of each roots, given known shear 
        displacements and shear zone thicknesses.

        Parameters
        ----------
        shear_displacement : Quantity
            (Current) shear displacement, as a scalar Quantity
        shear_zone_thickness : Quantity
            (Current) shear zone thickness, as a scalar Quantity
        jacobian : bool, optional
            If True, function also returns the partial derivatives of the 
            calculated fractionswith respect to both the shear zone displacement
            and the shear zone thickness. By default False

        Returns
        -------
        dict
            Results dictionary, with keys:
            * orientation: Quantity
              Array (size nroot * 3) with force decomposition fractions for
              every root
            * dorientation_dshear_displacement: Quantity
              Array (size nroot * 3). Only returned if jacobian = True
            * dorientation_dshear_zone_thickness: Quantity
              Array (size nroot * 3). Only returned if jacobian = True
        """
        ones = np.ones_like(self.roots.diameter)
        zeros = np.zeros_like(self.roots.diameter)
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            dict_out = {'orientation': np.stack((ones, zeros, zeros), axis = -1)}
        else:
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x ** 2 + length_y ** 2 + length_z ** 2)
            dict_out = {'orientation': np.stack((
                            length_x / length, 
                            length_y / length, 
                            length_z / length
                            ), axis = -1)}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                dict_out['dorientation_dshear_zone_thickness'] = np.stack((
                    zeros / shear_zone_thickness.units,
                    zeros / shear_zone_thickness.units,
                    zeros / shear_zone_thickness.units
                    ), axis = -1)
                dict_out['dorientation_dshear_displacement'] = np.stack((
                    ones / shear_displacement.units,
                    zeros / shear_displacement.units,
                    zeros / shear_displacement.units
                    ), axis = -1)
            else:
                tmp1 = length_x * shear_displacement / (length**3 * shear_zone_thickness)
                dict_out['dorientation_dshear_zone_thickness'] = np.stack((
                    tmp1 * length_x - shear_displacement / (length * shear_zone_thickness),
                    tmp1 * length_y,
                    tmp1 * length_z
                    ), axis = -1)
                dict_out['dorientation_dshear_displacement'] = np.stack((
                    1.0 / length * (1.0 - length_x**2 / length**2),
                    -length_y * length_x / length**3,
                    -length_z * length_x / length**3
                    ), axis = -1)
        return(dict_out)


    def calc_shear_displacement_from_pullout(
            self,
            pullout_displacement: Quantity,
            shear_zone_thickness: Quantity
            ) -> Quantity:
        elongation = pullout_displacement / self.length_distribution_factor
        length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
        length = length_initial + elongation                        
        length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
        length_z = shear_zone_thickness
        length_x = np.sqrt(length**2 - length_y**2 - length_z**2)
        return(length_x - shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2])


    def calc_displacement_to_rootpeak(
            self,
            shear_zone_thickness: Quantity
            ) -> Quantity:
        """Calculate shear displacement at peak reinforcements of individual roots

        Calculate the shear displacement associated with each root reaching its
        maximum tensile force, i.e. at the point of breakage or at the onset
        of slippage. The associated pull-out displacement is calculated, and 
        this is then converter back to shear displacements

        Returns
        -------
        Quantity
            Array with shear displacements
        """
        pullout_displacement = self.pullout.calc_displacement_to_peak()
        elongation = pullout_displacement / self.length_distribution_factor
        if (shear_zone_thickness.magnitude <= 0.0):
            return(elongation)
        else:
            length_initial =  shear_zone_thickness / self.roots.orientation[..., 2]
            length_x0 = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2]
            length_y0 = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z0 = shear_zone_thickness
            length = length_initial + elongation
            length_x = np.sqrt(length**2 - length_y0**2 - length_z0**2)
            return(length_x - length_x0)


###########################
### WALDRON-TYPE MODELS ###
###########################

class Waldron(DirectShearBase):
    """Waldron model class

    Class for Waldron-type models, prediction soil reinforcement as function
    of direct shear displacements in the soil. 

    This class incorporates all versions of these type of models. For example, 
    it can allow for root breakage and/or slippage (Waldron and Dakessian, 
    1981), two- or three-dimensional initial root orientations (Grey, 
    Meijer et al. 2022), and elasto-plasticity (Meijer et al, 2022)
    
    This class inherits from the general direct shear model class:
    '_DirectShear'.

    Attributes
    ----------
    roots
        MultipleRoots object
    interface
        Interface object 
    soil_profile
        SoilProfile object
    failure_surface
        FailureSurface object
    slipping
        Boolean indicating whether slipping behaviour is included. Default is
        True
    breakage
        Boolean indicating whether breakage behaviour is included. Default is 
        True
    elastoplastic
        Boolean indicating whether roots behave elasto-plastically (True) or 
        linear elastic (False). Default is False
    weibull_shape
        Weibull shape parameter for root breakage (Weibull survival function).
        if None, roots break 'instantly' (i.e. shape parameter is infinite),
        Default is None

    Methods
    -------
    calc_k(shear_displacement, ...)
        calculate the WWM k-factor for each root at any given level of shear
        displacement
    calc_reinforcement(shear_displacement, ...)
        calculate reinforcement at given level(s) of shear displacement
    calc_peak_reinforcement()
        calculate peak reinforcement
    plot(...)
        show how reinforcement mobilises with shear displacement
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            breakage: bool = True,
            slipping: bool = True,
            elastoplastic: bool = False,
            weibull_shape: float | int | None = None
    ):
        """Initialise a Waldron model class object

        Parameters
        ----------
        roots : MultipleRoots
            MultipleRoots object, containing root properties
        interface : Interface
            Interface object, containing properties of root--soil interface
        soil_profile : SoilProfile
            SoilProfile object
        failure_surface : FailureSurface
            FailureSurface object
        slipping : bool, optional
            Include root slippage? By default True. This requires the 'length'
            attribute in 'roots'
        breakage : bool, optional
            Include root breakage? By default True. This requires the 
            'tensile_strength' attribute in 'roots'
        elastoplastic : bool, optional
            Include elasto-plastic root behaviour? By default False. This 
            requires the 'yield_strength' and 'plastic_modulus' attributes
            in roots
        weibull_shape : float | int | None, optional
            Weibull shape parameter for root modelling breakage (when 
            breakage = True). By default None. If 'None', all roots are 
            assumed to break instantly (like in the original Waldron-type 
            models).
        """
        super().__init__(roots, interface, soil_profile, failure_surface)
        if isinstance(weibull_shape, int) | isinstance(weibull_shape, float):
            if weibull_shape <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            elif np.isinf(weibull_shape):
                raise ValueError('weibull_shape must have finite value. Set to None for sudden breakages')
        else:
            if weibull_shape is not None:
                raise TypeError('weibull_shape must be an int, float or None')
        self.weibull_shape = weibull_shape
        self.slipping = slipping
        self.breakage = breakage
        self.elastoplastic = elastoplastic
        self.pullout = AxialPullout(
            roots, 
            interface,
            surface = False, 
            breakage = breakage, 
            slipping = slipping, 
            elastoplastic = elastoplastic, 
            weibull_shape = weibull_shape
            )


    def calc_k(
            self,
            shear_displacement: Quantity,
            jacobian: bool = False,
            ) -> dict:
        """Calculate WWM k-factor for each root, at given shear displacement

        Parameters
        ----------
        shear_displacement : Quantity
            Current shear displacement (scalar)
        jacobian : bool, optional
            If True, return the derivative of the k-factor with respect to 
            shear displacement

        Returns
        -------
        dict
            Dictionary with reinforcement results. Has keys:
            
            'k' : Quantity
                WWM k-factor for each root
            'dk_dshear_displacement' : Quantity
                Partial derivative of k with respect to shear displacement, for
                each root. Only returned when `jacobian = True`.
        """
        res_orientation = self.calc_displaced_orientation(
            shear_displacement,
            self.failure_surface.shear_zone_thickness,
            jacobian = jacobian
            )
        tanphi = np.tan(self.soil_profile.get_soil(self.failure_surface.depth).friction_angle)
        dict_out = {'k': res_orientation['orientation'][..., 0] +  res_orientation['orientation'][..., 2] * tanphi}
        if jacobian is True:
            dict_out['dk_dshear_displacement'] = (
                res_orientation['dorientation_dshear_displacement'][..., 0]
                + res_orientation['dorientation_dshear_displacement'][..., 2]
                * tanphi
                )
        return(dict_out)


    def calc_reinforcement(
            self,
            shear_displacement: Quantity | Parameter,
            total: bool = True,
            jacobian: bool = False,
            squeeze: bool = True,
            sign: int | float = 1.0
            ) -> dict:
        """
        Calculate root reinforcement at specified level(s) of shear 
        displacement.

        Parameters
        ----------
        shear_displacement : Quantity | Parameter
            soil shear displacement. Can be either a scalar or a vector of
            displacements 
        total : bool, optional
            if True, returns total reinforcement by all roots. If False, return
            reinforcement for each root seperately. By default True
        jacobian : bool, optional
            additionally return the derivative of reinforcement with respect 
            to shear displacement. By default False
        squeeze : bool, optional
            If `True`, strip all dimensions with length `1` out of the various
            results arrays. By default True
        sign : int, float, optional
            Multiplication factor for all result returned by the function. 
            This is used to be able to use minimisation algorithms in order
            to find the global maximum force, see function `self.peak_force()`. 
            Default = 1.0

        Returns
        -------
        dict
            Dictionary with reinforcement results. Has keys:

            'displacement' : Quantity
                Array with all shear displacement steps
            'reinforcement' : Quantity
                shear reinforcements. Has shape (`n*m`) where `n` is the number 
                of displacement steps and m the number of roots. If `total` is 
                True, `m = None`
            'behaviour_types' : np.ndarray
                list of root behaviour type names. 
            'behaviour_fraction' : np.ndarray
                fraction of total root cross-sectional area that behaves
                according to each of the types in 'behaviour_types'. Has shape 
                (`n*p*m`) where `n` is the number of dispalcement steps, `p` the 
                number of behaviour types, and `m` the number of roots. If 
                `total = True`, `m = None`
            'dreinforcement_ddisplacement': Quantity
                derivative of reinforcement output with respect to the shear 
                displacement. Only returned when `jacobian = True`. Has shape 
                (`n*m`) where `n` is the number of displacement stes and `m` the 
                number of roots. If total = True, `m = None`.

        """
        shear_displacement = create_quantity(shear_displacement, check_unit = 'mm')
        if np.isscalar(shear_displacement.magnitude):
            shear_displacement = np.array([shear_displacement.magnitude]) * shear_displacement.units
        ndisplacement = len(shear_displacement)
        nbehaviour = len(self.pullout.behaviour_types)
        nroots = len(self.roots.xsection)
        cr = np.zeros((ndisplacement, nroots)) * units('kPa')
        xsection_fractions = np.zeros((ndisplacement, nbehaviour, nroots))
        if jacobian is True:
            dcr_dus = np.zeros((ndisplacement, nroots)) * units('kPa/mm')
    
        for us, i in zip(shear_displacement, np.arange(ndisplacement)):
            res_up = self.calc_pullout_displacement(
                us,
                self.failure_surface.shear_zone_thickness,
                jacobian = jacobian
            )
            res_Tp = self.pullout.calc_force(
                res_up['pullout_displacement'], 
                jacobian = jacobian
                )
            res_k = self.calc_k(
                us,
                jacobian = jacobian
                )
            cr[i, ...] = sign * res_k['k'] * res_Tp['force'] / self.failure_surface.cross_sectional_area
            xsection_fractions[i, res_Tp['behaviour_index'], np.arange(nroots)] = (
                res_Tp['survival_fraction'] 
                * self.roots.xsection.magnitude 
                / np.sum(self.roots.xsection.magnitude)
                )
            if jacobian is True:
                dcr_dus[i, ...] = sign / self.failure_surface.cross_sectional_area * (
                    res_k['dk_dshear_displacement'] * res_Tp['force']
                     + res_k['k'] * res_Tp['dforce_ddisplacement'] 
                     * res_up['dpullout_displacement_dshear_displacement']
                    )

        dict_out = {
            'displacement': shear_displacement,
            'behaviour_types': self.pullout.behaviour_types            
            }
        if total is True:
            dict_out['reinforcement'] = cr.sum(axis = -1)
            dict_out['behaviour_fraction'] = xsection_fractions.sum(axis = -1)
        else:
            dict_out['reinforcement'] = cr
            dict_out['behaviour_fraction'] = xsection_fractions
        if jacobian is True:
            if total is True:
                dict_out['dreinforcement_ddisplacement'] = dcr_dus.sum(axis = -1)
            else:
                dict_out['dreinforcement_ddisplacement'] = dcr_dus
        if squeeze is True:
            dict_out['displacement'] = dict_out['displacement'].squeeze()
            dict_out['reinforcement'] = dict_out['reinforcement'].squeeze()
            dict_out['behaviour_fraction'] = dict_out['behaviour_fraction'].squeeze()
            if jacobian is True:
                dict_out['dreinforcement_ddisplacement'] = dict_out['dreinforcement_ddisplacement'].squeeze()
        return(dict_out)    


    def calc_peak_reinforcement(
            self, 
            factor: int | float = 1.15
            ) -> dict:
        """Calculate the magnitude and displacement at maximum root reinforcement

        Calculate the maximum root reinforcement and associated shear 
        displacement. 

        An estimation of the shear displacement domain is made using the 
        pull-out displacements at which each root reaches its maximum force,
        either at the point of breakage or at the onset of root slippage. These
        are then transformed to shear displacements using the function
        'calc_displacement_to_rootpeak'. 

        The maximum reinforcement is found by using scipy's evolutionary
        optimiser (scipy.optimise.differential_evolution) on the domain from
        zero to the largest value of shear displacement for any root peak.
        
        Parameters
        ----------
        factor : int | float, optional
            Multiplier for shear displacement that is searched by the 
            evolutionary solver (to make sure peak is within search domain),
            by default 1.15

        Returns
        -------
        dict
            Dictionary with peak reinforcement results. Has keys:
            
            'reinforcement' : Quantity
                maximum value of the root reinforcement at any shear 
                displacement
            'displacement' : Quantity
                the value of the shear displacement at which the peak 
                reinforcement is mobilised
        """
        shear_displacement_max = factor * np.max(self.calc_displacement_to_rootpeak(self.failure_surface.shear_zone_thickness))
        shear_displacement_units = shear_displacement_max.units
        def fun_to_optimize(x):
            return(self.calc_reinforcement(
                x * shear_displacement_units,
                jacobian = False,
                total = True,
                sign = -1.0
                )['reinforcement'].magnitude)
        sol = differential_evolution(
            fun_to_optimize,
            bounds = [(0.0, shear_displacement_max.magnitude)]
            )
        displacement_peak = sol.x[0] * shear_displacement_max.units
        return({
            'displacement': displacement_peak,
            'reinforcement': self.calc_reinforcement(displacement_peak, jacobian = False)['reinforcement']
            })


    def plot(
            self,
            ax = None,
            n: int = 251,
            stack = False,
            peak: bool = True,
            margin_axis: int | float = 0.20,
            labels = True,
            margin_label: int | float = 0.05,
            xlabel: str = 'Shear displacement',
            ylabel: str = 'Reinforcement',
            xunit: str = 'mm',
            yunit: str = 'kPa'            
            ):
        """Plot how reinforcements in Waldron model mobilise with displacements

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object to plot on. If not defined, plots on the axis
            in plot currently open. By default None
        n : int, optional
            number of displacement positions to plot, by default 251
        stack : bool, optional
            shows contributions of all individual roots by means of a 
            stackplot. By default False
        peak : bool, optional
            show the location of the peak using a scatter point. By default 
            True
        margin_axis : int | float, optional
            Add some extra displacement range so failure in roots nicely shows
            up in plot. Defined as a fraction of the chosen displacement range
            based on peak (function _get_displacement_root_peak()). By default
            0.20.
        labels : bool | list, optional
            labels to plot on contribution of each root, by default False.
            If False, no labels are plotted. If True, labels are plotted using
            the index of the root in the MultipleRoots object. Custom labels 
            can be inputted using a list, which must have the same length as 
            the number of roots in the bundle.
        margin_label : int | float, optional
            Fraction of plot width to offset plotting labels from moment
            of failure (breakage, slipping). By default 0.10.
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
        if self.breakage is False and self.slipping is False:
            shear_displacement_max = 100.0 * units('mm')
        else:
            shear_displacement_rootpeak = self.calc_displacement_to_rootpeak(self.failure_surface.shear_zone_thickness)
            shear_displacement_max = np.max(shear_displacement_rootpeak)
        shear_displacement = np.linspace(0.0 * shear_displacement_max, shear_displacement_max * (1.0 + margin_axis), n)
        results = self.calc_reinforcement(shear_displacement, jacobian = False, total = False)
        if self.roots.xsection.shape == (1, ):
            total_reinforcement_magnitude = results['reinforcement'].to(yunit).magnitude
        else:
            total_reinforcement_magnitude = np.sum(results['reinforcement'], axis = 1).to(yunit).magnitude
        
        if ax is None:
            ax = plt.gca()
        shear_displacement_magnitude = shear_displacement.to(xunit).magnitude
        ax.plot(
            shear_displacement_magnitude,
            total_reinforcement_magnitude,
            c = 'black'
            )

        if stack is True:
            reinforcement_perroot_magnitude = results['reinforcement'].to(yunit).magnitude
            ax.stackplot(shear_displacement_magnitude, reinforcement_perroot_magnitude.transpose())
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
                if (self.slipping is False) and (self.breakage is False):
                    labels_x = shear_displacement[int((1.0 - margin_label) * n)]
                    labels_y_tmp = self.calc_reinforcement(labels_x, total = False)['reinforcement'].to(yunit).magnitude
                    labels_y_magnitude = np.cumsum(labels_y_tmp, axis = 1) - 0.5 * labels_y_tmp
                    labels_x_magnitude = np.full(len(labels_y_magnitude), labels_x.to(xunit).magnitude)
                else:
                    labels_x_tmp = shear_displacement_rootpeak - margin_label * np.max(shear_displacement_rootpeak)
                    labels_y_tmp = self.calc_reinforcement(labels_x_tmp, total = False)['reinforcement'].to(yunit).magnitude
                    labels_x_magnitude = labels_x_tmp.to(xunit).magnitude
                    labels_y_tmp2 = np.tril(labels_y_tmp)
                    labels_y_magnitude = np.sum(labels_y_tmp2, axis = 1) - 0.5 * np.diag(labels_y_tmp2)
                for xi, yi, li in zip(labels_x_magnitude, labels_y_magnitude, labels):
                    ax.annotate(
                        li, 
                        xy = (xi, yi), 
                        ha = 'center', 
                        va = 'center', 
                        bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                        fontsize = 'small'
                        )

        if peak is True:
            if self.breakage is True or self.slipping is True:
                peak_results = self.calc_peak_reinforcement()
                ax.scatter(
                    peak_results['displacement'].to(xunit).magnitude,
                    peak_results['reinforcement'].to(yunit).magnitude,
                    c = 'black'
                    )

        ax.set_xlabel(xlabel + ' [' + str(xunit) + ']')
        ax.set_ylabel(ylabel + ' [' + str(yunit) + ']')
        return(ax)


###########################################
### Dundee Root Analytical Model (DRAM) ###
###########################################

class Dram(DirectShearBase):
    """Dram model class

    Class for the Dundee Root Analytical Model (DRAM), prediction soil 
    reinforcement as function of direct shear displacements in the soil, and 
    incorporating shear zone thickness increase during the test based on 
    satisfying the perfectly-plastic Mohr-Coulomb failure criterion on the 
    sheaer plane

    This class incorporates all versions of these type of models. For example, 
    it can allow for root breakage and/or slippage (Waldron and Dakessian, 
    1981), two- or three-dimensional initial root orientations (Grey, 
    Meijer et al. 2022), and elasto-plasticity (Meijer et al, 2022)
    
    This class inherits from the general direct shear model class:
    '_DirectShear'.

    Attributes
    ----------
    roots
        MultipleRoots object
    interface
        Interface object 
    soil_profile
        SoilProfile object
    failure_surface
        FailureSurface object
    slipping
        Boolean indicating whether slipping behaviour is included. Default is
        True
    breakage
        Boolean indicating whether breakage behaviour is included. Default is 
        True
    elastoplastic
        Boolean indicating whether roots behave elasto-plastically (True) or 
        linear elastic (False). Default is False
    weibull_shape
        Weibull shape parameter for root breakage (Weibull survival function).
        if None, roots break 'instantly' (i.e. shape parameter is infinite),
        Default is None

    Methods
    -------
    calc_reinforcement(shear_displacement, ...)
        calculate reinforcement at given level(s) of shear displacement
    calc_peak_reinforcement()
        calculate peak reinforcement
    plot(...)
        show how reinforcement mobilises with shear displacement
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            breakage: bool = True,
            slipping: bool = True,
            elastoplastic: bool = False,
            weibull_shape: float | int | None = None
            ):
        """Initialise a Waldron model class object

        Parameters
        ----------
        roots : MultipleRoots
            MultipleRoots object, containing root properties
        interface : Interface
            Interface object, containing properties of root--soil interface
        soil_profile : SoilProfile
            SoilProfile object
        failure_surface : FailureSurface
            FailureSurface object
        slipping : bool, optional
            Include root slippage? By default True. This requires the 'length'
            attribute in 'roots'
        breakage : bool, optional
            Include root breakage? By default True. This requires the 
            'tensile_strength' attribute in 'roots'
        elastoplastic : bool, optional
            Include elasto-plastic root behaviour? By default False. This 
            requires the 'yield_strength' and 'plastic_modulus' attributes
            in roots
        weibull_shape : float | int | None, optional
            Weibull shape parameter for root modelling breakage (when 
            breakage = True). By default None. If 'None', all roots are 
            assumed to break instantly (like in the original Waldron-type 
            models).
        """
        super().__init__(roots, interface, soil_profile, failure_surface)
        if isinstance(weibull_shape, int) | isinstance(weibull_shape, float):
            if weibull_shape <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            elif np.isinf(weibull_shape):
                raise ValueError('weibull_shape must have finite value. Set to None for sudden breakages')
        else:
            if weibull_shape is not None:
                raise TypeError('weibull_shape must be an int, float or None')
        self.weibull_shape = weibull_shape
        self.slipping = slipping
        self.breakage = breakage
        self.elastoplastic = elastoplastic
        self.pullout = AxialPullout(
            roots, 
            interface,
            surface = False, 
            breakage = breakage, 
            slipping = slipping, 
            elastoplastic = elastoplastic, 
            weibull_shape = weibull_shape
            )


    def calc_single_step(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            soil_shear_strength: Quantity,
            soil_friction_angle: Quantity,
            total: bool = True,
            jacobian: bool = False
            ):
        dict_ori = self.calc_displaced_orientation( 
            shear_displacement,
            shear_zone_thickness,
            jacobian = jacobian
            )
        dict_up = self.calc_pullout_displacement(
            shear_displacement,
            shear_zone_thickness,
            jacobian = jacobian
            )
        dict_Tp = self.pullout.calc_force(   # calculate tensile force
            dict_up['pullout_displacement'],
            jacobian = jacobian
            )
        root_force_x = dict_Tp['force'] * dict_ori['orientation'][:, 0] 
        root_force_z = dict_Tp['force'] * dict_ori['orientation'][:, 2] * np.tan(soil_friction_angle)
        dict_out = {'yield_value': np.sum(root_force_x - root_force_z) / self.failure_surface.cross_sectional_area - soil_shear_strength}
        if total is True:
            dict_out['reinforcement'] = np.sum(root_force_x + root_force_z) / self.failure_surface.cross_sectional_area
        else:
            dict_out['reinforcement'] = (root_force_x + root_force_z) / self.failure_surface.cross_sectional_area
        if jacobian is True:
            droot_load_dshear_zone_thickness = np.sum(
                dict_Tp['dforce_ddisplacement'] 
                * dict_up['dpullout_displacement_dshear_zone_thickness']
                * dict_ori['orientation'][:, 0]
                + dict_Tp['force']
                * dict_ori['dorientation_dshear_zone_thickness'][:, 0]
                ) / self.failure_surface.cross_sectional_area
            droot_resistance_dshear_zone_thickness = np.sum(
                dict_Tp['dforce_ddisplacement'] 
                * dict_up['dpullout_displacement_dshear_zone_thickness']
                * dict_ori['orientation'][:, 2]
                + dict_Tp['force']
                * dict_ori['dorientation_dshear_zone_thickness'][:, 2]
                ) * np.tan(soil_friction_angle) / self.failure_surface.cross_sectional_area
            dict_out['dyield_value_dshear_zone_thickness'] = droot_load_dshear_zone_thickness - droot_resistance_dshear_zone_thickness
        return(dict_out)

    
    def calc_reinforcement(
            self,
            max_shear_displacement: Parameter | Quantity | None = None,
            n: int = 251,
            algorithm: str = 'bracket',
            total: bool = True,
            initial_shear_displacement: None | Quantity = None,
            initial_shear_zone_thickness: None | Quantity = None
            ) -> dict:
        """
        Calculate shear reinforcement as function of shear displacement

        Iterate through a range of soil shear displacements, ranging from zero
        (default) to a specified maximum shear displacement. For each 
        displacement step, calculate the  root reinforcement, shear zone 
        thickness and behaviour of roots.

        Parameters
        ----------
        max_shear_displacement : Parameter | Quantity | None, optional
            Maximum shear displacement, by default None. If None, an automatic
            reasonable guess.
        n : int, optional
            Number of (equally-spaced) discrete displacement steps to use, 
            by default 251
        algorithm : str, optional
            root solve method used by the `scipy.optimize.root_solve()` function
            used to find the new shear zone thickness, in case of shear zone
            instability, by default 'bracket'
        total : bool, optional
            if True, returns total reinforcement by all roots. If False, return
            reinforcement for each root seperately. By default True
        initial_shear_displacement : None | Quantity, optional
            Initial shear displacement, by default None, in which case it is 
            assumed as 0.0. Can be used to start the iterative solving process
            at displacements other than zero, but should normally not be used. 
        initial_shear_zone_thickness : None | Quantity, optional
            Initial shear zone thickness, by default None, in which case it is 
            assumed from `self.failure_surface.shear_zone_thickness`. 
            Can be used to start the iterative solving process at displacements 
            other than zero if needed, but should normally not be used.

        Returns
        -------
        dict
            Dictionary with reinforcement results. Has keys:

            'displacement' : Quantity
                Array with all shear displacement steps
            'reinforcement' : Quantity
                shear reinforcements. Has shape (`n*m`) where `n` is the number 
                of displacement steps and m the number of roots. If `total` is 
                True, `m = None`
            'shear_zone_thickness' : Quantity
                shear zone thickness at each shear displacement step
            'behaviour_types' : np.ndarray
                list of root behaviour type names. 
            'behaviour_fraction' : np.ndarray
                fraction of total root cross-sectional area that behaves
                according to each of the types in 'behaviour_types'. Has shape 
                (`n*p*m`) where `n` is the number of dispalcement steps, `p` the 
                number of behaviour types, and `m` the number of roots. If 
                `total = True`, `m = None`
            'dreinforcement_ddisplacement': Quantity
                derivative of reinforcement output with respect to the shear 
                displacement. Only returned when `jacobian = True`. Has shape 
                (`n*m`) where `n` is the number of displacement stes and `m` the 
                number of roots. If total = True, `m = None`.
        """
        if is_namedtuple(max_shear_displacement):
            max_shear_displacement = max_shear_displacement.value * units(max_shear_displacement.unit)
        if initial_shear_displacement is None:
            initial_shear_displacement = 0.0 * max_shear_displacement.units
        else:
            initial_shear_displacement = initial_shear_displacement.to(max_shear_displacement.units)
        # initiate arrays for outputs
        shear_displacement = max_shear_displacement.units * np.linspace(
            initial_shear_displacement.magnitude,
            max_shear_displacement.magnitude,
            n)
        if initial_shear_zone_thickness is None:
            initial_shear_zone_thickness = self.failure_surface.shear_zone_thickness        
        shear_zone_thickness = np.full(n, initial_shear_zone_thickness.magnitude) * initial_shear_zone_thickness.units
        if total is True:
            reinforcement = np.zeros(n) * units('kPa')
        else:
            reinforcement = np.zeros((n, len(self.roots.xsection))) * units('kPa')
        # soil properties
        soil_shear_strength = self.soil_profile.calc_shear_strength(self.failure_surface.depth)
        soil_friction_angle = self.soil_profile.get_soil(self.failure_surface.depth).friction_angle
        # loop through all displacement steps
        for i in np.arange(n):
            if shear_displacement[i].magnitude > 0.0:
                # calculate results
                res = self.calc_single_step(
                    shear_displacement[i],
                    shear_zone_thickness[i - 1],
                    soil_shear_strength,
                    soil_friction_angle,
                    total = total,
                    jacobian = False
                    )
                if res['yield_value'].magnitude < 0.0:
                    # stable -> assign output
                    reinforcement[i, ...] = res['reinforcement']
                    shear_zone_thickness[i] = shear_zone_thickness[i - 1]
                else:
                    if np.isclose(shear_zone_thickness[i - 1], self.failure_surface.max_shear_zone_thickness):
                        # shear zone at max thickness
                        reinforcement[i, ...] = res['reinforcement']
                        shear_zone_thickness[i] = shear_zone_thickness[i - 1]
                    else:
                        # check if possible to get a stable shear plane at the maximum shear zone thickness
                        res_max = self.calc_single_step(
                            shear_displacement[i],
                            self.failure_surface.max_shear_zone_thickness,
                            soil_shear_strength,
                            soil_friction_angle,
                            total = total,
                            jacobian = False
                            )
                        if res_max['yield_value'].magnitude >= 0.0:
                            # unstable at max -> set shear zone to shear_zone_max
                            reinforcement[i, ...] = res_max['reinforcement']
                            shear_zone_thickness[i] = self.failure_surface.max_shear_zone_thickness
                        else:
                            # stable at max - iterate to find new shear zone thickness that makes yield_value zero
                            if algorithm == 'bracket':
                                sol = root_scalar(
                                    lambda x: self.calc_single_step(
                                        shear_displacement[i],
                                        x * units('mm'),
                                        soil_shear_strength,
                                        soil_friction_angle,
                                        total = False,
                                        jacobian = False
                                        )['yield_value'].magnitude,
                                    bracket = [
                                        shear_zone_thickness[i - 1].to('mm').magnitude,
                                        self.failure_surface.max_shear_zone_thickness.to('mm').magnitude
                                        ]
                                    )
                                shear_zone_thickness[i] = sol.root * units('mm')
                            elif algorithm == 'gradient':
                                def root_function(x):
                                    res = self.calc_single_step(
                                        shear_displacement[i],
                                        x * units('mm'),
                                        soil_shear_strength,
                                        soil_friction_angle,
                                        total = False,
                                        jacobian = True
                                        )
                                    return(
                                        res['yield_value'].magnitude,
                                        res['dyield_value_dshear_zone_thickness'].magnitude
                                        )
                                initial_guess = (
                                    2.0 * shear_zone_thickness[i - 1].to('mm').magnitude
                                    - shear_zone_thickness[i - 2].to('mm').magnitude
                                    )
                                sol = root_scalar(
                                    root_function,
                                    x0 = initial_guess,
                                    fprime = True                                
                                    )
                                shear_zone_thickness[i] = sol.root * units('mm')
                            res_solved = self.calc_single_step(
                                shear_displacement[i],
                                shear_zone_thickness[i],
                                soil_shear_strength,
                                soil_friction_angle,
                                total = total,
                                jacobian = False
                                )
                            reinforcement[i, ...] = res_solved['reinforcement']
        # return results dictionary
        return({
            'displacement': shear_displacement,
            'reinforcement': reinforcement,
            'shear_zone_thickness': shear_zone_thickness
            })
    
    
    def calc_peak_reinforcement(
            self,
            n = 51,
            passes = 3
            ):
        shear_displacement_max = max(self.calc_displacement_to_rootpeak(self.failure_surface.max_shear_zone_thickness))
        shear_displacement_min = 0.0 * shear_displacement_max.units
        shear_zone_thickness = self.failure_surface.shear_zone_thickness
        for i in np.arange(passes):
            res = self.calc_reinforcement(
                shear_displacement_max, 
                n = n,
                initial_shear_displacement = shear_displacement_min,
                initial_shear_zone_thickness = shear_zone_thickness
                )
            index_peak = np.argmax(res['reinforcement'])
            if index_peak == 0:
                index_previous = 0
                index_next = 1
            elif index_peak == (n - 1):
                index_previous = index_peak - 1
                index_next = index_peak
            else:
                index_previous = index_peak - 1
                index_next = index_peak + 1
            shear_displacement_min = res['displacement'][index_previous]
            shear_displacement_max = res['displacement'][index_next]
            shear_zone_thickness = res['shear_zone_thickness'][index_previous]
        return({
            'displacement': res['displacement'][index_peak],
            'reinforcement': res['reinforcement'][index_peak]
            })
    

    def plot(
            self,
            ax = None,
            n: int = 251,
            stack = False,
            peak: bool = True,
            margin_axis: int | float = 0.10,
            labels = True,
            margin_label: int | float = 0.05,
            xlabel: str = 'Shear displacement',
            ylabel: str = 'Reinforcement',
            xunit: str = 'mm',
            yunit: str = 'kPa'            
            ):
        """Plot how forces in the Waldron model mobilise with displacements

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
        margin_axis : int | float, optional
            Add some extra displacement range so failure in roots nicely shows
            up in plot. Defined as a fraction of the chosen displacement range
            based on peak (function _get_displacement_root_peak()). By default
            0.10.
        labels : bool | list, optional
            labels to plot on contribution of each root, by default False.
            If False, no labels are plotted. If True, labels are plotted using
            the index of the root in the MultipleRoots object. Custom labels 
            can be inputted using a list, which must have the same length as 
            the number of roots in the bundle.
        margin_label : int | float, optional
            Fraction of plot width to offset plotting labels from moment
            of failure (breakage, slipping). By default 0.10.
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
        if self.breakage is False and self.slipping is False:
            shear_displacement_max = 100.0 * units('mm')
        else:
            shear_displacement_rootpeak = self.calc_displacement_to_rootpeak(self.failure_surface.max_shear_zone_thickness)
            shear_displacement_max = np.max(shear_displacement_rootpeak) * (1.0 + margin_axis)
        results = self.calc_reinforcement(shear_displacement_max, n = n, total = False)
        if self.roots.xsection.shape == (1, ):
            total_reinforcement_magnitude = results['reinforcement'].to(yunit).magnitude
        else:
            total_reinforcement_magnitude = np.sum(results['reinforcement'], axis = 1).to(yunit).magnitude
        
        if ax is None:
            ax = plt.gca()
        shear_displacement = results['displacement']
        shear_displacement_magnitude = shear_displacement.to(xunit).magnitude
        ax.plot(
            shear_displacement_magnitude,
            total_reinforcement_magnitude,
            c = 'black'
            )

        if stack is True:
            reinforcement_perroot_magnitude = results['reinforcement'].to(yunit).magnitude
            ax.stackplot(shear_displacement_magnitude, reinforcement_perroot_magnitude.transpose())
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
                if (self.slipping is False) and (self.breakage is False):
                    labels_x = shear_displacement[int((1.0 - margin_label) * n)]
                    labels_y_tmp = self.calc_reinforcement(labels_x, total = False)['reinforcement'].to(yunit).magnitude
                    labels_y_magnitude = np.cumsum(labels_y_tmp, axis = 1) - 0.5 * labels_y_tmp
                    labels_x_magnitude = np.full(len(labels_y_magnitude), labels_x.to(xunit).magnitude)
                else:
                    labels_x_tmp = shear_displacement_rootpeak - margin_label * np.max(shear_displacement_rootpeak)
                    labels_y_tmp = self.calc_reinforcement(labels_x_tmp, total = False)['reinforcement'].to(yunit).magnitude
                    labels_x_magnitude = labels_x_tmp.to(xunit).magnitude
                    labels_y_tmp2 = np.tril(labels_y_tmp)
                    labels_y_magnitude = np.sum(labels_y_tmp2, axis = 1) - 0.5 * np.diag(labels_y_tmp2)
                for xi, yi, li in zip(labels_x_magnitude, labels_y_magnitude, labels):
                    ax.annotate(
                        li, 
                        xy = (xi, yi), 
                        ha = 'center', 
                        va = 'center', 
                        bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                        fontsize = 'small'
                        )

        if peak is True:
            if self.breakage is True or self.slipping is True:
                peak_results = self.calc_peak_reinforcement(n = n, passes = 3)
                ax.scatter(
                    peak_results['displacement'].to(xunit).magnitude,
                    peak_results['reinforcement'].to(yunit).magnitude,
                    c = 'black'
                    )

        ax.set_xlabel(xlabel + ' [' + str(xunit) + ']')
        ax.set_ylabel(ylabel + ' [' + str(yunit) + ']')
        return(ax)