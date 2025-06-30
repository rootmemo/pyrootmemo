import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma
from scipy.optimize import minimize, differential_evolution
from pyrootmemo.helpers import units, Parameter, create_quantity, solve_quadratic, solve_cubic
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.tools.utils_rotation import axisangle_rotate
from pyrootmemo.tools.utils_plot import round_range
from pint import Quantity
import warnings


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
    peak_force()
        Calculate peak force in the bundle
    peak_reinforcement(failure_surface, k)
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

    def peak_force(self) -> np.ndarray:
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

    def peak_reinforcement(
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
        return(k * self.peak_force() / failure_surface.cross_sectional_area)


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
            unit: str = 'N',
            reference_diameter: Quantity | Parameter = (1.0, 'mm'),
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
        unit : str, optional
            unit used for plotting forces, by default 'N'
        reference_diameter : Quantity | Parameter(float | int, str), optional
            value of the reference diameter, by default 1.0 mm
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
        peak_results = self.peak_force()
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
        force = self.force(displacement, total = True)
        
        fig, ax = plt.subplots()
        if stack is True:
            ax.stackplot(
                displacement.to(xunit).magnitude, 
                self.force(displacement, total = False).to(yunit).magnitude
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
            labels_y_dimensional_all = self.force(labels_x_dimensional, total = False)
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

class _DirectShear():
    """Base class for direct shear displacement-driven models
    
    Serves as a base clas for models in which reinforcement is mobilised
    as function of direct shear displacement, such as the different iterations
    of Waldron's models, or DRAM.

    An addition to the following attributes, also sets some useful attributes
    to the input arguments:

    roots.orientation
        unit vector describing the initial orientation of each roots in 
        'roots'. numpy array with size (number of roots, 3)
    failure_surface.tanphi
        the value of tan(friction angle) for the soil that is present at the
        failure surface

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
    distribution_factor
        Distribution factor for assigning root elongation to pullout 
        displacement

    Methods
    -------
    TODO: update methods
    __init__(roots, interface, soil_profile, failure_surface, distribution_factor, **kwargs)
        Constructor
    get_initial_root_orientations()
        Defined initial orientations of all roots relative to the shear zone
    get_orientation_parameters(displacement, shear_zone_thickness, jac)
        Calculate root elongations in shear zone and k-factors 
    """

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            distribution_factor: float | int = 0.5
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
        distribution_factor : float | int, optional
            distribution factor determining how much of the root elongation in
            the shear zone to assign to each side, by default 0.5. 0.5 
            corresponds with symmetry, i.e. root segments on either side of
            the shear zone behave identically in terms of mobilising forces

        Raises
        ------
        TypeError
            _description_
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
        if not(isinstance(distribution_factor, int) | isinstance(distribution_factor, float)):
            raise TypeError('distribution factor must be int or float')
        self.distribution_factor = distribution_factor
        self.roots.orientation = self.calc_initial_root_orientations()
        self.failure_surface.tanphi = np.tan(
            soil_profile
            .get_soil(failure_surface.depth)
            .friction_angle
            .to('rad')
            )
    
    def calc_initial_root_orientations(
            self
            ) -> np.ndarray:
        """Calculate initial root orientations relative to the shear direction

        Orientations are defined as 3-dimensional orientations **relative to** 
        the failure surface so that:

        * local x = direction of shearing
        * local y = perpendicular to x on shear plane
        * local z = pointing downwards into the soil
        
        Orientations are defined in terms of 3-dimensional unit vectors.
                
        The object 'roots' may contain some information about the **global**
        initial orientation of the roots. This is defined in a **global** 
        right-handed Cartesian coordinate system, with z-axis pointing down 
        into the ground. Orientations are assumed to be defined in a spherical
        coordinate system where:
        
        * azimuth angle = angle from x-axis to projection of root vector on 
          the x-y plane
        * elevation angle = angle from z-axis to root vector
        
        If the initial root orientations are not defined, it is assume they 
        are all **perpedicular** to the shear zone.

        The direction of failure failure surface (taken from FailureSurface 
        object) is assumed to be defined as the angle of the surface in x-z 
        plane, defined (positive) from x to z, i.e. the 'dip angle'.

        Returns
        -------
        np.ndarray
            Numpy array with size (nroots, 3) with the relative 3-D root 
            orientations defined as unit vectors
        """
        # shape of root vector ('number of roots')
        roots_shape = self.roots.diameter.magnitude.shape
        # root orientations not defined - assume all perpendicular to failure surface
        if (not hasattr(self.roots, 'azimuth_angle')) & (not hasattr(self.roots, 'elevation_angle')):
            return(np.stack((
                np.zeros(*roots_shape),
                np.zeros(*roots_shape),
                np.ones(*roots_shape)                    
                ), axis = -1))
        # (partial) angles provided -> rotate to local coordinate system
        else:
            if not hasattr(self.roots, 'azimuth_angle'):
                self.roots.azimuth_angle = np.zeros(*roots_shape) * units('deg')
            if not hasattr(self.roots, 'elevation_angle'):
                self.roots.elevation_angle = np.zeros(*roots_shape) * units('deg')
            # get global root orientations
            root_orientation_global = np.stack((
                np.cos(self.roots.azimuth_angle.magnitude) 
                * np.sin(self.roots.elevation_angle.magnitude),
                np.sin(self.roots.azimuth_angle.magnitude) 
                * np.sin(self.roots.elevation_angle.magnitude),
                np.cos(self.roots.elevation_angle.magnitude)
            ), axis = -1)
            # rotate to local coordinate system and set unit vectors
            if hasattr(self.failure_surface, 'orientation'):
                axisangle = np.array([0.0, -self.failure_surface.orientation.to('rad'), 0.0])
            else:
                axisangle = np.array([0.0, 0.0, 0.0])
            # rotate and return
            return(axisangle_rotate(root_orientation_global, axisangle))


    def calc_pullout_displacement(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5,
            jacobian: bool = False
            ) -> dict:
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            ones = np.ones(*self.roots.xsection.shape)
            dict_out = {'pullout_displacement': distribution_factor * shear_displacement * ones}
        else:
            length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            dict_out = {'pullout_displacement': distribution_factor * (length - length_initial)}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                dict_out['dpullout_displacement_dshear_displacement'] = distribution_factor * ones * units('mm/mm')
                dict_out['dpullout_displacement_dshear_zone_thickness'] = 0.0 * ones * units('mm/mm')
            else:
                dict_out['dpullout_displacement_dshear_displacement'] = distribution_factor * length_x / length
                dict_out['dpullout_displacement_dshear_zone_thickness'] = distribution_factor * length / shear_zone_thickness
        return(dict_out)


    def calc_orientation_factor(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jacobian: bool = False
            ) -> dict:
        if np.isclose(shear_zone_thickness.magnitude, 0.0):
            ones = np.ones(*self.roots.xsection.shape)
            dict_out = {'k': ones}
        else:
            length_x = shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2] + shear_displacement
            length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            length_z = shear_zone_thickness
            length = np.sqrt(length_x**2 + length_y**2 + length_z**2)
            dict_out = {'k': (length_x + length_z * self.failure_surface.tanphi) / length}
        if jacobian is True:
            if np.isclose(shear_zone_thickness.magnitude, 0.0):
                dict_out['dk_dshear_displacement'] = 0.0 * ones / shear_displacement.units
                if np.isclose(shear_displacement.magnitude, 0.0):
                    dict_out['dk_dshear_zone_thickness'] = 0.0 * ones / shear_zone_thickness.units
                else:
                    dict_out['dk_dshear_zone_thickness'] = -np.inf * ones / shear_zone_thickness.units
            else:
                dict_out['dk_dshear_displacement'] = 1.0 / length - dict_out['k'] * length_x / length**2
                dict_out['dk_dshear_zone_thickness'] = -shear_displacement / (shear_zone_thickness * length)
        return(dict_out)


    def calc_shear_from_pullout_displacement(
            self,
            pullout_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5
            ) -> Quantity:
        elongation = pullout_displacement / distribution_factor
        length_initial = shear_zone_thickness / self.roots.orientation[..., 2]
        length = length_initial + elongation                        
        length_y = shear_zone_thickness * self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
        length_z = shear_zone_thickness
        length_x = np.sqrt(length**2 - length_y**2 - length_z**2)
        return(length_x - shear_zone_thickness * self.roots.orientation[..., 0] / self.roots.orientation[..., 2])


    def calc_orientation_parameters(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            distribution_factor: int | float = 0.5,
            jac: bool = False
            ) -> dict:
        """Calculate root pullout displacement and k-factor

        Calculates the pull-out displacement and the WWM orientation factor k
        for each root. 

        The pull-out displacement is defined as the axial movement of a 
        (segment of) root on one side of the shear zone. 

        The WWM orientation factor k is defined as the ratio between the amount 
        of root reinforcement each root generates (in terms of force) and the 
        current tensile force in that root.

        This function requires the root orientation - relative to the shear
        direction - to be known (in terms of a unit vector).
        
        Parameters
        ----------
        shear_displacement : Quantity
            Current level of shear displacement (scalar)
        shear_zone_thickness : Quantity
            shear zone thickness (scalar)
        distribution_factor : int | float, optional
            assumed ratio between root pull-out displacement and root 
            elongation within the shear zone, by default 0.5. When 0.5 means
            root segments on either side of the shear zone pull out by the
            same amount
        jac : bool, optional
            If True, return derivatives of pull-out displacement and k-factors
            with respect to shear displacement and shear zone thickness. By 
            default False

        Returns
        -------
        dict
            dictionary with fields:

            * 'pullout_displacement': level of pull-out displacement for each 
              root
            * 'k': WWM orientation factor for each root
            * 'dup_dus': derivative of pull-out displacement with respect to
              the shear displacement. Only returned when jac = True.
            * 'dup_dh': derivative of pull-out displacement with respect to
              the shear zone thickness. Only returned when jac = True. 
            * 'dk_dus': derivative of orientation factor k with respect to
              the shear displacement. Only returned when jac = True.
            * 'dk_dh': derivative of orientation factor k with respect to
              the shear zone thickness. Only returned when jac = True.

        """
        init_vector_x = (
            shear_zone_thickness
            * self.roots.orientation[..., 0]
            / self.roots.orientation[..., 2]
            )
        init_vector_y = (
            shear_zone_thickness
            * self.roots.orientation[..., 1]
            / self.roots.orientation[..., 2]
        )
        init_vector_z = shear_zone_thickness * np.ones_like(init_vector_x)
        if shear_zone_thickness.magnitude >= 0.0:
            init_length = shear_zone_thickness / self.roots.orientation[..., 2]
            displaced_length = np.sqrt(
                (init_vector_x + shear_displacement)**2 
                + init_vector_y**2 
                + init_vector_z**2
                )
        else:
            init_length = 0.0 * shear_zone_thickness * self.roots.orientation[..., 2]
            displaced_length = shear_displacement * np.ones_like(init_vector_x)
        pullout_displacement = distribution_factor * (displaced_length - init_length)
        k = (
            (init_vector_x + shear_displacement) 
            + (init_vector_z * self.failure_surface.tanphi)
            ) / displaced_length
        if jac is False:
            return({
                'pullout_displacement': pullout_displacement,
                'k': k
                })
        else:
            # calculate derivatives with respect to:
            # * shear displacement: us
            # * shear zone thickness: h
            divx_dh = self.roots.orientation[..., 0] / self.roots.orientation[..., 2]
            divy_dh = self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            divz_dh = np.ones_like(init_vector_z)
            if shear_zone_thickness.magnitude >= 0.0:
                dL0_dh = 1.0 / self.roots.orientation[..., 2]
                dL_dus = (init_vector_x + shear_displacement) / displaced_length
                dL_dv0x = (init_vector_x + shear_displacement) / displaced_length
                dL_dv0y = init_vector_y / displaced_length
                dL_dv0z = init_vector_z / displaced_length
            else:
                dL0_dh = 0.0 * self.roots.orientation[..., 2]
                dL_dus = np.ones_like(init_vector_z)
                dL_dv0x = np.ones_like(init_vector_z)
                dL_dv0y = np.ones_like(init_vector_z)
                dL_dv0z = np.ones_like(init_vector_z)
            dup_dus = distribution_factor * dL_dus
            dL_dh = (
                dL_dv0x * divx_dh
                + dL_dv0y * divy_dh
                + dL_dv0z * divz_dh
                )
            dup_dh = distribution_factor * (dL_dh - dL0_dh)
            dk_dus = (
                1.0 / displaced_length
                - k / displaced_length * dL_dus
            )
            dk_dh = (
                (divx_dh + divz_dh * self.failure_surface.tanphi) 
                / displaced_length
                - k / displaced_length * dL_dh
            )
            return({
                'pullout_displacement': pullout_displacement,
                'k': k,
                'dup_dus': dup_dus,
                'dup_dh': dup_dh,
                'dk_dus': dk_dus,
                'dk_dup': dk_dh
                })


###########################
### WALDRON-TYPE MODELS ###
###########################

# Waldron class
class Waldron(_DirectShear):
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
    distribution_factor
        Distribution factor for assigning root elongation to pullout 
        displacement
    slipping
        Boolean indicating whether slipping behaviour is included
    breakage
        Boolean indicating whether breakage behaviour is included
    elastoplastic
        Boolean indicating whether roots behave elasto-plastically (True) or 
        linear elastic (False)
    weibull_shape
        Weibull shape parameter for root breakage (Weibull survival function).
        if None, roots break 'instantly' (i.e. shape parameter is infinite)

    Methods
    -------
    reinforcement(shear_displacement, ...)
        calculate reinforcement at given level(s) of shear displacement
    peak_reinforcement()
        calculate peak reinforcement
    plot(...)
        show how reinforcement mobilises with shear displacement
    """

    # initiate model
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

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        TypeError
            _description_
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
        
    # calculate root reinforcement at given displacement step
    def calc_reinforcement(
            self,
            shear_displacement: Quantity | Parameter,
            total: bool = True,
            jacobian: bool = False,
            squeeze: bool = True,
            sign: int | float = 1.0
            ) -> dict:
        """Calculate root reinforcement at current level of displacement

        Parameters
        ----------
        shear_displacement : Quantity | Parameter
            soil shear displacement. If not set as a Quantity (with dimension
            length), TODO: UPDATE TEXT HERE
        total : bool, optional
            if True, returns total reinforcement by all roots. If False, return
            reinforcement for each root seperately.
        jacobian : bool, optional
            additionally return the derivative of reinforcement with respect 
            to shear displacement. By default False
        squeeze : bool, optional
            If True, strip all dimensions with length '1' out of the various
            results arrays. By default True
        sign : int, float, optional
            Multiplication factor for all result returned by the function. 
            This is used to be able to use minimisation algorithms in order
            to find the global maximum force, see function self.peak_force(). 
            Default = 1.0

        Returns
        -------
        dict
            Dictionary with reinforcement results. Has keys:
            
            'reinforcement' : Quantity
                shear reinforcements. Has shape (n*m) where n is the number of displacement steps
                and m the number of roots. If total is True, m = 1
            'behaviour_types' : np.ndarray
                list of root behaviour type names. 
            'behaviour_fraction' : np.ndarray
                fraction of total root cross-sectional area that behaves
                according to each of the types in 'behaviour_types'. Has shape (n*p*m) where
                n is the number of dispalcement steps, p the number of behaviour types, and 
                m the number of roots. If total = True, m = 1
            'dreinforcement_ddisplacement': Quantity
                derivative of reinforcement output with respect to the shear 
                displacement. Only returned when jacobian = True. Has shape (n*m) where
                n is the number of displacement stes and m the number of roots. If total
                = True, m = 1.

        """
        shear_displacement = create_quantity(shear_displacement, check_unit = 'mm')
        if np.isscalar(shear_displacement.magnitude):
            shear_displacement = np.array([shear_displacement.magnitude]) * shear_displacement.units
        ndisplacement = len(shear_displacement)
        nbehaviour = len(self.pullout.behaviour_types)
        nroots = self.roots.xsection.shape
        cr = np.zeros((ndisplacement, *nroots)) * units('kPa')
        xsection_fractions = np.zeros((ndisplacement, nbehaviour, *nroots))
        if jacobian is True:
            dcr_dus = np.zeros((ndisplacement, *nroots)) * units('kPa/mm')
    
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
            res_k = self.calc_orientation_factor(
                us,
                self.failure_surface.shear_zone_thickness,
                jacobian = jacobian
                )
            cr[i, ...] = sign * res_k['k'] * res_Tp['force'] / self.failure_surface.cross_sectional_area
            xsection_fractions[i, ...] = np.bincount(
                res_Tp['behaviour_index'], 
                weights = (
                    res_Tp['survival_fraction'] 
                    * self.roots.xsection.magnitude 
                    / np.sum(self.roots.xsection.magnitude)
                    ),
                minlength = nbehaviour
                )
            if jacobian is True:
                dcr_dus[i, ...] = sign / self.failure_surface.cross_sectional_area * (
                    res_k['dk_dshear_displacement'] * res_Tp['force']
                     + res_k['k'] * res_Tp['dforce_ddisplacement'] * res_up['dpullout_displacement_dshear_displacement']
                    )

        dict_out = {'behaviour_types': self.pullout.behaviour_types}
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
            dict_out['reinforcement'] = dict_out['reinforcement'].squeeze()
            dict_out['behaviour_fraction'] = dict_out['behaviour_fraction'].squeeze()
            if jacobian is True:
                dict_out['dreinforcement_ddisplacement'] = dict_out['dreinforcement_ddisplacement'].squeeze()
        return(dict_out)    

    def calc_peak_reinforcement(self) -> dict:
        # it is possible that at root breakage, force is not max, due to rotation of forces

        # algorithm:
        # - find shear displacements to peak for each root
        # - calculate forces and derivatives
        # - get candidate options 
        #   (may need to offset displacement ever so slightly, in order to not get
        #    into floating precision issues, i.e. roots *just* broken even though they shouldn't)
        # - maximise using gradient search,
        #   if weibull=False, only needed if gradient = negative
        #   if weibull=True, need to account for direction  
        # - 
        # - 

        # root displacements to peak
        None


    # find soil shear displacement at peak force of each root
    def _get_displacement_root_peak_old(self) -> Quantity:
        """Estimate shear displacement at which each root reaches peak

        The 'peak' is defined as either the point of sudden breakage (
        tensile strength is exceeded) and/or the point at which the root 
        starts to slip (whichever happens first).

        Returns
        -------
        Quantity
            shear displacement at which each root reaches its peak
        """

        # get force at failure in each root
        if self.breakage is True:
            force_breakage = self.roots.xsection * self.roots.tensile_strength
        if self.slipping is True:
            force_slipping = self.roots.length * self.roots.circumference * self.interface.shear_strength
        # select limiting case
        if self.breakage is True:
            if self.slipping is True:
                force = np.minimum(force_breakage, force_slipping)
            else:
                force = force_breakage
        else:
            if self.slipping is True:
                force = force_slipping
            else:
                return(None)
        # get index of correct behaviour
        if self.elastoplastic is False:
            i = np.flatnonzero(self.pullout.behaviour_types == 'Anchored, elastic')[0]
            pullout_displacement = (
                self.pullout.coefficients[0][..., i] * force**2
                + self.pullout.coefficients[1][..., i] * force
                + self.pullout.coefficients[2][..., i]
                )
        else:
            i = np.flatnonzero(self.pullout.behaviour_types == 'Anchored, plastic')[0]
            print('Not yet implemented')
        # calculate shear displacement at failure
        elongation = pullout_displacement / self.distribution_factor
        vx = self.roots.orientation[..., 0]
        vy = self.roots.orientation[..., 1]
        vz = self.roots.orientation[..., 2]
        h = self.failure_surface.shear_zone_thickness
        shear_displacement = (
            np.sqrt((elongation + h / vz)**2 - h**2 * (1.0 + vy**2 / vz**2))
            - h * vx / vz
        )
        # return
        return(shear_displacement)

    # find peak reinforcement
    def calc_peak_reinforcement_old(self) -> dict:
        """Find peak reinforcement in Waldron's model

        Function first estimates a feasible range of shear displacements, 
        based on when roots start breaking and or slipping.

        Subsequently, a optimisation using Scipy's differential_evolution()
        function is performed to find the peak.

        Returns
        -------
        dict
            Dictionary with peak force results. Has keys:

            'displacement'
                Shear displacement at peak reinforcement
            'reinforcement'
                Peak reinforcement

        """
        # no root breakage or slipping -> infinite reinforcement
        if self.breakage is False and self.slipping is False:
            warnings.warn('No breakage or slipping - peak reinforcement is infinite!')
            return({
                'displacement': np.inf * units('m'),
                'reinforcement': np.inf * units('kPa')
                })
        # get range of displacements, and get reinforcement for each
        margin = 0.1  # add little bit of extra to range, just to be sure :-)
        us_all = self._get_displacement_root_peak()
        us_start_unit = 'mm'
        us_min = (1.0 - margin) * np.min(us_all).to(us_start_unit).magnitude
        us_max = (1.0 + margin) * np.max(us_all).to(us_start_unit).magnitude
        # function to optimise
        def fun(x):
            return(self.reinforcement(
                x * units(us_start_unit),
                jac = False,
                sign = -1.0
                )['reinforcement'].magnitude)
        # optimise
        sol = differential_evolution(
            fun,
            bounds = [(us_min, us_max)]
            )
        # extract 
        us_peak = sol.x[0] * units(us_start_unit)
        cr_peak = self.reinforcement(us_peak, jac = False)['reinforcement']
        # return dictionary with results
        return({
            'displacement': us_peak, 
            'reinforcement': cr_peak
            })

    # plot shear displacement versus reinforcement 
    def plot(
            self,
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
        """Plot how forces in the Waldron model mobilise with displacements

        Parameters
        ----------
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
        # no root breakage or slipping -> infinite reinforcement
        if self.breakage is False and self.slipping is False:
            us_peak = 100.0 * units('mm')
        else:
            us_peak_all = self._get_displacement_root_peak()
            us_peak = np.max(us_peak_all)
        us = np.linspace(0.0 * us_peak, us_peak * (1.0 + margin_axis), n)
        # total reinforcement, per root, for each displacement step
        res = self.reinforcement(us, jac = False, total = False)
        cr_total = np.sum(res['reinforcement'], axis = 0).to(yunit).magnitude
        # plot trace
        fig, ax  = plt.subplots()
        us_plot = us.to(xunit).magnitude
        ax.plot(
            us_plot,
            cr_total,
            c = 'black'
            )
        # stack plot
        if stack is True:
            cr = res['reinforcement'].to(yunit).magnitude
            ax.stackplot(us_plot, cr)
            # labels
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
                # label x-positions 
                if (self.slipping is False) and (self.breakage is False):
                    labels_x = us[int((1.0 - margin_label) * n)]
                    labels_y_tmp = self.reinforcement(labels_x, total = False)['reinforcement'].to(yunit).magnitude
                    labels_y = np.cumsum(labels_y_tmp) - 0.5 * labels_y_tmp
                    labels_x = np.full(len(labels_y), labels_x)
                else:
                    labels_x = us_peak_all - margin_label * np.max(us_peak_all)
                    labels_y_tmp = self.reinforcement(labels_x, total = False)['reinforcement'].to(yunit).magnitude
                    labels_x = labels_x.to(xunit).magnitude
                    labels_y_tmp = np.triu(labels_y_tmp)
                    labels_y = np.sum(labels_y_tmp, axis = 0) - 0.5 * np.diag(labels_y_tmp)
                # add labels to plot
                for xi, yi, li in zip(labels_x, labels_y, labels):
                    ax.annotate(
                        li, 
                        xy = (xi, yi), 
                        ha = 'center', 
                        va = 'center', 
                        bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                        fontsize = 'small'
                        )
        # plot peak
        if peak is True:
            peak = self.peak_reinforcement()
            plt.scatter(
                peak['displacement'].to(xunit).magnitude,
                peak['reinforcement'].to(yunit).magnitude,
                c = 'black'
                )
        # set labels
        ax.set_xlabel(xlabel + ' [' + str(xunit) + ']')
        ax.set_ylabel(ylabel + ' [' + str(yunit) + ']')
        # return
        return(fig, ax)
