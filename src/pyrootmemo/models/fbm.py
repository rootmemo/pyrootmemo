import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from pyrootmemo import Parameter
from pyrootmemo.helpers import create_quantity
from pyrootmemo.geometry import FailureSurface
from pyrootmemo.materials import MultipleRoots
from pyrootmemo.tools.utils_plot import round_range
from pint import Quantity


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
        output : dict
            Dictionary with all calculation results

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
        self.output = {}

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

    def calc_peak_force(self) -> None:
        """Calculate peak force in the fibre bundle

        Calculat the peak force that the entire fibre bundle can carry. This
        reinforcement is calculated using the 'matrix method' as described 
        by Yildiz and Meijer.
        
        Attributes Modified
        -------------------
        output : dict
            adds a `peak_force` item to the output dictionary

        Returns
        -------
        None
        """
        peak_force = np.max(np.sum(self.matrix, axis = 0))
        self.output['peak_force'] = peak_force

    def calc_peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: int | float = 1.0
            ) -> None:
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

        Attributes Modified
        -------------------
        output : dict
            adds a `peak_reinforcement` item to the output dictionary
            
        Returns
        -------
        None
        """
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('Failure surface does not contain attribute cross_sectional_area')
        if not (isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be an scalar integer or float')
        if not 'peak_force' in self.output:
            self.calc_peak_force()
        self.output['peak_reinforcement'] = (
            k *  self.output['peak_force'] 
            / failure_surface.cross_sectional_area
            )
    
    def calc_reduction_factor(self) -> None:
        """Calculate FBM peak reinforcement relative to WWM

        Calculate the ratio between fibre bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0.0 and 1.0. '1.0' indicates all roots break simultaneously.

        Attributes Modified
        -------------------
        output : dict
            adds a `reduction_factor` item to the output dictionary
        
        Returns
        -------
        None
        """
        if not 'peak_force' in self.output:
            self.peak_force()
        force_sum = np.sum(self.roots.xsection * self.roots.tensile_strength)
        factor = self.output['peak_force'] / force_sum
        if isinstance(factor, Quantity):
            factor = factor.magnitude
        self.output['reduction_factor'] = factor
    
    def plot(
            self,
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
        ax : matplotlib.axes.Axes, optional
            matplotlib axis object to plot on. If not defined, the current axis
            is used. By default None
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
        
        if not isinstance(ax, Axes):
            ax = plt.gca()

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
        
        return(ax)
