# import packages and functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pint import Quantity
from pyrootmemo.utils_plot import round_range
from pyrootmemo.materials import MultipleRoots
from pyrootmemo.geometry import FailureSurface
from pyrootmemo.tools.helpers import units


###########
### FBM ###
###########


# FBM class
class Fbm():
    """
    Class for fibre bundle models.
    
    This class uses the "matrix" method as described in Yildiz & Meijer. This
    method follows the following steps:
    
    1. Roots are sorted in order of breakage. The 'sorting order', i.e. the 
       list of indices to sort root properties into the correct order, is 
       stored in class attribute 'sort_order'. The 'breakage order', i.e. the 
       list of indices describing the order of breakage for each root, is 
       stored in class attribute 'breakage_order'.
     
    2. A matrix is generated that calculates the force in every root (rows), 
       at the moment of breakage of any root (columns). This matrix is stored
       as the class attribute 'matrix', and assumes roots have already been 
       sorted in order of breakage
       
    3. Peak forces can now easily be termined by finding the column in the 
       matrix that has the largest sum of forces
       
       
    The class constains some additional methods:
    
    * 'peak_force()': calculate peak force in root bundle
    * 'peak_reinforcement()': calculate peak root reinforcement, given a known
      failure surface area and a Wu/Waldron orientation factor 'k'
    * 'plot()': stackplot showing how forces in each root are gradually 
      mobilised    
    
    """
    
    # initialise class
    def __init__(
            self, 
            roots: MultipleRoots, 
            load_sharing: float | int
            ):
        """
        Initiate fibre bundle model class

        Parameters
        ----------
        roots : instance of MultipleRoots class. 
            Must contain fields 'diameter', 'xsection' and 'tensile_strength'
        loadsharing : float
            fibre bundle model load sharing parameter.

        Returns
        -------
        None.

        """
        # check roots input
        if not isinstance(roots, MultipleRoots):
            TypeError('roots must be object of class MultipleRoots')
        attributes_required = ['diameter', 'xsection', 'tensile_strength']
        for i in attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots must contain ' + str(i) + ' attribute')
        # check if loadsharing parameter is a finite, scalar value
        if not (isinstance(load_sharing, int) | isinstance(load_sharing, float)):
            raise ValueError('load_sharing must be a scalar integer or float')
        if np.isinf(load_sharing):
            raise ValueError('load_sharing must have finite value')
        # set parameters
        self.roots = roots
        self.load_sharing = load_sharing
        # get sorting order (get roots in order of failure
        self.sort_order = np.argsort(
            self._tensile_capacity() 
            / (roots.diameter ** load_sharing)
            )
        # get breakage order (order of root breakages
        self.breakage_order = np.argsort(self.sort_order)
        # get force matrix
        self.matrix = self._get_matrix()
        
        
    # tensile capacity - force at which roots break
    def _tensile_capacity(self) -> Quantity:
        """
        Calculate force at which each roots breaks
        """
        return(self.roots.tensile_strength * self.roots.xsection)
    
    
    # matrix  - matrix 
    def _get_matrix(self) -> Quantity:
        """
        Generate matrix for force in each root (rows) at breakage of each 
        root (columns). Assumes roots are sorted in order of breakage
        """
        # root capacity (force)
        force = self._tensile_capacity()
        # get units
        force_unit = force.units
        # sort data
        y_sorted = (force.magnitude)[self.sort_order]
        x_sorted = (self.roots.diameter.magnitude)[self.sort_order]
        # forces in each root (rows) as function of breaking root (columns)
        matrix = np.outer(
            x_sorted ** self.load_sharing,
            y_sorted / (x_sorted ** self.load_sharing)
            )
        # remove roots that have broken (upper triagle of matrix, so keep 
        # lower triangle)
        matrix_broken = np.tril(matrix)
        # return with units added back
        return(matrix_broken * force_unit)


    # peak force
    def peak_force(self) -> Quantity:
        """
        Calculate peak force (largest force at any point) in the fibre bundle

        Returns
        -------
        float
            peak force.

        """
        return(np.max(np.sum(self.matrix, axis = 0)))


    # reinforcement
    def peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: int | float = 1.0
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
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('Failure surface does not contain attribute cross_sectional_area')
        # check k-factor
        if not (isinstance(k, int) | isinstance(k, float)):
            TypeError('k must be an scalar integer or float')
        # return
        return(k * self.peak_force() / failure_surface.cross_sectional_area)
    
    
    # reduction factor
    def reduction_factor(self) -> float:
        """
        Calculate the ratio between bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0.0 and 1.0. '1.0' indicates all roots break simultaneously.

        Returns
        -------
        float
            reduction factor.

        """
        force_fbm = self.peak_force()
        force_sum = np.sum(self.roots.xsection * self.roots.tensile_strength)
        return((force_fbm / force_sum).magnitude)
    
    def plot(
            self,
            unit: str = 'N',
            reference_diameter = 1.0,
            reference_diameter_unit: str = 'mm',
            stack: bool = False,
            peak: bool = True,
            labels: list | bool = False, 
            label_margin: float = 0.05, 
            xlabel: str = 'Force in reference root', 
            ylabel: str = 'Total force in root bundle'      
            ):
        """
        Generate a matplotlib plot showing how forces in each roots are 
        mobilised, as function of the force in the reference root

        Parameters
        ----------
        labels : bool, optional
            labels for individual roots, If False, no labels are plotted. If
            True, labels are plotted as numbers indicating the order in which
            roots are defined in the input. Can be defined as a list of 
            character strings to specify individual labels for each root.
            The default is True.
        label_margin : float, optional
            controls the location for plotting labels. Defined as the fraction
            of the x-axis size. Labels are plotted on the right-hand size of
            the force triangles, and centred vertically. The default is 0.05.
        xlabel : chr, optional
            x-axis label. The default is 'Force in reference root'.
        ylabel : chr, optional
            y-axis label. The default is 'Total force in root bundle'.

        Returns
        -------
        tuple
            Tuple containing a figure and an axis object.

        """
        # force matrix - force in each root (row) at breakage of each root (columns) - sorted
        M = self.matrix.to(unit).magnitude
        # diameters, normalised and sorted in order of breakage
        diameter = self.roots.diameter.to(reference_diameter_unit).magnitude[self.sort_order]
        # force in reference root at moments of breakage, and just after
        x_before = np.diag(M) * (reference_diameter / diameter)**self.load_sharing
        x_after = x_before + 1.0e-12 * np.max(x_before)
        x_all = np.append(0.0, np.stack((x_before, x_after)).ravel(order = 'F'))
        # total reinforcement
        y_before = np.sum(M, axis = 0)
        y_after = y_before - np.diag(M)
        y_all = np.append(0.0, np.stack((y_before, y_after)).ravel(order = 'F'))
        # plot
        fig, ax = plt.subplots()
        if stack is True:
            # forces in individual roots
            y_before_i = M
            y_after_i = M - np.diag(np.diag(M))
            y_all_i = np.concatenate((
                np.zeros((M.shape[0], 1)),
                np.stack((y_before_i, y_after_i), axis = -1).reshape(M.shape[0], 2 * M.shape[1]),
                ), axis = -1)
            # reverse y-values (forces), so plot lines are stacked in order from last breaking to first breaking
            y_all_i = np.flip(y_all_i, axis = 0)
            # colour order - use default matplotlib colors, but order in which roots are defined
            prop_cycle = mpl.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            n_color = len(colors)
            colors_new = np.array(colors * int(np.ceil(M.shape[0] / n_color)))[np.flip(self.sort_order)]
            # plot stacked traces
            ax.stackplot(x_all, y_all_i, colors = colors_new)
            # plot total trace
            ax.plot(x_all, y_all, c = 'black')
            # label text
            if labels is True:
                labels = list(self.sort_order + 1)
                plot_labels = True
            elif isinstance(labels, list):
                if len(labels) == M.shape[0]:
                    labels = np.array(labels)[self.sort_order]
                    plot_labels = True
                else:
                    plot_labels = False
            else:
                plot_labels = False
            # add labels to plot
            if plot_labels is True:
                labels_x = x_before - label_margin * np.max(x_before)
                labels_y = (y_before - 0.5 * np.diag(M)) * labels_x / x_before
                for xi, yi, li in zip(labels_x, labels_y, labels):
                    ax.annotate(
                        li, xy = (xi, yi), 
                        ha = 'center', 
                        va = 'center', 
                        bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
                        fontsize = 'small'
                        )
        else:
            # bundle force at moments of breakage, and just after
            y_before = np.sum(M, axis = 0)
            y_after = y_before - np.diag(M)
            # bundle force array for plotting
            y_all = np.append(0.0, np.stack((y_before, y_after)).ravel(order = 'F'))
            # plot
            ax.plot(x_all, y_all)
        # add peak reinforcement
        if peak is True:
            Msum0 = np.sum(M, axis = 0)
            i_peak = np.argmax(Msum0)
            x_peak = x_before[i_peak]
            y_peak = Msum0[i_peak]
            plt.scatter(x_peak, y_peak, c = 'black')
        # axes
        ax.set_xlabel(xlabel + " [" + unit + "]")
        ax.set_ylabel(ylabel + " [" + unit + "]")
        ax.set_xlim(round_range(x_all, limits = [0.0, None])['limits'])
        ax.set_ylim(round_range(
            self.peak_force().to(unit).magnitude, 
            limits = [0.0, None]
            )['limits'])
        # return plotted object
        return(fig, ax)

