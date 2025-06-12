import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma
from scipy.optimize import minimize, differential_evolution
from pyrootmemo.tools.helpers import units
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.pullout import PulloutEmbeddedElastic, PulloutEmbeddedElasticSlipping, PulloutEmbeddedElasticBreakage, PulloutEmbeddedElasticBreakageSlipping, PulloutEmbeddedElastoplastic, PulloutEmbeddedElastoplasticSlipping, PulloutEmbeddedElastoplasticBreakage, PulloutEmbeddedElastoplasticBreakageSlipping
from pyrootmemo.tools.utils_rotation import axisangle_rotate
from pyrootmemo.tools.utils_plot import round_range
from pint import Quantity, DimensionalityError
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
    
    # initialise class
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
        # check cross-sectional area correctly defined
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('failure_surface must contain attribute "cross_sectional_area"')
        # check k-factor
        if not (isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be an scalar integer or float')
        # return
        return(k * self.peak_force() / failure_surface.cross_sectional_area)



##########################
### FIBRE BUNDLE MODEL ###
##########################


# FBM class
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
    peak_force()
        Calculate peak force in bundle
    peak_reinforcement(failure_surface, k)
        Calculate peak reinforcement by bundle
    reduction_factor()
        FBMw reinforcement relative to WWM reinforcement
    plot(**kwargs)
        Generate plot showing how reinforcement mobilises
    """
    
    # initialise class
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

        Raises
        ------
        AttributeError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        # check roots input
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be object of class MultipleRoots')
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
        """Calculate force at which each roots breaks"""
        return(self.roots.tensile_strength * self.roots.xsection)
    
    # matrix  - matrix 
    def _get_matrix(self) -> Quantity:
        """Create force matrix

        Generate matrix for force in each root (rows) at breakage of each 
        root (columns). Assumes roots are pre-sorted in order of breakage, 
        so that broken roots can be easily removed by only keeping the 
        lower triangle of the matrix.
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
        """Calculate peak force in the fibre bundle

        Returns
        -------
        Quantity
            Peak force, i.e. largest force the entire bundle can carry
        """
        return(np.max(np.sum(self.matrix, axis = 0)))

    # reinforcement
    def peak_reinforcement(
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
        # check cross-sectional area correctly defined
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('Failure surface does not contain attribute cross_sectional_area')
        # check k-factor
        if not (isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be an scalar integer or float')
        # return
        return(k * self.peak_force() / failure_surface.cross_sectional_area)
    
    # reduction factor
    def reduction_factor(self) -> float:
        """FBM reinforcement relative to WWM

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
    
    # plot how roots mobilise, according to FBM
    def plot(
            self,
            unit: str = 'N',
            reference_diameter: int | float | Quantity = 1.0,
            reference_diameter_unit: str = 'mm',
            stack: bool = False,
            peak: bool = True,
            labels: list | bool = False, 
            label_margin: float = 0.05, 
            xlabel: str = 'Force in reference root', 
            ylabel: str = 'Total force in root bundle'      
            ):
        """Plot FBM force mobilisation

        Generate a matplotlib plot showing how forces in each roots are 
        mobilised, as function of the force in the reference root

        Parameters
        ----------
        unit : str, optional
            unit used for plotting forces, by default 'N'
        reference_diameter : int | float | Quantity, optional
            value of the reference diameter, by default 1.0
        reference_diameter_unit : str, optional
            unit of the reference diameter, by default 'mm'
        stack : bool, optional
            show contributions of each individual root, by default False
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
            
        Raises
        ------
        DimensionalityError
            _description_
        ValueError
            _description_
        """
        # force matrix - force in each root (row) at breakage of each root (columns) - sorted
        M = self.matrix.to(unit).magnitude
        # diameters, normalised and sorted in order of breakage
        if isinstance(reference_diameter, Quantity):
            if reference_diameter.dimensionality != units('mm').dimensionality:
                raise DimensionalityError('dimensionality of reference_diameter must be length')
            if not np.isscalar(reference_diameter.magnitude):
                raise ValueError('reference diameter must be a scalar')
        elif not(isinstance(reference_diameter, int) | isinstance(reference_diameter, float)):
            raise TypeError('reference_diameter must be float, int or Quantity')            
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



############
### RBMw ###
############


# RBMw class
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
    force(displacement)
        Calculate force in bundle at given displacement
    peak_force()
        Calculate peak force in bundle
    peak_reinforcement(failure_surface, k)
        Calculate peak reinforcement by bundle
    reduction_factor()
        RBMw reinforcement relative to WWM reinforcement
    plot(**kwargs)
        Generate plot showing how reinforcements mobilises with displacement
    """

    # initialise class
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
        # check if roots of correct class
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be instance of class MultipleRoots')
        # calculate weibull scale parameter (if not already explicitly defined)
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
        # check if roots contains all required instances
        for i in root_attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots must contain ' + i + ' values')
        # set roots
        self.roots = roots
        # check and set weibull shape parameter
        if not (isinstance(weibull_shape, float) | isinstance(weibull_shape, int)):
            raise ValueError('weibull_shape must be a scalar value')
        if weibull_shape <= 0.0:
            raise ValueError('weibull_shape must exceed zero ')
        if np.isinf(weibull_shape):
            raise ValueError('weibull_shape must have a finite value')
        self.weibull_shape = weibull_shape

    # forces in roots at current level of axial displacement
    def force(
            self,
            displacement: Quantity | int | float | np.ndarray,
            displacement_unit = 'm',
            total: bool = True,
            deriv: int = 0,
            sign: int | float = 1.0
            ) -> Quantity:
        """Calculate RBMw force at given level of displacement

        Parameters
        ----------
        displacement : Quantity | int | float | np.ndarray
            Quantity object with displacements. Should have unit with dimension 
            length. Can also be defined at integer, float or numpy array, in
            which case the 'displacement_unit' argument is used as unit.
        displacement_unit : str, optional
            The unit of displacement, in case displacement not inputted as a 
            Quantity
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
            (sum of) forces (or derivatives) in the root bundle

        Raises
        ------
        DimensionalityError
            _description_
        TypeError
            _description_
        ValueError
            _description_
        """
        # if displacement is not defined with a unit, make into a Quantity
        if isinstance(displacement, Quantity):
            if displacement.dimensionality != units('m').dimensionality:
                raise DimensionalityError('dimensionality of displacement must be length') 
        elif isinstance(displacement, int) | isinstance(displacement, float) | isinstance(displacement, np.ndarray):
            warnings.warn('Unit of displacement not defined. Assumed as ' + displacement_unit)
            displacement = displacement * units(displacement_unit)
        else:
            raise TypeError('displacement must be of type Quantity, float, int or np.ndarray')
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
             / self.roots.length)[:, np.newaxis]
        b = (self.roots.length
             * self.roots.tensile_strength
             / self.roots.elastic_modulus 
             * self.weibull_scale
            )[:, np.newaxis]
        x = displacement[np.newaxis, :]
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
                    return(sign * np.sum(y, axis = 0))
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
                    return(sign * np.sum(dy_dx, axis = 0))
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
                    return(sign * np.sum(dy2_dx2, axis = 0))
            else:
                return(sign * dy2_dx2.squeeze())
        # higher order derivatives
        else:
            raise ValueError('Only deriv = 0, 1 or 2 are currently available.')

    # Calculate peak force 
    def peak_force(
            self,
            method: str = 'Newton-CG'
            ) -> dict:
        """Calculate RBMw peak force

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
        method : str, optional
            Method to use in the scipy.optimize.minimize algorithm.
            Default is 'Newton-CG'. Analytical jacobian and hessian are 
            analytically known (see function self.force(), which allows for
            returning derivatives).

        Returns
        -------
        dict
            Dictionary with keys 'displacement' and 'force', 
            containing the Quantity of the displacement and force at 
            peak.   
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
        return({
            'force': peak_force_options[index],
            'displacement': displacement_options[index] 
        })

    # reinforcement
    def peak_reinforcement(
            self, 
            failure_surface: FailureSurface,
            k: float = 1.0
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
        k : float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.0.

        Returns
        -------
        dict
            Dictionary with keys 'displacement' and 'reinforcement', 
            containing the Quantity of the displacement and reinforcement at 
            peak.            
        """
        # check cross-sectional area correctly defined
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be intance of FailureSurface class')
        if not hasattr(failure_surface, 'cross_sectional_area'):
            raise AttributeError('failure_surface must contain attribute "cross_sectional_area"')
        # check k-value
        if not(isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be scalar integer or float')
        # peak force
        peak = self.peak_force()
        return({
            'reinforcement': k * peak['force'] / failure_surface.cross_sectional_area,
            'displacement': peak['displacement']
        })    
    
    # reduction factor
    def reduction_factor(
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
        force_rbmw = self.peak_force()['force']
        force_root = np.sum(self.roots.xsection * self.roots.tensile_strength)
        return((force_rbmw / force_root).magnitude)
    
    # plot RBMw force as function of pullout displacement
    def plot(
            self,
            n: int = 251,
            stack: bool = False,
            peak: bool = True,
            fraction: int | float = 0.75,  # minimum fraction of roots broken in each diameter,
            labels: list | bool = False, 
            xlabel: chr = 'Pull-out displacement', 
            ylabel: chr = 'Total force in root bundle',
            xunit: chr = 'mm',
            yunit: chr = 'N'
            ): 
        """Plot how forces in the RBMw mobilise with displacements

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
            'fraction' fraction of roots have broken, for all roots defined
        labels : bool | list, optional
            labels to plot on contribution of each root, by default False.
            If False, no labels are plotted. If True, labels are plotted using
            the index of the root in the MultipleRoots object. Custom labels 
            can be inputted using a list, which must have the same length as 
            the number of roots in the bundle.
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
        # calculate peak force and displacement
        peak_results = self.peak_force()
        # displacement to average root failure
        displacement_average = (
            self.roots.tensile_strength 
            / self.roots.elastic_modulus 
            * self.roots.length
            )
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
            ax.stackplot(displacement_magnitude, force_each_magnitude)
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
            labels_x_dimensional = (
                self.weibull_scale * (1.0 / self.weibull_shape)**(1.0 / self.weibull_shape)
                * self.roots.tensile_strength
                / self.roots.elastic_modulus
                * self.roots.length
                )
            labels_x = labels_x_dimensional.to(xunit).magnitude
            # labels y-positions - halfway up each stacking instance
            labels_y_dimensional_all = self.force(labels_x_dimensional, total = False)
            labels_y_all = np.triu(labels_y_dimensional_all.to(yunit).magnitude)
            labels_y = np.sum(labels_y_all, axis = 0) -  0.5 * np.diag(labels_y_all)
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



##########################################
### BASE CLASS FOR DIRECT SHEAR MODELS ###
##########################################

class _DirectShear():
    """Base class for direct shear models (Waldron, DRAM)
    
    Serves as a base clas for models in which reinforcement is mobilised
    as function of direct shear displacement.

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
    __init__(roots, interface, soil_profile, failure_surface, distribution_factor, **kwargs)
        Constructor
    get_initial_root_orientations()
        Defined initial orientations of all roots relative to the shear zone
    get_orientation_parameters(displacement, shear_zone_thickness, jac)
        Calculate root elongations in shear zone and k-factors 
    """

    # initisaliser
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
        # check instances are of the correct class
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be instance of class MultipleRoots')
        if not isinstance(interface, Interface):
            raise TypeError('interface must be instance of class Interface')
        if not isinstance(soil_profile, SoilProfile):
            raise TypeError('soil_profile must be instance of class SoilProfile')
        if not isinstance(failure_surface, FailureSurface):
            raise TypeError('failure_surface must be instance of class FailureSurface')
        # assign input
        self.roots = roots
        self.interface = interface
        self.soil_profile = soil_profile
        self.failure_surface = failure_surface
        # distribution parameter
        if not(isinstance(distribution_factor, int) | isinstance(distribution_factor, float)):
            raise TypeError('distribution factor must be int or float')
        self.distribution_factor = distribution_factor
        # set root orientations (relative to failure surface)
        self.roots.orientation = self.get_initial_root_orientations()
        # set friction angle at failure surface
        self.failure_surface.tanphi = np.tan(
            soil_profile
            .get_soil(failure_surface.depth)
            .friction_angle
            .to('rad')
            )
    
    # set root orientations as unit vectors, relative to direction of shearing
    def get_initial_root_orientations(
            self
            ) -> np.ndarray:
        """Set initial root orientations relative to shear direction

        returns a numpy array with 
        size (number of roots, 3).
        
        function set 3-D root orientations **relative to** failure surface so that:
        * local x = direction of shearing
        * local y = perpendicular to x on shear plane
        * local z = pointing downwards into the soil
        orientations are defined in terms of 3-dimensional unit vectors
                
        Roots are assumed to be defined in a **global** coordinate system
        * right-handed Cartesian coordinate system, with z-axis pointing down into the ground
        * azimuth angle = angle from x-axis to projection of root vector on the x-y plane
        * elevation angle = angle from z-axis to root vector
        
        If the initial root orientations are not defined, assume they are all 
        **perpedicular** to the shear zone.

        direction of failure failure surface (taken from FailureSurface object)
        * assume angle is defined as angle in x-z plane, defined (positive) from x to z

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

    # function to get 
    def get_orientation_parameters(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            jac: bool = False
            ) -> dict:
        # vector components of initial root orientation in shear zone
        v0x = (
            shear_zone_thickness
            * self.roots.orientation[..., 0]
            / self.roots.orientation[..., 2]
            )
        v0y = (
            shear_zone_thickness
            * self.roots.orientation[..., 1]
            / self.roots.orientation[..., 2]
        )
        v0z = shear_zone_thickness * np.ones_like(v0x)
        # length in shear zone
        if shear_zone_thickness.magnitude >= 0.0:
            L0 = shear_zone_thickness / self.roots.orientation[..., 2]
            L = np.sqrt((v0x + shear_displacement)**2 + v0y**2 + v0z**2)
        else:
            L0 = 0.0 * shear_zone_thickness * self.roots.orientation[..., 2]
            L = shear_displacement * np.ones_like(v0x)
        # pullout displacement
        up = self.distribution_factor * (L - L0)
        # orientation factor
        k = ((v0x + shear_displacement) + (v0z * self.failure_surface.tanphi)) / L
        # return
        if jac is False:
            return({
                'pullout_displacement': up,
                'k': k
                })
        else:
            # calculate derivatives with respect to:
            # * shear displacement: us
            # * shear zone thickness: h
            dv0x_dh = self.roots.orientation[..., 0] / self.roots.orientation[..., 2]
            dv0y_dh = self.roots.orientation[..., 1] / self.roots.orientation[..., 2]
            dv0z_dh = np.ones_like(v0z)
            if shear_zone_thickness.magnitude >= 0.0:
                dL0_dh = 1.0 / self.roots.orientation[..., 2]
                dL_dus = (v0x + shear_displacement) / L
                dL_dv0x = (v0x + shear_displacement) / L
                dL_dv0y = v0y / L
                dL_dv0z = v0z / L
            else:
                dL0_dh = 0.0 * self.roots.orientation[..., 2]
                dL_dus = np.ones_like(v0z)
                dL_dv0x = np.ones_like(v0z)
                dL_dv0y = np.ones_like(v0z)
                dL_dv0z = np.ones_like(v0z)
            dup_dus = self.distribution_factor * dL_dus
            dL_dh = (
                dL_dv0x * dv0x_dh
                + dL_dv0y * dv0y_dh
                + dL_dv0z * dv0z_dh
                )
            dup_dh = self.distribution_factor * (dL_dh - dL0_dh)
            dk_dus = (
                1.0 / L
                - k / L * dL_dus
            )
            dk_dh = (
                (dv0x_dh + dv0z_dh * self.failure_surface.tanphi) / L
                - k / L * dL_dh
            )
            # return values and derivatives in dictionary
            return({
                'pullout_displacement': up,
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

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    TypeError
        _description_
    DimensionalityError
        _description_
    TypeError
        _description_
    """

    # initiate model
    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            slipping: bool = True,
            breakage: bool = True,
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
        # call __init__ from parent class
        super().__init__(roots, interface, soil_profile, failure_surface)
        # set flags for analysis type (allowing for slippage, breakage and/or elastoplasticity)
        self.slipping = slipping
        self.breakage = breakage
        self.elastoplastic = elastoplastic
        self.weibull_shape = weibull_shape
        # check weibull shape
        if isinstance(weibull_shape, int) | isinstance(weibull_shape, float):
            if weibull_shape <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            elif np.isinf(weibull_shape):
                raise ValueError('weibull_shape must have finite value. Set to None for sudden breakages')
        else:
            if weibull_shape is not None:
                raise TypeError('weibull_shape must be an int, float or None')
        # set correct pullout object, depending on cases (slipping, breakage etc)
        if slipping is True:
            if breakage is True:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplasticBreakageSlipping(
                        roots, interface, weibull_shape = weibull_shape)
                else:
                    self.pullout = PulloutEmbeddedElasticBreakageSlipping(
                        roots, interface, weibull_shape = weibull_shape)
            else:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplasticSlipping(
                        roots, interface)
                else:
                    self.pullout = PulloutEmbeddedElasticSlipping(
                        roots, interface)
        else:
            if breakage is True:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplasticBreakage(
                        roots, interface, weibull_shape = weibull_shape)
                else:
                    self.pullout = PulloutEmbeddedElasticBreakage(
                        roots, interface, weibull_shape = weibull_shape)
            else:
                if elastoplastic is True:
                    self.pullout = PulloutEmbeddedElastoplastic(
                        roots, interface)
                else:
                    self.pullout = PulloutEmbeddedElastic(
                        roots, interface)

    # calculate root reinforcement at given displacement step
    def reinforcement(
            self,
            shear_displacement: Quantity | float | int | np.ndarray,
            shear_displacement_unit: str = 'm',
            total: bool = True,
            jac: bool = False,
            sign: int | float = 1.0
            ) -> dict:
        """Calculate root reinforcement at current level of displacement

        Parameters
        ----------
        shear_displacement : Quantity | float | int | np.ndarray
            soil shear displacement. If not set as a Quantity (with dimension
            length), the unit is assumed as 'shear_displacement_unit'.
        shear_displacement_unit : str, optional
            assumed shear displacement unit in case 'shear_displacement' not
            defined as a Quantity, by default 'm'
        total : bool, optional
            if True, returns total reinforcement by all roots. If False, return
            reinforcement for each root seperately as a 2-D matrix (axis 0 = 
            roots, axis 1 = displacement steps). By default True
        jac : bool, optional
            additionally return the derivative of reinforcement with respect 
            to shear displacement. By default False
        sign : int, float, optional
            Multiplication factor for all result returned by the function. 
            This is used to be able to use minimisation algorithms in order
            to find the global maximum force, see function self.peak_force(). 
            Default = 1.0

        Returns
        -------
        dict
            Dictionary with reinforcement results. Has keys:
            
            'reinforcement': 
                shear reinforcement
            'behaviour_type':
                list of root behaviour type names
            'behaviour_fraction': 
                fraction of total root cross-sectional area that behaves
                according to each of the types in 'behaviour_type'
            'derivative': 
                derivative of reinforcement output with respect to the shear 
                displacement. Only returned when jac = True

        Raises
        ------
        DimensionalityError
            _description_
        TypeError
            _description_
        """
        # set displacement value with units - and make array (if not already so)
        if isinstance(shear_displacement, Quantity):
            if shear_displacement.dimensionality != units('m').dimensionality:
                raise DimensionalityError('dimensionality of displacement must be length')
            if np.isscalar(shear_displacement.magnitude):
               shear_displacement = np.array([shear_displacement.magnitude]) * shear_displacement.units
        elif isinstance(shear_displacement, int) | isinstance(shear_displacement, float):
            warnings.warn('Unit of displacement not defined. Assumed as ' + shear_displacement_unit)
            shear_displacement = np.array([shear_displacement]) * units(shear_displacement_unit)
        elif isinstance(shear_displacement, list):
            warnings.warn('Unit of displacement not defined. Assumed as ' + shear_displacement_unit)
            shear_displacement = np.array(shear_displacement) * units(shear_displacement_unit)
        elif isinstance(shear_displacement, np.ndarray):
            warnings.warn('Unit of displacement not defined. Assumed as ' + shear_displacement_unit)
            shear_displacement = shear_displacement * units(shear_displacement_unit)
        else:
            raise TypeError('input type for displacement not recognised')
        # relative cross-sectional area of each root
        xsection_rel = self.roots.xsection.magnitude / np.sum(self.roots.xsection.magnitude)
        # loop through all displacement steps
        cr_all = []
        xsection_fractions_all = []
        if jac is True:
            dcr_dus_all = []
        for us in shear_displacement:
            # pullout displacement (up) and orientation factors (k)
            res = self.get_orientation_parameters(
                us,
                self.failure_surface.shear_zone_thickness,
                jac = jac
            )
            # pullout force (Tp), survival fraction (S) and behaviour type index (b)
            Tp, dTp_dup, S, b = self.pullout.force(
                res['pullout_displacement'], 
                jac = jac
                )
            # reinforcement
            cr = sign * res['k'] * Tp / self.failure_surface.cross_sectional_area
            if total is True:
                cr = np.sum(cr)
            cr_all.append(cr)
            # fraction of root cross-sectional area for each behaviour types
            xsection_fractions = np.bincount(
                b, 
                weights = S * xsection_rel,
                minlength = len(self.pullout.behaviour_types)
                )
            xsection_fractions_all.append(xsection_fractions)
            # return jacobian if requested
            if jac is True:
                dcr_dus = sign * (
                    (res['dk_dus'] * Tp + res['k'] * dTp_dup * res['dup_dus']) 
                    / self.failure_surface.cross_sectional_area
                    )
                if total is True:
                    dcr_dus = np.sum(dcr_dus)
                dcr_dus_all.append(dcr_dus)
        # create output
        cr_out = np.array([i.magnitude for i in cr_all]).transpose().squeeze() * cr_all[0].units
        xsection_fractions_out = np.array(xsection_fractions_all).transpose().squeeze()
        out = {
            'reinforcement': cr_out, 
            'behaviour_type': self.pullout.behaviour_types,
            'behaviour_fraction': xsection_fractions_out
            }
        # add derivative of reinforcement with respect to shear displacement
        if jac is True:
            out['derivative'] = (
                np.array([i.magnitude for i in dcr_dus_all]).transpose().squeeze 
                * dcr_dus_all[0].units
                )
        # return
        return(out)    

    # find soil shear displacement at peak force of each root
    def _get_displacement_root_peak(self) -> Quantity:
        """Estimate shear displacement at which each root reaches peak

        The 'peak' is defined as either the point of sudden breakage (
        tensile strength is exceeded) and/or the point at which the root 
        starts to slip.

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
    def peak_reinforcement(self) -> dict:
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
