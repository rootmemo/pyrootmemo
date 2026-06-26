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
from pyrootmemo.models import AxialPullout
from pint import Quantity
from .direct_shear import _DirectShear

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
    calc_reinforcement(shear_displacement, ...)
        calculate reinforcement at given level(s) of shear displacement
    calc_peak_reinforcement()
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
        
    def calc_reinforcement(
            self,
            shear_displacement: Quantity | Parameter,
            total: bool = True,
            jacobian: bool = False,
            squeeze: bool = True,
            sign: int | float = 1.0
            ) -> dict:
        """Calculate root reinforcement given level(s) of displacement

        Parameters
        ----------
        shear_displacement : Quantity | Parameter
            soil shear displacement.
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
            res_k = self.calc_orientation_factor(
                us,
                self.failure_surface.shear_zone_thickness,
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

    def calc_displacement_to_rootpeak(self) -> Quantity:
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
        shear_zone_thickness = self.failure_surface.shear_zone_thickness
        pullout_displacement = self.pullout.calc_displacement_to_peak()
        elongation = pullout_displacement / self.distribution_factor
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
        shear_displacement_max = factor * np.max(self.calc_displacement_to_rootpeak())
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
            fig = None,
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
        """Plot how forces in the Waldron model mobilise with displacements

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
            shear_displacement_rootpeak = self.calc_displacement_to_rootpeak()
            shear_displacement_max = np.max(shear_displacement_rootpeak)
        shear_displacement = np.linspace(0.0 * shear_displacement_max, shear_displacement_max * (1.0 + margin_axis), n)
        results = self.calc_reinforcement(shear_displacement, jacobian = False, total = False)
        if self.roots.xsection.shape == (1, ):
            total_reinforcement_magnitude = results['reinforcement'].to(yunit).magnitude
        else:
            total_reinforcement_magnitude = np.sum(results['reinforcement'], axis = 1).to(yunit).magnitude
        
        if fig is None and ax is None:
            fig, ax  = plt.subplots()
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
        return(fig, ax)
