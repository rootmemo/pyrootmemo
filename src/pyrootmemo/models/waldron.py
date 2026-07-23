import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from pyrootmemo import Parameter
from pyrootmemo.helpers import units, create_quantity, Results, ResultsType
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
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

    def __init__(
            self,
            roots: MultipleRoots,
            interface: Interface,
            soil_profile: SoilProfile,
            failure_surface: FailureSurface,
            breakage: bool | None = None,
            slipping: bool | None = None,
            elastoplastic: bool | None = None,
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
        slipping : bool | None, optional
            Include root slippage? By default None. If None, it set to `True` 
            when possible, which requires the `length` attribute in `roots`, 
            and otherwise set to `False`. This default behaviour can be 
            overruled by setting `True` or `False` manually.
        breakage : bool | None, optional
            Include root slippage? By default None. If None, it set to `True` 
            when possible, which requires the `tensile_strength` attribute in 
            `roots`, and otherwise set to `False`. This default behaviour can be 
            overruled by setting `True` or `False` manually.
        elastoplastic : bool | None, optional
            Include elasto-plastic root behaviour? By default None. If None, 
            it set to `True` when possible, which requires the `plastic_modulus`
            and `yield_strength` attributes in `roots`, and otherwise set to 
            `False`. This default behaviour can be overruled by setting `True` 
            or `False` manually.
        weibull_shape : float | int | None, optional
            Weibull shape parameter for root modelling breakage (when 
            breakage = True). By default None. If 'None', all roots are 
            assumed to break instantly (like in the original Waldron-type 
            models).
        """
        super().__init__(roots, interface, soil_profile, failure_surface)
        if breakage is None:
            if hasattr(roots, 'tensile_strength'):
                self.breakage = True
            else:
                self.breakage = False
        elif isinstance(breakage, bool):
            self.breakage = breakage
        else:
            raise TypeError("'breakage' must be boolean or None")
        if slipping is None:
            if hasattr(roots, 'length'):
                self.slipping = True
            else:
                self.slipping = False
        elif isinstance(slipping, bool):
            self.slipping = slipping
        else:
            raise TypeError("'slipping' must be boolean or None")
        if elastoplastic is None:
            if hasattr(roots, 'plastic_modulus') and hasattr(roots, 'yield_strength'):
                self.elastoplastic = True
            else:
                self.elastoplastic = False
        elif isinstance(elastoplastic, bool):
            self.elastoplastic = elastoplastic
        else:
            raise TypeError("'elastoplastic' must be boolean or None")
        if isinstance(weibull_shape, int) | isinstance(weibull_shape, float):
            if weibull_shape <= 0.0:
                raise ValueError('weibull_shape must exceed zero')
            elif np.isinf(weibull_shape):
                raise ValueError('weibull_shape must have finite value. Set to None for sudden breakages')
        else:
            if weibull_shape is not None:
                raise TypeError('weibull_shape must be an int, float or None')
            self.weibull_shape = weibull_shape
        self.pullout = AxialPullout(
            roots, 
            interface,
            surface = False, 
            breakage = self.breakage, 
            slipping = self.slipping, 
            elastoplastic = self.elastoplastic, 
            weibull_shape = weibull_shape
            )

    def calc_reinforcement(
            self,
            displacement: Quantity | Parameter,
            jacobian: bool = False,
            squeeze: bool = True,
            multiplier: int | float = 1.0,
            results: str = "attribute"
            ):
        """Calculate root reinforcement given level(s) of displacement

        Parameters
        ----------
        displacement : Quantity | Parameter
            soil shear displacement.
        jacobian : bool, optional
            additionally return the derivative of reinforcement with respect 
            to shear displacement. By default False
        squeeze : bool, optional
            If True, strip all dimensions with length '1' out of the various
            results arrays. By default True
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

        Function creates a dictionary with the keys:
           
            * 'displacement' : Quantity
                Direct shear displacements (size n)
            * 'reinforcement' : Quantity
                shear reinforcements. Has shape (n)
            * 'reinforcement_per_root' : Quantity
                shear reinforcements. Has shape (n*m) where n is the number of displacement steps
                and m the number of roots.
            * 'behaviour_types' : np.ndarray
                list of root behaviour type names. 
            * 'behaviour_fraction' : np.ndarray
                fraction of total root cross-sectional area that behaves
                according to each of the types in 'behaviour_types'. Has shape (n*p) where
                n is the number of displacement steps and p the number of behaviour types, 
                (defined in the `behaviour_types` field).
            * 'dreinforcement_ddisplacement': Quantity
                derivative of reinforcement output with respect to the shear 
                displacement. Only returned when jacobian = True. Has shape (n)
        
        How this dictionary returned depends on the value of the `results` 
        argument.
                
        """
        displacement = create_quantity(displacement, check_unit = 'mm')
        if np.isscalar(displacement.magnitude):
            displacement = np.array([displacement.magnitude]) * displacement.units
        ndisplacement = len(displacement)
        nbehaviour = len(self.pullout.behaviour_types)
        nroots = len(self.roots.xsection)
        reinforcement_per_root = np.zeros((ndisplacement, nroots)) * units('kPa')
        xsection_fractions_per_root = np.zeros((ndisplacement, nbehaviour, nroots))
        if jacobian is True:
            dreinforcement_per_root_dshear_displacement = np.zeros((ndisplacement, nroots)) * units('kPa/mm')
    
        for us, i in zip(displacement, np.arange(ndisplacement)):
            dict_pullout_disp = self.calc_pullout_displacement(
                us,
                self.failure_surface.shear_zone_thickness,
                jacobian = jacobian,
            )
            dict_pullout_force = self.pullout.calc_force(
                dict_pullout_disp['pullout_displacement'], 
                jacobian = jacobian,
                results = 'return'
                )
            dict_k = self.calc_orientation_factor(
                us,
                self.failure_surface.shear_zone_thickness,
                jacobian = jacobian
                )
            reinforcement_per_root[i, ...] = (
                multiplier 
                * dict_k['k'] 
                * dict_pullout_force['force_per_root'] 
                / self.failure_surface.cross_sectional_area
                )
            xsection_fractions_per_root[i, dict_pullout_force['behaviour_index'], np.arange(nroots)] = (
                dict_pullout_force['survival_fraction'] 
                * self.roots.xsection.magnitude 
                / np.sum(self.roots.xsection.magnitude)
                )
            if jacobian is True:
                dreinforcement_per_root_dshear_displacement[i, ...] = (
                    multiplier 
                    / self.failure_surface.cross_sectional_area * (
                        dict_k['dk_dshear_displacement'] * dict_pullout_force['force_per_root']
                        + dict_k['k'] * dict_pullout_force['dforce_per_root_ddisplacement'] 
                        * dict_pullout_disp['dpullout_displacement_dshear_displacement']
                    )
                )
        
        xsection_fractions = xsection_fractions_per_root.sum(axis = -1)
        behaviour_types_unique = np.unique(self.pullout.behaviour_types)
        behaviour_fraction_unique = np.stack(
            [np.sum(xsection_fractions[:, self.pullout.behaviour_types == b], axis = 1) 
             for b in behaviour_types_unique],
             axis = 1)
        behaviour_types_unique = np.append(behaviour_types_unique, 'Broken')
        behaviour_fraction_unique = np.concatenate(
            (behaviour_fraction_unique, 1.0 - np.sum(behaviour_fraction_unique, axis = 1)[:, np.newaxis]),
            axis = 1
            )

        output = {
            'displacement': displacement,
            'reinforcement_per_root': reinforcement_per_root,
            'reinforcement': reinforcement_per_root.sum(axis = -1),
            'behaviour_types': behaviour_types_unique,
            'behaviour_fraction': behaviour_fraction_unique
            }
        if jacobian is True:
            output['dreinforcement_ddisplacement'] = dreinforcement_per_root_dshear_displacement.sum(axis = -1)
        if squeeze is True:
            output['displacement'] = output['displacement'].squeeze()
            output['reinforcement'] = output['reinforcement'].squeeze()
            output['reinforcement_per_root'] = output['reinforcement_per_root'].squeeze()
            output['behaviour_fraction'] = output['behaviour_fraction'].squeeze()
            if jacobian is True:
                output['dreinforcement_ddisplacement'] = output['dreinforcement_ddisplacement'].squeeze()
        
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output   

    def calc_displacement_to_rootpeak(
            self,
            ) -> Quantity:
        """Calculate shear displacement at peak reinforcements of individual roots

        Calculate the shear displacement associated with each root reaching its
        maximum tensile force, i.e. at the point of breakage or at the onset
        of slippage. The associated pull-out displacement is calculated, and 
        this is then converted back to shear displacements and returned.

        Returns
        -------
        Quantity
            Array with shear displacements
        """
        shear_zone_thickness = self.failure_surface.shear_zone_thickness
        pullout_displacement = self.pullout.calc_displacement_to_peak(results = 'return')
        elongation = pullout_displacement['peak_displacement_per_root'] / self.distribution_factor
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
            extend_multiplier: int | float = 1.15,
            results: str = "attribute"
            ):
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
        
        Calculate shear displacement at peak reinforcements of individual roots

        Calculate the shear displacement associated with each root reaching its
        maximum tensile force, i.e. at the point of breakage or at the onset
        of slippage. The associated pull-out displacement is calculated, and 
        this is then converter back to shear displacements

        Function creates a dictionary with the keys:
           
            * 'peak_reinforcement' : Quantity
                maximum value of the root reinforcement at any shear 
                displacement
            * 'peak_displacement' : Quantity
                the value of the shear displacement at which the peak 
                reinforcement is mobilised

        How this dictionary returned depends on the value of the `results` 
        argument.
        
        Parameters
        ----------
        extend_multiplier : int | float, optional
            Multiplier for shear displacement that is searched by the 
            evolutionary solver (to make sure peak is within search domain),
            by default 1.15
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.
        """
        if (self.breakage is False) and (self.slipping is False):
            output = {
                'peak_displacement': np.inf * units('mm'),
                'peak_reinforcement': np.inf * units('kPa')
                }
        else:
            max_displacement_per_root = self.calc_displacement_to_rootpeak()
            shear_displacement_max = extend_multiplier * np.max(max_displacement_per_root)
            shear_displacement_units = shear_displacement_max.units
            def fun_to_optimize(x):
                return(self.calc_reinforcement(
                    x * shear_displacement_units,
                    jacobian = False,
                    multiplier = -1.0,
                    results = 'return'
                    )['reinforcement'].magnitude)
            sol = differential_evolution(
                fun_to_optimize,
                bounds = [(0.0, shear_displacement_max.magnitude)]
                )
            displacement_peak = sol.x[0] * shear_displacement_max.units
            dict_results_peak = self.calc_reinforcement(
                displacement_peak,
                jacobian = False,
                results = 'return'
                )
            output = {
                'peak_displacement': dict_results_peak['displacement'],
                'peak_reinforcement': dict_results_peak['reinforcement']
                }
            
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
        if (self.breakage is False) and (self.slipping is False):
            shear_displacement_max = 100.0 * units('mm')
        else:
            shear_displacement_rootpeak = self.calc_displacement_to_rootpeak()
            shear_displacement_max = np.max(shear_displacement_rootpeak)
        shear_displacement = np.linspace(0.0 * shear_displacement_max, shear_displacement_max * (1.0 + margin_axis), n)
        dict_results = self.calc_reinforcement(shear_displacement, jacobian = False, results = 'return')
        reinforcement_magnitude = dict_results['reinforcement'].to(yunit).magnitude

        if ax is None:
            ax = plt.gca()

        shear_displacement_magnitude = shear_displacement.to(xunit).magnitude
        ax.plot(
            shear_displacement_magnitude,
            reinforcement_magnitude,
            c = 'black'
            )

        if stack is True:
            reinforcement_perroot_magnitude = dict_results['reinforcement_per_root'].to(yunit).magnitude
            ax.stackplot(shear_displacement_magnitude, reinforcement_perroot_magnitude.transpose())
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
                if (self.slipping is False) and (self.breakage is False):
                    labels_x = shear_displacement[int((1.0 - margin_label) * n)]
                    dict_results_label =  self.calc_reinforcement(labels_x, results = 'return')
                    labels_y_tmp = dict_results_label['reinforcement_per_root'].to(yunit).magnitude
                    labels_y_magnitude = np.cumsum(labels_y_tmp, axis = 1) - 0.5 * labels_y_tmp
                    labels_x_magnitude = np.full(len(labels_y_magnitude), labels_x.to(xunit).magnitude)
                else:
                    labels_x_tmp = shear_displacement_rootpeak - margin_label * np.max(shear_displacement_rootpeak)
                    dict_results_label =  self.calc_reinforcement(labels_x_tmp, results = 'return')
                    labels_y_tmp = dict_results_label['reinforcement_per_root'].to(yunit).magnitude
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
                if ('peak_diplacement' in self.output) and ('peak_reinforcement' in self.output):
                    peak_displacement = self.output['peak_displacement'].to(xunit).magnitude
                    peak_reinforcement = self.output['peak_reinforcement'].to(yunit).magnitude
                else:
                    dict_peak_results = self.calc_peak_reinforcement(results = 'return')
                    peak_displacement = dict_peak_results['peak_displacement'].to(xunit).magnitude
                    peak_reinforcement = dict_peak_results['peak_reinforcement'].to(yunit).magnitude
                ax.scatter(peak_displacement, peak_reinforcement, c = 'black')

        ax.set_xlabel(xlabel + ' [' + str(xunit) + ']')
        ax.set_ylabel(ylabel + ' [' + str(yunit) + ']')
        return(ax)
