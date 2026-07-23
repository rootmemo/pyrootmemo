import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from pyrootmemo import Parameter
from pyrootmemo.helpers import units, create_quantity, Results, ResultsType
from pyrootmemo.geometry import SoilProfile, FailureSurface
from pyrootmemo.materials import MultipleRoots, Interface
from pyrootmemo.models import AxialPullout
from pint import Quantity
from .direct_shear import _DirectShear

class Dram(_DirectShear):
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
            breakage: bool = None,
            slipping: bool = None,
            elastoplastic: bool = None,
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
        self.roots.length_surface = 0.5 * np.ones(roots.xsection.shape) * failure_surface.shear_zone_thickness
        self.pullout = AxialPullout(
            roots, 
            interface,
            surface = True, 
            breakage = breakage, 
            slipping = slipping, 
            elastoplastic = elastoplastic, 
            weibull_shape = weibull_shape
            )
        if not hasattr(self.failure_surface, 'max_shear_zone_thickness'):
            self.failure_surface.max_shear_zone_thickness = np.inf

    def calc_single_step(
            self,
            shear_displacement: Quantity,
            shear_zone_thickness: Quantity,
            soil_shear_strength: Quantity,
            soil_friction_angle: Quantity,
            jacobian: bool = False
            ) -> dict:
        dict_orientation = self.calc_displaced_orientation( 
            shear_displacement,
            shear_zone_thickness,
            jacobian = jacobian
            )
        dict_pullout_disp = self.calc_pullout_displacement(
            shear_displacement,
            shear_zone_thickness,
            jacobian = jacobian
            )
        dict_pullout_force = self.pullout.calc_force(   
            dict_pullout_disp['pullout_displacement'],
            jacobian = jacobian,
            results = 'return'
            )
        root_force_x = dict_pullout_force['force_per_root'] * dict_orientation['orientation'][:, 0] 
        root_force_z = dict_pullout_force['force_per_root'] * dict_orientation['orientation'][:, 2] * np.tan(soil_friction_angle)
        output = {'yield_value': np.sum(root_force_x - root_force_z) / self.failure_surface.cross_sectional_area - soil_shear_strength}
        output['reinforcement_per_root'] = (root_force_x + root_force_z) / self.failure_surface.cross_sectional_area
        output['reinforcement'] = np.sum(output['reinforcement_per_root'])
        if jacobian is True:
            dload_dshear_zone_thickness = np.sum(
                dict_pullout_force['dforce_per_root_ddisplacement'] 
                * dict_pullout_disp['dpullout_displacement_dshear_zone_thickness']
                * dict_orientation['orientation'][:, 0]
                + dict_pullout_force['force_per_root']
                * dict_orientation['dorientation_dshear_zone_thickness'][:, 0]
                ) / self.failure_surface.cross_sectional_area
            dresistance_dshear_zone_thickness = np.sum(
                dict_pullout_force['dforce_per_root_ddisplacement'] 
                * dict_pullout_disp['dpullout_displacement_dshear_zone_thickness']
                * dict_orientation['orientation'][:, 2]
                + dict_pullout_force['force_per_root']
                * dict_orientation['dorientation_dshear_zone_thickness'][:, 2]
                ) * np.tan(soil_friction_angle) / self.failure_surface.cross_sectional_area
            output['dyield_value_dshear_zone_thickness'] = dload_dshear_zone_thickness - dresistance_dshear_zone_thickness
        return(output)

    def calc_reinforcement(
            self,
            max_shear_displacement: Parameter | Quantity | None = None,
            n: int = 100,
            algorithm: str = 'bracket',
            initial_shear_displacement: None | Quantity = None,
            initial_shear_zone_thickness: None | Quantity = None,
            results: str = "attribute"
            ):
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
            by default 100
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
        results : int | str
            Controls how results are returned, by default "attribute":
            * `results = "attribute" or `results = 0` adds calculated results to 
            the `output` dictionary attribute of the model instance.
            * `results = "return"` or `results = 1` returns the dictionary 
            instead. 
            * `results = "both"` or `results = 2` does both at the same time.

        Function creates a dictionary with the keys:

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
                (`n*p`) where `n` is the number of dispalcement steps and `p` 
                the number of behaviour types.
            'dreinforcement_ddisplacement': Quantity
                derivative of reinforcement output with respect to the shear 
                displacement. Only returned when `jacobian = True`. Has shape 
                (`n`) where `n` is the number of displacement steps.
        """
        if isinstance(max_shear_displacement, Parameter):
            max_shear_displacement = max_shear_displacement.value * units(max_shear_displacement.unit)
        if initial_shear_displacement is None:
            initial_shear_displacement = 0.0 * max_shear_displacement.units
        else:
            initial_shear_displacement = initial_shear_displacement.to(max_shear_displacement.units)

        shear_displacement = max_shear_displacement.units * np.linspace(
            initial_shear_displacement.magnitude,
            max_shear_displacement.magnitude,
            n)
        if initial_shear_zone_thickness is None:
            initial_shear_zone_thickness = self.failure_surface.shear_zone_thickness        
        shear_zone_thickness = np.full(n, initial_shear_zone_thickness.magnitude) * initial_shear_zone_thickness.units
        reinforcement = np.zeros(n) * units('kPa')

        soil_shear_strength = self.soil_profile.calc_shear_strength(
            self.failure_surface.depth,
            orientation = np.cos(self.failure_surface.elevation_angle) if hasattr(self.failure_surface, 'elevation_angle') else 0.0
            )
        soil_friction_angle = self.soil_profile.get_soil(self.failure_surface.depth).friction_angle
        
        # loop through all displacement steps
        for i in np.arange(n):
            if shear_displacement[i].magnitude > 0.0:
                # calculate results
                dict_res = self.calc_single_step(
                    shear_displacement[i],
                    shear_zone_thickness[i - 1],
                    soil_shear_strength,
                    soil_friction_angle,
                    jacobian = False
                    )
                if dict_res['yield_value'].magnitude < 0.0:
                    # stable -> assign output
                    reinforcement[i, ...] = dict_res['reinforcement']
                    shear_zone_thickness[i] = shear_zone_thickness[i - 1]
                else:
                    if np.isclose(shear_zone_thickness[i - 1], self.failure_surface.max_shear_zone_thickness):
                        # shear zone at max thickness
                        reinforcement[i, ...] = dict_res['reinforcement']
                        shear_zone_thickness[i] = shear_zone_thickness[i - 1]
                    else:
                        # check if possible to get a stable shear plane at the maximum shear zone thickness
                        dict_max = self.calc_single_step(
                            shear_displacement[i],
                            self.failure_surface.max_shear_zone_thickness,
                            soil_shear_strength,
                            soil_friction_angle,
                            jacobian = False
                            )
                        if dict_max['yield_value'].magnitude >= 0.0:
                            # unstable at max -> set shear zone to shear_zone_max
                            reinforcement[i, ...] = dict_max['reinforcement']
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
                                    dict_res = self.calc_single_step(
                                        shear_displacement[i],
                                        x * units('mm'),
                                        soil_shear_strength,
                                        soil_friction_angle,
                                        jacobian = True
                                        )
                                    return(
                                        dict_res['yield_value'].magnitude,
                                        dict_res['dyield_value_dshear_zone_thickness'].magnitude
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
                            dict_solved = self.calc_single_step(
                                shear_displacement[i],
                                shear_zone_thickness[i],
                                soil_shear_strength,
                                soil_friction_angle,
                                jacobian = False
                                )
                            reinforcement[i, ...] = dict_solved
        
        output = {
            'displacement': shear_displacement,
            'reinforcement': reinforcement,
            'shear_zone_thickness': shear_zone_thickness
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
            n: int = 51,
            passes: int = 3,
            results: str = "attribute"
            ):
        if np.isfinite(self.failure_surface.max_shear_zone_thickness):
            shear_displacement_max = max(self.calc_displacement_to_rootpeak(self.failure_surface.max_shear_zone_thickness))
        else:
            shear_displacement_max = 1.5 * max(self.calc_displacement_to_rootpeak(self.failure_surface.shear_zone_thickness))
        shear_displacement_min = 0.0 * shear_displacement_max.units
        shear_zone_thickness = self.failure_surface.shear_zone_thickness
        for i in np.arange(passes):
            dict_res = self.calc_reinforcement(
                shear_displacement_max, 
                n = n,
                initial_shear_displacement = shear_displacement_min,
                initial_shear_zone_thickness = shear_zone_thickness
                )
            index_peak = np.argmax(dict_res['reinforcement'])
            if index_peak == 0:
                index_previous = 0
                index_next = 1
            elif index_peak == (n - 1):
                index_previous = index_peak - 1
                index_next = index_peak
            else:
                index_previous = index_peak - 1
                index_next = index_peak + 1
            shear_displacement_min = dict_res['displacement'][index_previous]
            shear_displacement_max = dict_res['displacement'][index_next]
            shear_zone_thickness = dict_res['shear_zone_thickness'][index_previous]
        output = {
            'displacement': dict_res['displacement'][index_peak],
            'reinforcement': dict_res['reinforcement'][index_peak]
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