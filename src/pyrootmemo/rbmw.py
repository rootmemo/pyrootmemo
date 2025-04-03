# import packages and functions
from scipy.special import gamma
from scipy.optimize import root_scalar
import numpy as np
from pyrootmemo.tools.helpers import units
from pint import Quantity
import matplotlib.pyplot as plt
from pyrootmemo.utils_plot import round_range


# RBMw class
class Rbmw():
       
    # initialise class
    def __init__(
            self, 
            roots,
            weibull_shape: float | int, 
            weibull_scale = None
            ):
        """
        Initiate RBMw bundle model class

        Parameters
        ----------
        roots : instance of MultipleRoots class. 
            Must contain fields 'diameter', 'xsection', 'tensile_strength', 'length', 'elastic_modulus'
        weibull_shape : float
            Weibull shape parameter

        Returns
        -------
        None.

        """
        # set parameters
        self.roots = roots
        self.weibull_shape = weibull_shape
        # calculate weibull scale parameter
        if weibull_scale is None: 
            self.weibull_scale = 1. / gamma(1. + 1. / weibull_shape)
    
    # survival fraction - Weibull
    def _weibull_survival(
            self, 
            x: int | float,
            deriv: int = 0
            ):
        # x = relative capacity of root strength mobilised (ratio of stress / average strength)
        # survival fraction
        if deriv == 0:
            return(np.exp(-(x / self.weibull_scale) ** self.weibull_shape))
        elif deriv == 1:
            return(
                -self.weibull_shape / self.weibull_scale
                * (x / self.weibull_scale) ** (self.weibull_shape - 1)
                * np.exp(-(x / self.weibull_scale) ** self.weibull_shape)
                )

    # forces in roots at current level of axial displacement
    def force(
            self,
            displacement,
            total: bool = True,
            deriv: int = 0
            ):
        # force in each root - ignoring failure
        strain = displacement / self.roots.length
        force_unbroken = self.roots.xsection * self.roots.elastic_modulus * strain
        # tensile capacity
        capacity = self.roots.xsection * self.roots.tensile_strength
        # survival fraction
        survival = self._weibull_survival(force_unbroken / capacity)
        # force per root
        if deriv == 0:
            # average force per root
            values = force_unbroken * survival
        # derivative of force per root
        elif deriv == 1:
            dstrain_ddisplacement = 1. / self.roots.length
            dforceunbroken_ddisplacement = self.roots.xsection * self.roots.elastic_modulus * dstrain_ddisplacement
            dsurvival_dforceunbroken = self._weibull_survival(force_unbroken / capacity, deriv = 1) / capacity
            dsurvival_ddisplacement = dsurvival_dforceunbroken * dforceunbroken_ddisplacement
            values = dforceunbroken_ddisplacement * survival + force_unbroken * dsurvival_ddisplacement
        # return
        if total is True:
            return(np.sum(values))
        else:
            return(values)

    # Calculate peak force 
    def peak_force(
            self
            ) -> float:        
        # displacements until peak reinforcement of each root
        displacement_peaks = (
            self.roots.tensile_strength / self.roots.elastic_modulus * self.roots.length
            / self.weibull_shape ** (1. / self.weibull_shape)
            * self.weibull_scale
        )
        displacement_peaks = np.sort(np.unique(displacement_peaks.magnitude)) * displacement_peaks.units
        # force at each peak
        forces = np.array([self.force(ui, total = True, deriv = 0).magnitude for ui in displacement_peaks])
        # get force gradients at each of these displacements
        gradients = [self.force(ui, total = True, deriv = 1).magnitude for ui in displacement_peaks]
        # forwards and backwards predictions
        disp_diff = np.diff(displacement_peaks.magnitude)
        forces_forwards = np.append(forces[:-1] + disp_diff * gradients[:-1], 0.0)
        forces_backwards = np.append(0.0, forces[1:] - disp_diff * gradients[1:])
        forces_max = np.max(np.row_stack((forces_forwards, forces_backwards)), axis = 0)
        # select which starting guesses to keep (max force that can be achieved from starting point must be larger than max from <forces> array)
        keep = (forces_max >= np.max(forces))
        # starting guesses for displacement at peak
        displacement_guesses = displacement_peaks[keep]
        # find displacement at peaks, for each of the starting guesses, using root solving
        unit = displacement_guesses.units
        displacement_options = np.array([
            root_scalar(
                lambda x: self.force(x * unit, total = True, deriv = 1).magnitude,
                x0 = i.magnitude
            ).root
            for i in displacement_guesses
        ]) * unit
        # calculate forces at each displacement option (local peaks)
        peak_force_options = [
            self.force(i, total = True, deriv = 0)
            for i in displacement_options
        ]
        # index with maximum force (global peak)
        index = np.argmax(np.array([i.magnitude for i in peak_force_options]))
        # return max force, and displacement at max force
        return(
            peak_force_options[index],
            displacement_options[index]            
        )

    # reinforcement
    def peak_reinforcement(
            self, 
            soil_area = 1.0 * units("m^2"),
            k: float = 1.0
            ) -> float:
        """
        Calculate peak reinforcement (largest soil reinforcement at any point)
        generated by the fibre bundle

        Parameters
        ----------
        soil_area : float, optional
            Soil cross-sectional area that contains the roots defined. 
            The default is 1.0 m^2.
        k : float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.0.

        Returns
        -------
        float
            peak root reinforcement.

        """
        # convert area
        if not isinstance(soil_area, Quantity):
            if isinstance(soil_area, int) or isinstance(soil_area, float):
                Warning("soil area unit not defined - assumed as m^2")
                soil_area = soil_area * units("m^2")
            else:
                TypeError("soil area must be defined as integer or float")
        # return
        return(k * self.peak_force() / soil_area)
    
    
    # reduction factor
    def reduction_factor(
            self
            ) -> float:
        """
        Calculate the ratio between bundle peak force and the sum of 
        individual fibre strengths. Function will thus return a value between
        0 and 1. '1' indicates all roots break simultaneously.

        Returns
        -------
        float
            reduction factor.

        """
        force_rbmw = self.peak_force()
        force_root = np.sum(self.roots.xsection * self.roots.tensile_strength)
        return(force_rbmw / force_root)
    
    
    def plot(
            self,
            n = 101,
            fraction = 0.9,  # minimum fraction of roots broken in each diameter,
            xlabel: chr = 'Displacement', 
            ylabel: chr = 'Total force in root bundle',
            xunit = 'mm',
            yunit = 'N',
            peak = True
            ):
        # calculate peak force and displacement
        peak_force, peak_displacement = self.peak_force()
        # displacement to average root failure
        displacement_average = self.roots.tensile_strength / self.roots.elastic_modulus * self.roots.length
        displacement_max = (
            np.max(displacement_average) 
            * self.weibull_scale
            * (-np.log(1. - fraction)) ** (1. / self.weibull_shape)
        )
        # displacement range
        displacement = np.linspace(0, displacement_max, n)
        # calculate forces
        force = [self.force(i, total = True) for i in displacement]
        # generate plot object
        fig, ax = plt.subplots()
        # convert values to magnitudes, for plotting
        peak_displacement_magnitude = peak_displacement.to(xunit).magnitude
        peak_force_magnitude = peak_force.to(yunit).magnitude
        displacement_magnitude = displacement.to(xunit).magnitude
        force_magnitude = np.array([i.to(yunit).magnitude for i in force])
        # plot line
        ax.plot(displacement_magnitude, force_magnitude, '-')
        # plot peak reinforcement
        if peak is True:
            plt.scatter(peak_displacement_magnitude, peak_force_magnitude)
        # axis limits
        ax.set_xlim(round_range(displacement_max.to(xunit).magnitude, limits = [0, None])['limits'])
        ax.set_ylim(round_range(peak_force.to(yunit).magnitude, limits = [0., None])['limits'])
        # axis labels
        ax.set_xlabel(xlabel + ' [' + xunit + ']')
        ax.set_ylabel(ylabel + ' [' + yunit + ']')
        # return
        return(fig, ax)


    # # plot
    # def plot(
    #         self,
    #         labels: bool = True, 
    #         margin: float = 0.05, 
    #         xlabel: chr = 'Displacement', 
    #         ylabel: chr = 'Total force in root bundle',
    #         xunit: str = "N"
    #         ) -> tuple:
    #     """
    #     Generate a matplotlib plot showing how forces in each roots are 
    #     mobilised.

    #     Parameters
    #     ----------
    #     labels : bool, optional
    #         labels for individual roots, If False, no labels are plotted. If
    #         True, labels are plotted as numbers indicating the order in which
    #         roots are defined in the input. Can be defined as a list of 
    #         character strings to specify individual labels for each root.
    #         The default is True.
    #     margin : float, optional
    #         controls the location for plotting labels. Defined as the fraction
    #         of the x-axis size. Labels are plotted on the right-hand size of
    #         the force triangles, and centred vertically. The default is 0.05.
    #     xlabel : chr, optional
    #         x-axis label. The default is 'Force in reference root'.
    #     ylabel : chr, optional
    #         y-axis label. The default is 'Total force in root bundle'.

    #     Returns
    #     -------
    #     tuple
    #         Tuple containing a figure and an axis object.

    #     """
    #     # units
    #     #force_unit = self._tensile_capacity().to_reduced_units().u
    #     #force_unit_str = format(force_unit, '~L')
    #     # number of roots
    #     n_root = len(self.roots.diameter)
    #     # sorted caa
    #     capacity_sorted = (self._tensile_capacity().to(unit))[self.sort_order]
    #     diameter_sorted = self.roots.diameter[self.sort_order]
    #     # convert to unit of choice
    #     # x-values at breakage of each root (forces in reference root = first root defined in list)
    #     xb = capacity_sorted * (self.roots.diameter[0] / diameter_sorted) ** self.load_sharing
    #     xb = xb.magnitude
    #     # array with all x-values (start + before and after breakage)
    #     dx = 1.e-6
    #     x = np.append(0., np.repeat(xb, 2) * np.tile(np.array([1., 1. + dx]), n_root))
    #     # forces in each root (start + before and after breakage)
    #     y = np.hstack((np.zeros((n_root, 1)), np.repeat(self.matrix.to(unit).magnitude, 2, axis = 1)))
    #     y[np.arange(n_root), 2 * np.arange(n_root) + 2] = 0.
    #     # reverse y-values (forces), so plot lines are stacked in order from last breaking to first breaking
    #     y = np.flip(y, axis = 0)
    #     # colour order - use default matplotlib colors, but order in which roots are defined
    #     prop_cycle = mpl.rcParams['axes.prop_cycle']
    #     colors = prop_cycle.by_key()['color']
    #     n_color = len(colors)
    #     colors_new = np.array(colors * int(np.ceil(n_root / n_color)))[np.flip(self.sort_order)]
    #     # create new figure 
    #     fig, ax = plt.subplots()
    #     # plot figure
    #     ax.stackplot(x, y, colors=colors_new)
    #     ax.set_xlabel(xlabel + " [" + unit + "]")
    #     ax.set_ylabel(ylabel + " [" + unit + "]")
    #     ax.set_xlim(round_range(x, limits = [0, None])['limits'])
    #     ax.set_ylim(round_range(self.peak_force().to(unit).magnitude, limits = [0., None])['limits'])
    #     # label text
    #     if labels is True:
    #         labels = self.sort_order + 1
    #         plot_labels = True
    #     elif isinstance(labels, list):
    #         labels = np.array(labels)[self.sort_order]
    #         plot_labels = True
    #     else:
    #         plot_labels = False
    #     # add labels to plot
    #     if plot_labels is True:
    #         # labels positions
    #         labels_x = xb - margin*np.max(xb)
    #         labels_y = ((np.sum(self.matrix.to(unit).magnitude, axis = 0) 
    #                     - 0.5 * np.diag(self.matrix.to(unit).magnitude))
    #                     * (labels_x/xb))
    #         # add to plot
    #         for xi, yi, li in zip(labels_x, labels_y, labels):
    #             ax.annotate(
    #                 li, xy = (xi, yi), 
    #                 ha = 'center', 
    #                 va = 'center', 
    #                 bbox = dict(boxstyle = 'round', fc = 'white', alpha = 0.5),
    #                 fontsize = 'small'
    #                 )
    #     # return figure
    #     return(fig, ax)
        

############
### FBMC ###
############

class Fbmc:

    # initialise class
    def __init__(
            self, 
            area,   # power law distribution: diameter--root crosssectional area
            strength,  # power law: diameter--root tensile strength
            load_sharing: float | int,
            root_area_ratio = 1.0
            ):
        # set parameters
        self.load_sharing = load_sharing
        # check diameter - must be power law distribution
        area_types_accepted = [PowerFit, PowerFitBinned]
        if not any([isinstance(area, i) for i in area_types_accepted]):
            raise TypeError(f'diameter distribution must be of type {area_types_accepted}')
        else:
            self.area = area
        # check tensile strength power law
        self.strength = strength

    # calculate zeta factors
    def zeta(self):
        z1 = ((2. + self.strength.exponent - self.loadsharing)
              * np.log(self.area.upper / self.area.lower))
        z2 = ((1. + self.area.exponent + self.rar.exponent)
              * np.log(self.area.upper / self.area.lower))
        return(np.array([z1, z2]))
    
    # FBM reduction factor - at peak or at defined strain levels
    def reduction_factor(
            self, 
            strain: np.ndarray = None
            ):
        # get zeta values
        z1, z2 = self.zeta()
        # decide on case - strain defined or not
        if strain is None:
            ## return strain and reinfocement at peak
            # if zeta1 < 0, swap signs for zeta1 and zeta2
            if z1 < 0.0:
                z1 = -z1
                z2 = -z2
            # strength reduction factor, based on zeta-values
            if z1 > 0.:
                if z2 > 0.:
                    if np.isclose(z1, z2):
                        if z1 <= 1.:
                            reduction = z2 * np.exp(-z2) / (1. - np.exp(-z2))
                        else:
                            reduction = np.exp(-1.) / (1. - np.exp(-z2))
                    else:
                        if (z1 / z2)**(1. / (z2 - z1) >= np.exp(1.)):
                            reduction = (z1 / z2)**(z1 / (z2 - z1)) / (1. - np.exp(-z2))
                        else:
                            reduction = z2 / (z1 - z2) * (np.exp(-z2) - np.exp(-z1)) / (1. - np.exp(-z2))
                elif (z2 == 0.):
                    reduction = (1. - np.exp(-z1)) / z1
                else:
                    reduction = z2 / (z1 - z2) * (np.exp(-z2) - np.exp(-z1)) / (1. - np.exp(-z2))
            else:
                reduction = 1.            
            ## strain at peak
            # breakage strains of smallest and largest root
            strain_min = (
                (self.area.lower / self.area.x0)
                ** (2. + self.strength.exponent - self.loadsharing)
                )
            strain_max = (
                (self.area.upper / self.area.x0)
                ** (2. + self.strength.exponent - self.loadsharing)
                )
            # peak strain multiplier
            if (z1 * z2) <= 0:
                if strain_max > strain_min:
                    tmp1 = np.exp(-z1)
                else:
                    tmp1 = np.exp(z1)
            else:
                tmp1 = (z1 / z2) ** (z1 / (z2 - z1))
            # temp value 2
            if strain_max > strain_min:
                tmp2 = np.exp(-z1)
            else:
                tmp2 = np.exp(z1)
            # strain at peak reinforcement
            strain_peak = max(strain_min, strain_max) * max(tmp1, tmp2)
            # return
            return(strain_peak, reduction)            
        else:
            ## return value at specific values of strain. 
            ## `strain' is defined as the ratio between strain applied to the 
            ## reference root and its failure strain
            # WWM part (divided by k*tru0*phir0)
            if np.isclose(self.strength.exponent + self.area.exponent, -1.):

    # -> fix reference diameters, may be different for 'area' and 'strength'
                wwm = self.reference_diameter*np.log(self.rar.upper/self.rar.lower)
            else:
                wwm = ((self.rar.upper*(self.rar.upper/self.reference_diameter)
                        **(self.tensile.exponent + self.rar.exponent)
                       - self.rar.lower*(self.rar.lower/self.reference_diameter)
                       **(self.tensile.exponent + self.rar.exponent))
                       / (1. + self.tensile.exponent + self.rar.exponent))
            # calculate strain to failure in thicknest and thinnest root
            strain_min = (
                (self.area.lower / self.area.x0)
                ** (2. - self.loadsharing + self.strength.exponent)
                )
            strain_max = (
                (self.area.upper / self.area.x0)
                ** (2. - self.loadsharing + self.strength.exponent)
                )
            # calculate reinforcements
            if np.isclose(self.loadsharing, 2. + self.strength.exponent) or np.isclose(self.area.lower, self.area.upper):
                ## WWM solution - all roots break at the same time
                return(strain / strain_min * (strain <= strain_min))
            else:
                ## FBMc solution 
                if (self.loadsharing < (2. + self.strength.exponent)):
                    ## Thin roots will break first
                    # domains
                    t1 = (strain > 0.) & (strain <= strain_min)
                    t2 = (strain > strain_min) & (strain < strain_max)
                    # lower diameter integration limit
                    d1 = self.area.upper * np.ones(len(strain))
                    d1[t1] = self.area.lower
                    d1[t2] = (
                        (self.area.x0 * strain[t2])
                        ** (-1. / (self.loadsharing - 2. - self.strength.exponent))
                        )
                    # upper diameter integration limit
                    d2 = self.area.upper
                else:
                    ## Thick roots will break first
                    # domains
                    t1 = (strain > 0.) & (strain <= strain_max)
                    t2 = (strain > strain_max) & (strain < strain_min)
                    # lower diameter integration limit
                    d1 = self.rar.lower
                    # upper diameter integration limit
                    d2 = self.rar.lower*np.ones(len(strain))
                    d2[t1] = self.rar.upper
                    d2[t2] = (self.reference_diameter*strain[t2]
                              **(-1./(self.loadsharing - 2. - self.tensile.exponent)))
                # integrate over all intact diameters
                if np.isclose(self.loadsharing - 1. + self.rar.exponent, 0.):
                    fbmc = self.reference_diameter*np.log(self.rar.upper/self.rar.lower)*strain
                else:
                    fbmc = (1./(self.loadsharing - 1. + self.rar.exponent)
                            * (d2*(d2/self.reference_diameter)
                               **(self.loadsharing- 2. + self.rar.exponent) 
                               - d1*(d1/self.reference_diameter)
                               **(self.loadsharing - 2. + self.rar.exponent))
                            * strain)
                # return scaled value
                return(fbmc/wwm)