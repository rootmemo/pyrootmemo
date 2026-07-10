import numpy as np
from pyrootmemo.geometry import FailureSurface
from pyrootmemo.materials import MultipleRoots
from pyrootmemo.helpers import Results, ResultsType

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
        MultipleRoots object containing properties of all roots considered for
        reinforcement calculations
    output : dict
        Dictionary with all calculation results
            
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
            MultipleRoots class. Must contain attributes 'diameter', 
            'xsection' and 'tensile_strength'.
        """
        if not isinstance(roots, MultipleRoots):
            raise TypeError('roots must be an object of class MultipleRoots')
        attributes_required = ['diameter', 'xsection', 'tensile_strength']
        for i in attributes_required:
            if not hasattr(roots, i):
                raise AttributeError('roots object must contain ' + str(i) + ' attribute')
        self.roots = roots
        self.output = {}


    def calc_peak_force(
            self,
            results: int | str = "attribute"
            ):
        """
        Calculates WWM peak force.

        This is defined as the sum of the maximum tensile forces that can 
        be mobilised in all roots.

        Function creates a dictionary with the key `peak_force`. How this is 
        returned depends on the value of the `results` argument.

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
        peak_force = np.sum(self.roots.xsection * self.roots.tensile_strength)
        output = {'peak_force': peak_force}
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
            k: int | float = 1.2,
            results: str = 'attribute'
            ):
        """
        Calculate peak reinforcement (largest soil reinforcement at any point)
        according to the WWM model

        Function creates a dictionary with the key `peak_reinforcement`. How 
        this is returned depends on the value of the `results` argument.
        
        Parameters
        ----------
        failure_surface : FailureSurface
            "FailureSurface" class object. Must contain the attribute 
            "cross_sectional_area" that contains the cross-sectinonal area 
            of the failure surface
        k : float, optional
            Wu/Waldron reinforcement orientation factor. The default is 1.2.
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
        if not (isinstance(k, int) | isinstance(k, float)):
            raise TypeError('k must be an scalar integer or float')
        if not 'peak_force' in self.output:
            self.calc_peak_force()
        output = {'peak_reinforcement': (
            k *  self.output['peak_force'] 
            / failure_surface.cross_sectional_area
            )}
        match Results(results).how:
            case ResultsType.ATTRIBUTE:
                self.output.update(output)
            case ResultsType.RETURN:
                return output
            case ResultsType.BOTH:
                self.output.update(output)
                return output
