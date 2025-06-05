# load
import csv
from pyrootmemo.materials import MultipleRoots
from collections import namedtuple

# set parameter named tuple
Parameter = namedtuple("parameter", "value unit")


# read csv with units to dictionary  - CRUDE FUNCTION, mostly written for SBEE2025 Workshop purposes (GJM)
def csv2dict(
        path: str,
        delimiter: str = ','
        ) -> dict:
    # read csv 
    # * file is assumed to have a single row containing headers
    # * each column header string is assumed to consist of 'Parameter name' + underscore + unit
    # * outputs dictionary with parameter names as keys, and a 'values' + 'units' dictionary as values
    # * 
    with open(path, newline = '') as csvfile:
        # read all csv data
        reader = csv.reader(csvfile, delimiter = delimiter, quotechar = '|')
        items = []
        for row in reader:
            items.append(row)
        # split column headers into parameters and units
        headers = items[0]
        parameters = [i[: i.rindex('_')] for i in headers]
        units = [i.split('_')[-1] for i in headers]
        if not isinstance(headers, list):
            headers = [headers]
            units = [units]
            data = [[i] for i in items[1:, ]]
        else:
            data = list(map(list, zip(*items[1: ])))  # (list of list of strings)
        # return  dictionary with parameters, values and units
        return({p: {'values': d, 'units': u} for p, d, u in zip(parameters, data, units)})


# Read root data from a csv file and generate MultipleRoots object
def read_csv_roots(
        path: str,
        delimiter: str = ',',
        species: str = 'test_species'
        ) -> MultipleRoots:
    """Load MultipleRoots data from a .csv file

    The file is assumed to only contain numerical data that can be converted
    to floats. The column headers (top row of the csv file) must be specified
    as <parameter name>_<unit>, e.g. "diameter_mm" or "tensile_strength_MPa".

    Parameters
    ----------
    path : str
        path to .csv file to load
    delimiter : str, optional
        csv delimiter, by default ','
    species : str, optional
        name of the plant species, by default 'test_species'

    Returns
    -------
    MultipleRoots
        Root properties
    """

    # it is assumed that **all** data is convertable to numeric type
    # get dictionary with data
    dic_raw = csv2dict(path, delimiter = delimiter)
    # convert to parameter type
    dic = {k: Parameter([float(i) for i in v['values']], v['units']) for k, v in dic_raw.items()}
    # add species name
    dic['species'] = species
    # return MultipleRoots object
    return(MultipleRoots(**dic))