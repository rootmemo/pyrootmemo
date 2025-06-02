# load
import csv
from pyrootmemo.materials import MultipleRoots
from collections import namedtuple

# set parameter named tuple
Parameter = namedtuple("parameter", "value unit")

# read csv with units
def csv2dict(
        path
        ):
    # read csv 
    # * file is assumed to have a single row containing headers
    # * each column header string is assumed to consist of 'Parameter name' + underscore + unit
    # * outputs dictionary with parameter names as keys, and a 'values' + 'units' dictionary as values
    # * 
    with open(path, newline = '') as csvfile:
        # read all csv data
        # dialect = csv.Sniffer().sniff(csvfile.read(1024))
        # csvfile.seek(0)
        # reader = csv.reader(csvfile, dialect)
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
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
        path,
        species = 'test_species'
        ):
    # it is assumed that all data is convertable to numeric type
    # get dictionary with data
    dic_raw = csv2dict(path)
    # convert to parameter type
    dic = {k: Parameter([float(i) for i in v['values']], v['units']) for k, v in dic_raw.items()}
    # add species name
    dic['species'] = species
    # return MultipleRoots object
    return(MultipleRoots(**dic))