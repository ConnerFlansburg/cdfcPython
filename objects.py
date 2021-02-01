"""
objects.py contains several of the data structures used by both cdfcProject & cdfc. It exists primarily for
organizational purposes, rather than structural necessity.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import logging as log
import sys
import traceback
import typing as typ

import numpy as np

from formatting import printError

# OPS is the list of valid operations on the tree
OPS: typ.Final = ['add', 'subtract', 'times', 'max', 'if']


def countNodes() -> typ.Generator:
    """Used to generate unique node IDs."""
    
    num = 0
    while True:
        yield num
        num += 1


NodeCount = countNodes()  # create a generator for node count


def countTrees() -> typ.Generator:
    """Used to generate unique root IDs."""
    num = 1
    while True:
        yield num
        num += 1


TreeCount = countTrees()  # create a generator for tree count


class WrapperInstance:
    """
    Instance is an instance, or row, within our data set. It represents a single
    record of whatever we are measuring. This replaces the earlier row struct.
    
    :var className: The class id of the class that the record is in (this class ids start at 0).
    :var attributes: This dictionary stores the features values in the instance, keyed by index (these start at 0)
    :var vList: The list of the values stored in the dictionary (used to speed up iteration)
    
    :type className: int
    :type attributes: dict
    :type vList: list
    """
    
    def __init__(self, className: int, values: np.ndarray):
        # this stores the name of the class that the instance is in
        self.className: int = className
        # this stores the values of the features, keyed by index, in the instance
        self.attributes: typ.Dict[int, float] = dict(zip(range(values.size), values))
        # this creates a list of the values stored in the dictionary for when iteration is wanted
        self.vList: typ.List[float] = values.tolist()  # + this will be a list[float], ignore warnings
        
        # ! For Debugging Only
        # try:  # check that all the feature values are valid
        #     if None in self.vList:  # if a None is in the list of feature values
        #         raise Exception('Tried to create an Instance obj with a None feature value')
        # except Exception as err:
        #     log.error(str(err))
        #     tqdm.write(str(err))
        #     traceback.print_stack()
        #     sys.exit(-1)  # exit on error; recovery not possible
    
    def __array__(self) -> np.array:
        """Converts an Instance to an Numpy array."""
        return np.array([float(self.className)] + self.vList)


class cdfcInstance:
    """
    Instance is an instance, or row, within our data set. It represents a single
    record of whatever we are measuring. This replaces the earlier row struct.

    :var className: The class id of the class that the record is in (this class ids start at 0).
    :var attributes: This dictionary stores the features values in the instance, keyed by index (these start at 0)
    :var vList: The list of the values stored in the dictionary (used to speed up iteration)
    
    :type className: int
    :type attributes: dict
    :type vList: list
    """
    
    def __init__(self, className: int, attributes: typ.Dict[int, float], vList: typ.Optional[typ.List[float]] = None):
        # this stores the name of the class that the instance is in
        self.className: int = className
        # this stores the values of the features, keyed by index, in the instance
        self.attributes: typ.Dict[int, float] = attributes
        # this creates a list of the values stored in the dictionary for when iteration is wanted
        if vList is None:
            self.vList: typ.List[float] = list(self.attributes.values())  # ? is this to expensive?
        else:
            self.vList: typ.List[float] = vList
        
        try:  # check that all the feature values are valid
            if None in self.attributes.values():  # if a None is in the list of feature values
                raise Exception('Tried to create an Instance obj with a None feature value')
        except Exception as err:
            log.error(str(err))
            printError(str(err))
            traceback.print_stack()  # print stack trace so we know how None is reaching Instance
            sys.exit(-1)  # exit on error; recovery not possible
    
    def __array__(self) -> np.array:
        """Converts an Instance to an Numpy array."""
        return np.array([float(self.className)] + self.vList)
