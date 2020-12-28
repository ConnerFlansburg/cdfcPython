""" This file contains the distance functions used by CDFC & exists so that the distance calculation is more modular.
    Currently Distances.py supports the Correlation, Czekanowski, and Euclidean functions.
    Note that when possible, SciPy libraries should often be replaced with Numpy equivalents as they tend to be faster.
    
    Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

    Github Repo: https://github.com/brom94/cdfcPython.git
"""
import logging as log
import typing as typ
import numpy as np
import scipy.spatial.distance as sp


def computeDistance(func: str, Vi: typ.List[float], Vj: typ.List[float]) -> typ.Union[float, int]:
    """ Computes the distance using the provided distance function.

    :param func: Lowercase string name of the function to use.
    :type func: str
    :param Vi: First 1d vector
    :type Vi: typ.List[float]
    :param Vj: Second 1d vector
    :type Vj: typ.List[float]
    :return: vector of distance values
    :rtype: typ.Union[float, int]
    """
    
    if func == "czekanowski":  # if the function provided was Czekanowski,
        return __Czekanowski(Vi, Vj)  # run Czekanowski
    
    elif func == "euclidean":  # if the euclidean distance was requested
        return __Euclidean(Vi, Vj)
    
    elif func == "correlation":  # if the correlation distance/value was requested
        return sp.correlation(Vi, Vj)
    
    else:  # if no valid distance function was provided, default to the euclidean distance
        return __Euclidean(Vi, Vj)


def __Euclidean(Vi: typ.List[float], Vj: typ.List[float]) -> typ.Union[float, int]:
    """ Used to by computeDistance to calculate the Euclidean distance.
    
    :param Vi: First 1d vector
    :type Vi: typ.List[float]
    :param Vj: Second 1d vector
    :type Vj: typ.List[float]
    :return: vector of distance values
    :rtype: typ.Union[float, int]
    """
    
    # convert the vectors to numpy arrays
    Ai: np.array = np.array(Vi)
    Aj = np.array(Vj)
    
    # compute the distance & return
    return np.linalg.norm(Ai - Aj)


def __Czekanowski(Vi: typ.List[float], Vj: typ.List[float]) -> typ.Union[float, int]:
    """ Used to by computeDistance to calculate the Czekanowski distance.

    :param Vi: First 1d vector
    :type Vi: typ.List[float]
    :param Vj: Second 1d vector
    :type Vj: typ.List[float]
    :return: vector of distance values
    :rtype: typ.Union[float, int]
    """
    
    log.debug('Starting Czekanowski() method')
    
    # ************************** Error checking ************************** #
    # ! For Debugging Only
    # try:
    #     if len(Vi) != len(Vj):
    #         log.error(f'In Czekanowski Vi[d] & Vi[d] are not equal Vi = {Vi}, Vj = {Vj}')
    #         raise Exception(f'ERROR: In Czekanowski Vi[d] & Vi[d] are not equal Vi = {Vi}, Vj = {Vj}')
    #     if None in Vi:
    #         log.error(f'In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
    #         raise Exception(f'ERROR: In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
    #     if None in Vj:
    #         log.error(f'In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
    #         raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
    # except Exception as err:
    #     lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
    #     printError(str(err) + f', line = {lineNm}')
    #     sys.exit(-1)  # recovery impossible, exit with an error
    # ******************************************************************** #
    
    top: typ.Union[int, float] = 0
    bottom: typ.Union[int, float] = 0
    
    # + range(len(self.features)) loops over the number of features the hypothesis has.
    # + Vi & Vj are lists of the instances from the original data, that have been transformed
    # + by the hypothesis. We want to loop over them in parallel because that we will compare
    # + feature 1 in Vi with feature 1 in Vj
    
    for i, j in zip(Vi, Vj):  # zip Vi & Vj so that we can iterate in parallel
        
        # **************************************** Error Checking **************************************** #
        # ! For Debugging Only
        # if Vi == [None, None] and Vj == [None, None]:
        #     log.error(f'In Czekanowski Vj ({Vj}) & Vi ({Vi}) was found to be a \'None type\'')
        #     raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) & Vi ({Vi}) was found to be a \'None type\'')
        #
        # elif Vj == [None, None]:
        #     log.error(f'In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
        #     raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
        #
        # elif Vi == [None, None]:
        #     log.error(f'In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
        #     raise Exception(f'ERROR: In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
        # ************************************************************************************************ #
        
        top += min(i, j)  # get the top of the fraction
        bottom += (i + j)  # get the bottom of the fraction
    # exit loop
    
    try:  # attempt division
        
        value = 1 - ((2 * top) / bottom)  # create the return value
    
    except ZeroDivisionError:
        # resolve 0/0 case
        if top == 0:  # if the numerator & the denominator both 0, set to zero
            value = 0
        # resolve the n/0 case
        else:  # if only the denominator is 0, set it to a small number
            bottom = pow(10, -6)  # + Experiment with this value
            value = 1 - ((2 * top) / bottom)  # create the return value
    
    log.debug('Finished Czekanowski() method')
    
    return value
