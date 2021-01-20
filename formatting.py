"""
formatting.py is a library of data formatting & reshaping utilities used by both cdfc & cdfcProject.
It should only be called from within the project.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import logging as log
import sys
import traceback
import typing as typ

# from pprint import pprint
import numpy as np
import pandas as pd

# console formatting strings
HDR = '*' * 6
SUCCESS = u' \u2713\n'
OVERWRITE = '\r' + HDR
SYSOUT = sys.stdout


def printError(message: str) -> None:
    """
    printError is used for coloring error messages red.
    
    :param message: The message to be printed.
    :type message: str
    
    :return: printError does not return, but rather prints to the console.
    :rtype: None
    """
    print("\033[91;1m {}\033[00m" .format(message))


def flattenTrainingData(trainList: typ.List[typ.List[np.ndarray]]) -> np.ndarray:
    """
    __flattenTrainingData reduces the dimension of the provided training data. These additional
    dimensions are added during cross fold validation and are not needed by other routines/functions.
    This function turns a list of lists of NumPy arrays into a NumPy n-dimensional array.
    
    :param trainList: The current training data.
    :type trainList: List[List[np.ndarray]]
    
    :return: The reduced training data.
    :rtype: np.ndarray
    """
    
    train = []  # currently training is a list of lists of lists because of the buckets.
    for lst in trainList:  # we can now remove the buckets by concatenating the lists of instance
        train += lst  # into one list of instances, flattening our data, & making it easier to work with
    
    # __transform the training & testing data into numpy arrays & free the List vars to be reused
    train = np.array(train)  # turn training data into a numpy array
    
    return train


def formatForSciKit(data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    __formatForSciKit takes the input data and converts it into a form that can
    be understood by the sklearn package.
    
    :param data: The input data, from a read in CSV.
    :type data: np.ndarray
    
    :return: The input file in a form parsable by sklearn.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    
    # create the label array Y (the target of our training)
    # from all rows, pick the 0th column
    try:
        # + data[:, :1] get every row but only the first column
        flat = np.ravel(data[:, :1])  # get a list of all the labels as a list of lists & then flatten it
        labels = np.array(flat)  # convert the label list to a numpy array
        # create the feature matrix X ()
        # + data[:, 1:] get every row but drop the first column
        ftrs = np.array(data[:, 1:])  # get everything BUT the labels/ids
    
    except (TypeError, IndexError) as err:
        lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        msg = f'{str(err)}, line {lineNm}:\ndata = {data}\ndimensions = {data.ndim}'
        log.error(msg)  # log the error
        printError(msg)  # print message
        traceback.print_stack()  # print stack trace
        sys.exit(-1)  # exit on error; recovery not possible
    
    return ftrs, labels


def buildAccuracyFrame(accuracyScores: typ.List[float], K: int, learningModel: str) -> pd.DataFrame:
    """
    __buildAccuracyFrame creates the Panda dataframe used by __accuracyFrameToLatex.
    
    :param accuracyScores: The accuracy list created by __buildModel() in cdfcProject
    :param K: The number of "buckets" (folds) used in K Fold Cross Validation.
    :param learningModel:
    
    :type accuracyScores:
    :type K: int
    :type learningModel: str
    
    :return: pd.DataFrame
    :rtype: pd.DataFrame
    """
    
    # *** Build the Labels *** #
    # this will create labels for each bucket ("fold") of the model, of the form "Fold i Accuracy"
    columnList = [f"Fold {i} Accuracy" for i in range(1, K + 1)]
    
    # this will create the labels for stats we will calculate and add them to the end of columnList
    stats: typ.List[str] = ["min", "median", "mean", "max"]
    columnList += stats
    
    # *** Calculate the Statistics *** #
    mn = min(accuracyScores)
    median = np.median(accuracyScores)
    mean = np.mean(accuracyScores)
    mx = max(accuracyScores)
    
    # put the calculated stats into a list
    statsList = [mn, median, mean, mx]
    accuracyScores += statsList  # add the stats to the accuracy scores

    # *** Build the Dataframe *** #
    try:
        # create the dataframe
        df = pd.DataFrame(accuracyScores, index=columnList, columns=[f'{learningModel} Accuracy'])
    
        # format the dataframe
        df *= 100         # turn the decimal into a percent
        df = df.round(1)  # round to 1 decimal place
        df = df.T         # transpose the dataframe, making it ready for latex
    
    except Exception as err:
        lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        msg: str = f'ERROR is formatting.py, line {lineNm}\n{str(err)}'  # create the message
        printError(msg)  # print the message
        print(f'columnList = {columnList}\nstatsList = {statsList}')  # print objects/data
        print(f'accuracyScores = {accuracyScores}')
        printError(traceback.format_exc())  # print stack trace
        sys.exit(-1)  # exit on error; recovery not possible
    
    return df
