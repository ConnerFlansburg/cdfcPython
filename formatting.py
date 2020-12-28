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

ModelList = typ.Tuple[typ.List[float], typ.List[float], typ.List[float]]  # type hinting alias


def printError(message: str) -> None:
    """
    printError is used for coloring error messages red.
    
    :param message: The message to be printed.
    :type message: str
    
    :return: printError does not return, but rather prints to the console.
    :rtype: None
    """
    print("\033[91m {}\033[00m" .format(message))


def __flattenTrainingData(trainList: typ.List[typ.List[np.ndarray]]) -> np.ndarray:
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


def __formatForSciKit(data: np.ndarray) -> (np.ndarray, np.ndarray):
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


def __buildAccuracyFrame(modelsTuple: ModelList, K: int) -> pd.DataFrame:
    """
    __buildAccuracyFrame creates the Panda dataframe used by __accuracyFrameToLatex.
    
    :param modelsTuple: The results of the classification models (KNN, Decision Tree, Naive Bayes)
    :param K: The number of "buckets" (folds) used in K Fold Cross Validation.
    
    :type modelsTuple: ModelList
    :type K: int
    
    :return: pd.DataFrame
    :rtype: pd.DataFrame
    """
    
    SYSOUT.write("Creating accuracy dataframe...\n")  # update user
    
    # get the models from the tuple
    knnAccuracy = modelsTuple[0]
    dtAccuracy = modelsTuple[1]
    nbAccuracy = modelsTuple[2]
    
    # SYSOUT.write(HDR + ' Creating dictionary ......')
    # use accuracy data to create a dictionary. This will become the frame
    accuracyList = {'KNN': knnAccuracy, 'Decision Tree': dtAccuracy, 'Naive Bayes': nbAccuracy}
    # SYSOUT.write(OVERWRITE + ' Dictionary created '.ljust(50, '-') + SUCCESS)
    
    # SYSOUT.write(HDR + ' Creating labels ......')
    # this will create labels for each instance ("fold") of the model, of the form "Fold i"
    rowList = [["Fold {}".format(i) for i in range(1, K + 1)]]
    # SYSOUT.write(OVERWRITE + ' Labels created successfully '.ljust(50, '-') + SUCCESS)
    
    # SYSOUT.write(HDR + ' Creating dataframe ......')
    # create the dataframe. It will be used both by the latex & the plot exporter
    df = pd.DataFrame(accuracyList, columns=['KNN', 'Decision Tree', 'Naive Bayes'], index=rowList)
    # SYSOUT.write(OVERWRITE + ' Dataframe created successfully '.ljust(50, '-') + SUCCESS)
    
    # SYSOUT.write('Accuracy Dataframe created without error\n')  # update user
    
    return df


def __accuracyFrameToLatex(modelsTuple: ModelList, df: pd.DataFrame) -> pd.DataFrame:
    """
    __accuracyFrameToLatex modifies a Panda dataframe, making it ready to be
    converted to Latex. It returns the modified dataframe, and does not convert
    it to Latex (so frame.to_latex() must still be called).
    
    :param modelsTuple: The results of the classification models (KNN, Decision Tree, Naive Bayes)
    :param df: The Panda dataframe to be converted.
    
    :type modelsTuple: ModelList
    :type df: pd.DataFrame
    
    :return: pd.DataFrame
    :rtype: pd.DataFrame
    """
    
    SYSOUT.write("\nConverting frame to LaTeX...\n")  # update user
    SYSOUT.write(HDR + ' Transposing dataframe')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    # transposing passes a copy, so as to avoid issues with plot (should we want it)
    frame: pd.DataFrame = df.transpose()  # update user
    
    SYSOUT.write(OVERWRITE + ' Dataframe transposed '.ljust(50, '-') + SUCCESS)
    
    # get the models from the tuple
    knnAccuracy = modelsTuple[0]
    dtAccuracy = modelsTuple[1]
    nbAccuracy = modelsTuple[2]
    
    SYSOUT.write(HDR + ' Calculating statistics...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    mn = [min(knnAccuracy), min(dtAccuracy), min(nbAccuracy)]  # create the new min,
    median = [np.median(knnAccuracy), np.median(dtAccuracy), np.median(nbAccuracy)]  # median,
    mean = [np.mean(knnAccuracy), np.mean(dtAccuracy), np.mean(nbAccuracy)]  # mean,
    mx = [max(knnAccuracy), max(dtAccuracy), max(nbAccuracy)]  # & max columns
    
    SYSOUT.write(OVERWRITE + ' Statistics calculated '.ljust(50, '-') + SUCCESS)  # update user
    
    SYSOUT.write(HDR + ' Adding statistics to dataframe...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    frame['min'] = mn  # add the min,
    frame['median'] = median  # median,
    frame['mean'] = mean  # mean,
    frame['max'] = mx  # & max columns to the dataframe
    
    SYSOUT.write(OVERWRITE + ' Statistics added to dataframe '.ljust(50, '-') + SUCCESS)  # update user
    SYSOUT.write(HDR + ' Converting dataframe to percentages...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    frame *= 100  # turn the decimal into a percent
    frame = frame.round(1)  # round to 1 decimal place
    
    SYSOUT.write(OVERWRITE + ' Converted dataframe values to percentages '.ljust(50, '-') + SUCCESS)  # update user
    
    return frame
