"""
cdfcProject.py serves as the primary entry point into the project for wrapper main.py. It reads in CSV files, parses
them, creates constants, trains the models, creates & trains cdfc, and exports data. The primary purpose of cdfcProject
is to coordinate all other files in the project.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import collections as collect
import copy
import math
import os
import pickle
import time as time
# import traceback
import tkinter as tk
from pathlib import Path

# from tkinter import messagebox
from tkinter import filedialog
from alive_progress import config_handler
from pyfiglet import Figlet
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from cdfc import cdfc
from formatting import *
from formatting import buildAccuracyFrame, formatForSciKit, flattenTrainingData
from objects import WrapperInstance as Instance

# ********************************************* Constants used by Parser ********************************************* #
BETA: typ.Final = 2              # BETA is a constant used to calculate the pop size
R: typ.Final = 2                 # R is the ratio of number of CFs to the number of classes (features/classes)
PASSED_FUNCTION = None           # PASSED_FUNCTION is the distance function that was passed (defaults to Euclidean)
LEARN = None                     # LEARN is the type of learning model that should be used (defaults to KNN)
# ************************************************** Random Seeding ************************************************** #
SEED = 498
# random.seed(SEED)
# ********************************************* Constants used by Logger ********************************************* #
# create the file path for the log file & configure the logger
logPath = str(Path.cwd() / 'logs' / 'cdfc.log')
# log.basicConfig(level=log.ERROR, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')
# ******************************************** Constants used for Writing ******************************************** #
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'     # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout
# ****************************************** Constants used by Type Hinting ****************************************** #
# ! change back to 10 after testing
K: typ.Final[int] = 10  # set the K for k fold cross validation
ModelList = typ.Tuple[typ.List[float], typ.List[float], typ.List[float]]          # type hinting alias
ModelTypes = typ.Union[KNeighborsClassifier, GaussianNB, DecisionTreeClassifier]  # type hinting alias
ScalarsIn = typ.Union[None, StandardScaler]
# ****************************************** Configuration of Progress Bar ****************************************** #
config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', title_length=0, length=20)  # the global config for the loading bars
# config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', force_tty=True, title_length=0, length=10)  # the global config for the loading bars
# ******************************************************************************************************************** #


# * Next Steps
# TODO add doc strings
# TODO add more unit tests


def parseFile(train: np.ndarray) -> typ.Dict[any, any]:
    """ parseFile takes in training data as a numpy array, and parses it for CDFC. This
        involves getting the constants & creating the data structures that CDFC will need.
    
    Variables:
        train (numpy array): numpy array created by reading in the file.
 
    Return:
        constants (dictionary): dictionary with the constants & data structures that CDFC needs.
    """
    
    # + Remember this parses bucket(s) now, not files
    # *** the constants to be set * #
    INSTANCES_NUMBER: int = 0
    rows: typ.List[Instance] = []
    ENTROPY_OF_S: int = 0  # * used in entropy calculation * #
    CLASS_DICTS = {}
    # *** (the rest are set after the loop) *** #
    
    classes = []  # this will hold classIds and how often they occur
    classSet = set()  # this will hold how many classes there are
    classToOccur = {}  # maps a classId to the number of times it occurs
    ids = []  # this will be a list of all labels/ids with no repeats
    
    # set global variables using the now transformed data
    # each line in train will be an instances
    SYSOUT.write(HDR + ' Setting global variables...')
    for line in train:
        
        # parse the file
        name = line[0]
        # ? I am correctly setting the classId/name so that it works with using FEATURE_NUMBER?
        # ? Who do I get the Attribute ids? The must be the index of the attribute
        # ****************** Check that the ClassID/ClassName is an integer ****************** #
        try:
            if np.isnan(name):  # if it isn't a number
                raise Exception(f'ERROR: Parser expected an integer, got a NaN of value:{line[0]}')
            elif not (type(name) is int):  # if it is a number, but not an integer
                log.debug(f'Parser expected an integer class ID, got a float: {line[0]}')
                name = int(name)  # caste to int
        except ValueError:  # if casting failed
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Parse could not cast {name} to integer, line = {lineNm}')
            tqdm.write(f'ERROR: parser could not cast {name} to integer, line = {lineNm}')
        except Exception as err:  # catch NaN exception
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Parser expected an integer classId, got a NaN: {name}, line = {lineNm}')
            tqdm.write(str(err))
        # ************************************************************************************ #
        # now that we know the classId/className (they're the same thing) is an integer, continue parsing
        rows.append(Instance(name, line[1:]))  # reader[0] = classId, reader[1:] = attribute values
        classes.append(name)
        classSet.add(name)
        INSTANCES_NUMBER += 1
        
        # *** Create a Dictionary of Attribute Values Keyed by the Attribute's Index *** #
        # ! check that this is working correctly
        attributeNames = range(len(line[1:]))  # create names for the attributes number from 0 to len()
        attributeValues = line[1:]             # grab all the attribute values in this instance
        
        try:  # ++++ Do dictionary Creation & Modification Inside Try/Catch ++++
            if len(attributeValues) == len(attributeNames):  # check that the lengths of both names & values are equal
                # if they are equal, turn attribute values into a list of lists: [[value1],[value2],[value3],...]
                # this is so we can append values to that list without overwriting everytime
                attributeValues = [[val] for val in attributeValues]
                # create a new dictionary using (attributeName, attributeValue) pairs (in a tuple)
                newDict: typ.Dict[int, typ.List[float]] = dict(zip(attributeNames, attributeValues))
            else:  # if they are not equal, log it, throw an exception, and exit
                msg = f'Parser found more attribute names ({len(attributeNames)}) than values ({attributeValues})'
                log.error(msg)
                raise AssertionError(f'ERROR: {msg}')
            
            # *** Merge the Old Dictionary with the New One *** #
            # track how many unique/different class IDs there are & create dictionaries for
            # ? update() replaces old values so we have to loop over, can this be done faster?
            if name in ids:  # if we've found an instance in this class before
                oldDict = CLASS_DICTS[name]       # get the old dictionary for the class
                for att in oldDict.keys():        # we need to update every attribute value list so loop
                    oldDict[att] += newDict[att]  # for each value list, concatenate the old & new lists
            
            # *** Insert the Dictionary into CLASS_DICTS *** #
            else:  # if this is the first instance in the class
                CLASS_DICTS[name] = dict(newDict)  # insert the new dictionary using the key classID
                ids.append(name)                   # add classId to ids -- a list of unique classIDs
                
                log.debug(f'Parser created dictionary for classId {name}')  # ! for debugging
        
        except IndexError:                         # catch error thrown by dictionary indexing
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Parser encountered an Index Error on line {lineNm}. name={name}, line[0]={line[0]}')
            tqdm.write(f'ERROR: Parser encountered an Index Error on line {lineNm}. name={name}, line[0]={line[0]}')
            sys.exit(-1)                           # recovery impossible, exit
        except AssertionError as err:              # catch the error thrown by names/value length check
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            tqdm.write(str(err) + f', line = {lineNm}')
            sys.exit(-1)                           # recovery impossible, exit
        except Exception as err:                   # catch any other error that might be thrown
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            tqdm.write(str(err) + f', line = {lineNm}')
            sys.exit(-1)                           # recovery impossible, exit
        
        # ********* The Code Below is Used to Calculated Entropy  ********* #
        # this will count the number of times a class occurs in the provided data
        # dictionary[classId] = counter of times that class is found
        
        if classToOccur.get(name):      # if we have encountered the class before
            classToOccur[line[0]] += 1  # increment
        else:  # if this is the first time we've encountered the class
            classToOccur[line[0]] = 1  # set to 1
        # ****************************************************************** #
    
    CLASS_IDS = ids                           # this will collect all the feature names
    FEATURE_NUMBER = len(rows[0].attributes)  # get the number of features in the data set
    POPULATION_SIZE = FEATURE_NUMBER * BETA   # set the pop size
    LABEL_NUMBER = len(ids)                   # get the number of classes in the data set
    M = R * LABEL_NUMBER                      # get the number of constructed features
    
    # ********* The Code Below is Used to Calculated Entropy  ********* #
    # loop over all classes
    for i in classToOccur.keys():
        pi = classToOccur[i] / INSTANCES_NUMBER  # compute p_i
        ENTROPY_OF_S -= pi * math.log(pi, 2)     # calculation entropy summation
    # ***************************************************************** #

    # this dictionary will hold the parsed constants that will be sent to cdfc
    constants = {
        'FEATURE_NUMBER': FEATURE_NUMBER,
        'CLASS_IDS': CLASS_IDS,
        'POPULATION_SIZE': POPULATION_SIZE,
        'INSTANCES_NUMBER': INSTANCES_NUMBER,
        'LABEL_NUMBER': LABEL_NUMBER,
        'M': M,
        'rows': rows,
        'ENTROPY_OF_S': ENTROPY_OF_S,
        'CLASS_DICTS': CLASS_DICTS,
        'DISTANCE_FUNCTION': PASSED_FUNCTION,
    }

    # ! For Debugging Only
    # try:
    #     if not (type(constants) is dict):
    #         raise Exception(f'parseFile set constants to something other than a dictionary')
    # except Exception as err:
    #     SYSOUT.write(str(err) + f'\n constants = {constants}, \n feature num = {FEATURE_NUMBER}, \n'
    #                             f'class ids = {CLASS_IDS}, \n pop size = {POPULATION_SIZE}, \n')

    SYSOUT.write(OVERWRITE + ' Global variables set '.ljust(50, '-') + SUCCESS)
    return constants


def valuesInClass(classId: int, attribute: int, constants) -> typ.Tuple[typ.List[float], typ.List[float]]:
    """valuesInClass determines what values of an attribute occur in a class
        and what values do not

    Arguments:
        classId {String or int} -- This is the identifier for the class that
                                    should be examined
        attribute {int} -- this is the attribute to be investigated. It should
                            be the index of the attribute in the Instance
                            namedtuple in the rows list

    Returns:
        inClass -- This holds the values in the class.
        notInClass -- This holds the values not in the class.
    """
    CLASS_DICTS = constants['CLASS_DICTS']
    LABEL_NUMBER = constants['LABEL_NUMBER']
    CLASS_IDS = constants['CLASS_IDS']
    
    inDict = CLASS_DICTS[classId]                 # get the dictionary of attribute values for this class
    inClass: typ.List[float] = inDict[attribute]  # now get feature/attribute values that appear in the class
    
    # ******************** get all the values from all the other dictionaries for this feature ******************** #
    classes = CLASS_IDS.copy()  # make a copy of the list of unique classIDs
    classes.remove(classId)     # remove the class we're looking at from the list
    out: typ.List[float] = []   # this is a temporary variable that will hold the out list while it's constructed
    try:
        if LABEL_NUMBER == 2:          # * get the other score
            index = classes.pop(0)     # pop the first classId in the list (should be the only item in the list)
            spam = CLASS_DICTS[index]  # get the dictionary for the other class
            out = spam[attribute]      # get the feature/attribute values for the other class
            assert len(classes) == 0   # if classes is not empty now, an error has occurred
        
        elif LABEL_NUMBER == 3:        # * get the other 2 scores
            index = classes.pop(0)     # pop the first classId in the list (should only be 2 items in the list)
            spam = CLASS_DICTS[index]  # get the dictionary for the class
            out = spam[attribute]      # get the feature/attribute values for the class
            
            index = classes.pop(0)     # pop the 2nd classId in the list (should be the only item in the list)
            spam = CLASS_DICTS[index]  # get the dictionary for the class
            out += spam[attribute]     # get the feature/attribute values for the class
            assert len(classes) == 0   # if classes is not empty now, an error has occurred
        
        elif LABEL_NUMBER == 4:        # * get the other 3 scores
            index = classes.pop(0)     # pop the first classId in the list (should only be 3 items in the list)
            spam = CLASS_DICTS[index]  # get the dictionary for the class
            out = spam[attribute]      # get the feature/attribute values for the class
            
            index = classes.pop(0)     # pop the 2nd classId in the list (should only be 2 items in the list)
            spam = CLASS_DICTS[index]  # get the dictionary for the class
            out += spam[attribute]     # get the feature/attribute values for the class
            
            index = classes.pop(0)     # pop the 3rd classId in the list (should only be 1 items in the list)
            spam = CLASS_DICTS[index]  # get the dictionary for the class
            out += spam[attribute]     # get the feature/attribute values for the class
            assert len(classes) == 0   # if classes is not empty now, an error has occurred
        
        else:                          # * if there's more than 4 classes
            for i in CLASS_DICTS:      # loop over all the list of dicts
                if i == classId:       # when we hit the dict that's in the class, skip
                    continue
                else:
                    spam = CLASS_DICTS[i]   # get the dictionary for the class
                    out += spam[attribute]  # get the feature/attribute values for the class
        
        notInClass: typ.List[float] = out   # set the attribute values that do not appear in the class using out
    
    except AssertionError as err:              # catches error thrown by the elif statements
        lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
        log.error(f'ValuesInClass found more classIds than expected. Unexpected class(es) found: {classes}\n'
                  + f'Label Number = {LABEL_NUMBER}, Classes = {classes}, Class Id = {classId}, line = {lineNm}')
        tqdm.write(f'ValuesInClass found more classIds than expected. Unexpected class(es) found: {classes}')
        tqdm.write(str(err) + f'line{lineNm}')
        sys.exit(-1)
    # ************************************************************************************************************* #

    # ! For Debugging Only
    # try:
    #     if not inClass and not notInClass:
    #         log.debug('The valuesInClass method has found that both inClass & notInClass are empty')
    #         raise Exception('valuesInClass() found notInClass[] & inClass[] to be empty')
    #     elif not inClass:     # if inClass is empty
    #         log.debug('The valuesInClass method has found that inClass is empty')
    #         raise Exception('valuesInClass() found inClass[] to be empty')
    #     elif not notInClass:  # if notInClass is empty
    #         log.debug('The valuesInClass method has found that notInClass is empty')
    #         raise Exception('valuesInClass() found notInClass[] to be empty')
    # except Exception as err:
    #     lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
    #     tqdm.write(str(err) + f', line = {lineNm}')
    #     sys.exit(-1)  # exit on error; recovery not possible
    
    return inClass, notInClass  # return inClass & notInClass


def __discretization(data: np.ndarray) -> np.ndarray:
    
    for index, item in np.ndenumerate(data):

        if item < -0.5:
            data[index] = -1

        elif item > 0.5:
            data[index] = 1

        else:
            data[index] = 0

    return data


def __transform(entries: np.ndarray, scalarPassed: ScalarsIn) -> typ.Tuple[np.ndarray, StandardScaler]:
    
    # NOTE: this should only ever be called if useNormalize is true!

    # remove the class IDs so they don't get normalized
    noIds = np.array(entries[:, 1:])

    # noIds is a list of instances without IDs of the form:
    # [ [value, value, value, ...],
    #   [value, value, value, ...], ...]

    # set stdScalar either using parameter in the case of testing data,
    # or making it in the case of training data
    # if we are dealing with training data, not testing data
    if scalarPassed is None:

        # *** __transform the data *** #
        scalar = StandardScaler()        # create a standard scalar object (this will make the mean 0 & the sd 1)
        scalar.fit(noIds)                # fit the scalar's distribution using the data
        tData = scalar.transform(noIds)  # __transform the data to fit the scalar's distribution
        
        # since we are using normalize, perform a discrete transformation on the data
        tData = __discretization(tData)

    # if we are dealing with testing data, we don't want to fit the scalar; just __transform the fit
    else:

        # *** __transform the data *** #
        scalar = scalarPassed            # used the passed scalar object
        tData = scalar.transform(noIds)  # __transform the scalar using passed fit

        # since we are using normalize, perform a discrete transformation on the data
        tData = __discretization(tData)

    # *** add the IDs back on *** #
    entries[:, 1:] = tData  # this overwrites everything in entries BUT the ids

    # stdScalar - Standard Scalar, will used to on test & training data
    return entries, scalar


def terminals(classId: int, constants) -> typ.List[int]:
    """terminals creates the list of relevant terminals for a given class.

    Arguments:
        classId {int} -- classId is the identifier for the class for
                         which we want a terminal set

    Returns:
        terminalSet {list[int]} -- terminals returns the highest scoring features as a list.
                                   The list will have a length of FEATURE_NUMBER/2, and will
                                   hold the indexes of the features.
    """
    log.debug('Starting terminals() method')
    
    FEATURE_NUMBER = constants['FEATURE_NUMBER']
    
    Score = collect.namedtuple('Score', ['Attribute', 'Relevancy'])
    ScoreList = typ.List[Score]
    scores: ScoreList = []
    
    # FEATURE_NUMBER is the number of features in the data. This means that it can also be a list of
    # the indexes of the features (the feature IDs). Subtract it by 1 to make 0 a valid feature ID
    for i in range(FEATURE_NUMBER):
        inClass, notIn = valuesInClass(classId, i, constants)  # find the values of attribute i in/not in class classId
        
        # get the t-test & complement of the p-value for the feature
        # tValue will be zero when the lists have the same mean
        # pValue will only be 1 when tValue is 0
        tValue, pValue = stats.ttest_ind(inClass, notIn, equal_var=False)
        
        # ****************** Check that valuesInClass & t-test worked as expected ****************** #
        # ! For Debugging Only
        # try:
        #     # __transform into numpy arrays which are easier to test
        #     inside_of_class = np.array(inClass)
        #     not_inside_of_class = np.array(notIn)
        #
        #     # *** Check if pValue is 1 *** #
        #     if pValue == 1:  # if pValue is 1 then inClass & notIn are the same. Relevancy should be zero
        #         log.debug(f'pValue is 1 (inClass & notIn share the same mean), feature {i} should be ignored')
        #
        #     # *** Check that inClass is not empty *** #
        #     if not inClass:
        #         log.error(f'inClass was empty, ninClass={inClass}, notIn={notIn}, classId={classId}, attribute={i}')
        #         raise Exception(f'ERROR: inClass was empty,'
        #                         f'\ninClass={inClass}, notIn={notIn}, classId={classId}, attribute={i}')
        #     # + if inClass is empty tValue is inaccurate, don't run other checks + #
        #
        #     # *** Check that inClass & notIn aren't equal *** #
        #     elif np.array_equal(inside_of_class, not_inside_of_class):
        #         log.error(f'inClass & notIn are equal, inClass{inside_of_class}, notIn{not_inside_of_class}')
        #         raise Exception(f'inClass & notIn are equal, inClass{inside_of_class}, '
        #                         f'notIn{not_inside_of_class}')
        #
        #     # *** Check that inClass & notIn aren't equivalent *** #
        #     elif np.array_equiv(inside_of_class, not_inside_of_class):
        #         log.error(f'inClass & notIn are equivalent (but not equal, their shapes are different), '
        #                   f'inClass{inside_of_class}, notIn{not_inside_of_class}')
        #         raise Exception(f'inClass & notIn are equivalent, inClass{inside_of_class}, '
        #                         f'notIn{not_inside_of_class}')
        #
        #     # *** Check that tValue was set & is a finite number  *** #
        #     elif tValue is None or math.isnan(tValue) or math.isinf(tValue):
        #         log.error(f'tValue computation failed, expected a finite number got {tValue}')
        #         raise Exception(f'ERROR: tValue computation failed, expected a finite number got {tValue}')
        #
        #     # *** Check that pValue was set & is a number  *** #
        #     elif pValue is None or math.isnan(pValue) or math.isinf(pValue):
        #         log.error(f'pValue computation failed, expected a finite number got {pValue}')
        #         raise Exception(f'ERROR: pValue computation failed, expected a finite number got {pValue}')
        
        # except Exception as err:
        #     tqdm.write(str(err))
        #     sys.exit(-1)  # exit on error; recovery not possible
        # ******************************************************************************************* #
        
        # calculate relevancy for a single feature (if the mean is the same for inClass & notIn, pValue=1)
        if pValue >= 0.05:  # if pValue is greater than 0.05 then the feature is not relevant
            relevancy: float = 0.0  # because it's not relevant, set relevancy score to 0
            scores.append(Score(i, relevancy))  # add relevancy score to the list of scores
        
        # otherwise
        else:
            
            try:
                relevancy: float = np.divide(np.absolute(tValue), pValue)  # set relevancy using t-value/p-value
                
                # *************************** Check that division worked *************************** #
                # ! For Debugging Only
                # if math.isinf(relevancy):  # check for n/0
                #     log.error(
                #         f'Relevancy is infinite; some non-zero was divided by 0 -- tValue={tValue} pValue={pValue}')
                #     raise Exception(f'ERROR: relevancy is infinite, tValue={tValue} pValue={pValue}')
                #
                # elif math.isnan(relevancy):  # check for 0/0
                #     log.error(f'Relevancy is infinite; 0/0 -- tValue={tValue} pValue={pValue}')
                #     raise Exception(f'ERROR: relevancy is NaN (0/0), tValue={tValue} pValue={pValue}')
                # if pValue == 1:
                #     log.error('pValue is 1, but was not caught by if pValue >= 0.05')
                #     raise Exception('ERROR: pValue is 1, but was not caught by if pValue >= 0.05')
                # ********************************************************************************** #
                
                # if division worked, add relevancy score to the list of scores
                scores.append(Score(i, relevancy))
            
            except Exception as err:
                tqdm.write(str(err))
                sys.exit(-1)  # exit on error; recovery not possible
    
    ordered: ScoreList = sorted(scores, key=lambda s: s.Relevancy)  # sort the features by relevancy scores
    
    terminalSet: typ.List[int] = []  # this will hold relevant terminals
    top: int = len(ordered) // 2  # find the halfway point
    relevantScores: ScoreList = ordered[:top]  # slice top half
    
    for i in relevantScores:  # loop over relevant scores
        terminalSet.append(i.Attribute)  # add the attribute number to the terminal set
    
    # ************************* Test if terminalSet is empty ************************* #
    # ! For Debugging Only
    # try:
    #     if not terminalSet:      # if terminalSet is empty
    #         log.error('Terminals calculation failed: terminalSet is empty')
    #         raise Exception('ERROR: Terminals calculation failed: terminalSet is empty')
    #     if None in terminalSet:  # if terminalSet contains a None
    #         log.error('Terminals calculation failed: terminalSet contains a None')
    #         raise Exception('ERROR: Terminals calculation failed: terminalSet contains a None')
    # except Exception as err:
    #     lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
    #     tqdm.write(f'{str(err)}, line = {lineNm}')
    #     sys.exit(-1)  # exit on error; recovery not possible
    # ********************************************************************************* #
    
    log.debug('Finished terminals() method')
    
    return terminalSet


def __mapInstanceToClass(entries: np.ndarray) -> typ.Dict[int, typ.List[typ.Union[int, typ.List[float]]]]:
    # *** connect a class to every instance that occurs in it ***
    # this will map a classId to a 2nd dictionary that will hold the number of instances in that class &
    # the instances
    # classToInstances[classId] = list[counter, instance1[], instance2[], ...]
    classToInstances: typ.Dict[int, typ.List[typ.Union[int, typ.List[float]]]] = {}
    
    for e in entries:
        label = e[0]  # get the class id
        dictPosition: typ.List[typ.Union[int, typ.List[float]]] = classToInstances.get(label)  # get the position for the class in the dictionary
        
        # if we already have an entry for that classId, append to it
        if dictPosition:
            dictPosition[0] += 1            # increment instance counter
            idx = dictPosition[0]           # get the index that the counter says is next
            eggs: typ.List[typ.List[float]] = e.tolist()  # get the instances from the entry
            # noinspection PyTypeChecker
            dictPosition.insert(idx, eggs)  # at that index insert a list representing the instance
        
        # if this is the first time we've seen the class, create a new list for it
        else:
            # add a list, at index 0 put the counter, and at index 1 put a list containing the instance (values & label)
            classToInstances[label] = [1, e.tolist()]

    return classToInstances


def __getPermutation(instances: typ.List[typ.List[typ.Union[int, float]]], seed: int = None):
    # *** create a permutation of a class's instances *** #
    # permutation is a 2D numpy array, where each line is an instance
    rand = np.random.default_rng(seed)  # create a numpy random generator with/without seed
    permutation = rand.permutation(instances)  # shuffle the instances
    return permutation


def __dealToBuckets(classToInstances: typ.Dict[int, typ.List[typ.Union[int, typ.List[float]]]]) -> typ.List[typ.List[np.ndarray]]:
    # *** Now create a random permutation of the instances in a class, & put them in buckets ***
    buckets = [[] for _ in range(K)]  # create a list of empty lists that we will "deal" our "deck" of instances to
    index = 0  # this will be the index of the bucket we are "dealing" to
    
    # classId is a int
    for classId in classToInstances.keys():
        
        # *** for each class use the class id to get it's instances ***
        # instances is of the form [instance1, instance2, ...]  (the counter has been removed)
        # where instanceN is a numpy array and where instanceN[0] is the classId & instanceN[1:] are the values
        permutation = __getPermutation(classToInstances[classId][1:], SEED)

        # *** assign instances to buckets *** #
        # loop over every instance in the class classId, in what is now a random order
        # p will be a single Instance from permutation & so will be a 1D numpy array representing an instance
        for p in permutation:  # for every Instance in permutation
            buckets[index].append(p)  # add the random instance to the bucket at index
            index = (index + 1) % K  # increment index in a round robin style
    
    # *** The buckets are now full & together should contain every instance *** #
    return buckets


def __fillBuckets(entries: np.ndarray) -> typ.List[typ.List[np.ndarray]]:
    
    classToInstances = __mapInstanceToClass(entries)
    buckets = __dealToBuckets(classToInstances)

    return buckets


# TODO check what this function is doing & update docstring
def __buildModel(buckets: typ.List[typ.List[np.ndarray]], mType: str, useNormalize: bool) -> typ.List[float]:
    """
    __buildModel
    
    :param buckets:
    :param mType: type of model that will make use of the feature reduction done by CDFC
    :param useNormalize: should the original data be normalized
    
    :type buckets:
    :type mType: str  (valid options are: 'KNN', 'Naive Bayes', or 'Decision Tree' )
    :type useNormalize: bool
    
    :return: the classifications of the instances
    :rtype: typ.List[float]
    """

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #
    oldR = 0                            # used to remember previous r in loop
    testingList = None                  # used to keep value for testing after a loop
    # TODO: try ot find a way to avoid a deepcopy of buckets
    trainList: typ.List[typ.List[np.ndarray]] = copy.deepcopy(buckets)  # make a copy of buckets so we don't override it
    accuracy: typ.List[float] = []      # this will store the details about the accuracy of our hypotheses

    # TODO: change pickle name so that it won't read in "pickles" from different data sets
    # *** Open the Pickle Jar file *** #
    pth = Path.cwd() / 'jar' / 'features'  # create the file path
    try:

        if os.path.isfile(str(pth)):           # if the file does exist
            with open(str(pth), 'rb') as fl:   # try to open the file
                pickles = pickle.load(fl)      # load the file into pickles
            wasPickle = True                   # since we read in the file set to True
            print(NO_OVERWRITE + ' Pickle File Read '.ljust(50, '-') + SUCCESS)

        else:                                  # if the file didn't exist
            pickles = {}                       # set pickles to be an empty dict
            wasPickle = False                  # set wasPickle to False, since we didn't read
    
    except (FileNotFoundError, IOError):       # if we encountered an error while reading
        lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
        print(f'Pickle encountered an error while reading in the file, line = {lineNm}')
        wasPickle = False                      # since we don't know if we read in, set to false and
        pickles = {}                           # set pickles to empty to avoid data corruption errors

    # *** Divide them into training & test data K times ***
    # loop over all the random index values.
    # + Each loop will create a new learning model using the fit method from SciKit

    iteration: int = 0  # used to count iterations so progress can be printed
    for r in range(K):  # len(r) = K so this will be done K times
        print('\n\n' + f' Starting Fold {iteration + 1}/{K} '.center(58, '*'))

        # ************ Create a New Model ************ #
        # determine the type of model we need to create
        if mType == 'KNN':
            model = KNeighborsClassifier(n_neighbors=3)

        elif mType == 'Naive Bayes':
            model = GaussianNB()

        else:  # mType == 'Decision Tree'
            model = DecisionTreeClassifier(random_state=0)
        
        # print('models created')  # ! debugging only!
        # ********** Get the Training & Testing Data ********** #
        # the Rth bucket becomes our testing data, everything else becomes training data
        # this is done in order to prevent accidental overwrites
        if testingList is None:                  # if this is not the first time though the loop
            testingList = trainList.pop(r)       # then set train & testing
            oldR = r                             # save the current r value for then next loop
            
        else:                                    # if we've already been through the loop at least once
            trainList.insert(oldR, testingList)  # add testing back into train
            testingList = trainList.pop(r)       # then set train & testing
            oldR = r                             # save the current r value for then next loop

        # print('got training models')  # ! debugging only!
        # ********** Flatten the Training Data ********** #
        train: np.ndarray = flattenTrainingData(trainList)  # remove buckets to create a single pool

        testing: np.ndarray = np.array(testingList)  # turn testing data into a numpy array, testing doesn't need to be flattened

        # print('flattened training data')  # ! debugging only!
        # ********** 3A Normalize the Training Data (if useNormalize is True) ********** #
        # this is also fitting the data to a distribution
        if useNormalize:
            train, scalar = __transform(train, None)  # now normalize the training, and keep the scalar used
        else:  # this case is not strictly needed, but helps the editor/IDE not throw errors
            scalar = None

        # print('normalized training data')  # ! debugging only!
        # ********** 3B Train the CDFC Model & Transform the Training Data using It ********** #

        # SYSOUT.write(f"\nTraining CDFC for {mType}...\n")  # update user
        
        if wasPickle:                                                            # if there was a saved data object
            data = pickles[r]                                                    # read in from it
        else:                                                                    # if there wasn't a saved data object,
            constants = parseFile(train)                                         # parse the file to get the constants
            relevant: typ.Dict[int, typ.List[int]] = {}                          # this will hold the relevant features found by terminals
            for classId in constants['CLASS_IDS']:                               # loop over all class ids and get the relevant features for each one
                relevant[classId] = terminals(classId, constants)                # store the relevant features for a class using the classId as a key
            # __sanityCheckDictionary(relevant)
            data = (constants, relevant)                                         # data[0] = constants to be set, data[1] = TERMINALS
            pickles[r] = data

        # now that we have our train & test data create our hypothesis (train CDFC)
        # print('calling CDFC...')  # ! debugging only!
        CDFC_Hypothesis = cdfc(data, PASSED_FUNCTION)
        SYSOUT.write(OVERWRITE + ' CDFC Trained '.ljust(50, '-') + SUCCESS)      # replace starting with complete
        
        SYSOUT.write(HDR + ' Transforming Training Data ......')                 # print
        SYSOUT.flush()
        train: np.array = CDFC_Hypothesis.runCDFC(train)                         # __transform data using the CDFC model
        SYSOUT.write(OVERWRITE + ' Data Transformed '.ljust(50, '-') + SUCCESS)  # replace starting with complete

        # ********** 3C Train the Learning Algorithm ********** #
        SYSOUT.write(HDR + f' Training {mType} Model ......')
        # format data for SciKit Learn
        ftrs, labels = formatForSciKit(train)
        
        # now that the data is formatted, run the learning algorithm.
        # if useNormalize is True the data has been transformed to fit
        # a scalar & gone through discretization. If False, it has not
        model.fit(ftrs, labels)
        SYSOUT.write(OVERWRITE + f' {mType} Model Trained '.ljust(50, '-') + SUCCESS)  # replace starting with complete
    
        # ********** 3D.1 Normalize the Testing Data ********** #
        # if use
        if useNormalize:
            testing, scalar = __transform(testing, scalar)
    
        # ********** 3D.2 Reduce the Testing Data Using the CDFC Model ********** #
        testing = CDFC_Hypothesis.runCDFC(testing)  # use the cdfc model to reduce the data's size
    
        # format testing data for SciKit Learn
        ftrs, trueLabels = formatForSciKit(testing)
    
        # ********** 3D.3 Feed the Testing Data into the Model & get Accuracy ********** #
        labelPrediction = model.predict(ftrs)  # use model to predict labels
        # compute the accuracy score by comparing the actual labels with those predicted
        score = accuracy_score(trueLabels, labelPrediction)
        accuracy.append(score)  # add the score to the list of scores

        # *** print the computed accuracy *** #
        percentScore: float = round(score * 100, 1)  # turn the score into a percent with 2 decimal places
        
        if percentScore > 75:         # > 75 print in green
            SYSOUT.write(f'\r\033[32;1m{mType} Accuracy is: {percentScore}%\033[00m\n')
            SYSOUT.flush()

        elif 45 < percentScore < 75:  # > 45 and < 75 print yellow
            SYSOUT.write(f'\r\033[33;1m{mType} Accuracy is: {percentScore}%\033[00m\n')
            SYSOUT.flush()

        elif percentScore < 45:       # < 45 print in red
            SYSOUT.write(f'\r\033[91;1m{mType} Accuracy is: {percentScore}%\033[00m\n')
            SYSOUT.flush()
        
        else:  # don't add color, but print accuracy
            SYSOUT.write(f'{mType} Accuracy is: {percentScore}%\n')
            SYSOUT.flush()
        
        iteration += 1  # update iteration
    
    if not wasPickle:  # if there wasn't a saved object, save
        try:  # *** Save the Features/Terminals using Pickle *** #
            
            with open(str(pth), 'wb') as fl:  # open the file with write binary privileges
                pickle.dump(pickles, fl)      # pickle the dict, storing it in a file
                
        except FileNotFoundError:  # if the file wasn't found, and it couldn't be created
            log.error(f'Pickle could not open or create file {pth}')         # log the error & print it
            SYSOUT.write(f'\nPickle could not open or create file {pth}\n')  # to console, but continue

    SYSOUT.write('\n' + NO_OVERWRITE + ' K Fold Validation Complete '.ljust(50, '-') + SUCCESS)  # update user
    
    # *** Return Accuracy *** #
    return accuracy


# ! for testing purposes only!
def __sanityCheckDictionary(d: typ.Dict[int, typ.List[int]]) -> None:
    """
    A Sanity check for the dictionary that cdfcProject will send to cdfc.
    This is used in testing only, and checks that the dictionary is non-empty.
    
    :param d: dictionary to be tested.
    :type d: dict
    
    :return: either raises an exception or returns None if dictionary is non-empty.
    :rtype: None
    """

    log.debug('Starting Dictionary Sanity Check...')
    
    try:
        if None in d.values():
            raise Exception('ERROR: buildModel() created a relevant value dictionary that includes \'None\'')
    
    except Exception as err:
        lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
        msg = f'{str(err)}, line = {lineNm}'  # create the error/log message
        log.error(msg)
        tqdm.write(msg)
        traceback.print_stack()
        sys.exit(-1)  # exit on error; recovery not possible
    
    log.debug('Dictionary Sanity Check Passed')


def __runSciKitModels(entries: np.ndarray, useNormalize: bool) -> typ.List[float]:
    """
    __runSciKitModels builds the non-CDFC models & returns the
    constructed models as a list.
    
    :param entries: the parsed input file
    :param useNormalize: a flag that is true if the data should be normalized
    
    :type entries: np.ndarray
    :type useNormalize: bool
    
    :return: the constructed models
    :rtype: typ.List[typ.List[float]]
    """
    
    # NOTE adding more models requires updating the Models type at top of file
    # hdr, overWrite, & success are just used to format string for the console
    # accuracy is a float list, each value is the accuracy for a single run
    
    # ***** create a set of K buckets filled with our instances ***** #
    SYSOUT.write("\nBuilding buckets...")  # update user
    buckets = __fillBuckets(entries)  # using the parsed data, fill the k buckets (once for all models)
    SYSOUT.write(OVERWRITE + " Buckets built ".ljust(50, '-') + SUCCESS)  # update user
    
    if LEARN == "DT":
        
        # ************ Decision Tree Classifier ************ #
        accuracy: typ.List[float] = __buildModel(buckets, 'Decision Tree', useNormalize)  # build the model
    
    elif LEARN == "NB":
        
        # ************ Gaussian Classifier (Naive Bayes) ************ #
        accuracy: typ.List[float] = __buildModel(buckets, 'Naive Bayes', useNormalize)  # build the model
    
    else:  # do the defualt (KNN)
        
        # ************ Kth Nearest Neighbor Classifier ************ #
        accuracy: typ.List[float] = __buildModel(buckets, 'KNN', useNormalize)  # build the model

    # SYSOUT.write("Model ran\n\n")  # update user

    # TODO change calling function so it can accept a single return
    return accuracy


def run(fnc: str, mdl: str) -> None:
    """
    run is the entry point for main.py. It prompts the user for a file, parses it, builds the required models,
    formats the data frames, and general coordinates the other models with CDFC.
    
    :param fnc: the distance function to be used. It should be provided via command line flags (see main.py).
    :param mdl:
    
    :type fnc: str
    :type mdl:
    
    :return: run either crashes or returns a None on a success.
    :rtype: None
    """
    title: str = Figlet(font='larry3d').renderText('C D f C')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message
    SYSOUT.write("\033[32;1mProgram Initialized Successfully\033[00m\n")
    
    parent = tk.Tk()            # prevent root window caused by Tkinter
    parent.overrideredirect(1)  # Avoid it appearing and then disappearing quickly
    parent.withdraw()           # Hide the window

    # *** set the passed functions value *** #
    # NOTE: for some reason just passing the fnc string was causing a key error
    global PASSED_FUNCTION
    global LEARN
    
    LEARN = mdl  # set the value of LEARN using provided input
    
    if fnc == "correlation":    # use the Correlation function
        PASSED_FUNCTION = "correlation"
    elif fnc == "czekanowski":  # use the Czekanowski function
        PASSED_FUNCTION = "czekanowski"
    elif fnc == "cosine":    # use the Euclidean function
        PASSED_FUNCTION = "cosine"
    else:                       # default to the Euclidean function
        PASSED_FUNCTION = "euclidean"
    
    SYSOUT.write('\nGetting File...\n')
    try:
        inPath = Path(filedialog.askopenfilename(parent=parent))  # prompt user for file path
        SYSOUT.write(f"{HDR} File {inPath.name} selected......")
    except PermissionError:
        sys.stderr.write(f"\n{HDR} Permission Denied, or No File was Selected\nExiting......")  # exit gracefully
        sys.exit("Could not access file/No file was selected")

    SYSOUT.write(OVERWRITE + f' File {inPath.name} Found '.ljust(50, '-') + SUCCESS)

    # useNormalize = messagebox.askyesno('CDFC - Transformations', 'Do you want to __transform the data before using it?', parent=parent)  # Yes / No
    useNormalize = True  # use during debugging to make runs faster
    
    # *** Read the file into a numpy 2d array *** #
    SYSOUT.write(HDR + ' Reading in .csv file...')  # update user
    entries = np.genfromtxt(inPath, delimiter=',', skip_header=1)  # + this line is used to read .csv files
    SYSOUT.write(OVERWRITE + ' .csv file read in successfully '.ljust(50, '-') + SUCCESS)  # update user
    SYSOUT.write('\r\033[32;1mFile Found & Loaded Successfully\033[00m\n')                   # update user
    
    # *** Build the Models *** #
    accuracy: typ.List[float] = __runSciKitModels(entries, useNormalize)  # knnAccuracy, dtAccuracy, nbAccuracy

    # NOTE: everything below just formats & outputs the results
    # *** Create a Dataframe that Combines the Accuracy of all the Models *** #
    accuracyFrame = buildAccuracyFrame(accuracy, K, LEARN)
    
    # *** Export the Dataframe as a LaTeX File *** #
    SYSOUT.write(HDR + ' Exporting LaTeX dataframe...')
    
    # will be of the form MODEL_DATASET, for example: KNN_Colon
    title = f'{LEARN}_{inPath.name}'
    
    try:  # attempt to convert the dataframe to latex
        output: str = accuracyFrame.to_latex(label=title)
    except Exception as err:
        lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        msg: str = f'ERROR is formatting.py, line {lineNm}\n{str(err)}'  # create the message
        printError(msg)                        # print the message
        print(f'dataframe = {accuracyFrame}')  # print the dataframe
        printError(traceback.format_exc())     # print stack trace
        sys.exit(-1)  # exit on error; recovery not possible
    
    # set the output file path
    if useNormalize:                       # if we are using the transformations
        stm = inPath.stem + 'Transformed'  # add 'Transformed' to the file name
    
    else:                          # if we are not transforming the data
        stm = inPath.stem + 'Raw'  # add 'Raw' to the file name
    
    outPath = Path.cwd() / 'data' / 'outputs' / (title + stm + '.tex')  # create the file path

    with open(outPath, "w") as texFile:                           # open the selected file
        print(output, file=texFile)  # & write dataframe to it, converting it to latex
    texFile.close()                                               # close the file
    
    SYSOUT.write(OVERWRITE + ' Export Successful '.ljust(50, '-') + SUCCESS)
    # SYSOUT.write('\033[32;1m Dataframe converted to LaTeX & Exported\n'+'\033[0m')
    
    return


if __name__ == "__main__":
    
    start = time.time()        # get the start time
    run("cosine", 'KNN')    # run CDFC
    end = time.time() - start  # get the elapsed
    print(f'Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(end))}')  # print the elapsed time
