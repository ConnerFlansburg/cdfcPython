import copy
import sys
import logging as log
import collections as collect
import cProfile
import math
from tqdm import tqdm
from tqdm import trange
from scipy import stats
from cdfcFmt import *
import tkinter as tk
import typing as typ
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from pyfiglet import Figlet
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from cdfc import cdfc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from cdfcFmt import __buildAccuracyFrame, __accuracyFrameToLatex, __formatForSciKit, __flattenTrainingData

'''
                                       CSVs should be of the form

           |  label/id   |   attribute 1   |   attribute 2   |   attribute 3   |   attribute 4   | ... |   attribute k   |
--------------------------------------------------------------------------------------------------------------------------
instance 1 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
instance 2 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
instance 3 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
instance 4 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
    ...    |    ...      |      ...        |       ...       |       ...       |       ...       | ... |       ...       |
--------------------------------------------------------------------------------------------------------------------------
instance n | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------

'''
# ********************************************* Constants used by Parser ********************************************* #
row = collect.namedtuple('row', ['className', 'attributes'])  # a single line in the csv, representing a record/instance
BETA: typ.Final = 2              # BETA is a constant used to calculate the pop size
R: typ.Final = 2                 # R is the ratio of number of CFs to the number of classes (features/classes)
# ******************************************** Constants used by Profiler ******************************************** #
profiler = cProfile.Profile()                       # create a profiler to profile cdfc during testing
statsPath = str(Path.cwd() / 'logs' / 'stats.log')  # set the file path that the profiled info will be stored at
# ********************************************* Constants used by Logger ********************************************* #
# create the file path for the log file & configure the logger
logPath = str(Path.cwd() / 'logs' / 'cdfc.log')
log.basicConfig(level=log.DEBUG, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')
# ******************************************** Constants used for Writing ******************************************** #
HDR = '*' * 6
SUCCESS = u' \u2713\n'
OVERWRITE = '\r' + HDR
SYSOUT = sys.stdout
# ****************************************** Constants used by Type Hinting ****************************************** #
K: typ.Final[int] = 10  # set the K for k fold cross validation
ModelList = typ.Tuple[typ.List[float], typ.List[float], typ.List[float]]          # type hinting alias
ModelTypes = typ.Union[KNeighborsClassifier, GaussianNB, DecisionTreeClassifier]  # type hinting alias
ScalarsOut = typ.Tuple[np.ndarray, StandardScaler]
ScalarsIn = typ.Union[None, StandardScaler]
# ******************************************************************************************************************** #

# * Next Steps
# TODO get CDFC working & use it to reduce data
# TODO add doc strings
# TODO add more unit tests


def parseFile(train: np.ndarray):
    # TODO: review for bugs. Remember this parses bucket(s) now, not files
    # *** the constants to be set * #
    INSTANCES_NUMBER = 0
    rows: typ.List[row] = []
    ENTROPY_OF_S = 0  # * used in entropy calculation * #
    CLASS_DICTS = {}
    # *** (the rest are set after the loop) *** #
    
    classes = []  # this will hold classIds and how often they occur
    classSet = set()  # this will hold how many classes there are
    classToOccur = {}  # maps a classId to the number of times it occurs
    ids = []  # this will be a list of all labels/ids with no repeats
    
    # set global variables using the now transformed data
    # each line in train will be an instances
    for line in tqdm(train, desc="Setting Global Variables", ncols=25):
        
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
        rows.append(row(name, line[1:]))  # reader[0] = classId, reader[1:] = attribute values
        classes.append(name)
        classSet.add(name)
        INSTANCES_NUMBER += 1
        
        # *** Create a Dictionary of Attribute Values Keyed by the Attribute's Index *** #
        # ! check that this is working correctly
        attributeNames = range(len(line[1:]))  # create names for the attributes number from 0 to len()
        attributeValues = line[1:]  # grab all the attribute values in this instance
        
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
                oldDict = CLASS_DICTS[name]  # get the old dictionary for the class
                for att in oldDict.keys():  # we need to update every attribute value list so loop
                    oldDict[att] += newDict[att]  # for each value list, concatenate the old & new lists
            
            # *** Insert the Dictionary into CLASS_DICTS *** #
            else:  # if this is the first instance in the class
                CLASS_DICTS[name] = dict(newDict)  # insert the new dictionary using the key classID
                ids.append(name)  # add classId to ids -- a list of unique classIDs
                
                log.debug(f'Parser created dictionary for classId {name}')  # ! for debugging
                # tqdm.write(f'Parser created dictionary for classId {name}')  # ! for debugging
        
        except IndexError:  # catch error thrown by dictionary indexing
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Parser encountered an Index Error on line {lineNm}. name={name}, line[0]={line[0]}')
            tqdm.write(f'ERROR: Parser encountered an Index Error on line {lineNm}. name={name}, line[0]={line[0]}')
            sys.exit(-1)  # recovery impossible, exit
        except AssertionError as err:  # catch the error thrown by names/value length check
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            tqdm.write(str(err) + f', line = {lineNm}')
            sys.exit(-1)  # recovery impossible, exit
        except Exception as err:  # catch any other error that might be thrown
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            tqdm.write(str(err) + f', line = {lineNm}')
            sys.exit(-1)  # recovery impossible, exit
        
        # ********* The Code Below is Used to Calculated Entropy  ********* #
        # this will count the number of times a class occurs in the provided data
        # dictionary[classId] = counter of times that class is found
        
        if classToOccur.get(name):  # if we have encountered the class before
            classToOccur[line[0]] += 1  # increment
        else:  # if this is the first time we've encountered the class
            classToOccur[line[0]] = 1  # set to 1
        # ****************************************************************** #
    
    CLASS_IDS = ids  # this will collect all the feature names
    FEATURE_NUMBER = len(rows[0].attributes)  # get the number of features in the data set
    POPULATION_SIZE = FEATURE_NUMBER * BETA  # set the pop size
    LABEL_NUMBER = len(ids)  # get the number of classes in the data set
    M = R * LABEL_NUMBER  # get the number of constructed features
    
    # ********* The Code Below is Used to Calculated Entropy  ********* #
    # loop over all classes
    for i in classToOccur.keys():
        pi = classToOccur[i] / INSTANCES_NUMBER  # compute p_i
        ENTROPY_OF_S -= pi * math.log(pi, 2)  # calculation entropy summation
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
    }
    
    return constants
    

def __discretization(data: np.ndarray) -> np.ndarray:
    
    for index, item in np.ndenumerate(data):

        if item < -0.5:
            data[index] = -1

        elif item > 0.5:
            data[index] = 1

        else:
            data[index] = 0

    return data


def __transform(entries: np.ndarray, scalarPassed: ScalarsIn) -> ScalarsOut:
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

        # *** transform the data *** #
        scalar = StandardScaler()        # create a standard scalar object (this will make the mean 0 & the sd 1)
        scalar.fit(noIds)                # fit the scalar's distribution using the data
        tData = scalar.transform(noIds)  # transform the data to fit the scalar's distribution
        
        # since we are using normalize, perform a discrete transformation on the data
        tData = __discretization(tData)

    # if we are dealing with testing data, we don't want to fit the scalar; just transform the fit
    else:

        # *** transform the data *** #
        scalar = scalarPassed            # used the passed scalar object
        tData = scalar.transform(noIds)  # transform the scalar using passed fit

        # since we are using normalize, perform a discrete transformation on the data
        tData = __discretization(tData)

    # *** add the IDs back on *** #
    entries[:, 1:] = tData  # this overwrites everything in entries BUT the ids

    # stdScalar - Standard Scalar, will used to on test & training data
    return entries, scalar


# TODO change so terminals is done before calling cdfc & the result gets passed (as a dict?)
def terminals(classId: int) -> typ.List[int]:
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
    
    Score = collect.namedtuple('Score', ['Attribute', 'Relevancy'])
    ScoreList = typ.List[typ.Union[typ.List, Score]]
    scores: ScoreList = []
    
    # FEATURE_NUMBER is the number of features in the data. This means that it can also be a list of
    # the indexes of the features (the feature IDs). Subtract it by 1 to make 0 a valid feature ID
    for i in range(FEATURE_NUMBER):
        inClass, notIn = valuesInClass(classId, i)  # find the values of attribute i in/not in class classId
        
        # get the t-test & complement of the p-value for the feature
        # tValue will be zero when the lists have the same mean
        # pValue will only be 1 when tValue is 0
        tValue, pValue = stats.ttest_ind(inClass, notIn, equal_var=False)
        
        # ****************** Check that valuesInClass & t-test worked as expected ****************** #
        try:
            # transform into numpy arrays which are easier to test
            inside_of_class = np.array(inClass)
            not_inside_of_class = np.array(notIn)
            
            # *** Check if pValue is 1 *** #
            if pValue == 1:  # if pValue is 1 then inClass & notIn are the same. Relevancy should be zero
                log.debug(f'pValue is 1 (inClass & notIn share the same mean), feature {i} should be ignored')
            
            # *** Check that inClass is not empty *** #
            if not inClass:
                log.error(f'inClass was empty, ninClass={inClass}, notIn={notIn}, classId={classId}, attribute={i}')
                raise Exception(f'ERROR: inClass was empty,'
                                f'\ninClass={inClass}, notIn={notIn}, classId={classId}, attribute={i}')
            # + if inClass is empty tValue is inaccurate, don't run other checks + #
            
            # *** Check that inClass & notIn aren't equal *** #
            elif np.array_equal(inside_of_class, not_inside_of_class):
                log.error(f'inClass & notIn are equal, inClass{inside_of_class}, notIn{not_inside_of_class}')
                raise Exception(f'inClass & notIn are equal, inClass{inside_of_class}, '
                                f'notIn{not_inside_of_class}')
            
            # *** Check that inClass & notIn aren't equivalent *** #
            elif np.array_equiv(inside_of_class, not_inside_of_class):
                log.error(f'inClass & notIn are equivalent (but not equal, their shapes are different), '
                          f'inClass{inside_of_class}, notIn{not_inside_of_class}')
                raise Exception(f'inClass & notIn are equivalent, inClass{inside_of_class}, '
                                f'notIn{not_inside_of_class}')
            
            # *** Check that tValue was set & is a finite number  *** #
            elif tValue is None or math.isnan(tValue) or math.isinf(tValue):
                log.error(f'tValue computation failed, expected a finite number got {tValue}')
                raise Exception(f'ERROR: tValue computation failed, expected a finite number got {tValue}')
            
            # *** Check that pValue was set & is a number  *** #
            elif pValue is None or math.isnan(pValue) or math.isinf(pValue):
                log.error(f'pValue computation failed, expected a finite number got {pValue}')
                raise Exception(f'ERROR: pValue computation failed, expected a finite number got {pValue}')
        
        except Exception as err:
            tqdm.write(str(err))
            sys.exit(-1)  # exit on error; recovery not possible
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
                if math.isinf(relevancy):  # check for n/0
                    log.error(
                        f'Relevancy is infinite; some non-zero was divided by 0 -- tValue={tValue} pValue={pValue}')
                    raise Exception(f'ERROR: relevancy is infinite, tValue={tValue} pValue={pValue}')
                
                elif math.isnan(relevancy):  # check for 0/0
                    log.error(f'Relevancy is infinite; 0/0 -- tValue={tValue} pValue={pValue}')
                    raise Exception(f'ERROR: relevancy is NaN (0/0), tValue={tValue} pValue={pValue}')
                if pValue == 1:
                    log.error('pValue is 1, but was not caught by if pValue >= 0.05')
                    raise Exception('ERROR: pValue is 1, but was not caught by if pValue >= 0.05')
                # ********************************************************************************** #
                
                else:  # if division worked
                    scores.append(Score(i, relevancy))  # add relevancy score to the list of scores
            
            except Exception as err:
                tqdm.write(str(err))
                sys.exit(-1)  # exit on error; recovery not possible
    
    ordered: ScoreList = sorted(scores, key=lambda s: s.Relevancy)  # sort the features by relevancy scores
    
    terminalSet: typ.Union[int, typ.List] = []  # this will hold relevant terminals
    top: int = len(ordered) // 2  # find the halfway point
    relevantScores: ScoreList = ordered[:top]  # slice top half
    
    for i in relevantScores:  # loop over relevant scores
        # ? this is where the terminal index is added. Is the index correct?
        terminalSet.append(i.Attribute)  # add the attribute number to the terminal set
    
    # ************************* Test if terminalSet is empty ************************* #
    try:
        if not terminalSet:  # if terminalSet is empty
            log.error('Terminals calculation failed: terminalSet is empty')
            raise Exception('ERROR: Terminals calculation failed: terminalSet is empty')
    except Exception as err:
        tqdm.write(str(err))
        sys.exit(-1)  # exit on error; recovery not possible
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
    seed = 498
    
    # classId is a int
    for classId in classToInstances.keys():
        
        # *** for each class use the class id to get it's instances ***
        # instances is of the form [instance1, instance2, ...]  (the counter has been removed)
        # where instanceN is a numpy array and where instanceN[0] is the classId & instanceN[1:] are the values
        permutation = __getPermutation(classToInstances[classId][1:], seed)

        # *** assign instances to buckets *** #
        # loop over every instance in the class classId, in what is now a random order
        # p will be a single row from permutation & so will be a 1D numpy array representing an instance
        for p in permutation:  # for every row in permutation
            buckets[index].append(p)  # add the random instance to the bucket at index
            index = (index + 1) % K  # increment index in a round robin style
    
    # *** The buckets are now full & together should contain every instance *** #
    return buckets


def __fillBuckets(entries: np.ndarray) -> typ.List[typ.List[np.ndarray]]:
    
    classToInstances = __mapInstanceToClass(entries)
    buckets = __dealToBuckets(classToInstances)

    return buckets


def __buildModel(buckets, model: ModelTypes, useNormalize) -> typ.List[float]:
    # TODO: change to use new flow
    # TODO: call parser to parse bucket, then pass returned dictionary (& the logger) to CDFC
    # ? should the parser be called on the entries or the buckets? i.e. should the transformation be aware
    # ?   of the whole data set when trained?

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #
    oldR = 0                            # used to remember previous r in loop
    testingList = None                  # used to keep value for testing after a loop
    # TODO: try ot find a way to avoid a deepcopy of buckets
    trainList = copy.deepcopy(buckets)  # make a copy of buckets so we don't override it
    accuracy = []                       # this will store the details about the accuracy of our hypotheses

    # *** Divide them into training & test data K times ***
    # loop over all the random index values
    for r in range(0, K):  # len(r) = K so this will be done K times
    
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
    
        # ********** Flatten the Training Data ********** #
        train = __flattenTrainingData(trainList)  # remove buckets to create a single pool

        testing = np.array(testingList)  # turn testing data into a numpy array, testing doesn't need to be flattened

        # ********** 3A Normalize the Training Data (if useNormalize is True) ********** #
        # this is also fitting the data to a distribution
        if useNormalize:
            train, scalar = __transform(train, None)  # now normalize the training, and keep the scalar used
        else:  # this case is not strictly needed, but helps the editor/IDE not throw errors
            scalar = None
    
        # ********** 3B Train the CDFC Model & Transform the Training Data using It ********** #
        SYSOUT.write(HDR + ' Training CDFC ......')  # print \tab * Training CDFC ....
        SYSOUT.flush()
        CDFC_Hypothesis = cdfc(train)  # now that we have our train & test data create our hypothesis
        SYSOUT.write(OVERWRITE + ' CDFC Trained '.ljust(50, '-') + SUCCESS)      # replace starting with complete
        
        SYSOUT.write(HDR + ' Transforming Training Data ......')                 # print
        SYSOUT.flush()
        train = CDFC_Hypothesis.transform(train)                                 # transform data using the CDFC model
        SYSOUT.write(OVERWRITE + ' Data Transformed '.ljust(50, '-') + SUCCESS)  # replace starting with complete

        # ********** 3C Train the Learning Algorithm ********** #
        # format data for SciKit Learn
        ftrs, labels = __formatForSciKit(train)
        
        # now that the data is formatted, run the learning algorithm.
        # if useNormalize is True the data has been transformed to fit
        # a scalar & gone through discretization. If False, it has not
        model.fit(ftrs, labels)
    
        # ********** 3D.1 Normalize the Testing Data ********** #
        # if use
        if useNormalize:
            testing, scalar = __transform(testing, scalar)
    
        # ********** 3D.2 Reduce the Testing Data Using the CDFC Model ********** #
        # + testing = CDFC_Hypothesis.transform(testing)  # use the cdfc model to reduce the data's size
    
        # format testing data for SciKit Learn
        ftrs, trueLabels = __formatForSciKit(testing)
    
        # ********** 3D.3 Feed the Training Data into the Model & get Accuracy ********** #
        labelPrediction = model.predict(ftrs)  # use model to predict labels
        # compute the accuracy score by comparing the actual labels with those predicted
        accuracy.append(accuracy_score(trueLabels, labelPrediction))
    
    # *** Return Accuracy *** #
    return accuracy


def __runSciKitModels(entries: np.ndarray, useNormalize: bool) -> ModelList:
    # NOTE adding more models requires updating the Models type at top of file
    # hdr, overWrite, & success are just used to format string for the console
    # accuracy is a float list, each value is the accuracy for a single run
    
    # ***** create a set of K buckets filled with our instances ***** #
    SYSOUT.write("\nBuilding buckets...\n")  # update user
    buckets = __fillBuckets(entries)  # using the parsed data, fill the k buckets (once for all models)
    SYSOUT.write("Buckets built\n\n")  # update user

    SYSOUT.write("\nBuilding models...\n")  # update user
    
    # ************ Kth Nearest Neighbor Classifier ************ #
    SYSOUT.write(HDR + ' KNN model starting ......')                        # print \tab * KNN model starting......
    SYSOUT.flush()                                                          # since there's no newline push buffer to console
    knnAccuracy: typ.List[float] = __buildModel(buckets, KNeighborsClassifier(n_neighbors=3), useNormalize)  # build the model
    SYSOUT.write(OVERWRITE+' KNN model completed '.ljust(50, '-')+SUCCESS)  # replace starting with complete

    # ************ Decision Tree Classifier ************ #
    SYSOUT.write(HDR + ' Decision Tree model starting ......')              # print \tab * dt model starting......
    SYSOUT.flush()                                                          # since there's no newline push buffer to console
    dtAccuracy: typ.List[float] = __buildModel(buckets, DecisionTreeClassifier(random_state=0), useNormalize)  # build the model
    SYSOUT.write(OVERWRITE +                                                # replace starting with complete
                 ' Decision Tree model built '.ljust(50, '-') + SUCCESS)

    # ************ Gaussian Classifier (Naive Bayes) ************ #
    SYSOUT.write(HDR + ' Naive Bayes model starting ......')                # print \tab * nb model starting......
    SYSOUT.flush()                                                          # since there's no newline push buffer to console
    nbAccuracy: typ.List[float] = __buildModel(buckets, GaussianNB(), useNormalize)       # build the model
    SYSOUT.write(OVERWRITE +                                                # replace starting with complete
                 ' Naive Bayes model built '.ljust(50, '-') + SUCCESS)
    
    SYSOUT.write("Models built\n\n")  # update user

    return knnAccuracy, dtAccuracy, nbAccuracy


def main() -> None:
    SYSOUT.write(Figlet(font='larry3d').renderText('C D F C'))  # formatted start up message
    SYSOUT.write("Program Initialized Successfully\n")
    
    parent = tk.Tk()            # prevent root window caused by Tkinter
    parent.overrideredirect(1)  # Avoid it appearing and then disappearing quickly
    parent.withdraw()           # Hide the window

    SYSOUT.write('\nGetting File...\n')
    try:
        inPath = Path(filedialog.askopenfilename(parent=parent))  # prompt user for file path
        SYSOUT.write(f"{HDR} File {inPath.name} selected......")
    except PermissionError:
        sys.stderr.write(f"\n{HDR} Permission Denied, or No File was Selected\nExiting......")  # exit gracefully
        sys.exit("Could not access file/No file was selected")

    SYSOUT.write(f"{OVERWRITE} File {inPath.name} found ".ljust(57, '-') + SUCCESS)

    useNormalize = messagebox.askyesno('CDFC - Transformations', 'Do you want to transform the data before using it?', parent=parent)  # Yes / No
    
    # *** Read the file into a numpy 2d array *** #
    SYSOUT.write(HDR + ' Reading in .csv file...')  # update user
    entries = np.genfromtxt(inPath, delimiter=',', skip_header=1)  # + this line is used to read .csv files
    SYSOUT.write(OVERWRITE + ' .csv file read in successfully '.ljust(50, '-') + SUCCESS)  # update user
    SYSOUT.write('\rFile Found & Loaded Successfully\n')  # update user
    
    # *** Build the Models *** #
    modelsTuple = __runSciKitModels(entries, useNormalize)  # knnAccuracy, dtAccuracy, nbAccuracy

    # NOTE: everything below just formats & outputs the results
    # *** Create a Dataframe that Combines the Accuracy of all the Models *** #
    accuracyFrame = __buildAccuracyFrame(modelsTuple, K)

    # *** Modify the Dataframe to Match our LaTeX File *** #
    latexFrame = __accuracyFrameToLatex(modelsTuple, accuracyFrame)
    
    # *** Export the Dataframe as a LaTeX File *** #
    SYSOUT.write(HDR + ' Exporting LaTeX dataframe...')
    
    # set the output file path
    if useNormalize:                       # if we are using the transformations
        stm = inPath.stem + 'Transformed'  # add 'Transformed' to the file name
    
    else:                          # if we are not transforming the data
        stm = inPath.stem + 'Raw'  # add 'Raw' to the file name
    
    outPath = Path.cwd() / 'data' / 'outputs' / (stm + '.tex')  # create the file path

    with open(outPath, "w") as texFile:             # open the selected file
        print(latexFrame.to_latex(), file=texFile)  # & write dataframe to it, converting it to latex
    texFile.close()                                 # close the file
    
    SYSOUT.write(OVERWRITE + ' Export Successful '.ljust(50, '-') + SUCCESS)
    SYSOUT.write('Dataframe converted to LaTeX & Exported\n')
    
    # *** Exit *** #
    SYSOUT.write('\nExiting')
    sys.exit(0)  # close program
    

if __name__ == "__main__":

    main()
