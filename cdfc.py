"""
cdfc.py creates, and evolves a genetic program using Class Dependent Feature Select.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import collections as collect
import logging as log
import math
# import pprint
import random
import sys
import traceback
import typing as typ
import warnings
from pathlib import Path
import Distances as Dst
import numpy as np
from alive_progress import alive_bar, config_handler
from decimal import Decimal
# from pyitlib import discrete_random_variable as drv
# import pprint
from formatting import printError
from _collections import defaultdict
from Tree import Tree
from Node import Node
from objects import cdfcInstance as Instance
import uuid
from copy import deepcopy

# ! Next Steps
# TODO fix accuracy issue
# TODO update comments on new code
# TODO add exceptions to comments

# TODO check copyright on imported packages
# TODO add testing functions

# **************************** Constants/Globals **************************** #
DISTANCE_FUNCTION = "euclidean"               # DISTANCE_FUNCTION is the distance function that should be used
ALPHA: typ.Final = 0.8                        # ALPHA is the fitness weight alpha
BARCOLS = 25                                  # BARCOLS is the number of columns for the progress bar to print
CROSSOVER_RATE: typ.Final = 0.8               # CROSSOVER_RATE is the chance that a candidate will reproduce
ELITISM_RATE: typ.Final = 1                   # ELITISM_RATE is the elitism rate
GENERATIONS: typ.Final = 50                   # GENERATIONS is the number of generations the GP should run for
# GENERATIONS: typ.Final = 25                    # ! value for testing/debugging to increase speed
MAX_DEPTH: typ.Final = 8                      # MAX_DEPTH is the max depth trees are allowed to be & is used in grow/full
# MAX_DEPTH: typ.Final = 3                      # ! value for testing/debugging to make trees more readable
MUTATION_RATE: typ.Final = 0.2                # MUTATION_RATE is the chance that a candidate will be mutated
# ! changes here must also be made in the tree object, and the grow & full functions ! #
OPS: typ.Final = ['add', 'subtract',          # OPS is the list of valid operations on the tree
                  'times', 'max', 'if']
NUM_TERMINALS = {'add': 2, 'subtract': 2,     # NUM_TERMINALS is a dict that, when given an OP as a key, give the number of terminals it needs
                 'times': 2, 'max': 2, 'if': 3}
# ! set the value of R for every new dataset, it is NOT set automatically ! #
TERMINALS: typ.Dict[int, typ.List[int]] = {}  # TERMINALS is a dictionary that maps class ids to their relevant features
TOURNEY: typ.Final = 7                        # TOURNEY is the tournament size
ENTROPY_OF_S = 0                              # ENTROPY_OF_S is used for entropy calculation
FEATURE_NUMBER = 0                            # FEATURE_NUMBER is the number of features in the data set
LABEL_NUMBER = 0                              # LABEL_NUMBER is the number of classes/labels in the data
CLASS_IDS: typ.List[int] = []                 # CLASS_IDS is a list of all the unique class ids
INSTANCES_NUMBER = 0                          # INSTANCES_NUMBER is  the number of instances in the training data
M = 0                                         # M is the number of constructed features (R * Label Number)
POPULATION_SIZE = 0                           # POPULATION_SIZE is the population size
CL_DICTION = typ.Dict[int, typ.Dict[int, typ.List[float]]]
CLASS_DICTS: CL_DICTION = {}                  # CLASS_DICTS is a list of dicts (indexed by classId) mapping attribute values to classes
SEED = 498                                    # SEED the seed used for random values
# random.seed(SEED)  # WARNING: setting this may cause issues with node ID generation!
GLOBAL_COUNTER: typ.Dict[str, int] = {}       # GLOBAL_COUNTER is used during debugging to know how often an event happens
# ++++++++++++++++++++++++ console formatting strings +++++++++++++++++++++++++ #
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'     # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout
# ++++++++++++++++++++++++ configurations & file paths ++++++++++++++++++++++++ #
sys.setrecursionlimit(10000)                                  # set the recursion limit for the program

# np.seterr(divide='ignore')                                    # suppress divide by zero warnings from numpy
np.seterr(all='ignore')
# suppressMessage = 'invalid value encountered in true_divide'  # suppress the divide by zero error from Python
# warnings.filterwarnings('ignore', message=suppressMessage)
suppressMessage2 = 'RuntimeWarning: overflow encountered in float_power'
warnings.filterwarnings('ignore', message=suppressMessage2)

config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', title_length=0, length=20)  # the global config for the loading bars
# config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', force_tty=True, title_length=0, length=10)  # the global config for the loading bars

logPath = str(Path.cwd() / 'logs' / 'cdfc.log')               # create the file path for the log file & configure the logger
# log.basicConfig(level=log.ERROR, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')
log.basicConfig(level=log.DEBUG, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')

# profiler = cProfile.Profile()                                 # create a profiler to profile cdfc during testing
# statsPath = str(Path.cwd() / 'logs' / 'stats.log')            # set the file path that the profiled info will be stored at
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# *********************** Namespaces/Structs & Objects ************************ #
rows: typ.List[Instance] = []  # this will store all of the records read in (the training dat) as a list of rows
# ***************************************************************************** #


class ConstructedFeature:
    # noinspection PyUnresolvedReferences
    """
        Constructed Feature is used to represent a single constructed feature in
        a hypothesis. It contains the tree representation of the feature along
        with additional information about the feature.
    
        :var className: Class id of the class that the feature is meant to distinguish.
        :var tree: Constructed feature's binary decision tree.
        :var size: Number of nodes in the tree
        :var relevantFeatures: List of terminal characters relevant to the feature's class.
        
        :type className: int
        :type tree: Tree
        :type size: int
        :type relevantFeatures: list[int]
        
    """
    USED_IDS: typ.List[str] = []

    def __init__(self, className: int, tree: Tree) -> None:
        """Constructor for the ConstructedFeature object"""
        self.className = className                    # the name of the class this tree is meant to distinguish
        self.tree = tree                              # the root node of the constructed feature
        # noinspection PyTypeChecker
        self.size = tree.size                         # the individual size (the size of the tree)
        self.relevantFeatures = TERMINALS[className]  # holds the indexes of the relevant features
        
        ID: str = str(uuid.uuid4())  # create a unique ID
        if ID in self.USED_IDS:  # if this ID has been used before, print
            printError('WARNING: Duplicate Feature created')
        else:
            self.USED_IDS += ID  # mark ID as used
        self._ID = ID
        # sanityCheckCF(self)  # ! testing purposes only!

    def __str__(self):
        # + simple
        strValue: str = f'[ID {self.ID}|Class {self.className}|Size: {self.size}]'
        # + verbose
        # strValue: str = f'CF -- Class:{self.className}\n{str(self.tree)}'
        return strValue
    
    def __repr__(self):
        return self.__str__()

    @property
    def ID(self):
        return self._ID

    def transform(self, instance: Instance) -> float:
        """
        Takes an instance, transforms it using the decision tree, and return the value computed.
        
        :param instance: Instance to be transformed.
        :type instance: WrapperInstance
        
        :return: The new value computed by running the tree's operations on the provided instance.
        :rtype: float
        """
        
        # Send the tree a list of all the attribute values in a single instance
        featureValues: typ.Dict[int, float] = instance.attributes
        return self.tree.runTree(featureValues, self.className, TERMINALS)


class Hypothesis:
    # noinspection PyUnresolvedReferences
    """
        Hypothesis is a single hypothesis (a GP individual), and will contain a list of constructed features. It
        should have the same number of constructed features for every class id, and should have at least one for
        each class id.
        
        :var features: Dictionary of the constructed features, keyed by class ids, for this hypothesis.
        :var size: Sum of the constructed feature's sizes.
        :var fitness: Calculated fitness score.
        :var distance: Calculated distance value.
        :var averageInfoGain: Average of every features info gain.
        :var maxInfoGain: Largest info gain for any feature.
        
        :type features: dict[int][list[ConstructedFeature]]
        :type size: int
        :type fitness: float
        :type distance: float
        :type averageInfoGain: float
        :type maxInfoGain: float
        
        
        
        Methods:
            getFitness: Get the fitness score of the Hypothesis
            __transform: Transforms a dataset using the trees in the constructed features.
    
        """
    USED_IDS: typ.List[str] = []
    # * type hinting aliases * #
    fDictType = typ.Optional[typ.Dict[int, typ.List[ConstructedFeature]]]
    cfsType = typ.List[ConstructedFeature]

    _fitness: typ.Union[None, int, float] = None  # the fitness score
    _distance: typ.Union[float, int] = 0          # the distance function score
    averageInfoGain: typ.Union[float, int] = -1  # the average info gain of the hypothesis
    maxInfoGain: typ.Union[float, int] = -1      # the max info gain in the hypothesis
    features: typ.Dict[int, typ.List[ConstructedFeature]] = {}
    cfList: typ.List[ConstructedFeature] = []
    # + averageInfoGain & maxInfoGain must be low enough that they will always be overwritten + #
    
    def __init__(self, size: int, cfs: cfsType = None, fDict: fDictType = None) -> None:
        """Constructor for the Hypothesis object"""
        self._size: int = size                                         # the number of nodes in all the cfs

        if (fDict is None) and (cfs is None):
            raise Exception  # both can't be none
        
        if cfs is None:
            cfs = []
            for ls in list(fDict.values()):
                cfs.extend(ls)
        
        # if the dictionary was not passed, then we need to build it
        if fDict is None:

            ftrDictionary: defaultdict[int, list] = defaultdict(list)
            for ftr in cfs:  # add the CFs to the dictionary, keyed by their class id
                classID = ftr.className  # get the class ID of the feature
                # add the feature to a list of features of the same class
                ftrDictionary[classID].append(ftr)

            # change it from a defaultdict to a dict
            fDict = dict(ftrDictionary)

        # set fDict & cfList
        self.features = fDict
        self.cfList = cfs

        ID: str = str(uuid.uuid4())  # create a unique ID
        if ID in self.USED_IDS:  # if this ID has been used before, print
            printError('WARNING: Duplicate Hypothesis created')
        else:
            self.USED_IDS += ID  # mark ID as used
        self._ID = ID

    @property
    def ID(self):
        return self._ID

    def __lt__(self, hyp2: "Hypothesis"):
        return self.fitness < hyp2.fitness

    def __le__(self, hyp2):
        return self.fitness <= hyp2.fitness

    def __eq__(self, hyp2: "Hypothesis"):
        return self.fitness == hyp2.fitness

    def __ne__(self, hyp2: "Hypothesis"):
        return self.fitness != hyp2.fitness

    def __gt__(self, hyp2: "Hypothesis"):
        return self.fitness > hyp2.fitness

    def __ge__(self, hyp2: "Hypothesis"):
        return self.fitness >= hyp2.fitness
    
    def __str__(self) -> str:
        # + Sparse Print
        # strValue: str = self.__print_sparse()
        # + Default Print
        strValue: str = self.__print_basic()
        # + Verbose Print (includes cfList)
        # strValue: str = self.__print_verbose()
        
        return strValue
    
    def __print_sparse(self):
        return f'Hypothesis: {self.ID} | Fitness: {self.fitness}'
    
    def __print_basic(self) -> str:
        strValue: str = f'Hypothesis {self.ID}\n'
        # strValue += f'\tFitness: {self.fitness}\n'    # print fitness
        for k in self.features.keys():  # for each key
            strValue += f'\tClass {k} CFs:\n'  # print the key
        
            for ftr in self.features[k]:  # loop over the feature list
                strValue += f'\t\t{ftr}\n'  # convert each CF
    
        return strValue
    
    def __print_verbose(self) -> str:
        strValue: str = f'Hypothesis\n'
        # strValue += f'\tFitness: {self.fitness}\n'    # print fitness
        for k in self.features.keys():  # for each key
            strValue += f'\tClass {k}:\n'  # print the key
        
            for ftr in self.features[k]:  # loop over the feature list
                strValue += f'\t\t{ftr}\n'  # convert each CF
    
        strValue += f'\tCF List:\n'
    
        for cf in self.cfList:  # loop over the cf list
            strValue += f'\t\t{cf}\n'  # print the cf
    
        return strValue
    
    def print_inside_population(self) -> str:
        """ Used to print inside of Population """
        strValue: str = ''
        for k in self.features.keys():  # for each key
            strValue += f'\t\tClass {k} CFs:\n'  # print the key
        
            for ftr in self.features[k]:  # loop over the feature list
                strValue += f'\t\t\t{ftr}\n'  # convert each CF
    
        return strValue
    
    def __repr__(self):
        return self.__str__()

    def getFeatures(self, classId: int) -> typ.Tuple[ConstructedFeature]:
        """ Gets a list of CFs for a given class"""
        return tuple(self.features[classId])

    @property
    def size(self) -> int:
        return self._size
    
    def updateSize(self) -> None:
        size: int = 0
        for cf in self.cfList:
            size += cf.size
        self._size = size

    @property
    def fitness(self) -> typ.Union[int, float]:
        """
        Getter for fitness & a wrapper for getFitness() which calculates the fitness value.
        
        :return: Hypothesis's fitness.
        :rtype: float or int
        """

        # self._fitness = self.__newFitness()  # ! This is used to test if fitness is being calculated correctly
        
        if self._fitness is None:                # if fitness isn't set
            self._fitness = self.__newFitness()  # set the fitness score
        return self._fitness                     # either way return fitness

    def updateFitness(self) -> None:
        """
        This should be used instead of __newFitness in order to force a new fitness calculation

        """
        
        self._fitness = self.__newFitness()  # set the fitness score
        
        return
    
    def __newFitness(self) -> float:
        """
        __newFitness uses several helper functions to calculate the fitness of a Hypothesis. This
        should only be called by it's wrapper function fitness(). This is because __newFitness will
        calculate a new fitness value everytime it's called, even if the fitness already has a value
        
        :return: Fitness value of a Hypothesis.
        :rtype: float
        """
        # !!! debugging !!! #
        # global GLOBAL_COUNTER
        # # if None set, otherwise increment
        # if GLOBAL_COUNTER.get('newFitness'):
        #     GLOBAL_COUNTER['newFitness'] = 1
        # else:
        #     GLOBAL_COUNTER['newFitness'] += 1
        # !!! debugging !!! #
        
        def DbCalculation(S: typ.List[Instance]) -> float:
            """ Used to calculate Db """

            index = 0
            minimum: typ.Dict[int, float] = {}
            # loop over every instance
            for i in S:
                
                Ci = i.className  # get the class name of the instance
                
                differentClasses: typ.List[typ.Optional[Instance]] = []

                for j in S:
                    Cj = j.className  # get the class name of the instance

                    if Ci != Cj:  # if the instances i & j are from different classes
                        differentClasses.append(j)

                # different classes now contains all the instance that are in different classes
                minimum[index] = Dst.computeDistance(DISTANCE_FUNCTION, i.vList, differentClasses[0].vList)
                
                for j in differentClasses:  # create the list on mins
                    
                    distance = Dst.computeDistance(DISTANCE_FUNCTION, i.vList, j.vList)

                    # if we have found a lower value for the distance of i, update min
                    if distance < minimum[index]:
                        minimum[index] = distance

                index += 1
            # sum everything & return
            value: typ.Union[int, float] = sum(minimum.values())
            return (1 / len(S)) * value

        def DwCalculation(S: typ.List[Instance]) -> float:
            """ Used to calculate Dw """

            index = 0
            maximum: typ.Dict[int, float] = {}
            # loop over every instance
            for i in S:
        
                Ci = i.className  # get the class name of the instance
        
                sameClasses: typ.List[typ.Optional[Instance]] = []
        
                for j in S:
                    Cj = j.className  # get the class name of the instance
            
                    if (Ci == Cj) and (i is not j):  # if the instances i & j are from different classes
                        sameClasses.append(j)
        
                # same classes now contains all the instances that are in the same classe
                maximum[index] = Dst.computeDistance(DISTANCE_FUNCTION, i.vList, sameClasses[0].vList)
        
                for j in sameClasses:  # create the list on maxs

                    distance = Dst.computeDistance(DISTANCE_FUNCTION, i.vList, j.vList)
            
                    # if we have found a higher value for the distance of i, update max
                    if distance > maximum[index]:
                        maximum[index] = distance

                index += 1
            # sum everything & return
            value = sum(maximum.values())
            return (1 / len(S)) * value
        
        def Distance(values: typ.List[Instance]) -> float:
            """"
            Distance calculates the distance value of the Hypothesis
            
            :param values: List of the instances who's distance we wish to compute.
            :type values: list
            
            :return: Distance value calculated using the chosen distance function.
            :rtype: float
            """

            # ******** Calculate Db & Dw ******** #
            Db: float = DbCalculation(values)
            Dw: float = DwCalculation(values)
            
            # *************** Create Exponent *************** #
            exp: float = -5 * (Db - Dw)  # create the exponent
            # log.debug(f'Exponent: -5 * ({Db} - {Dw}) = {exp}')
            
            # *************** Calculate Power *************** #
            try:
                # Note: Python supports arbitrarily large ints, but not floats
                pwr = np.float_power(np.e, exp)  # this can cause overflow
                # pwr = np.power(np.e, exp)
            except (OverflowError, RuntimeWarning):
                try:  # attempt recovery
                    # try using Decimal type to hold large floats
                    pwrFix = Decimal(exp).exp()  # e**exp
                    print('OVERFLOW ERROR occurred during distance calculation')
                    print(f'value: {pwrFix.to_eng_string()}')
                    # try to round the Decimal to 4 places & then convert to a float
                    r = 1 / (1 + float(pwrFix.quantize(Decimal('1.000000000000'))))
                    return r
                except (RuntimeError, TypeError, OverflowError):
                    print('Recovery was impossible, exiting')
                    sys.exit(-1)  # exit with an error
            
            pwr = np.round(pwr, 12)
            # log.debug(f'Power of e = {pwr}')

            return 1 / (1 + pwr)
    
        def __entropy(partition: typ.List[Instance]) -> float:
            """
            Calculates the entropy of a Hypothesis
            
            :param partition: A section of the input data.
            :type partition: list
            
            :return: Entropy value of the partition.
            :rtype: float
            """
            
            p: typ.Dict[int, int] = {}   # p[classId] = number of instances in the class in the partition sv
            # for instance i in a partition sv
            for i in partition:  # type: Instance
                if i.className in p:       # if we have already found the class once,
                    p[i.className] += 1  # increment the counter
                    
                else:                   # if we have not yet encountered the class
                    p[i.className] = 1  # set the counter to 1

            calc = 0
            for c in p.keys():  # for class in the list of classes in the partition sv
                # perform entropy calculation
                pi = p[c] / len(partition)
                calc -= pi * math.log(pi, 2)

            # log.debug('Finished entropy() method')

            return calc

        def __conditionalEntropy(feature: ConstructedFeature) -> float:
            """
            Calculates the conditional entropy of a Hypothesis
            
            :param feature: Constructed feature who's conditional entropy is needed.
            :type feature: ConstructedFeature
            
            :return: Entropy value of the passed feature.
            :rtype: float
            """
    
            # log.debug('Starting conditionalEntropy() method')

            # this is a feature struct that will be used to store feature values
            # with their indexes/IDs in CFs
            ft = collect.namedtuple('ft', ['id', 'value'])
            
            # key = CF(Values), Entry = instance in training data
            partition: typ.Dict[float, typ.List[Instance]] = {}
            
            s = 0                                # used to sum CF's conditional entropy
            used = TERMINALS[feature.className]  # get the indexes of the used features
            v = []                               # this will hold the used features ids & values
            for i in rows:                       # type: Instance
                # loop over all instances
                # get CF(v) for this instance (i is a Instance struct which is what __transform needs)
                cfv = feature.transform(i)  # needs the values for an instance

                # get the values in this instance i of the used feature
                for u in used:  # loop over all the indexes of used features
                    # create a ft where the id is u (a used index) &
                    # where the value is from the instance
                    v.append(ft(u, i.attributes[u]))

                if cfv in partition:            # if the partition exists
                    partition[cfv].append(i)  # add the instance to it
                else:                     # if the partition doesn't exist
                    partition[cfv] = [i]  # create it

            for e in partition.keys():
                s += (len(partition[e])/INSTANCES_NUMBER) * __entropy(partition[e])

            # log.debug('Finished conditionalEntropy() method')

            return s  # s holds the conditional entropy value

        gainSum = 0  # the info gain of the hypothesis
        
        try:
            # ! this is throwing a key error. Why?
            # ! this does iterate through the class ids correctly...
            for classId in CLASS_IDS:  # Loop over all the class ids
                for f in self.features[classId]:  # Loop over all the features for that class id
    
                    # ********* Entropy calculation ********* #
                    condEntropy = __conditionalEntropy(f)  # find the conditional entropy
        
                    # ******** Info Gain calculation ******* #
                    # ? maybe the problem is with ENTROPY_OF_S? should it be a dict?
                    # ?  Not all classes have the same entropy, right?
                    f.infoGain = ENTROPY_OF_S - condEntropy  # H(class) - H(class|f)
                    gainSum += f.infoGain                    # update the info sum
        
                    # updates the max info gain of the hypothesis if needed
                    if self.maxInfoGain < f.infoGain:
                        self.maxInfoGain = f.infoGain
        except KeyError as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            # printError(''.join(traceback.format_stack()))  # print stack trace
            
            printError(f'Encountered KeyError: {str(err)}, on line: {lineNm}')  # print the error
            # printError(f"Encountered during fitness calculation number {GLOBAL_COUNTER['newFitness']}")
            printError(f'Class IDs: {CLASS_IDS}')
            printError(f'Class Keys in Hypothesis: {list(self.features.keys())}')
            printError('Features in Hypothesis:')
            
            for k in self.features.keys():       # for each key
                printError(f'Key {k}:')                 # print the key
                for ftr in self.features[k]:  # loop over the feature list
                    printError(f'\t{ftr}')
            sys.exit(-1)
            
        # calculate the average info gain using formula 3
        term1 = gainSum+self.maxInfoGain
        term2 = (M+1)*(math.log(LABEL_NUMBER, 2))
        self.averageInfoGain += term1 / term2

        # set size
        # * this must be based off the number of nodes a tree has because
        # * the depth will be the same for all of them

        # *********  Distance Calculation ********* #
        self._distance = Distance(self.__transform())  # calculate the distance using the transformed values

        # ********* Final Calculation ********* #
        term1 = ALPHA*self.averageInfoGain
        term2 = (1-ALPHA)*self._distance
        term3 = (math.pow(10, -7)*self.size)
        final = term1 + term2 - term3
        # ********* Finish Calculation ********* #

        # log.debug('Finished getFitness() method')
        self._fitness = final
        return final

    def runCDFC(self, data: np.array) -> np.array:
        """
        runCDFC transforms a dataset using the trees in the constructed features, and is use by cdfcProject
        to convert (reduce) data using class dependent feature construction.
        
        :param data: A dataset to be converted by the CDFC algorithm.
        :type data: np.array

        :return: A dataset that has been converted by the algorithm.
        :rtype: np.array
        """

        # this is a type hint alias for the values list where: [classID(int), value(float), value(float), ...]
        valueList = typ.List[typ.Union[int, float]]
        # a list of values lists
        transformedData: typ.List[valueList] = []  # this will hold the new transformed dataset after as it's built

        # loop over each row/instance in data and transform each row using each constructed feature.
        # We want to transform a row once, FOR EACH CF in a Hypothesis.
        for d in data:  # type: np.array
            # This will hold the transformed values for each constructed feature until we have all of them.
            values: valueList = [d[0]]  # values[0] = class name(int), values[0:] = transformed values (float)

            # here we want to create a np array version of an Instance object of the form
            # (classID, values[]), for each row/instance
            
            # for each row, convert that row using each constructed feature (where f is a constructed feature)
            for f in self.cfList:  # type: ConstructedFeature
                # convert the numpy array to an instance & transform it
                currentLine: float = f.transform(Instance(d[0], dict(zip(range(len(d[1:])), d[1:])), d[1:]))
                # add the value of the transformation to the values list
                values.append(currentLine)
            
            # NOTE: now we want to add the np array Instance to the array of all the transformed Instances to create
            # a new data set
            transformedData.append(values)

        # convert the data set from a list of lists to a numpy array
        return np.array([np.array(x) for x in transformedData])
        
    def __transform(self) -> typ.List[Instance]:
        """
        __transform transforms a dataset using the trees in the constructed features. This is used internally
        during training, and will be done over the Rows constant. This is produces data of a different format
        then runCDFC.
            
        :return: A new dataset, created by transforming the original one.
        :rtype: list
        """
        
        transformed: typ.List[Instance] = []  # this will hold the transformed values
    
        for r in rows:    # for each Instance in the provided training data
            values = []   # this will hold the calculated values for all the constructed features

            for f in self.cfList:            # __transform the original input using each constructed feature
                values.append(f.transform(r))  # append the transformed values for a single CF to values
            
            # each Instance will hold the new values for an Instance & className, and
            # transformed will hold all the instances for a hypothesis
            vls = dict(zip(range(len(values)), values))  # create a dict of the values keyed by their index
            transformed.append(Instance(r.className, vls, values))

        # log.debug('Finished __transform() method')
        
        return transformed  # return the list of all instances


class Population:
    """
    Population is a list of Hypothesis, and a generation number. It is largely
    just a namespace.
    
    :var candidateHypotheses: list of hypotheses
    :var generationNumber: current generation number
    
    :type candidateHypotheses: list
    :type generationNumber: int
    """

    def __init__(self, candidateHypotheses: typ.List[Hypothesis], generationNumber: int) -> None:
        """Constructor for the Population object"""
        
        # a list of all the candidate hypotheses
        self.candidateHypotheses: typ.List[Hypothesis] = candidateHypotheses
        # this is the number of this generation
        self.generation = generationNumber
        # this will store the elite
        self.elite: typ.Optional[Hypothesis] = None

    def __str__(self) -> str:
        # + Print Default
        out: str = self.__print_basic()
        # + Print Verbose (prints Hypothesis details)
        # out: str = self.__print_verbose()
        
        return out
        
    def __repr__(self) -> str:
        return self.__str__()

    def __print_verbose(self):
        out: str = f'Population {self.generation}\n'
        out += f'\tElite:      {self.elite.ID} | Fitness: {self.elite.fitness}\n'
        for h in self.candidateHypotheses:
            out += f'\tHypothesis: {h.ID} | Fitness: {h.fitness}\n'
            out += f'{h.print_inside_population()}\n'
    
        return out
    
    def __print_basic(self):
        out: str = f'Population\n'
        out += f'\tElite:      {self.elite.ID} | Fitness: {self.elite.fitness}\n'
        for h in self.candidateHypotheses:
            out += f'\tHypothesis: {h.ID} | Fitness: {h.fitness}\n'
    
        return out

    def __tournament(self) -> Hypothesis:  # ! check for reference issues here
        """
        Used by evolution to selection the parent(s)

        :return: The best hypothesis that tournament found.
        :rtype: Hypothesis
        """
    
        # **************** Tournament Selection **************** #
        # get a list including every valid index in candidateHypotheses
        positions: typ.List[int] = list(range(len(self.candidateHypotheses)))
        first: typ.Optional[Hypothesis] = None  # the tournament winner
        score = 0  # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
        
            randomIndex: int = random.choice(positions)  # choose a random index in p.candidateHypotheses
        
            candidate: Hypothesis = self.candidateHypotheses[randomIndex]  # get the hypothesis at the random index
            
            # ! Remove Elite Test ! #
            if self.elite is not None:
                while candidate.ID == self.elite.ID:
                    randomIndex: int = random.choice(positions)  # get a new random index
                    candidate: Hypothesis = self.candidateHypotheses[randomIndex]  # get another Hypoth
            # ! ! ! ! ! ! ! ! ! ! ! #
        
            positions.remove(randomIndex)  # remove the chosen value from the list of indexes (avoids duplicates)
            fitness = candidate.fitness  # get that hypothesis's fitness score
        
            if first is None:  # if first has not been set,
                first = candidate  # then  set it
        
            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it,
                score = fitness  # update the score to higher fitness,
    
        try:
            if first is None:
                raise Exception(f'ERROR: Tournament could not set first correctly, first = {first}')
        except Exception as err2:
            lineNm2 = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Tournament could not set first correctly, first = {first}, line number = {lineNm2}')
            print(f'{str(err2)}, line = {lineNm2}')
            sys.exit(-1)  # exit on error; recovery not possible
    
        # log.debug('Finished Tournament method')
    
        # print('Tournament Finished')  # ! for debugging only!
    
        return first
        # ************ End of Tournament Selection ************* #

    def __crossoverTournament(self) -> typ.Tuple[Hypothesis, Hypothesis]:
        """
        Used by crossover to selection the parents. It differs from the normal tournament
        because it will return two unique hypotheses.

        :return: Two hypothesis that tournament found.
        :rtype: typ.Tuple[Hypothesis, Hypothesis]
        """
    
        # **************** Tournament Selection **************** #
        # get a list including every valid index in candidateHypotheses
        positions: typ.List[int] = list(range(len(self.candidateHypotheses)))
        first = None  # the tournament winner
        firstIndex = None  # the index of the winner
        score = 0  # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
        
            randomIndex: int = random.choice(positions)  # choose a random index in p.candidateHypotheses
            candidate: Hypothesis = self.candidateHypotheses[randomIndex]  # get the hypothesis at the random index
            
            # ! Remove Elite Test ! #
            if self.elite is not None:
                while candidate.ID == self.elite.ID:
                    randomIndex: int = random.choice(positions)  # get a new random index
                    candidate: Hypothesis = self.candidateHypotheses[randomIndex]  # get another Hypoth
            # ! ! ! ! ! ! ! ! ! ! ! #
            
            positions.remove(randomIndex)  # remove the chosen value from the list of indexes (avoids duplicates)
            fitness = candidate.fitness  # get that hypothesis's fitness score
        
            if (first is None) or (score < fitness):  # if first has not been set, or candidate if more fit
                first = candidate  # then update it
                score = fitness  # then update the score to higher fitness
                firstIndex = randomIndex  # finally update the index of the winner
    
        positions = list(range(len(self.candidateHypotheses)))
        try:
            positions.remove(firstIndex)  # remove the last winner from consideration
        except ValueError as err:
            print(str(err))
            print(f'index is {firstIndex}\nlist of positions is {positions}')
            sys.exit(-1)  # exit on error; recovery not possible
    
        second = None  # the 2nd tournament winner
        secondIndex = None  # the index of the winner
        score = 0  # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
        
            randomIndex: int = random.choice(positions)  # choose a random index in p.candidateHypotheses
            candidate: Hypothesis = self.candidateHypotheses[randomIndex]  # get the hypothesis at the random index
            
            # ! Remove Elite Test ! #
            if self.elite is not None:
                while candidate.ID == self.elite.ID:
                    randomIndex: int = random.choice(positions)  # get a new random index
                    candidate: Hypothesis = self.candidateHypotheses[randomIndex]  # get another Hypoth
            # ! ! ! ! ! ! ! ! ! ! ! #
            
            positions.remove(randomIndex)  # remove the chosen value from the list of indexes (avoids duplicates)
            fitness = candidate.fitness  # get that hypothesis's fitness score
        
            if (second is None) or (score < fitness):  # if 2nd has not been set, or candidate is more fit
                second = candidate  # then update it
                score = fitness  # then update the score to higher fitness
                secondIndex = randomIndex  # get the index of the winner
    
        try:
            if first is None or second is None:
                raise Exception(f'ERROR: Tournament could not set first or second correctly, first = {first}')
        except Exception as err2:
            lineNm2 = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Tournament could not set first correctly, first = {first}, line number = {lineNm2}')
            print(f'{str(err2)}, line = {lineNm2}')
            sys.exit(-1)  # exit on error; recovery not possible
    
        # log.debug('Finished Tournament method')
        one: Hypothesis = self.candidateHypotheses[firstIndex]
        two = self.candidateHypotheses[secondIndex]
        if one.ID == two.ID:
            printError('CrossoverTournament failed to send to unique Hyptheses,\ninstead sent to duplicates')
        return one, two
        # ************ End of Tournament Selection ************* #

    def mutate(self) -> None:
        """
        Finds a random node and builds a new sub-tree starting at it. Currently mutate
        uses the same grow & full methods as the initial population generation without
        an offset. This means that mutate trees will still obey the max depth rule.
        This will create a new CF and overwrite the one it chose to mutate.
        """
    
        # * Get a Random Hypothesis using Tournament * #
        parent: Hypothesis = self.__tournament()
        
        # * From the Hypothesis, Get a Random CF * #
        randIndex: int = random.randint(0, M - 1)  # get a random index that's valid in cfList
        randCF: ConstructedFeature = parent.cfList[randIndex]  # use it to get a random CF
        
        # * From the CF, Get a Random Node * #
        tree: Tree = randCF.tree                   # get the tree from the CF
        nodeID: str = randCF.tree.getRandomNode()  # get a random node ID from the CF's tree
        node: Node = tree.getNode(nodeID)          # use the ID to get the Node object

        # TODO: Check remove children
        # * Remove the Children of the Node * #
        tree.removeChildren(nodeID)  # make sure the IDs of the children get deleted
        
        # * Perform Mutation on the Selected Node * #
        # if we are at max depth or choose TERM, place an index of a relevant feature
        if random.choice(['OPS', 'TERM']) == 'TERM' or tree.getDepth(nodeID, tree.root.ID) == MAX_DEPTH:
            terminals = randCF.relevantFeatures   # get the indexes of the relevant features
            node.data = random.choice(terminals)  # place a random index in the Node
            
        # if we are not at the max depth & have chosen OP, add a random operation & generate subtree
        else:
            node.data = random.choice(OPS)  # put a random operation in the node
    
            # randomly decide (50/50) which method to use to construct the new tree (grow or full)
            if random.choice(['Grow', 'Full']) == 'Grow':  # * Grow * #
                tree.grow(randCF.className, nodeID, MAX_DEPTH, TERMINALS, 0)
            else:                                          # * Full * #
                tree.full(randCF.className, nodeID, MAX_DEPTH, TERMINALS, 0)
    
        # * Force an Update of the Fitness Score * #
        parent.updateFitness()
    
        # * Handle Elitism * #
        # NOTE: because Hypothesis might be changed in place later, Elite must store a copy
        if self.elite is None:
            self.elite = deepcopy(parent)
        # if the the new Hypothesis has a better fitness, update Elite
        elif self.elite.fitness < parent.fitness:
            self.elite = deepcopy(parent)
    
        return
    
    def crossover(self) -> None:
        """Performs the crossover operation on two trees"""

        # ********** Get Two Random Hypotheses ********** #
        parent1: Hypothesis
        parent2: Hypothesis
        parent1, parent2 = self.__crossoverTournament()
        # *********************************************** #
        
        # ************************ Get a Random Class & Index ************************ #
        randIndex = random.randint(0, M - 1)  # get a random index that's valid in cfList
        classID = random.choice(CLASS_IDS)  # choose a random class
        # **************************************************************************** #
        
        # * Use the Random Class & Index to get a CF from each Hypothesis * #
        # + Feature 1
        feature1: ConstructedFeature = parent1.getFeatures(classID)[randIndex]
        tree1: Tree = feature1.tree  # get the tree
        # + Feature 2
        feature2: ConstructedFeature = parent2.getFeatures(classID)[randIndex]
        tree2: Tree = feature2.tree  # get the tree
        # ***************************************************************** #
    
        # * Check that the Hypotheses, Features, and Trees are not Duplicates * #
        try:
            if parent1.ID == parent2.ID:  # if the Hypotheses are the same
                raise AssertionError('In Crossover, parent 1 is parent 2')
            if feature1.ID == feature2.ID:  # if the Features are the same
                raise AssertionError('In Crossover, feature 1 is feature 2')
            if tree1 == tree2:  # if the Trees are the same
                raise AssertionError('In crossover, Tree1 is Tree2')
        except AssertionError as err:
            printError(str(err))
            print(f'Parent 1 ID: {parent1.ID}')
            print(f'Parent 2 ID: {parent2.ID}')
            print(f'Feature 1: {feature1}')
            print(f'Feature 2: {feature2}')
            print(f'Tree 1:\n{tree1}')
            print(f'Tree 2:\n{tree2}')
            sys.exit(-1)
        # ********************************************************************** #
    
        # ***** Get a Random Node from each Tree ***** #
        nodeID_1: str = tree1.getRandomNode()
        nodeID_2: str = tree2.getRandomNode()
        # ******************************************** #
        
        # *************** Use the Random Nodes to Create Two Subtrees *************** #
        branch1: str
        p1: str
        treeFromFeature1: Tree
        # Get the Branch & Parent of the Subtree from CF1. This will tell use where to add it in CF 2
        # Get the Subtree from CF1. This will be move to CF2 (nodeF1 will be root)
        treeFromFeature1, p1, branch1 = tree1.removeSubtree(nodeID_1)

        branch2: str
        p2: str
        treeFromFeature2: Tree
        # Get the Branch & Parent of the Subtree from CF1. This will tell use where to add it in CF 2
        # Get the Subtree from CF2. This will be move to CF1 (nodeF2 will be root)
        treeFromFeature2, p2, branch2 = tree2.removeSubtree(nodeID_2)
        # **************************************************************************** #

        # ************************** swap the two subtrees ************************** #
        # Add the Subtree from CF2 to the tree in CF1 (in the same location that the subtree1 was cut out)
        tree1.addSubtree(subtree=treeFromFeature2, newParent=p1, orphanBranch=branch1)
        # Add the Subtree from CF1 to the tree in CF2 (in the same location that the subtree2 was cut out)
        tree2.addSubtree(subtree=treeFromFeature1, newParent=p2, orphanBranch=branch2)
        # **************************************************************************** #
    
        # *************** Force the Hypotheses to Update their Values *************** #
        # + Parent 1
        parent1.updateSize()  # force an update of size & fitness
        parent1.updateFitness()
        # + Parent 2
        parent2.updateSize()  # force an update of size & fitness
        parent2.updateFitness()
        # **************************************************************************** #
 
        # * Deal with Elitism * #
        # Figure out which of the two changed Hypotheses has a higher fitness
        if parent1.fitness >= parent2.fitness:
            better: Hypothesis = parent1
        else:
            better: Hypothesis = parent2
        
        if self.elite is None:  # if elite hasn't been set yet, set it
            self.elite = deepcopy(better)
        # If one of the changed Hypotheses has a better fitness than elite, update it
        elif better.fitness > self.elite.fitness:
            self.elite = deepcopy(better)

        return

    def evolve(self, bar) -> None:
        
        bar.text('Starting new generation')
        
        for pop in range(POPULATION_SIZE):  # ? should this be self.hypotheses?
            
            probability = random.uniform(0, 1)  # get a random number between 0 & 1

            # ***************** Mutate ***************** #
            # if True:  # !! Debugging Only !!
            if probability < MUTATION_RATE:  # if probability is less than mutation rate, mutate
                bar.text('mutating...')      # update user
                self.mutate()                # perform mutation
            # ************* End of Mutation ************* #
            # **************** Crossover **************** #
            else:                            # if probability is greater than mutation rate, use crossover
                bar.text('crossing...')      # update user
                self.crossover()             # perform crossover operation
            # ************* End of Crossover ************* #
        bar.text('Generation complete')
        return
    # ***************** End of Namespaces/Structs & Objects ******************* #


def createInitialPopulation() -> Population:
    """
    Creates the initial population by calling createHypothesis() the needed number of times.
    
    :return: The initial population.
    :rtype: Population
    """
    
    def createHypothesis() -> Hypothesis:
        """
        Helper function that creates a single hypothesis
        
        :return: A new hypothesis
        :rtype: Hypothesis
        """
        
        cfDictionary = {}
        cfList = []
        size = 0

        # *** create M CFs for each class *** #
        for cid in CLASS_IDS:  # loop over the class ids
            
            # print(f'Creating CFs for class {cid}...')  # ! debugging only
            
            ftrs: typ.List[ConstructedFeature] = []  # empty the list of features
            
            for k in range(M):  # loop M times so M CFs are created
                
                tree = Tree()   # create an empty tree
                
                if random.choice([True, False]):         # *** use grow *** #
                    # print('Grow chosen')  # ! debugging
                    tree.grow(cid, tree.root.ID, MAX_DEPTH, TERMINALS, 0)  # create tree using grow
                    # print('Grow Finished')
                else:                                    # *** use full *** #
                    # print('Full chosen')  # ! debugging
                    tree.full(cid, tree.root.ID, MAX_DEPTH, TERMINALS, 0)    # create tree using full
                    # print('Full Finished')
    
                cf = ConstructedFeature(cid, tree)       # create constructed feature
                ftrs.append(cf)                          # add the feature to the list of features
                cfList.append(cf)

                # print(f'\t {k} CF is {cf}')  # ! debugging
                
                size += cf.size

            # add the list of features for class cid to the dictionary, keyed by cid
            cfDictionary[cid] = ftrs
            
        # !!! For debugging only !!!
        # print('Hypothesis Created')
        # for cf in cfList:  # for each cf
        #     cf.tree.checkForMissingKeys()
        # # if we pass this point then the key problem comes from changes made by mutation and crossover
        # print('\tNo missing keys detected\n')
        # !!! For debugging only !!!
        # create a hypothesis & return it
        return Hypothesis(cfs=cfList, fDict=cfDictionary, size=size)

    hypothesis: typ.List[Hypothesis] = []

    # print(f'Class IDs: {CLASS_IDS}')  # ! debugging
    with alive_bar(POPULATION_SIZE, title="Initial Population") as bar:  # declare your expected total
        # creat a number hypotheses equal to pop size
        for p in range(POPULATION_SIZE):           # iterate as usual
            hypothesis.append(createHypothesis())  # create & add the new hypothesis to the list
            bar()                                  # update progress bar

    pop = Population(hypothesis, 0)
    # print(pop.candidateHypotheses[2].cfList[1].tree)  # ! debugging only
    # sanityCheckPopReference(pop)  # ! testing purposes only!
    # sanityCheckPop(hypothesis)  # ! testing purposes only!
    return pop


# ********** Sanity Check Functions used for Debugging ********** #
# ! testing purposes only!
def sanityCheckPopReference(pop: Population):
    """ Used to make sure that every Hypothesis is unique"""
    log.debug('Starting Population Reference Check...')
    
    noDuplicates = []
    # loop over every GPI in the pop
    for i in pop.candidateHypotheses:  # type: Hypothesis
        
        sanityCheckHypReference(i)  # check that the hypoth is fine

        if i not in noDuplicates:  # if it isn't in the list, add it
            noDuplicates.append(i)
    
    # if there were duplicates, raise an error
    if len(noDuplicates) != len(pop.candidateHypotheses):
        raise AssertionError

    log.debug('Population Reference Check Passed')


def sanityCheckHypReference(hyp: Hypothesis):
    """ Used to make sure that every CF is unique"""
    noDuplicates = []
    # loop over every GPI in the pop
    for i in hyp.cfList:  # type: ConstructedFeature
        
        if i not in noDuplicates:
            noDuplicates.append(i)
    
    # if there were duplicates, raise an error
    if len(noDuplicates) != len(hyp.cfList):
        raise AssertionError


# ! testing purposes only!
def sanityCheckCF(cf: ConstructedFeature):
    """Used in debugging to check a Constructed Feature"""
    log.debug('Starting Constructed Feature Sanity Check...')
    cf.transform(rows[0])
    log.debug('Constructed Feature Sanity Check Passed')


def check_for_cf_copies(pop: Population):
    """
    Checks to make sure that none of the Constructed Features
    are copies of each other
    """
    
    ids: typ.Dict[str, Hypothesis] = {}
    
    for hyp in pop.candidateHypotheses:  # for each hypothesis
        
        for cf in hyp.cfList:  # loop over the CFs
            ID: str = cf.ID  # get the ID
            
            # if the ID is not in the dictionary then we have
            # not encountered it before so add it
            if ids.get(ID) is None:
                ids[ID] = hyp
            # if it is not None they it is a duplicate so throw an error
            else:
                printError('Check for CF Copies Failed')
                printError(f'Error: Duplicate Feature ID found during check, ID {ID}')
                print(hyp)  # print this hypoth
                print(ids.get(ID))  # print the hypoth with the duplicate
                sys.exit(-1)
    print('Check for CF Copies Passed!')
    return


def check_CF_number(hypoths: typ.List[Hypothesis]):
    
    result: typ.List[bool] = []
    badHypoths: typ.List[Hypothesis] = []
    
    # for each hypothesis: if the number of CFs is correct
    # add true, otherwise add false
    for h in hypoths:
        # get the keys of the dictionary storing the features as a list &
        # take the value in the first index
        index: int = list(h.features.keys())[0]
        # then use that index to access one of the feature lists for a class
        if len(h.features[index]) == M:
            result.append(True)
        else:
            result.append(False)
            badHypoths.append(h)
    
    # if there was a Hypoth with the wrong number of CFs
    if False in result:
        # get the the number of times a wrong size Hypoth was found
        occurs: int = result.count(False)
        try:
            raise AssertionError
        except AssertionError:
            msg: str = f"CF number check failed! Number of Hypotheses with incorrect size: {occurs}\n"
            printError(msg)
            log.error(msg)
            print(f"M = {M}")
            for h in badHypoths:  # print the incorrect hypoths
                print(h)
            
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
    else:
        print('CF number check passed!')
        return


def check_hypotheses(h1: Hypothesis, h2: Hypothesis):
    if h1.ID == h2.ID:
        printError("check Hypotheses found duplicates after Crossover!")
        sys.exit(-1)
    
    for cf1 in h1.cfList:  # loop over the list of CFs
        for cf2 in h2.cfList:  # and compare it to every CF in h2
            if cf1.ID == cf2.ID:
                printError("check Hypotheses found duplicates after Crossover!")
                sys.exit(-1)
# *************************************************************** #


def cdfc(dataIn, distanceFunction) -> Hypothesis:
    """
    cdfc is the 'main' of cdfc.py. It is called by cdfcProject which passes dataIn.
    It then creates an initial population & evolves several hypotheses. After going
    through a set amount ofr generations it returns a Hypothesis object.

    :param dataIn: Index 0 contains the values of the global constants that cdfc needs, and
                   index 1 contains the TERMINALS dictionary.
    :param distanceFunction:

    :type dataIn: tuple
    :type distanceFunction:
    
    :return: Hypothesis with the highest fitness score.
    :rtype: Hypothesis
    """

    values = dataIn[0]
    
    # makes sure we're using global variables
    global FEATURE_NUMBER
    global CLASS_IDS
    global POPULATION_SIZE
    global INSTANCES_NUMBER
    global LABEL_NUMBER
    global M
    global rows
    global ENTROPY_OF_S  # * used in entropy calculation * #
    global CLASS_DICTS
    global TERMINALS
    global DISTANCE_FUNCTION
    
    # print('setting variables')  # ! debugging only!
    
    # Read the values in the dictionary into the constants
    FEATURE_NUMBER = values['FEATURE_NUMBER']
    CLASS_IDS = values['CLASS_IDS']
    DISTANCE_FUNCTION = distanceFunction
    DISTANCE_FUNCTION = ['DISTANCE_FUNCTION']
    # POPULATION_SIZE = values['POPULATION_SIZE']
    POPULATION_SIZE = 10
    INSTANCES_NUMBER = values['INSTANCES_NUMBER']
    LABEL_NUMBER = values['LABEL_NUMBER']
    M = values['M']
    rows = values['rows']
    ENTROPY_OF_S = values['ENTROPY_OF_S']
    CLASS_DICTS = values['CLASS_DICTS']
    TERMINALS = dataIn[1]

    # print('variables set')  # ! debugging only!
    
    # *********************** Run the Algorithm *********************** #
    # print('creating initial pop')  # ! debugging only!
    currentPopulation: Population = createInitialPopulation()     # run initialPop/create the initial population

    # check_for_cf_copies(currentPopulation)  # ! Debugging Only !
    # check_CF_number(currentPopulation.candidateHypotheses)  # ! Debugging Only !
    
    SYSOUT.write(NO_OVERWRITE + ' Initial population generated '.ljust(50, '-') + SUCCESS)
    
    # loop, evolving each generation. This is where most of the work is done
    with alive_bar(GENERATIONS, title="Generations") as bar:

        for gen in range(GENERATIONS):  # iterate as usual
            
            currentPopulation.evolve(bar)  # * Update the Population in Place * #
            currentPopulation.generation += 1  # update the generation number
            # !!!!!!!!!!!!!!!!!!!!!!! Debugging Only !!!!!!!!!!!!!!!!!!!!!!! #
            # check_for_cf_copies(currentPopulation)
            print(str(currentPopulation))
            log.debug(str(currentPopulation))
            # print('Calling CF Number Check...')
            # check_CF_number(currentPopulation.candidateHypotheses)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

            bar()  # update bar now that a generation is finished
    # ***************************************************************** #

    # ****************** Return the Best Hypothesis ******************* #

    bestHypothesis: Hypothesis = max(currentPopulation.candidateHypotheses)
    bestHypothesis = max([bestHypothesis, currentPopulation.elite])

    return bestHypothesis  # return the best hypothesis generated
