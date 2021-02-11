"""
cdfc.py creates, and evolves a genetic program using Class Dependent Feature Select.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import collections as collect
import logging as log
import math
import random
import sys
# import traceback
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
from copy import deepcopy
# from copy import copy as cpy

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
# MAX_DEPTH: typ.Final = 8                      # MAX_DEPTH is the max depth trees are allowed to be & is used in grow/full
MAX_DEPTH: typ.Final = 3                      # ! value for testing/debugging to make trees more readable
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
M = 0                                         # M is the number of constructed features
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

np.seterr(divide='ignore')                                    # suppress divide by zero warnings from numpy
suppressMessage = 'invalid value encountered in true_divide'  # suppress the divide by zero error from Python
warnings.filterwarnings('ignore', message=suppressMessage)
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

    def __init__(self, className: int, tree: Tree) -> None:
        """Constructor for the ConstructedFeature object"""
        self.className = className                    # the name of the class this tree is meant to distinguish
        self.tree = tree                              # the root node of the constructed feature
        # noinspection PyTypeChecker
        self.size = tree.size                         # the individual size (the size of the tree)
        self.relevantFeatures = TERMINALS[className]  # holds the indexes of the relevant features
        # sanityCheckCF(self)  # ! testing purposes only!

    def __str__(self):
        # + simple
        strValue: str = f'||CF for Class {self.className}| Size: {self.size}||'
        # + verbose
        # strValue: str = f'CF -- Class:{self.className}\n{str(self.tree)}'
        return strValue
    
    def __repr__(self):
        return self.__str__()

    def transform(self, instance: Instance) -> float:
        """
        Takes an instance, transforms it using the decision tree, and return the value computed.
        
        :param instance: Instance to be transformed.
        :type instance: Instance
        
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
        strValue: str = f'Hypothesis\n'
        # strValue += f'\tFitness: {self.fitness}\n'    # print fitness
        for k in self.features.keys():                # for each key
            strValue += f'\tClass {k}:\n'             # print the key
            for ftr in self.features[k]:              # loop over the feature list
                strValue += f'\t\t{ftr}\n'            # convert each CF
        strValue += f'\tCF List:\n'
        for cf in self.cfList:  # loop over the cf list
            strValue += f'\t\t{cf}\n'  # print the cf
        
        return strValue

    def getFeatures(self, classId: int) -> typ.Tuple[ConstructedFeature]:
        """ Gets a list of CFs for a given class"""
        return tuple(self.features[classId])

    @property
    def size(self):
        return self._size
    
    def updateSize(self):
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
# *************************************************************** #


def evolve(population: Population, passedElite: Hypothesis, bar) -> Population:
    """
    evolve is used by CDFC during the evolution step to create the next generation of the algorithm.
    This is done by randomly choosing between mutation & crossover based on the mutation rate.
    
    Functions:
        tournament: Finds Constructed Features to be mutated/crossover-ed.
        mutate: Performs the mutation operation.
        crossover: Performs the crossover operation.
        
    :param population: Population to be evolved.
    :param passedElite: Highest scoring Hypothesis created so far.
    :param bar:
    
    :type population: Population
    :type passedElite: Hypothesis
    :type bar:
    
    :return: A new population (index 0) and the new elite (index 1).
    :rtype: tuple
    """

    elite: Hypothesis = passedElite

    def __tournament(p: Population) -> Hypothesis:    # ! check for reference issues here
        """
        Used by evolution to selection the parent(s)
        
        :param p: The current population of hypotheses.
        :type p: Population
        
        :return: The best hypothesis that tournament found.
        :rtype: Hypothesis
        """

        # **************** Tournament Selection **************** #
        # get a list including every valid index in candidateHypotheses
        positions: typ.List[int] = list(range(len(p.candidateHypotheses)))
        first: typ.Optional[Hypothesis] = None  # the tournament winner
        score = 0     # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
            
            randomIndex: int = random.choice(positions)   # choose a random index in p.candidateHypotheses
            
            candidate: Hypothesis = p.candidateHypotheses[randomIndex]   # get the hypothesis at the random index
            
            positions.remove(randomIndex)  # remove the chosen value from the list of indexes (avoids duplicates)
            fitness = candidate.fitness    # get that hypothesis's fitness score

            if first is None:      # if first has not been set,
                first = candidate  # then  set it

            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it,
                score = fitness    # update the score to higher fitness,
                
        try:
            if first is None:
                raise Exception(f'ERROR: Tournament could not set first correctly, first = {first}')
        except Exception as err2:
            lineNm2 = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Tournament could not set first correctly, first = {first}, line number = {lineNm2}')
            print(f'{str(err2)}, line = {lineNm2}')
            sys.exit(-1)                            # exit on error; recovery not possible
        
        # log.debug('Finished Tournament method')
        
        # print('Tournament Finished')  # ! for debugging only!
        
        return first
        # ************ End of Tournament Selection ************* #

    def __crossoverTournament(p: Population) -> typ.Tuple[Hypothesis, Hypothesis]:
        """
        Used by crossover to selection the parents. It differs from the normal tournament
        because it will return two unique hypotheses.

        :param p: The current population of hypotheses.
        :type p: Population

        :return: Two hypothesis that tournament found.
        :rtype: typ.Tuple[Hypothesis, Hypothesis]
        """
    
        # **************** Tournament Selection **************** #
        # get a list including every valid index in candidateHypotheses
        positions: typ.List[int] = list(range(len(p.candidateHypotheses)))
        first = None  # the tournament winner
        firstIndex = None  # the index of the winner
        score = 0  # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
        
            randomIndex: int = random.choice(positions)  # choose a random index in p.candidateHypotheses
            candidate: Hypothesis = p.candidateHypotheses[randomIndex]  # get the hypothesis at the random index
            positions.remove(randomIndex)  # remove the chosen value from the list of indexes (avoids duplicates)
            fitness = candidate.fitness  # get that hypothesis's fitness score
        
            if (first is None) or (score < fitness):  # if first has not been set, or candidate if more fit
                first = candidate  # then update it
                score = fitness  # then update the score to higher fitness
                firstIndex = randomIndex  # finally update the index of the winner

        positions = list(range(len(p.candidateHypotheses)))
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
            candidate: Hypothesis = p.candidateHypotheses[randomIndex]  # get the hypothesis at the random index
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

        return p.candidateHypotheses[firstIndex], p.candidateHypotheses[secondIndex]
        # ************ End of Tournament Selection ************* #

    def mutate() -> Hypothesis:
        """
        Finds a random node and builds a new sub-tree starting at it. Currently mutate
        uses the same grow & full methods as the initial population generation without
        an offset. This means that mutate trees will still obey the max depth rule.
        This will create a new CF and overwrite the one it chose to mutate.
        """
        
        # !!!!!!!!!!!!!!!!!!!!! Debugging Only !!!!!!!!!!!!!!!!!!!!! #
        # parent: Hypothesis = population.candidateHypotheses[2]
        # randIndex: int = 1
        # randCF: ConstructedFeature = parent.cfList[1]
        # randClass: int = randCF.className
        # terminals = randCF.relevantFeatures
        # tree: Tree = randCF.tree
        # nodeID: str = randCF.tree.getRandomNode()
        # node: Node = tree.getNode(nodeID)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        
        # ******************* Fetch Values Needed ******************* #
        parent: Hypothesis = __tournament(population)                # get copy of a parent Hypothesis using tournament
        randClass: int = random.choice(CLASS_IDS)                    # get a random class
        randIndex: int = random.randint(0, M-1)                      # get a random index that's valid in cfList
        randCF: ConstructedFeature = parent.cfList[randIndex]        # get a random Constructed Feature
        terminals = randCF.relevantFeatures                          # save the indexes of the relevant features
        tree: Tree = randCF.tree                                     # get the tree from the CF
        nodeID: str = randCF.tree.getRandomNode()                    # get a random node from the CF's tree
        node: Node = tree.getNode(nodeID)
        # *********************************************************** #
        
        # ************* Remove the Children of the Node ************* #
        tree.removeChildren(nodeID)  # delete all the children
        # *********************************************************** #
    
        # ************************* Mutate ************************* #
        if random.choice(['OPS', 'TERM']) == 'TERM' or tree.getDepth(nodeID, tree.root) == MAX_DEPTH:
            node.data = random.choice(terminals)  # if we are at max depth or choose TERM,
    
        else:  # if we choose to add an OP
            node.data = random.choice(OPS)  # give the node a random OP
        
            # randomly decide which method to use to construct the new tree (grow or full)
            if random.choice(['Grow', 'Full']) == 'Grow':  # * Grow * #
                tree.grow(randCF.className, nodeID, MAX_DEPTH, TERMINALS, 0)  # tree is changed in place starting with node
            else:                                          # * Full * #
                tree.full(randCF.className, nodeID, MAX_DEPTH, TERMINALS, 0)  # tree is changed in place starting with node
        # *********************************************************** #

        # tree.checkTree()     # ! For Testing purposes only !!
        # tree.sendToStdOut()  # ! For Testing purposes only !!
        
        # overwrite old CF with the new one
        parent.features[randClass][randIndex] = ConstructedFeature(randCF.className, randCF.tree)
        
        parent.updateFitness()  # force an update of the fitness score
        
        return parent
    
    def crossover() -> (Hypothesis, Hypothesis):
        """Performs the crossover operation on two trees"""
        
        # * Find Random Parents * #
        parent1, parent2 = __crossoverTournament(population)  # type: Hypothesis

        # !!!!! Debugging Only !!!!! #
        # parent1: Hypothesis = population.candidateHypotheses[2]
        # randIndex = 1
        # classID = parent1.cfList[1].className
        # !!!!! Debugging Only !!!!! #

        # * Get CFs from the Same Class * #
        randIndex = random.randint(0, M-1)    # get a random index that's valid in cfList
        classID = random.choice(CLASS_IDS)    # choose a random class

        # get a the chosen random features from both parents parent
        # + Feature 1
        feature1: ConstructedFeature = parent1.getFeatures(classID)[randIndex]
        tree1: Tree = feature1.tree  # get the tree
        
        # + Feature 2
        feature2: ConstructedFeature = parent2.getFeatures(classID)[randIndex]
        tree2: Tree = feature2.tree  # get the tree
    
        # *************** Find the Two Sub-Trees **************** #
        # Pick Two Random Nodes, one from CF1 & one from CF2

        branch1: str
        p1: str
        treeFromFeature1: Tree
        # Get the Branch & Parent of the Subtree from CF1. This will tell use where to add it in CF 2
        # Get the Subtree from CF1. This will be move to CF2 (nodeF1 will be root)
        treeFromFeature1, p1, branch1 = tree1.removeSubtree(tree1.getRandomNode())
        # tree1.checkForDuplicateKeys(treeFromFeature1)  # ! debugging
        
        branch2: str
        p2: str
        treeFromFeature2: Tree
        # Get the Branch & Parent of the Subtree from CF1. This will tell use where to add it in CF 2
        # Get the Subtree from CF2. This will be move to CF1 (nodeF2 will be root)
        treeFromFeature2, p2, branch2 = tree2.removeSubtree(tree2.getRandomNode())
        # tree2.checkForDuplicateKeys(treeFromFeature2)  # ! debugging
        # ******************************************************* #
    
        # !!!!!!!!!!!!!!!!!!! Debugging Only !!!!!!!!!!!!!!!!!!! #
        # print('Crossover Operation\nParent 1:')
        # print(parent1.cfList[1].tree)
        # print('Subtree from Parent 2:')
        # print(treeFromFeature2)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # TODO make sure we update the parents to point to new nodes
        
        # ************************** swap the two subtrees ************************** #
        # Add the Subtree from CF2 to the tree in CF1 (in the same location that the subtree1 was cut out)
        tree1.addSubtree(subtree=treeFromFeature2, newParent=p1, orphanBranch=branch1)
        # Add the Subtree from CF1 to the tree in CF2 (in the same location that the subtree2 was cut out)
        tree2.addSubtree(subtree=treeFromFeature1, newParent=p2, orphanBranch=branch2)
        # **************************************************************************** #

        # ************** Create Two New Constructed Features ************** #
        cf1: ConstructedFeature = ConstructedFeature(classID, tree1)
        cf2: ConstructedFeature = ConstructedFeature(classID, tree2)
        # ***************************************************************** #

        # ******************* Create Two New Hypotheses ******************* #
        # Get all the CFs of the old parent
        allCFs: typ.Dict[int, typ.List[ConstructedFeature]] = parent1.features
        # replace the one that changed
        allCFs[classID][randIndex] = cf1  # override the previous entry
        h1: Hypothesis = Hypothesis(size=0, fDict=deepcopy(allCFs))
        h1.updateSize()

        # Get all the CFs of the old parent
        allCFs: typ.Dict[int, typ.List[ConstructedFeature]] = parent2.features
        # replace the one that changed
        allCFs[classID][randIndex] = cf2  # override the previous entry
        h2: Hypothesis = Hypothesis(size=0, fDict=deepcopy(allCFs))
        h2.updateSize()

        # !!!!!!!!!!!!!!!!!!! Debugging Only !!!!!!!!!!!!!!!!!!! #
        # print('Crossover Complete\nParent 1:')
        # print(f'Parent ID of swapped Node:{p1}')
        # print(parent1.cfList[1].tree)
        
        # global GLOBAL_COUNTER
        # GLOBAL_COUNTER = 0     # (reset counter to find if problem is in the 1st fit call after crossover)
        # print('H1')
        # print(h1)
        # print('H2')
        # print(h2)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        
        h1.updateFitness()  # force an update of the fitness score
        h2.updateFitness()  # force an update of the fitness score
        # ***************************************************************** #

        return h1, h2

    # ******************* Evolution ******************* #
    # create a new population with no hypotheses (made here crossover & mutate can access it)
    newPopulation = Population([], population.generation + 1)
    
    # each iteration evolves 1 new candidate hypothesis, and we want to do this until
    # range(newPopulation.candidateHypotheses) = POPULATION_SIZE so loop over pop size
    for pop in range(POPULATION_SIZE):
        probability = random.uniform(0, 1)              # get a random number between 0 & 1

        # ! For Testing Only
        # mutate()
        # crossover()
        # bar()
    
        # ***************** Mutate ***************** #
        # if probability < MUTATION_RATE:            # if probability is less than mutation rate, mutate
        if False:  # ! debugging
            bar.text('mutating...')                # update user
            newHypoth = mutate()                   # perform mutation
            # add the new hypoth to the population
            newPopulation.candidateHypotheses.append(newHypoth)

            # ! debugging only ! #
            # print('Mutation created Hypothesis, checking...')
            # for cf in newHypoth.cfList:        # for each CF
            #     cf.tree.checkForMissingKeys()  # ! this one gets triggered
            # print('\tNo missing keys detected\n')
            # ! debugging only ! #
            # ****************** Elitism ****************** #
            if newHypoth.fitness > elite.fitness:  # if the new hypothesis has a better fitness
                elite = newHypoth                  # update elite
            # ************** End of Elitism *************** #
            
        # ************* End of Mutation ************* #

        # **************** Crossover **************** #
        else:                                             # if probability is greater than mutation rate, use crossover
            
            bar.text('crossing...')                       # update user
            newHypoth1, newHypoth2 = crossover()          # perform crossover operation

            # add the new hypoth to the population
            newPopulation.candidateHypotheses.append(newHypoth1)
            newPopulation.candidateHypotheses.append(newHypoth2)
            
            # ! debugging only ! #
            # print('Crossover created Hypothesis, checking...')
            # for cf in newHypoth1.cfList:       # for each CF
            #     cf.tree.checkForMissingKeys()  # check the tree
            # for cf in newHypoth2.cfList:       # for each CF
            #     cf.tree.checkForMissingKeys()  # check the tree
            # print('\tNo missing keys detected\n')
            # ! debugging only ! #

            # ****************** Elitism ****************** #
            if newHypoth1.fitness >= newHypoth2.fitness:  # if newHypoth1 has a greater or equal fitness
                betterH = newHypoth1                      # then set it as the one to be compared to elite
            else:                                         # otherwise,
                betterH = newHypoth2                      # set newHypoth2 as the one to be compared
            
            if betterH.fitness > elite.fitness:           # if one of the new hypotheses has a
                elite = betterH                           # better fitness, then update elite
            # ************** End of Elitism *************** #
            
        # ************* End of Crossover ************* #
    newPopulation.elite = elite
    return newPopulation  # ? should this pass a deepcopy of population?


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
    currentPopulation = createInitialPopulation()     # run initialPop/create the initial population
    SYSOUT.write(NO_OVERWRITE + ' Initial population generated '.ljust(50, '-') + SUCCESS)
    oldElite = currentPopulation.candidateHypotheses[0]  # init elitism

    # loop, evolving each generation. This is where most of the work is done
    
    # elites = [oldElite.fitness]  # ! debugging only!
    
    with alive_bar(GENERATIONS, title="Generations") as bar:  # declare your expected total

        for gen in range(GENERATIONS):  # iterate as usual
            
            newPopulation: Population  # type hinting
            newElite: Hypothesis       # type hinting
            
            # generate a new population by evolving the old one
            currentPopulation = evolve(currentPopulation, oldElite, bar)
            
            # elites.append(currentPopulation.elite.fitness)  # ! used in debugging

            bar()  # update bar now that a generation is finished

    # print('Fitness of the elites')  # ! used in debugging
    # pprint.pprint(elites, compact=True)  # ! used in debugging
    # SYSOUT.write(NO_OVERWRITE + ' Final Generation Reached'.ljust(50, '-') + SUCCESS)  # update user
    # ***************************************************************** #

    # ****************** Return the Best Hypothesis ******************* #

    bestHypothesis: Hypothesis = max(currentPopulation.candidateHypotheses)
    bestHypothesis = max([bestHypothesis, currentPopulation.elite])

    return bestHypothesis  # return the best hypothesis generated
