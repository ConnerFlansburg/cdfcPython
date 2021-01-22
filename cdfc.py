"""
cdfc.py creates, and evolves a genetic program using Class Dependent Feature Select.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import itertools
import collections as collect
import copy
import logging as log
import math
import random
import sys
import typing as typ
import warnings
from pathlib import Path
import Distances as Dst
import numpy as np
from alive_progress import alive_bar, config_handler
from treelib import Node as Node
from decimal import Decimal
# from pyitlib import discrete_random_variable as drv
import pprint


from objects import Tree
from objects import cdfcInstance as Instance

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
# MAX_DEPTH: typ.Final = 5                      # ! value for testing/debugging to make trees more readable
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
    
    def __init__(self, size: int, cfs: cfsType, fDict: fDictType = None) -> None:
        """Constructor for the Hypothesis object"""
        self.size: int = size                                         # the number of nodes in all the cfs

        if fDict:  # if a dictionary was passed, just copy the info in
            self.features = fDict
            self.cfList = cfs
            
        else:  # if a dictionary was not passed
            # create a list of all the Constructed Features (regardless of class)
            self.cfList = cfs
            
            for c in cfs:  # add the CFs to the dictionary, keyed by their class id
                if c.className in self.features.keys():   # if the entry already exists
                    self.features[c.className].append(c)  # append c to the list
                else:                                     # if the entry doesn't exist
                    self.features[c.className] = [c]  # create a list with c and add it to the dictionary

    def getFeatures(self, classId) -> typ.List[ConstructedFeature]:
        """ Gets a list of CFs for a given class"""
        return self.features[classId]

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
        
        def Distance(values: typ.List[Instance]) -> float:
            """"
            Distance calculates the distance value of the Hypothesis
            
            :param values: List of the instances who's distance we wish to compute.
            :type values: list
            
            :return: Distance value calculated using the chosen distance function.
            :rtype: float
            """
            
            minimum: typ.Optional[float] = 999  # smallest distance between classes that has been found
            maximum: typ.Optional[float] = -999  # largest distance within the same class that has been found
            
            # create a list containing every unique combination of Instances
            combined: typ.List[typ.Tuple[Instance]] = list(itertools.combinations(values, 2))
            
            # ********* Loop Over Every Combination of Instances ********* #
            # loop over the instance combinations
            for Vi, Vj in combined:  # type: Instance

                # ********** Compute Vid & Vjd ********** #
                # Vi.vList & Vj.vList will be list of floats
                dstValue = Dst.computeDistance(DISTANCE_FUNCTION, Vi.vList, Vj.vList)

                # *************** Determine Which to Calculate: Dw or Dj *************** #
                # if these instances are in the same class, and Db is less than dstValue, choose Dw
                if (Vi.className == Vj.className) and (maximum < dstValue):
                    maximum = dstValue
                
                # if these instances are in different classes, and Db is more than dstValue, choose Db
                elif (Vi.className != Vj.className) and (minimum > dstValue):
                    minimum = dstValue

            # ******** Multiply Both Db & Dw By 1/|S| ******** #
            Db: float = (1 / len(values)) * minimum
            Dw: float = (1 / len(values)) * maximum
            log.debug(f'min = {minimum}')
            log.debug(f'max = {maximum}')
            
            # *************** Create Exponent *************** #
            exp: float = -5 * (Db - Dw)  # create the exponent
            log.debug(f'Exponent: -5 * {Db} - {Dw} = {exp}')
            
            # *************** Calculate Power *************** #
            try:
                # Note: Python supports arbitrarily large ints, but not floats
                pwr = np.float_power(np.e, exp)  # this can cause overflow
                # pwr = np.power(np.e, exp)
            except OverflowError:
                # try using Decimal type to hold large floats
                pwrFix = Decimal(exp).exp()  # e**exp
                print('OVERFLOW ERROR occurred during distance calculation')
                print(f'value: {pwrFix.to_eng_string()}')
                # try to round the Decimal to 4 places & then convert to a float
                r = 1 / (1 + float(pwrFix.quantize(Decimal('1.0000'))))
                return r
                # sys.exit(-1)  # exit with an error
            
            pwr = np.round(pwr, 4)
            log.debug(f'Power of e = {pwr}')

            return 1 / (1 + pwr)
        
        def _averageInfoGain(cid: int) -> float:
            """ This calculates AvgIG for a single class """
            
            count: float = 0.0
            value: float = 0.0
            maxValue: float = -999
            
            # loop over every CF in the class cid (should loop m times)
            for ft in self.features[cid]:  # type: ConstructedFeature
                
                # ? what should I be sending here?
                # value = drv.information_mutual(cid, f.transform())  # get the mutual info gain
                count += value                    # add to the running total
                
                if value > maxValue:              # if this CF has a higher IG, update max
                    maxValue = value

            # since we didn't add fmax during the loop, add it here
            # (times the amount of times it should have been added)
            avgIG: float = value + (maxValue * M)
            
            # perform final calculations & return
            return avgIG/(M+1)
            
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

    # TODO check runCDFC & _transform
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

# ***************** End of Namespaces/Structs & Objects ******************* #


def __grow(classId: int, node: Node, tree: Tree) -> Node:
    """
    Grow creates a tree or sub-tree starting at the Node node, and using the Grow method.
    If node is a root Node, grow will build a tree, otherwise grow will build a sub-tree
    starting at node. Grow assumes that node's data has already been set & makes all
    changes in place.

    NOTE:
    During testing whatever calls grow should use the sanity check sanityCheckTree(newTree)

    :param classId: ID of the class that the tree should identify.
    :param node: The root node of the subtree __grow will create.
    :param tree: Tree that __grow is building or adding to.

    :type classId: int
    :type node: Node
    :type tree: Tree
    """
    
    coin = random.choice(['OP', 'TERM']) == 'TERM'  # flip a coin & decide OP or TERM
    
    # *************************** A Terminal was Chosen *************************** #
    # NOTE: check depth-1 because we will create children
    if coin == 'TERM' or (tree.getDepth(node) == MAX_DEPTH - 1):  # if we need to add terminals

        # pick the needed amount of terminals
        terms: typ.List[int] = random.choices(TERMINALS[classId], k=NUM_TERMINALS[node.data])
        
        if NUM_TERMINALS[node.data] == 2:                  # if the OP needs 2 children
            tree.addLeft(parent=node, data=terms.pop(0))   # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))  # create a new left node & add it
            
            return tree.getRoot()                          # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:                 # if the OP needs 3 children
            tree.addLeft(parent=node, data=terms.pop(0))    # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))   # create a new right node & add it
            tree.addMiddle(parent=node, data=terms.pop(0))  # create a new middle node & add it
            
            return tree.getRoot()                           # return the root node of the tree
        
        else:                                               # if NUM_TERMINALS was not 2 or 3
            raise IndexError("Grow could not find the number of terminals need")
    
    # *************************** A Operation was Chosen *************************** #
    else:  # if we chose to add an operation
        
        if NUM_TERMINALS[node.data] == 2:                              # if the number of terminals needed by node is two
            ops: typ.List[str] = random.choices(OPS, k=2)              # pick the needed amount of OPs
            
            left: Node = tree.addLeft(parent=node, data=ops.pop(0))    # add the new left node
            right: Node = tree.addRight(parent=node, data=ops.pop(0))  # add the new right node
            
            __grow(classId, left, tree)                                # call grow on left to set it's children
            __grow(classId, right, tree)                               # call grow on right to set it's children
            return tree.getRoot()                                      # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:                              # if the number of terminals needed by node is three
            ops: typ.List[str] = random.choices(OPS, k=3)                # pick the needed amount of OPs
            
            left: Node = tree.addLeft(parent=node, data=ops.pop(0))      # create & add the new left node to the tree
            right: Node = tree.addRight(parent=node, data=ops.pop(0))    # create & add the new right node to the tree
            middle: Node = tree.addMiddle(parent=node, data=ops.pop(0))  # create & add the new middle node to the tree
            
            __grow(classId, left, tree)                                  # call grow on left to set it's children
            __grow(classId, right, tree)                                 # call grow on right to set it's children
            __grow(classId, middle, tree)                                # call grow on middle to set it's children
            return tree.getRoot()                                        # return the root node of the tree
        
        else:  # if NUM_TERMINALS was not 1 or 2
            raise IndexError("Grow could not find the number of terminals need")


def __full(classId: int, node: Node, tree: Tree):
    """
    Full creates a tree or sub-tree starting at the Node node, and using the Full method.
    If node is a root Node, full will build a tree, otherwise full will build a sub-tree
    starting at node. Full assumes that node's data has already been set & makes all
    changes in place.
      
    NOTE:
    During testing whatever calls full should use the sanity check sanityCheckTree(newTree)
    
    :param classId: ID of the class that the tree should identify.
    :param node: The root node of the subtree __full will create.
    :param tree: Tree that __full is building or adding to.
    
    :type classId: int
    :type node: Node
    :type tree: Tree
    """
    
    # *************************** Max Depth Reached *************************** #
    if tree.getDepth(node) == MAX_DEPTH - 1:
        
        # pick the needed amount of terminals
        terms: typ.List[int] = random.choices(TERMINALS[classId], k=NUM_TERMINALS[node.data])
        
        if NUM_TERMINALS[node.data] == 2:      # if the OP needs 2 children
            tree.addLeft(parent=node, data=terms.pop(0))   # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))  # create a right left node & add it
            return tree.getRoot()              # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:     # if the OP needs 3 children
            tree.addLeft(parent=node, data=terms.pop(0))    # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))   # create a new right node & add it
            tree.addMiddle(parent=node, data=terms.pop(0))  # create a new middle node & add it
            return tree.getRoot()               # return the root node of the tree
        
        else:  # if NUM_TERMINALS was not 1 or 2
            raise IndexError("Grow could not find the number of terminals need")
    
    # *************************** If Not at Max Depth *************************** #
    else:  # if we haven't reached the max depth, add operations
        
        if NUM_TERMINALS[node.data] == 2:                              # if the number of terminals needed by node is two
            ops: typ.List[str] = random.choices(OPS, k=2)              # pick the needed amount of OPs

            left: Node = tree.addLeft(parent=node, data=ops.pop(0))    # add the new left node
            right: Node = tree.addRight(parent=node, data=ops.pop(0))  # add the new right node
            
            __full(classId, left, tree)                                # call grow on left to set it's children
            __full(classId, right, tree)                               # call grow on right to set it's children
            return tree.getRoot()                                      # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:                              # if the number of terminals needed by node is three
            ops: typ.List[str] = random.choices(OPS, k=3)                # pick the needed amount of OPs
            
            left: Node = tree.addLeft(parent=node, data=ops.pop(0))      # create & add the new left node to the tree
            right: Node = tree.addRight(parent=node, data=ops.pop(0))    # create & add the new right node to the tree
            middle: Node = tree.addMiddle(parent=node, data=ops.pop(0))  # create & add the new middle node to the tree
            
            __full(classId, left, tree)                                   # call grow on left to set it's children
            __full(classId, right, tree)                                  # call grow on right to set it's children
            __full(classId, middle, tree)                                 # call grow on middle to set it's children
            return tree.getRoot()                                         # return the root node of the tree
        
        else:  # if NUM_TERMINALS was not 1 or 2
            raise IndexError("Grow could not find the number of terminals need")


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
    
            ftrs: typ.List[ConstructedFeature] = []  # empty the list of features
            
            for _ in range(M):  # loop M times so M CFs are created
                
                tree = Tree()   # create an empty tree
                tree.addRoot()  # create a root node for the tree
                
                if random.choice([True, False]):         # *** use grow *** #
                    __grow(cid, tree.getRoot(), tree)    # create tree using grow
                else:                                    # *** use full *** #
                    __full(cid, tree.getRoot(), tree)    # create tree using full
    
                cf = ConstructedFeature(cid, tree)       # create constructed feature
                ftrs.append(cf)                          # add the feature to the list of features
                cfList.append(cf)
                
                size += size

            # add the list of features for class cid to the dictionary, keyed by cid
            cfDictionary[cid] = ftrs
            
        # create a hypothesis & return it
        return Hypothesis(cfs=cfList, fDict=cfDictionary, size=size)

    hypothesis: typ.List[Hypothesis] = []

    # creat a number hypotheses equal to pop size
    for __ in range(POPULATION_SIZE):  # iterate as usual
        hyp = createHypothesis()       # create a Hypothesis
        hypothesis.append(hyp)         # add the new hypothesis to the list

    pop = Population(hypothesis, 0)
    
    sanityCheckPopReference(pop)
    # sanityCheckPop(hypothesis)  # ! testing purposes only!
    return pop


# ********** Sanity Check Functions used for Debugging ********** #
# ! testing purposes only!
def sanityCheckPopReference(pop: Population):
    
    log.debug('Starting Population Reference Check...')
    # loop over every GPI in the pop
    for i in pop.candidateHypotheses:  # type: Hypothesis
        
        sanityCheckHypReference(i)  # check that the hypoth is fine
        
        count = 0  # rest the count for each GPI
        # compare that GPI with every other GPI
        for j in pop.candidateHypotheses:
            
            # there will always be one that is the same,
            # but throw an error if there is two
            if i is j:
                count += 1
            if count > 1:
                raise AssertionError

    log.debug('Population Reference Check Passed')


def sanityCheckHypReference(hyp: Hypothesis):
    # loop over every GPI in the pop
    for i in hyp.cfList:  # type: ConstructedFeature
        
        count = 0  # rest the count for each GPI
        # compare that GPI with every other GPI
        for j in hyp.cfList:  # type: ConstructedFeature
            
            # there will always be one that is the same,
            # but throw an error if there is two
            if i is j:
                count += 1
            if count > 1:
                raise AssertionError


# ! testing purposes only!
def sanityCheckCF(cf: ConstructedFeature):
    """Used in debugging to check a Constructed Feature"""
    log.debug('Starting Constructed Feature Sanity Check...')
    cf.transform(rows[0])
    log.debug('Constructed Feature Sanity Check Passed')


# ! testing purposes only!
def sanityCheckTree(tree: Tree, classId):
    """Used in debugging to check a Tree"""
    log.debug('Starting Tree Sanity Check...')
    tree.checkTree()
    tree.runTree(rows[0].attributes, classId, TERMINALS)
    log.debug('Tree Sanity Check Passed')
# *************************************************************** #


def evolve(population: Population, passedElite: Hypothesis, bar) -> typ.Tuple[Population, Hypothesis]:
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

    def __tournament(p: Population) -> Hypothesis:
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
        first = None  # the tournament winner
        score = 0     # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
            
            randomIndex: int = random.choice(positions)   # choose a random index in p.candidateHypotheses
            
            candidate: Hypothesis = p.candidateHypotheses[randomIndex]   # get the hypothesis at the random index
            
            positions.remove(randomIndex)  # remove the chosen value from the list of indexes (avoids duplicates)
            fitness = candidate.fitness    # get that hypothesis's fitness score

            if first is None:      # if first has not been set,
                first = candidate  # then  set it

            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it
                score = fitness    # then update the score to higher fitness
                
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
    
        return copy.deepcopy(p.candidateHypotheses[firstIndex]), copy.deepcopy(p.candidateHypotheses[secondIndex])
        # ************ End of Tournament Selection ************* #

    # ******************* Evolution ******************* #
    # create a new population with no hypotheses (made here crossover & mutate can access it)
    newPopulation = Population([], population.generation+1)

    def mutate() -> Hypothesis:  # ! check for reference issues here
        """
        Finds a random node and builds a new sub-tree starting at it. Currently mutate
        uses the same grow & full methods as the initial population generation without
        an offset. This means that mutate trees will still obey the max depth rule.
        """

        # TODO: change so it just creates a new CF(s) instead of working in place

        # ******************* Fetch Values Needed ******************* #
        parent: Hypothesis = __tournament(population)                # get copy of a parent Hypothesis using tournament
        randClass: int = random.choice(CLASS_IDS)                    # get a random class
        indexOptions = range(len(parent.features[randClass]))
        randIndex: int = random.choice(indexOptions)                 # get a random index
        randCF: ConstructedFeature = parent.cfList[randIndex]        # get a random Constructed Feature
        terminals = randCF.relevantFeatures                          # save the indexes of the relevant features
        tree: Tree = randCF.tree                                     # get the tree from the CF
        # tree.checkTree()     # ! For Testing purposes only !!
        # tree.sendToStdOut()  # ! For Testing purposes only !!
        node: Node = randCF.tree.getRandomNode()                     # get a random node from the CF's tree
        # *********************************************************** #
        
        # ************* Remove the Children of the Node ************* #
        children: typ.List[Node] = tree.children(node.identifier)   # get all the children
        [tree.remove_node(child.identifier) for child in children]  # delete all the children
        # *********************************************************** #
    
        # ************************* Mutate ************************* #
        if random.choice(['OPS', 'TERM']) == 'TERM' or tree.depth(node.identifier) == MAX_DEPTH:
            node.data = random.choice(terminals)  # if we are at max depth or choose TERM,
    
        else:  # if we choose to add an OP
            node.data = random.choice(OPS)  # give the node a random OP
        
            # randomly decide which method to use to construct the new tree (grow or full)
            if random.choice(['Grow', 'Full']) == 'Grow':  # * Grow * #
                __grow(randCF.className, node, tree)       # tree is changed in place starting with node
            else:                                          # * Full * #
                __full(randCF.className, node, tree)       # tree is changed in place starting with node
        # *********************************************************** #

        # tree.checkTree()     # ! For Testing purposes only !!
        # tree.sendToStdOut()  # ! For Testing purposes only !!
        
        # overwrite old CF with the new one
        parent.features[randClass][randIndex] = ConstructedFeature(randCF.className, randCF.tree)
        
        parent.updateFitness()  # force an update of the fitness score
        
        # add the mutated parent to the new pop (appending is needed because parent is a copy NOT a reference)
        newPopulation.candidateHypotheses.append(parent)
        return parent
    
    def crossover() -> (Hypothesis, Hypothesis):  # ! check for reference issues here
        """Performs the crossover operation on two trees"""

        # TODO: change so it just creates a new CF(s) instead of working in place
        
        # * Find Random Parents * #
        parent1, parent2 = __crossoverTournament(population)

        # * Get CFs from the Same Class * #
        randIndex = random.randint(0, M-1)  # get a random index that's valid in cfList
        randClass = random.choice(CLASS_IDS)  # choose a random class
        # Feature 1
        feature1: ConstructedFeature = parent1.getFeatures(randClass)[randIndex]  # get a random feature from the parent
        tree1: Tree = feature1.tree                                               # get the tree
        
        # tree1.checkTree()        # ! For Testing Only !!
        # tree1.sendToStdOut()     # ! For Testing Only !!

        # Feature 2
        # makes sure CFs are from/for the same class
        feature2: ConstructedFeature = parent2.getFeatures(randClass)[randIndex]
        tree2: Tree = feature2.tree  # get the tree
        
        # tree2.checkTree()        # ! For Testing Only !!
        # tree2.sendToStdOut()     # ! For Testing Only !!
    
        # *************** Find the Two Sub-Trees **************** #
        node1: Node = tree1.getRandomNode()           # get a random node
        branch1, p1 = tree1.getBranch(node1)          # get the branch string
        subTree1: Tree = tree1.remove_subtree(node1.identifier)  # get a sub-tree with node1 as root
        # subTree1.sendToStdOut()  # ! For Testing Only !!

        node2: Node = tree2.getRandomNode()           # get a random node
        branch2, p2 = tree2.getBranch(node2)          # get the branch string
        subTree2: Tree = tree2.remove_subtree(node2.identifier)  # get a sub-tree with node2 as root
        # subTree2.sendToStdOut()  # ! For Testing Only !!
        # ******************************************************* #
    
        # ************************** swap the two subtrees ************************** #
        tree1.addSubTree(parent=p1, branch=branch1, subtree=subTree1)  # update the first parent tree
        tree2.addSubTree(parent=p2, branch=branch2, subtree=subTree2)  # update the second parent tree
        # **************************************************************************** #

        # !!!!!!!!!!!!!! For Testing Only !!!!!!!!!!!!!! #
        # Print the trees to see if crossover broke them/performed correctly
        # print('Crossover Finished')
        # print('Parent Tree 1:')
        # print('1st Swapped Tree:')
        # tree1.sendToStdOut()
        # tree1.checkTree()
        # print('Parent Tree 2:')
        # print('2nd Swapped Tree:')
        # tree2.sendToStdOut()
        # tree2.checkTree()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    
        parent1.updateFitness()  # force an update of the fitness score
        parent2.updateFitness()  # force an update of the fitness score
        
        # parent 1 & 2 are both hypotheses and should have been changed in place,
        # but they refer to a copy made in tournament so add them to the new pop
        newPopulation.candidateHypotheses.append(parent1)
        newPopulation.candidateHypotheses.append(parent2)
        
        return parent1, parent2

    # each iteration evolves 1 new candidate hypothesis, and we want to do this until
    # range(newPopulation.candidateHypotheses) = POPULATION_SIZE so loop over pop size
    for pop in range(POPULATION_SIZE):
        probability = random.uniform(0, 1)              # get a random number between 0 & 1

        # ! For Testing Only
        # mutate()
        # crossover()
        # bar()
    
        # ***************** Mutate ***************** #
        if probability < MUTATION_RATE:            # if probability is less than mutation rate, mutate
            
            bar.text('mutating...')                # update user
            newHypoth = mutate()                   # perform mutation

            # ****************** Elitism ****************** #
            if newHypoth.fitness > elite.fitness:  # if the new hypothesis has a better fitness
                elite = newHypoth                  # update elite
            # ************** End of Elitism *************** #
            
        # ************* End of Mutation ************* #

        # **************** Crossover **************** #
        else:                                             # if probability is greater than mutation rate, use crossover
            
            bar.text('crossing...')                       # update user
            newHypoth1, newHypoth2 = crossover()          # perform crossover operation

            # ****************** Elitism ****************** #
            if newHypoth1.fitness >= newHypoth2.fitness:  # if newHypoth1 has a greater or equal fitness
                betterH = newHypoth1                      # then set it as the one to be compared to elite
            else:                                         # otherwise,
                betterH = newHypoth2                      # set newHypoth2 as the one to be compared
            
            if betterH.fitness > elite.fitness:           # if one of the new hypotheses has a
                elite = betterH                           # better fitness, then update elite
                
                # ! this is being hit meaning crossover is giving references to old CFs
                # print('Elitism Updated')  # ! debugging only
                # if betterH is elite:  # ! debugging only
                #     raise AssertionError  # ! debugging only
            # ************** End of Elitism *************** #
            
        # ************* End of Crossover ************* #

    return newPopulation, elite


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
    
    # Read the values in the dictionary into the constants
    FEATURE_NUMBER = values['FEATURE_NUMBER']
    CLASS_IDS = values['CLASS_IDS']
    DISTANCE_FUNCTION = distanceFunction
    # ! during testing we are just using Euclidean distance, so ignore any passed value
    DISTANCE_FUNCTION = 'euclidean'
    # POPULATION_SIZE = values['POPULATION_SIZE']
    POPULATION_SIZE = 10
    INSTANCES_NUMBER = values['INSTANCES_NUMBER']
    LABEL_NUMBER = values['LABEL_NUMBER']
    M = values['M']
    rows = values['rows']
    ENTROPY_OF_S = values['ENTROPY_OF_S']
    CLASS_DICTS = values['CLASS_DICTS']
    TERMINALS = dataIn[1]
    
    # *********************** Run the Algorithm *********************** #

    currentPopulation = createInitialPopulation()     # run initialPop/create the initial population
    SYSOUT.write(NO_OVERWRITE + ' Initial population generated '.ljust(50, '-') + SUCCESS)
    oldElite = currentPopulation.candidateHypotheses[0]  # init elitism

    # loop, evolving each generation. This is where most of the work is done
    
    elites = [oldElite.fitness]  # ! debugging only!
    
    with alive_bar(GENERATIONS, title="Generations") as bar:  # declare your expected total

        for gen in range(GENERATIONS):  # iterate as usual
            
            newPopulation, newElite = evolve(currentPopulation, oldElite, bar)  # generate a new population by evolving the old one
            # update currentPopulation to hold the new population
            # this is done in two steps to avoid potential namespace issues
            currentPopulation = newPopulation
            oldElite = newElite  # update elitism
            
            elites.append(newElite.fitness)  # ! used in debugging

            bar()  # update bar now that a generation is finished

    print('Fitness of the elites')  # ! used in debugging
    pprint.pprint(elites, compact=True)  # ! used in debugging
    # SYSOUT.write(NO_OVERWRITE + ' Final Generation Reached'.ljust(50, '-') + SUCCESS)  # update user
    # ***************************************************************** #

    # ****************** Return the Best Hypothesis ******************* #

    bestHypothesis: Hypothesis = max(currentPopulation.candidateHypotheses, key=lambda x: x.fitness)

    return bestHypothesis  # return the best hypothesis generated
