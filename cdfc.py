import cProfile
import collections as collect
import copy
import logging as log
import math
import random
import sys
import typing as typ
import warnings
from pathlib import Path
import traceback

import numpy as np
from tqdm import tqdm
from alive_progress import alive_bar, config_handler

# ! Next Steps
# TODO fix the exit code -1073741571 error in evolution
#  + is it stack overflow? it occurs during the setSize function

# TODO add docstrings
# TODO add testing functions

# **************************** Constants/Globals **************************** #
ALPHA: typ.Final = 0.8                        # ALPHA is the fitness weight alpha
BARCOLS = 25                                  # BARCOLS is the number of columns for the progress bar to print
CROSSOVER_RATE: typ.Final = 0.8               # CROSSOVER_RATE is the chance that a candidate will reproduce
ELITISM_RATE: typ.Final = 1                   # ELITISM_RATE is the elitism rate
GENERATIONS: typ.Final = 50                   # GENERATIONS is the number of generations the GP should run for
MAX_DEPTH: typ.Final = 8                      # MAX_DEPTH is the max depth trees are allowed to be & is used in grow/full
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
# ++++++++ console formatting strings ++++++++ #
HDR = '*' * 6
SUCCESS = u' \u2713\n'
OVERWRITE = '\r' + HDR
SYSOUT = sys.stdout
# ++++++++++++++++++++++++++++++++++++++++++++ #
# ************************ End of Constants/Globals ************************ #

# ********************** Namespaces/Structs & Objects ***********************


class Instance:
    # This class replace the older collect.namedtuple('Instance', ['className', 'values']) named tuple
    # NOTE: a row IS an instance
    
    def __init__(self, className: int, values: typ.Dict[int, float]):
        # this stores the name of the class that the instance is in
        self.className: int = className
        # this stores the values of the features, keyed by index, in the instance
        self.attributes: typ.Dict[int, float] = values
        # this creates a list of the values stored in the dictionary for when iteration is wanted
        self.vList: typ.List[float] = list(self.attributes.values())  # ? is this to expensive?
        
        try:                                  # check that all the feature values are valid
            if None in self.attributes.values():  # if a None is in the list of feature values
                raise Exception('Tried to create an Instance obj with a None feature value')
        except Exception as err:
            log.error(str(err))
            print(str(err))
            traceback.print_stack()           # print stack trace so we know how None is reaching Instance
            sys.exit(-1)                      # exit on error; recovery not possible


sys.setrecursionlimit(10000)

rows: typ.List[Instance] = []  # this will store all of the records read in (the training dat) as a list of rows

np.seterr(divide='ignore')  # suppress divide by zero warnings from numpy
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars')  # the global config for the loading bars

# create the file path for the log file & configure the logger
logPath = str(Path.cwd() / 'logs' / 'cdfc.log')
log.basicConfig(level=log.DEBUG, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')
# log.basicConfig(level=log.ERROR, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')

profiler = cProfile.Profile()                       # create a profiler to profile cdfc during testing
statsPath = str(Path.cwd() / 'logs' / 'stats.log')  # set the file path that the profiled info will be stored at


class Tree:
    """Tree is a binary tree data structure that is used to represent a
       constructed feature

    Variables:
        left: The left child of the tree. This will be a tree object.

        right: The right child of the tree. This will be a tree object.

        data: This will be either a terminal character or a function name.

    Methods:
        runTree: This is used to walk the tree until we reach a terminal
                 character. After we do it should return that character along
                 with a construct set of functions that should be operated on
                 that terminal character. These functions should be store in
                 the 'data' variable & be stored in as string function names.
    """
    
    def __init__(self, data: typ.Union[str, int], left: typ.Union[None, "Tree"] = None,
                 right: typ.Union[None, "Tree"] = None, middle: typ.Union[None, "Tree"] = None,) -> None:

        # *********************** Error Checking *********************** #
        try:
            
            if data is None:  # check that we aren't storing a none in data, if we are throw exception
                raise Exception('ERROR: Tree constructor tried to add a \'None\' to a tree as data')
            
            if data in OPS:   # check that we are not creating an OP with too few terminals
                if left is None or right is None:    # if left or right are not terminals, raise exception
                    raise Exception("ERROR: Tree constructor tried to create a terminal Tree node using an operation and 1 None")
                if left is None and right is None:   # if left & right are not terminals, raise exception
                    raise Exception("ERROR: Tree constructor tried to create a terminal Tree node using an operation and 2 Nones")
                if data == 'if' and middle is None:  # if we are using the if OP with no middle, raise exception
                    raise Exception("ERROR: Tree constructor tried to create an IF transformation with too few children")
        
        except Exception as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            msg = f'{str(err)}, line = {lineNm}'   # create the error/log message
            log.error(msg)                         # log the error
            print(msg)                        # print error to console
            traceback.print_stack()                # print stack trace so we know how None is reaching tree
            sys.exit(-1)                           # exit on error; recovery not possible

        # ************************ If data isn't null, we can build the tree ************************ #
        self.data: typ.Union[int, str] = data    # must either be a function or a terminal (if a terminal it should be it's index)
        self.left = left
        self.right = right
        self.middle = middle
        self.size = None             # initialize size to 0 & then call setSize to set it
        self.size = self.setSize(0)  # ? every time that we create a tree we call seSize, does this create too much overhead?
        
    def setLeft(self, left):
        self.left = left
        
    def setRight(self, right):
        self.left = right
        
    def getLeft(self) -> "Tree":
        try:
            if self.left is None:
                raise Exception('Tree tried to access a child that didn\'t exist')
            else:
                return self.left
        except Exception as err:
            log.error(str(err))
            print(str(err))
            traceback.print_stack()
            sys.exit(-1)  # exit on error; recovery not possible
        
    def getRight(self) -> "Tree":
        try:
            if self.right is None:
                raise Exception('Tree tried to access a child that didn\'t exist')
            else:
                return self.right
        except Exception as err:
            log.error(str(err))
            print(str(err))
            traceback.print_stack()
            sys.exit(-1)  # exit on error; recovery not possible
            
    def getMiddle(self) -> "Tree":
        try:
            if self.middle is None:
                raise Exception('Tree tried to access a child that didn\'t exist')
            else:
                return self.middle
        except Exception as err:
            log.error(str(err))
            print(str(err))
            traceback.print_stack()
            sys.exit(-1)  # exit on error; recovery not possible
        
    # running a tree should return a single value
    # featureValues -- the values of the relevant features keyed by their index in the original data
    def runTree(self, featureValues: typ.Dict[int, float]) -> float:
        return self.__runNode(featureValues)

    def __runNode(self, featureValues: typ.Dict[int, float]) -> typ.Union[int, float]:

        # + Try creating a unit test for this, grow, & full
        # if the node is an operation both of it's branches should have operations or terminals
        # either way calling __runNode() won't return a None
        log.debug('Attempting to run __runNode method...')
        try:
            if self.data not in OPS:  # if the node isn't in OPS, then it should be a terminal
    
                # *************************** Error Checking *************************** #
                if math.isnan(self.data):             # if the value stored is a NaN
                    log.error(f'NaN stored in tree. Expect OPS value or number, got {self.data}')
                    raise Exception(f'ERROR: NaN stored in tree. Expect OPS value or number, got {self.data}')
                
                if featureValues[self.data] is None:  # if the value stored is a None
                    raise Exception('featureValues contained a None at index self.data')
                # ********************************************************************** #
                
                return featureValues[self.data]  # if the terminal is valid, return it
            
            else:  # if this node is an operation, find which one & start recursion
    
                # *********************************** Error Checking ****************** #
                if self.left is None and self.right is not None:    # if one child is None, but not both
                    raise Exception('runNode found a node in OPS with 1 \'None\' child')
                
                elif self.right is None and self.left is not None:  # if one child is None, but not both
                    raise Exception('runNode found a node in OPS with 1 \'None\' child')
                
                if self.left is None and self.right is None:        # if both children are None
                    raise Exception('runNode found a node in OPS with 2 \'None\' children')
                
                if self.data == 'if' and self.middle is None:       # if the OP is IF and it has no middle
                    raise Exception('runNode found a node with a IF OP and no middle node')
                # ********************************************************************** #
                
                # *************** Determine Which OP is Stored & Run Recursion *************** #
                if self.data == 'add':         # if the OP was add
                    vl = self.left.__runNode(featureValues) + self.right.__runNode(featureValues)
                    return vl
                elif self.data == 'subtract':  # if the OP was subtract
                    vl = self.left.__runNode(featureValues) - self.right.__runNode(featureValues)
                    return vl
                elif self.data == 'times':     # if the OP was multiplication
                    vl = self.left.__runNode(featureValues) * self.right.__runNode(featureValues)
                    return vl
                elif self.data == 'max':       # if the OP was max
                    vl = max(self.left.__runNode(featureValues), self.right.__runNode(featureValues))
                    return vl
                elif self.data == 'if':        # if the OP was if
                    if self.left.__runNode(featureValues) >= 0:
                        vl = self.right.__runNode(featureValues)
                    else:
                        vl = self.getMiddle().__runNode(featureValues)
                    return vl
            
        except IndexError:
            lineNm = sys.exc_info()[-1].tb_lineno    # print line number error occurred on
            log.error(f'Index stored in tree was in range but did not exist. Value stored was:{self.data}, line = {lineNm}')
            print(f'ERROR: Index stored in tree was in range but did not exist. Value stored was:{self.data}, line = {lineNm}')
            sys.exit(-1)  # exit on error; recovery not possible
        except Exception as err:
            lineNm = sys.exc_info()[-1].tb_lineno    # print line number error occurred on
            print(str(err) + f', line = {lineNm}')
            traceback.print_stack()
            sys.exit(-1)  # exit on error; recovery not possible

    def setSize(self, counter) -> int:
        # BUG this is what is causing the exit code error
        counter += 1  # increment the counter

        leftCount = 0
        rightCount = 0
        
        if self.data not in OPS:  # if this is a terminal node
            return counter

        else:  # if this isn't a terminal walk the tree
            if self.left:                                 # if the left node is not null,
                leftCount = self.left.setSize(counter)    # then call recursively

            if self.right:                                # if the right node is not null,
                rightCount = self.right.setSize(counter)  # then call recursively
        
        # add the size of the left subtree to the right subtree to get the size
        # of everything below this node. Then return it up the recursive stack
        self.size = leftCount + rightCount
        return self.size


class ConstructedFeature:
    """Constructed Feature is used to represent a single constructed feature in
       a hypothesis. It contains the tree representation of the feature along
       with additional information about the feature.

    Variables:
        tree: This is the constructed features binary decision tree.

        className: The name/id of the class that the feature is meant to
                   distinguish.

        infoGain: The info gain of the feature.

        relevantFeatures: The list of terminal characters relevant to the
                          feature's class.

        transformedValues: The values of the original data after they have
                           been transformed by the decision tree.

    Methods:
        transform:  This takes the original data & transforms it using the
                    feature's decision tree.
    """

    infoGain = None           # the info gain for this feature
    usedFeatures = None       # the relevant features the cf actually uses
    transformedValues = None  # the values data after they have been transformed by the tree

    def __init__(self, className: int, tree: Tree, size: int = 0) -> None:
        self.className = className                    # the name of the class this tree is meant to distinguish
        self.tree = tree                              # the root node of the constructed feature
        self.size = size                              # the individual size  # TODO set size
        self.relevantFeatures = TERMINALS[className]  # holds the indexes of the relevant features
        # ! if tree sanity check passes then the constructed feature is fine & the error is somewhere else
        # sanityCheckCF(self)  # ! testing purposes only!

    def getUsedFeatures(self) -> typ.List[int]:
    
        # will hold the indexes found at each terminal node
        values = []  # type: typ.List[int]
        # ? do I need to be passing values instead of just using them?
        
        def __walk(node: Tree) -> None:
            # if this tree's node is a valid operation, keep walking done the tree
            if node.data in OPS:
                __walk(node.left)   # walk down the left branch
                __walk(node.right)  # walk down the right branch
                return              # now that I have walked down both branches return
        
            # if the node is not an operation, then it is a terminal index so add it to value
            else:
                values.append(node.data)
                return

        __walk(self.tree)  # walk the tree starting with the CF's root node
        
        try:
            if not values:  # if values is empty
                
                raise Exception('ERROR: The getUsedFeatures found no features (values list is empty)')
            return values      # values should now hold the indexes of the tree's terminals
        except Exception as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'The getUsedFeatures found no features (values list is empty), line number = {lineNm}')
            print(str(err) + f', line = {lineNm}')
            sys.exit(-1)  # exit on error; recovery not possible

    def transform(self, instance: Instance) -> float:
        # Send the tree a list of all the attribute values in a single instance
        featureValues: typ.Dict[int, float] = instance.attributes
        return self.tree.runTree(featureValues)

    def setSize(self):
        log.debug('Starting setSize function on Tree')
        self.size = self.tree.setSize(0)  # call getSize on the root of the tree
        log.debug('Completed setSize function on Tree')
        return


class Hypothesis:
    # a single hypothesis(a GP individual)
    fitness: typ.Union[None, int, float] = None  # the fitness score
    distance: typ.Union[float, int] = 0          # the distance function score
    averageInfoGain: typ.Union[float, int] = -1  # the average info gain of the hypothesis
    maxInfoGain: typ.Union[float, int] = -1      # the max info gain in the hypothesis
    # + averageInfoGain & maxInfoGain must be low enough that they will always be overwritten + #
    
    def __init__(self, features: typ.List[ConstructedFeature], size: int) -> None:
        self.features: typ.List[ConstructedFeature] = features  # a list of all the constructed features
        self.size: int = size                                   # the number of nodes in all the cfs

    def getFitness(self) -> float:
    
        log.debug('Starting getFitness() method')
        
        def __Czekanowski(Vi: typ.List[float], Vj: typ.List[float]) -> float:
            log.debug('Starting Czekanowski() method')

            # ************************** Error checking ************************** #
            # BUG Vi & Vj are [None, None]
            # ? How???
            try:
                if len(Vi) != len(Vj):
                    log.error(f'In Czekanowski Vi[d] & Vi[d] are not equal Vi = {Vi}, Vj = {Vj}')
                    raise Exception(f'ERROR: In Czekanowski Vi[d] & Vi[d] are not equal Vi = {Vi}, Vj = {Vj}')
                if None in Vi:
                    log.error(f'In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                    raise Exception(f'ERROR: In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                if None in Vj:
                    log.error(f'In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
                    raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
            except Exception as err:
                lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                print(str(err) + f', line = {lineNm}')
                sys.exit(-1)  # recovery impossible, exit with an error
            # ******************************************************************** #

            minSum: typ.Union[int, float] = 0
            addSum: typ.Union[int, float] = 0
            
            # + range(len(self.features)) loops over the number of features the hypothesis has.
            # + Vi & Vj are lists of the instances from the original data, that have been transformed
            # + by the hypothesis.
            try:

                for i, j in zip(Vi, Vj):                    # zip Vi & Vj so that we can iterate in parallel
                    
                    # **************************************** Error Checking **************************************** #
                    if Vi == [None, None] and Vj == [None, None]:
                        log.error(f'In Czekanowski Vj ({Vj}) & Vi ({Vi}) was found to be a \'None type\'')
                        raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) & Vi ({Vi}) was found to be a \'None type\'')
                    
                    elif Vj == [None, None]:
                        log.error(f'In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
                        raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
                    
                    elif Vi == [None, None]:
                        log.error(f'In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                        raise Exception(f'ERROR: In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                    # ************************************************************************************************ #
                    
                    top: typ.Union[int, float] = min(i, j)  # get the top of the fraction
                    bottom: typ.Union[int, float] = i + j   # get the bottom of the fraction
                    minSum += top                           # the top of the fraction
                    # ! addsum is zero a lot of the time
                    addSum += bottom                        # the bottom of the fraction
            
                if addSum == 0:  # BUG this attempts to divide by zero a lot; check that this is okay
                    raise RuntimeWarning('ERROR: Czekanowski attempted to divide by zero')
                else:
                    value = 1 - ((2*minSum) / addSum)           # capture the return value
            
            except RuntimeWarning as err:
                log.error(str(err))
                # lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                # print(str(err) + f', line = {lineNm}')
                value = 0                              # attempt recovery

            except Exception as err:
                lineNm = sys.exc_info()[-1].tb_lineno       # print line number error occurred on
                log.error(str(err))
                print(str(err) + f', line = {lineNm}')
                sys.exit(-1)                                # recovery impossible, exit with error

            log.debug('Finished Czekanowski() method')
            
            return value

        def Distance(values: typ.List[Instance]):
    
            log.debug('Starting Distance() method')
            
            # NOTE these must be high/low enough that it is always reset
            # NOTE these will be replaced when higher/lower values are found, they will NOT be added to
            # TODO ^^^^ check this ^^^^
            Db: typ.Union[int, float] = 2  # this will hold the lowest distance Czekanowski found
            Dw: typ.Union[int, float] = 0  # this will hold the highest distance Czekanowski found
    
            # ********** Compute Vi & Vj ********** #
            # the reason for these two loops is to allow us to compare vi with every other instance (vj)
            for vi in values:                                   # loop over all the training examples
                for vj in values:                               # loop over all the training examples

                    dist = __Czekanowski(vi.vList, vj.vList)  # compute the distance using the values
            
                    if vi.className == vj.className:            # if the instances vi & vj are from the same class, skip
                        continue
            
                    elif vi.attributes == vj.attributes:    # if vi & vj are not in the same class (Db), skip
                        continue
            
                    else:                                   # if vi & vj are valid
                        if dist > Dw:                       # replace the max if the current value is higher
                            Dw = dist
                            
                        if dist < Db:                       # replace the min if the current value is smaller
                            Db = dist

            # perform the final distance calculations
            Db *= (1 / len(values))  # multiply by 1/|S|
            Dw *= (1 / len(values))  # multiply by 1/|S|

            log.debug('Finished Distance() method')
            
            return 1 / (1 + math.pow(math.e, -5*(Db - Dw)))

        def __entropy(partition: typ.List[Instance]) -> float:
    
            log.debug('Starting entropy() method')
            
            p: typ.Dict[int, int] = {}   # p[classId] = number of instances in the class in the partition sv
            for i in partition:          # for instance i in a partition sv
                if i.className in p:       # if we have already found the class once,
                    p[i.className] += 1  # increment the counter
                    
                else:                   # if we have not yet encountered the class
                    p[i.className] = 1  # set the counter to 1

            calc = 0
            for c in p.keys():  # for class in the list of classes in the partition sv
                # perform entropy calculation
                pi = p[c] / len(partition)
                calc -= pi * math.log(pi, 2)

            log.debug('Finished entropy() method')

            return calc

        def __conditionalEntropy(feature: ConstructedFeature) -> float:
    
            log.debug('Starting conditionalEntropy() method')

            # this is a feature struct that will be used to store feature values
            # with their indexes/IDs in CFs
            ft = collect.namedtuple('ft', ['id', 'value'])
            
            # key = CF(Values), Entry = instance in training data
            partition: typ.Dict[float, typ.List[Instance]] = {}
            
            s = 0                             # used to sum CF's conditional entropy
            used = feature.getUsedFeatures()  # get the indexes of the used features
            v = []                            # this will hold the used features ids & values
            for i in rows:                    # loop over all instances

                # get CF(v) for this instance (i is a Instance struct which is what transform needs)
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

            log.debug('Finished conditionalEntropy() method')

            return s  # s holds the conditional entropy value

        gainSum = 0  # the info gain of the hypothesis
        for f in self.features:  # loop over all features & get their info gain

            # ********* Entropy calculation ********* #
            condEntropy = __conditionalEntropy(f)  # find the conditional entropy

            # ******** Info Gain calculation ******* #
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
        self.distance = Distance(self.transform())  # calculate the distance using the transformed values

        # ********* Final Calculation ********* #
        term1 = ALPHA*self.averageInfoGain
        term2 = (1-ALPHA)*self.distance
        term3 = (math.pow(10, -7)*self.size)
        final = term1 + term2 - term3
        # ********* Finish Calculation ********* #

        log.debug('Finished getFitness() method')
        return final

    # NOTE: this is the function used by cdfcProject
    def transform(self, data: typ.Union[None, np.array] = None) -> np.array:
    
        log.debug('Starting transform() method')
        
        transformed = []  # this will hold the transformed values
        
        # if data is None then we are transforming as part of the distance calculation
        # so we should use rows (the provided training data)
        if data is None:
    
            for r in rows:  # for each Instance
                values = []   # this will hold the calculated values for all the constructed features

                for f in self.features:            # transform the original input using each constructed feature
                    values.append(f.transform(r))  # append the transformed values for a single CF to values
                
                # each Instance will hold the new values for an Instance & className, and
                # transformed will hold all the instances for a hypothesis
                vls = dict(zip(range(len(values)), values))  # create a dict of the values keyed by their index
                transformed.append(Instance(r.className, vls))

            log.debug('Finished transform() method')

            return transformed  # return the list of all instances

        # if data is not None then we are predicting using an evolved model so we should use data
        # (this will be testing data from cdfcProject.py)
        else:
            
            for d in data:   # for each Instance
                values = []  # this will hold the calculated values for all the constructed features
                
                for f in self.features:            # transform the original input using each constructed feature
                    values.append(f.transform(d))  # append the transformed values for a single CF to values
                
                # each Instance will hold the new values for an Instance & className, and
                # transformed will hold all the instances for a hypothesis
                vls = dict(zip(range(len(values)), values))  # create a dict of the values keyed by their index
                transformed.append(Instance(d.className, vls))

            log.debug('Finished transform() method')

            return transformed  # return the list of all instances

    def setSize(self) -> None:
        for i in self.features:
            i.setSize()


class Population:
    # this will be the population of hypotheses. This is largely just a namespace
    def __init__(self, candidates: typ.List[Hypothesis], generationNumber: int) -> None:
        self.candidateHypotheses = candidates  # a list of all the candidate hypotheses
        self.generation = generationNumber     # this is the number of this generation
# ***************** End of Namespaces/Structs & Objects ******************* #


def __grow(relevantValues: typ.List[int], depth: int = 0) -> typ.Tuple[Tree, int]:
    # This function uses the grow method to generate an initial population
    # the last thing returned should be a trees root node
    if int == 0:  # only print to log on first call
        log.debug('The __grow method has been chosen to build a tree')

    depth += 1  # increase the depth by one

    if depth == MAX_DEPTH:  # if we've reached the max depth add a random terminal value and return
        return Tree(random.choice(relevantValues)), depth
    
    else:
        # unpack the list of terminals & operations, and then combine them
        ls: typ.List[typ.Union[int, str]] = [*OPS[:], *relevantValues]

        value: typ.Union[int, str] = random.choice(ls)   # get a random value from the combined list
    
        if value in relevantValues:     # if a terminal value was chosen
            return Tree(value), depth   # create a new tree & return it
        
        else:  # if the value was not a terminal value, then grow both children
            
            # change the below to an elif if more than 2/3 terminals are needed
            if NUM_TERMINALS[value] == 2:                          # if the number of terminals needed is two
                lft, leftDepth = __grow(relevantValues, depth)     # grow the left terminal or operation tree
                rgt, rightDepth = __grow(relevantValues, depth)    # grow the right terminal or operation tree
                newTree = Tree(value, lft, rgt)                    # create the new tree
                totalDepth = leftDepth + rightDepth                # update total size
                
            else:                                                  # if the number of terminals needed is three
                lft, leftDepth = __grow(relevantValues, depth)     # grow the left terminal or operation tree
                rgt, rightDepth = __grow(relevantValues, depth)    # grow the right terminal or operation tree
                mdl, middleDepth = __grow(relevantValues, depth)   # grow the right terminal or operation tree
                newTree = Tree(value, lft, rgt, mdl)               # create the new tree
                totalDepth = leftDepth + rightDepth + middleDepth  # update total size

            # sanityCheckTree(newTree)    # ! testing purposes only!
            # ! if tree sanity check passes then the tree is fine & error is in the ConstructedFeature()
            return newTree, totalDepth  # after the recursive calls have finished return the tree


def __full(relevantFeatures: typ.List[int], depth: int = 0) -> typ.Tuple[Tree, int]:
    # This function uses the full method to generate an initial population
    # the last thing returned should be a trees root node
    if int == 0:  # only print to log on first call
        log.debug('The __full method has been chosen to build a tree')

    depth += 1  # increase the depth by one

    if depth == MAX_DEPTH:  # if we've reached the max depth add a random terminal value and return
        # this tree should store a terminal, so no children should be created
        return Tree(random.choice(relevantFeatures)), depth
    
    else:  # if we haven't reached the max depth, we should create a tree with an OP
        value = random.choice(OPS)                             # select a random operation
        
        if NUM_TERMINALS[value] == 2:                          # if the number of terminals needed is two
            lft, leftDepth = __full(relevantFeatures, depth)   # grow the left terminal or operation tree
            rgt, rightDepth = __full(relevantFeatures, depth)  # grow the right terminal or operation tree
            newTree = Tree(value, lft, rgt)                    # create a new tree with Value (an operation) as it's data
            totalDepth = leftDepth + rightDepth                # update total size

        else:                                                   # if the number of terminals needed is three
            lft, leftDepth = __full(relevantFeatures, depth)    # grow the left terminal or operation tree
            rgt, rightDepth = __full(relevantFeatures, depth)   # grow the right terminal or operation tree
            mdl, middleDepth = __full(relevantFeatures, depth)  # grow the right terminal or operation tree
            newTree = Tree(value, lft, rgt, mdl)                # create the new tree
            totalDepth = leftDepth + rightDepth + middleDepth   # update total size

        # sanityCheckTree(newTree)    # ! testing purposes only!
        return newTree, totalDepth  # after the recursive calls have finished return the tree


def createInitialPopulation() -> Population:
    
    def createHypothesis() -> Hypothesis:
        # given a list of trees, create a hypothesis
        # NOTE this will make 1 tree for each feature, and 1 CF for each class

        classIds: typ.List[int] = copy.copy(CLASS_IDS)  # create a copy of all the unique class ids
        random.shuffle(classIds)                        # shuffle the class ids so their order is random

        ftrs: typ.List[ConstructedFeature] = []
        size = 0

        # ? should this be LABEL_NUMBER or FEATURE_NUMBER
        for nll in range(LABEL_NUMBER):
            # randomly decide if grow or full should be used.
            # Also randomly assign the class ID then remove that ID
            # so each ID may only be used once
            
            try:
                name = classIds.pop(0)  # get a random id
                
                if name not in CLASS_IDS:
                    raise Exception(f'createHypothesis got an invalid name ({name}) from classIds')
            
            except IndexError:                         # if classIds.pop() tried to pop an empty list, log error & exit
                lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                log.error(f'Index error encountered in createInitialPopulation (popped from an empty list), line {lineNm}')
                print(f'ERROR: Index error encountered in createInitialPopulation (popped from an empty list), line {lineNm}')
                sys.exit(-1)                           # exit on error; recovery not possible
            except Exception as err:                   # if class ids some how gave an invalid name
                lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                log.error(str(err))
                print(f'ERROR: {str(err)}, line {lineNm}')
                sys.exit(-1)                           # exit on error; recovery not possible
            # DEBUG if we pass this point we know that name is valid

            if random.choice([True, False]):           # *** use grow *** #
                tree, size = __grow(TERMINALS[name])   # create tree using grow
            else:                                      # *** use full *** #
                tree, size = __full(TERMINALS[name])   # create tree using full
            cf = ConstructedFeature(name, tree, size)  # create constructed feature
            ftrs.append(cf)                            # add the feature to the list of features

            size += size
            
        if classIds:  # if we didn't pop everything from the classIds list, raise an exception
            log.error(f'creatInitialPopulation didn\'t use all of classIds, classIds = {classIds}')
            raise Exception(f'ERROR: creatInitialPopulation didn\'t use all of classIds, classIds = {classIds}')
            
        # create a hypothesis & return it
        return Hypothesis(ftrs, size)

    hypothesis: typ.List[Hypothesis] = []

    # creat a number hypotheses equal to pop size
    with alive_bar(POPULATION_SIZE, title="Creating Initial Hypotheses") as bar:  # declare your expected total
        for __ in range(POPULATION_SIZE):  # iterate as usual
            hyp = createHypothesis()       # create a Hypothesis
            # sanityCheckHyp(hyp)          # ! testing purposes only!
            hypothesis.append(hyp)         # add the new hypothesis to the list
            bar()
    
    # sanityCheckPop(hypothesis)    # ! testing purposes only!
    
    return Population(hypothesis, 0)


# ********** Sanity Check Functions used for Debugging ********** #
# ! testing purposes only!
def sanityCheckPop(hypothesis: typ.List[Hypothesis]):
    log.debug('Starting Population Sanity Check...')
    for h in hypothesis:
        h.transform(rows)
    log.debug('Population Sanity Check Passed')


# ! testing purposes only!
def sanityCheckHyp(hyp: Hypothesis):
    log.debug('Starting Hypothesis Sanity Check...')
    hyp.transform()
    log.debug('Population Hypothesis Check Passed')


# ! testing purposes only!
def sanityCheckCF(cf: ConstructedFeature):
    log.debug('Starting Constructed Feature Sanity Check...')
    cf.transform(rows[0])
    log.debug('Constructed Feature Sanity Check Passed')


# ! testing purposes only!
def sanityCheckTree(tree: Tree):
    log.debug('Starting Tree Sanity Check...')
    tree.runTree(rows[0].attributes)
    log.debug('Tree Sanity Check Passed')
# *************************************************************** #


def evolve(population: Population, elite: Hypothesis) -> typ.Tuple[Population, Hypothesis]:

    def __tournament(p: Population) -> Hypothesis:
        # used by evolve to selection the parents
        # **************** Tournament Selection **************** #
        candidates: typ.List[Hypothesis] = copy.deepcopy(p.candidateHypotheses)  # copy to avoid overwriting
        first = None  # the tournament winner
        score = 0     # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
            randomIndex = random.choice(range(len(candidates)))   # get a random index value
            candidate: Hypothesis = candidates.pop(randomIndex)   # get the hypothesis at the random index
            # we pop here to avoid getting duplicates. The index uses candidates current size so it will be in range
            log.debug('Making getFitness method call in Tournament')
            fitness = candidate.getFitness()                      # get that hypothesis's fitness score
            log.debug('Finished getFitness method call in Tournament')

            if first is None:      # if first has not been set,
                first = candidate  # then  set it

            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it
                score = fitness    # then update the score to higher fitness
                
        try:
            if first is None:
                raise Exception(f'ERROR: Tournament could not set first correctly, first = {first}')
        except Exception as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Tournament could not set first correctly, first = {first}, line number = {lineNm}')
            print(str(err) + f', line = {lineNm}')
            sys.exit(-1)  # exit on error; recovery not possible
        
        log.debug('Finished Tournament method')
        return first
        # ************ End of Tournament Selection ************* #

    # ******************* Evolution ******************* #
    newPopulation = Population([], population.generation+1)  # create a new population with no hypotheses
    
    def getSubtree(ftr: Tree, terms: typ.List[int]) -> typ.Tuple[Tree, typ.Tuple[Tree, Tree]]:
        # walk the tree & find a random subtree for feature
        # TODO calculate size during this first walk

        oldFeature: Tree = ftr
        while True:
        
            # make a random decision
            choice = random.choice(["left", "right", "stop"])
            
            if choice == "stop" or ftr.data in terms:
                break  # if we chose stop or if we have encountered a terminal, break
                
            elif choice == "left" and ftr.left is None:
                break  # if choose to go left but there isn't a left, break
        
            elif choice == "right" and ftr.right is None:
                break  # if choose to go right but there isn't a right, break
        
            # NOTE: the code below is used to prevent a singular terminal node from being a valid subtree
            # elif choice == "left" and ftr.left.data in terms:
            #     break  # if we try to go left, but left is a terminal, break
        
            # elif choice == "right" and ftr.right.data in terms:
            #     break  # if we try to go right, but right is a terminal, break
        
            elif choice == "left" and ftr.left is not None:
                oldFeature = ftr  # ? will this be overwritten on next step?
                ftr = ftr.left  # if we chose left & left isn't None, go left
        
            elif choice == "right" and ftr.right is not None:
                oldFeature = ftr
                ftr = ftr.right  # if we chose right & right isn't None, go right
        
        return ftr, (oldFeature, oldFeature)  # return the subtree, and the parent tree
    
    # while the new population has fewer hypotheses than the max pop size
    while (len(newPopulation.candidateHypotheses)-1) < POPULATION_SIZE:
        
        probability = random.uniform(0, 1)  # get a random number between 0 & 1
        
        # **************** Mutate **************** #
        if probability < MUTATION_RATE:  # if probability is less than mutation rate, mutate
            
            # parent is from a copy of population made by tournament, NOT original pop
            parent: Hypothesis = __tournament(population)  # get parent hypothesis using tournament
            log.debug('Finished Tournament method call in evolve')
            
            rIndex: int = random.choice(range(len(parent.features)))     # get a random index
            randomFeature: ConstructedFeature = parent.features[rIndex]  # use the random index to get a feature from the hypothesis
            terminal: typ.List[int] = randomFeature.relevantFeatures     # get the indexes of the terminal values (for the feature)
            feature: Tree = randomFeature.tree                           # get the tree (for the feature)

            # randomly select a subtree in feature
            while True:  # walk the tree & find a random subtree
                
                decide = random.choice(["left", "right", "choose"])  # make a random decision

                if decide == "choose" or feature.data in terminal:
                    break
                
                elif decide == "left":   # go left
                    feature = feature.left

                elif decide == "right":  # go right
                    feature = feature.right
                    
            decideGrow = random.choice([True, False])  # randomly decide which method to use to construct the new tree
            
            # randomly generate subtree
            if decideGrow:  # use grow
                t, size = __grow(terminal)  # build the subtree

            else:  # use full
                t, size = __full(terminal)  # build the subtree
                
            cl = randomFeature.className                                # get the className of the feature
            parent.features[rIndex] = ConstructedFeature(cl, t, size)   # replace the parent with the mutated child
            newPopulation.candidateHypotheses.append(parent)            # add the parent to the new pop
            # appending is needed because parent is a copy made by tournament NOT a reference from the original pop
        # ************* End of Mutation ************* #

        # **************** Crossover **************** #
        else:
            
            # parent1 & parent2 are from a copy of population made by tournament, NOT original pop
            # because of this they should not be viewed as references
            parent1: Hypothesis = __tournament(population)
            parent2: Hypothesis = __tournament(population)
            #  check that each parent is unique
            # if they are the same they should reference the same object & so 'is' is used instead of ==
            while parent1 is parent2:
                parent2 = __tournament(population)

            # feature 1
            randomFeature: ConstructedFeature = random.choice(parent1.features)  # get a random feature from the parent
            terminals1: typ.List[int] = randomFeature.relevantFeatures           # get the indexes of the terminal values (for the feature)
            feature1: Tree = randomFeature.tree                                  # get the tree (for the feature)
            
            # ? should I get a new random feature or make sure they have the same classId? I don't think so
            # feature 2
            randomFeature = random.choice(parent2.features)             # get a random feature from the parent
            terminals2: typ.List[int] = randomFeature.relevantFeatures  # get the indexes of the terminal values (for the feature)
            feature2: Tree = randomFeature.tree                         # get the tree (for the feature)
            
            # *************** Find the Two Sub-Trees **************** #
            subTree1, parentTree1 = getSubtree(feature1, terminals1)
            subTree2, parentTree2 = getSubtree(feature2, terminals2)
            # ******************************************************* #
            
            # ************************** swap the two subtrees ************************** #
            # update the first parent tree
            if parentTree1[1] == 'left':         # if subTree 1 went left to find the subTree
                parentTree1[0].left = subTree2   # then replace the left with subTree 2
            else:                                # if subTree 1 went right to find the subTree
                parentTree1[0].right = subTree2  # then replace the right with subTree 2
            # update the second parent tree
            if parentTree2[1] == 'left':         # if subTree 2 went left to find the subTree
                parentTree2[0].left = subTree1   # then replace the left with subTree 1
            else:                                # if subTree 2 went right to find the subTree
                parentTree2[0].right = subTree1  # then replace the right with subTree 1
            # **************************************************************************** #

            # TODO change the size calculation so that we only walk the tree once
            # get the size of the new constructed features by walking the trees
            log.debug('Crossover is attempting to set the size for the new parents')
            # BUG this is what is causing the exit code error
            parent1.setSize()
            parent2.setSize()
            log.debug('Crossover completed setting the size for the new parents')
            
            # parent 1 & 2 are both hypotheses and should have been changed in place,
            # but they refer to a copy made in tournament so add them to the new pop
            newPopulation.candidateHypotheses.append(parent1)
            newPopulation.candidateHypotheses.append(parent2)
            # **************** End of Crossover **************** #
    
        # handle elitism
        newHypothFitness = newPopulation.candidateHypotheses[-1].getFitness()
        if newHypothFitness > elite.getFitness():
            elite = newPopulation.candidateHypotheses[-1]
        log.debug('Starting getFitness call in Evolution as part of elitism')
    
    print('Evolution Done')
    return newPopulation, elite


def cdfc(dataIn) -> Hypothesis:
    # Class Dependent Feature Construction

    values = dataIn[0]
    terminals = dataIn[1]
    
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
    
    # Read the values in the dictionary into the constants
    FEATURE_NUMBER = values['FEATURE_NUMBER']
    CLASS_IDS = values['CLASS_IDS']
    POPULATION_SIZE = values['POPULATION_SIZE']
    INSTANCES_NUMBER = values['INSTANCES_NUMBER']
    LABEL_NUMBER = values['LABEL_NUMBER']
    M = values['M']
    rows = values['rows']
    ENTROPY_OF_S = values['ENTROPY_OF_S']
    CLASS_DICTS = values['CLASS_DICTS']
    TERMINALS = terminals
    
    # *********************** Run the Algorithm *********************** #

    currentPopulation = createInitialPopulation()     # run initialPop
    SYSOUT.write(HDR + ' Initial population generated '.ljust(50, '-') + SUCCESS)
    # currentPopulation = createInitialPopulation()   # create initial population
    elite = currentPopulation.candidateHypotheses[0]  # init elitism

    # loop, evolving each generation. This is where most of the work is done
    log.debug('Starting generations stage...')
    with alive_bar(GENERATIONS, title="Generations") as bar:  # declare your expected total
        for __ in range(GENERATIONS):  # iterate as usual
            newPopulation, elite = evolve(currentPopulation, elite)  # generate a new population by evolving the old one
            # update currentPopulation to hold the new population
            # this is done in two steps to avoid potential namespace issues
            currentPopulation = newPopulation
            bar()
            
    log.debug('Finished evolution stage')
    SYSOUT.write(HDR + ' Final Generation Reached '.ljust(50, '-') + SUCCESS)  # update user
    # ***************************************************************** #

    # ****************** Return the Best Hypothesis ******************* #
    # check to see if the last generation has generated fitness scores
    log.debug('Finding best hypothesis')
    if currentPopulation.candidateHypotheses[2].fitness is None:
        # if not then generate them
        for i in currentPopulation.candidateHypotheses:
            i.getFitness()

    # now that we know each hypothesis has a fitness score, get the one with the highest fitness
    # ? check this works
    fitBar = tqdm(currentPopulation.candidateHypotheses, desc='Finding most fit hypothesis', unit='hyp', ncols=BARCOLS)
    bestHypothesis = max(fitBar, key=lambda x: x.fitness)
    log.debug('Found best hypothesis, returning...')
    return bestHypothesis  # return the best hypothesis generated
