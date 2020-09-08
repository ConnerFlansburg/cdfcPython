import copy
import math
import random
import sys
import warnings
from pathlib import Path
import logging as log
import numpy as np
import typing as typ
import collections as collect
from scipy import stats
from tqdm import tqdm
from tqdm import trange

# ! Next Steps
# TODO fix bug in relevancy
# TODO fix bug in run tree

# TODO write code for the if function in OPS
# TODO add docstrings
# TODO add testing functions

# **************************** Constants/Globals **************************** #
ALPHA: typ.Final = 0.8           # ALPHA is the fitness weight alpha
BETA: typ.Final = 2              # BETA is a constant used to calculate the pop size
CROSSOVER_RATE: typ.Final = 0.8  # CROSSOVER_RATE is the chance that a candidate will reproduce
ELITISM_RATE: typ.Final = 1      # ELITISM_RATE is the elitism rate
GENERATIONS: typ.Final = 50      # GENERATIONS is the number of generations the GP should run for
MAX_DEPTH: typ.Final = 8         # MAX_DEPTH is the max depth trees are allowed to be & is used in grow/full
MUTATION_RATE: typ.Final = 0.2   # MUTATION_RATE is the chance that a candidate will be mutated
# ! changes here must also be made in the runLeft & runRight functions in the tree object ! #
OPS: typ.Final = ['add', 'subtract', 'times', 'max', ]  # OPS is the list of valid operations on the tree
# ! set the value of R for every new dataset, it is NOT set automatically ! #
R: typ.Final = 2                 # R is the ratio of number of CFs to the number of classes (features/classes)
TOURNEY: typ.Final = 7           # TOURNEY is the tournament size
ENTROPY_OF_S = 0                 # ENTROPY_OF_S is used for entropy calculation
FEATURE_NUMBER = 0               # FEATURE_NUMBER is the number of features in the data set
INSTANCES_NUMBER = 0             # INSTANCES_NUMBER is  the number of instances in the training data
LABEL_NUMBER = 0                 # LABEL_NUMBER is the number of classes/labels in the data
M = 0                            # M is the number of constructed features
POPULATION_SIZE = 0              # POPULATION_SIZE is the population size
# console formatting strings
HDR = '*' * 6
SUCCESS = u' \u2713\n'
OVERWRITE = '\r' + HDR
SYSOUT = sys.stdout
# ************************ End of Constants/Globals ************************ #

# ********************** Namespaces/Structs & Objects *********************** #
row = collect.namedtuple('row', ['className', 'attributes'])  # a single line in the csv, representing a record/instance
rows: typ.List[row] = []  # this will store all of the records read in (the training dat) as a list of rows

np.seterr(divide='ignore')  # suppress divide by zero warnings from numpy
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# create the file path for the log file & configure the logger
logPath = str(Path.cwd() / 'logs' / 'cdfc.log')
log.basicConfig(level=log.DEBUG, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')


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
                 right: typ.Union[None, "Tree"] = None) -> None:
        self.data: typ.Union[int, str] = data    # must either be a function or a terminal (if a terminal it should be it's index)
        self.left = left
        self.right = right
        
    def setLeft(self, left):
        self.left = left
        
    def setRight(self, right):
        self.left = right
        
    # running a tree should return a single value
    # featureValues -- the values of the relevant features keyed by their index in the original data
    def runTree(self, featureValues):
        return self.__runNode(featureValues)

    def __runNode(self, featureValues):
        
        # BUG somehow +,-,*, and min() are being called on None type objects.
        # !   This is because for some reason it keeps trying to access the children that don't exist, even though it
        # !   encounters values before then - How??? If a node contains a number it should return a value & not run children
        # ? Maybe  when we have a value and try to index it using featureValues[self.data], we are getting a None?
        # ?   (i.e. the issue is that the instance has a None value at that index)
        # ? Maybe it's because featureValues is a Dictionary not a list? -- check
        # ? I think this error is a result of the error in the relevancy calculation that's preventing nodes from being set
        # + Try creating a unit test for this, grow, & full
        # if the node is an operation both of it's branches should have operations or terminals
        # either way calling __runNode() won't return a None
        log.debug('Attempting to run __runNode method...')
        try:
            if self.data in OPS:  # if this tree's node is a valid operation, then execute it
                log.debug('self.data was found in OPS...')
                # find out which operation it is & return it's value
                if self.data == 'add':
                    lft = self.left.__runNode(featureValues)
                    rgt = self.right.__runNode(featureValues)
                    vl = lft + rgt
                    return vl
        
                elif self.data == 'subtract':
                    lft = self.left.__runNode(featureValues)
                    rgt = self.right.__runNode(featureValues)
                    vl = lft - rgt
                    return vl
        
                elif self.data == 'times':
                    lft = self.left.__runNode(featureValues)
                    rgt = self.right.__runNode(featureValues)
                    vl = lft * rgt
                    return vl
                elif self.data == 'min':
                    lft = self.left.__runNode(featureValues)
                    rgt = self.right.__runNode(featureValues)
                    vl = min(lft, rgt)
                    return vl
            
            # if the data stored is not in OPS and is not a number raise an exception
            elif math.isnan(self.data):
                log.error(f'NaN stored in tree. Expect OPS value or number, got {self.data}')
                raise Exception(f'ERROR: NaN stored in tree. Expect OPS value or number, got {self.data}')
            # if the data stored is null, raise exception
            elif self.data is None:
                log.error(f'None stored in tree. Expect OPS value or number, got {self.data}')
                raise Exception(f'ERROR: None stored in tree. Expect OPS value or number, got {self.data}')
            
            # if the node is not an operation than it should be a terminal index. So using it on featureValues
            # should return a float or an int (the value of some feature in an instance) not a None
            else:  # if the node is not an operation and is a number, then it is a terminal index so return the value
                log.debug(f'__runNode found the feature value {featureValues[self.data]} in a node')
                return featureValues[self.data]

        except Exception as err:
            tqdm.write(str(err))

    def getSize(self, counter) -> int:

        counter += 1  # increment the counter

        leftCount = 0
        rightCount = 0

        if self.left:                               # if the left node is not null,
            leftCount = self.left.getSize(counter)  # then call recursively

        if self.right:                                # if the right node is not null,
            rightCount = self.right.getSize(counter)  # then call recursively

        # add the size of the left subtree to the right subtree to get the size
        # of everything below this node. Then return it up the recursive stack
        return leftCount + rightCount


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
        self.className = className  # the name of the class this tree is meant to distinguish
        self.tree = tree            # the root node of the constructed feature
        self.size = size            # the individual size
        
        # call terminals to create the terminal set
        self.relevantFeatures = terminals(className)  # holds the indexes of the relevant features

    def getUsedFeatures(self) -> typ.List[int]:
    
        # will hold the indexes found at each terminal node
        values = []  # type: typ.List[int]
        # ? do I need to be passing values instead of just using them?
        
        def __walk(node: Tree) -> None:
            # if this tree's node is a valid operation, keep walking done the tree
            if node.data in OPS:
                __walk(node.left)  # walk down the left branch
                __walk(node.right)  # walk down the right branch
                return  # now that I have walked down both branches return
        
            # if the node is not an operation, then it is a terminal index so add it to value
            else:
                values.append(node.data)
                return

        __walk(self.tree)  # walk the tree starting with the CF's root node
        
        if not values:  # if values is empty
            log.debug('The getUsedFeatures found no features (values list is empty)')
        return values      # values should now hold the indexes of the tree's terminals

    def transform(self, instance: row) -> float:
        # Send the tree a list of all the attribute values in a single instance
        return self.tree.runTree(instance.attributes)

    def setSize(self):
        self.size = self.tree.getSize(0)  # call getSize on the root of the tree
        return


class Hypothesis:
    # a single hypothesis(a GP individual)
    fitness = None          # the fitness score
    distance = 0            # the distance function score
    averageInfoGain = -1    # the average info gain of the hypothesis
    maxInfoGain = -1        # the max info gain in the hypothesis
    # + averageInfoGain & maxInfoGain must be low enough that they will always be overwritten + #
    
    def __init__(self, features, size) -> None:
        self.features = features  # a list of all the constructed features
        self.size = size          # the number of nodes in all the cfs

    def getFitness(self) -> float:

        def __Czekanowski(Vi: typ.List[typ.Union[int, float]], Vj: typ.List[typ.Union[int, float]]):
            minSum = 0
            addSum = 0

            for d in range(1, len(self.features)):  # loop over the number of features
                # BUG unsupported operand += for int & list
                top = min(Vi[d], Vj[d])  # get the top of the fraction
                bottom = Vi[d] + Vj[d]   # get the bottom of the fraction

                # ************************** Error checking ************************** #
                try:
                    if type(Vi[d]) is list:
                        log.error(f'In Czekanowski Vi[d] is a list. Vi[d] = {Vi[d]}')
                        raise Exception(f'ERROR: In Czekanowski Vi[d] is a list. Vi[d] = {Vi[d]}')
                    if type(Vj[d]) is list:
                        log.error(f'In Czekanowski Vj[d] is a list. Vj[d] = {Vj[d]}')
                        raise Exception(f'ERROR: In Czekanowski Vj[d] is a list. Vj[d] = {Vj[d]}')
                    if type(top) is list:  # check if min is a list
                        log.error(f'In Czekanowski min is a list {top}')
                        raise Exception(f'ERROR: In Czekanowski min is a list {top}')
                    if type(bottom) is list:  # check if min is a list
                        log.error(f'In Czekanowski bottom of fraction is a list {bottom}')
                        raise Exception(f'ERROR: In Czekanowski bottom of fraction is a list {bottom}')
                except Exception as err:
                    tqdm.write(str(err))
                # ******************************************************************** #
                
                minSum += top     # the top of the fraction
                addSum += bottom  # the bottom of the fraction
                
            return 1 - ((2*minSum) / addSum)  # calculate it & return

        def Distance(values):

            # ********** Compute Vi & Vj ********** #
            Db = 0  # the sum of mins
            Dw = 0  # the sum of maxs
            
            for i, vi in enumerate(values):  # loop over all the transformed values

                minSum = 2  # NOTE must be high enough that it is always reset
                maxSum = 0

                for j, vj in enumerate(values):

                    if i == j:    # if i equals j ignore this case
                        continue  # continue / skip / return go to next element

                    dist = __Czekanowski(vi, vj)  # compute the distance

                    if vi.inClass == vj.inClass:  # if vi & vj are in the same class (Dw), then
                        if dist > maxSum:         # replace the max if the current value is higher
                            maxSum = dist
                    else:                  # if vi & vj are not in the same class (Db), then
                        if dist < minSum:  # replace the min if the current value is smaller
                            minSum = dist

                Db += minSum  # update the min total
                Dw += maxSum  # update the max total

            # perform the final distance calculations
            t1 = Db / len(values)
            t2 = Dw / len(values)
            
            return 1 / (1 + math.pow(math.e, -5*(t1 - t2)))

        def __entropy(partition: typ.List[row]) -> float:
            
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

            return calc

        def __conditionalEntropy(feature: ConstructedFeature) -> float:

            # this is a feature struct that will be used to store feature values
            # with their indexes/IDs in CFs
            ft = collect.namedtuple('ft', ['id', 'value'])
            
            # key = CF(Values), Entry = instance in training data
            partition: typ.Dict[float, typ.List[row]] = {}
            
            s = 0                             # used to sum CF's conditional entropy
            used = feature.getUsedFeatures()  # get the indexes of the used features
            v = []                            # this will hold the used features ids & values
            for i in rows:                    # loop over all instances

                # get CF(v) for this instance (i is a row struct which is what transform needs)
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
        
        return final

    # ! this is the function used by cdfcProject
    def transform(self, data=None):

        instance = collect.namedtuple('instance', ['className', 'values'])
        
        transformed = []  # this will hold the transformed values
        
        # if data is None then we are transforming as part of evolution/training
        # so we should use rows (the provided training data)
        if data is None:
            
            rowBar = tqdm(rows, desc='Transforming as part of evolution')  # create progress bar
            
            for r in rowBar:    # for each instance
                values = []  # this will hold the calculated values for all the constructed features

                for f in self.features:            # transform the original input using each constructed feature
                    values.append(f.transform(r))  # append the transformed values for a single CF to values
                
                # each instance will hold the new values for an instance & className, and
                # transformed will hold all the instances for a hypothesis
                transformed.append(instance(r.className, values))

            return transformed  # return the list of all instances

        # if data is not None then we are predicting using an evolved model so we should use data
        # (this will be testing data from cdfcProject.py)
        else:
            
            bar = tqdm(data, desc='Transforming as part of evolution')  # create progress bar
            
            for d in bar:    # for each instance
                values = []  # this will hold the calculated values for all the constructed features
                
                for f in self.features:            # transform the original input using each constructed feature
                    values.append(f.transform(d))  # append the transformed values for a single CF to values
                
                # each instance will hold the new values for an instance & className, and
                # transformed will hold all the instances for a hypothesis
                transformed.append(instance(d.className, values))

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

    Score = collect.namedtuple('Score', ['Attribute', 'Relevancy'])
    ScoreList = typ.List[typ.Union[typ.List, Score]]
    scores: ScoreList = []

    # FEATURE_NUMBER is the number of features in the data. This means that it can also be a list of
    # the indexes of the features (the feature IDs). Subtract it by 1 to make 0 a valid feature ID
    for i in range(FEATURE_NUMBER-1):
        inClass, notIn = valuesInClass(classId, i)  # find the values of attribute i in/not in class classId

        # get the t-test & complement of the p-value for the feature
        # tValue will be zero when it's on the mean
        # pValue will only be zero when the instances are the same/equivalent
        # BUG pCompValue is 1, causing pValue to be 0 -- this makes relevancy a NaN
        # ? Do we actually want pCompValue or is it just pValue?
        # ! If stats.ttest_ind() is running correctly then the issue must be inClass & notIn,
        # !     and that would mean the error is in valuesInClass
        # the data has been standardized, so the variances are equal? (equal_var = True, which is the default)
        tValue, pCompValue = stats.ttest_ind(inClass, notIn)

        # ! p-Value is being set to 0 because the complement is 1 -- this makes relevancy a NaN
        pValue: float = np.subtract(1, pCompValue)  # get the actual p-values (since pCompValue is the complement, subtract 1)
        
        # ****************** Check that valuesInClass & t-test worked as expected ****************** #
        try:
            # transform into numpy arrays which are easier to test
            inside_of_class = np.array(inClass)
            not_inside_of_class = np.array(notIn)
            
            # *** Check that pCompValue won't be 0 *** #
            # ! currently this is triggered because pCompValue is 1
            if np.subtract(1, pCompValue) == 0:  # if pValue is zero then inClass & notIn are the same
                log.error(f'pCompValue is 1 making p-Value 0, pCompValue={pCompValue}')
                raise Exception(f'ERROR: p-Value is 0, pCompValue={pCompValue}')
            
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
            
            # *** Check that tValue was set & is a number  *** #
            elif tValue is None or math.isnan(tValue):
                log.error(f'tValue computation failed, expected a number got {tValue}')
                raise Exception(f'ERROR: tValue computation failed, expected a number got {tValue}')
            
            # *** Check that tValue is finite *** #
            elif math.isinf(tValue):
                log.error(f'tValue computation failed. tValues is a number, but is not finite. tValue = {tValue}')
                raise Exception(f'ERROR: tValue computation failed, expected a finite number got {tValue}')

            # *** Check that pValue was set & is a number  *** #
            elif tValue is None or math.isnan(pValue):
                log.error(f'pValue computation failed, expected a number got {pValue}')
                raise Exception(f'ERROR: pValue computation failed, expected a number got {pValue}')

            # *** Check that pValue is finite *** #
            elif math.isinf(pValue):
                log.error(f'pValue computation failed. pValues is a number, but is not finite. tValue = {pValue}')
                raise Exception(f'ERROR: tValue computation failed, expected a finite number got {pValue}')

        except Exception as err:
            tqdm.write(str(err))
        # ******************************************************************************************* #
        
        # calculate relevancy for a single feature
        if pValue >= 0.05:                      # if p-value is greater than 0.05 then the feature is not relevant
            relevancy: float = 0.0              # because it's not relevant, set relevancy score to 0
            scores.append(Score(i, relevancy))  # add relevancy score to the list of scores
            
        # otherwise
        else:
            
            try:
                relevancy: float = np.divide(np.absolute(tValue), pValue)  # set relevancy using t-value/p-value
                
                # *************************** Check that division worked *************************** #
                if math.isinf(relevancy):  # check for n/0
                    log.error(f'Relevancy is infinite; some non-zero was divided by 0 -- tValue={tValue} pValue={pValue}')
                    raise Exception(f'ERROR: relevancy is infinite, tValue={tValue} pValue={pValue}')
                
                # ! this gets thrown because pValue & tValue are 0
                elif math.isnan(relevancy):  # check for 0/0
                    log.error(f'Relevancy is infinite; 0/0 -- tValue={tValue} pValue={pValue}')
                    raise Exception(f'ERROR: relevancy is NaN (0/0), tValue={tValue} pValue={pValue}')
                # ********************************************************************************** #
                
                else:  # if division worked
                    scores.append(Score(i, relevancy))  # add relevancy score to the list of scores
            
            except Exception as err:
                tqdm.write(str(err))

    ordered: ScoreList = sorted(scores, key=lambda s: s.Relevancy)  # sort the features by relevancy scores

    terminalSet: typ.Union[int, typ.List] = []  # this will hold relevant terminals
    top: int = len(ordered)//2                  # find the halfway point
    relevantScores: ScoreList = ordered[:top]   # slice top half
    
    for i in relevantScores:             # loop over relevant scores
        # ? this is where the terminal index is added. Is the index correct?
        terminalSet.append(i.Attribute)  # add the attribute number to the terminal set
    
    # ************************* Test if terminalSet is empty ************************* #
    try:
        if not terminalSet:  # if terminalSet is empty
            log.error('Terminals calculation failed: terminalSet is empty')
            raise Exception('ERROR: Terminals calculation failed: terminalSet is empty')
    except Exception as err:
        tqdm.write(str(err))
    # ********************************************************************************* #

    return terminalSet


def valuesInClass(classId: int, attribute: int) -> typ.Tuple[typ.List[np.float64], typ.List[np.float64]]:
    """valuesInClass determines what values of an attribute occur in a class
        and what values do not

    Arguments:
        classId {String or int} -- This is the identifier for the class that
                                    should be examined
        attribute {int} -- this is the attribute to be investigated. It should
                            be the index of the attribute in the row
                            namedtuple in the rows list

    Returns:
        inClass -- This holds the values in the class.
        notInClass -- This holds the values not in the class.
    """
    # TODO add type hinting
    inClass: typ.Union[int, float, typ.List] = []     # attribute values that appear in the class
    notInClass: typ.Union[int, float, typ.List] = []  # attribute values that do not appear in the class

    for value in rows:  # loop over all the rows, where value is the row at the current index
        
        if value.className == classId:                   # if the class is the same as the class given, then
            inClass.append(value.attributes[attribute])  # add the feature's value to in
            
        else:                                               # if the class is not the same as the class given, then
            notInClass.append(value.attributes[attribute])  # add the feature's value to not in
            
    if not inClass:  # if inClass is empty
        log.debug('The valuesInClass method has found that inClass is empty')
    if not notInClass:  # if notInClass is empty
        log.debug('The valuesInClass method has found that notInClass is empty')
    
    return inClass, notInClass  # return inClass & notInClass


def __grow(relevantIndex: typ.List[np.float], depth: int = 0) -> typ.Tuple[Tree, int]:
    # This function uses the grow method to generate an initial population
    # the last thing returned should be a trees root node
    if int == 0:  # only print to log on first call
        log.debug('The __grow method has been chosen to build a tree')

    depth += 1  # increase the depth by one

    if depth == MAX_DEPTH:  # if we've reached the max depth add a random terminal value and return
        return Tree(random.randint(0, (len(relevantIndex) - 1))), depth
    else:
        # combine the list of terminals & operations
        ls: typ.List[typ.Union[str, float, int]] = OPS[:]
        ls.extend(relevantIndex)
    
        # get a random value from the combined list
        value = ls[random.randint(0, (len(ls) - 1))]
    
        # create a tree with value as it's data
        newTree = Tree(value)
    
        if value in relevantIndex:  # if the value added was a terminal value,
            return newTree, depth  # return the new tree
        else:  # if the value was not a terminal value, then grow both children
            newTree.left, leftDepth = __grow(relevantIndex, depth)
            newTree.right, rightDepth = __grow(relevantIndex, depth)
            totalDepth = leftDepth + rightDepth
            return newTree, totalDepth  # after the recursive calls have finished return the tree


def __full(relevantValues: typ.List[np.float], depth: int = 0) -> typ.Tuple[Tree, int]:
    # This function uses the full method to generate an initial population
    # the last thing returned should be a trees root node
    if int == 0:  # only print to log on first call
        log.debug('The __full method has been chosen to build a tree')

    depth += 1  # increase the depth by one

    if depth == MAX_DEPTH:  # if we've reached the max depth add a random terminal value and return
        return Tree(random.randint(0, (len(relevantValues) - 1))), depth
    else:
        # get a random operation
        value = OPS[random.randint(0, (len(OPS) - 1))]
    
        # create a tree with Value (an operation) as it's data
        newTree = Tree(value)
    
        # we didn't add a terminal so, then grow both children
        newTree.left, leftDepth = __full(relevantValues, depth)
        newTree.right, rightDepth = __full(relevantValues, depth)
        totalDepth = leftDepth + rightDepth
        return newTree, totalDepth  # after the recursive calls have finished return the tree


def createInitialPopulation() -> Population:

    def createHypothesis() -> Hypothesis:
        # given a list of trees, create a hypothesis
        # NOTE this will make 1 tree for each feature, and 1 CF for each class

        # get a list of all classIds
        classIds = list(range(LABEL_NUMBER))
        random.shuffle(classIds)

        ftrs: typ.List[ConstructedFeature] = []
        size = 0
        
        for nll in range(LABEL_NUMBER):
            # randomly decide if grow or full should be used.
            # Also randomly assign the class ID then remove that ID
            # so each ID may only be used once
            
            if random.choice([True, False]):
                name = classIds.pop(0)        # get a random id
                tree, size = __grow(terminals(name))  # create tree
                ftrs.append(ConstructedFeature(name, tree, size))
            else:
                name = classIds.pop(0)        # get a random id
                tree, size = __full(terminals(name))  # create tree
                ftrs.append(ConstructedFeature(name, tree, size))

            size += size
        # create a hypothesis & return it
        return Hypothesis(ftrs, size)

    hypothesis: typ.List[Hypothesis] = []

    # creat a number hypotheses equal to pop size
    for __ in trange(POPULATION_SIZE, desc="Creating Initial Hypotheses", unit="hyp"):
        hypothesis.append(createHypothesis())

    log.debug('The createInitialHypothesis() method has finished & is returning')
    return Population(hypothesis, 0)


def evolve(population: Population, elite: Hypothesis) -> typ.Tuple[Population, Hypothesis]:

    def __tournament(p: Population) -> Hypothesis:
        # used by evolve to selection the parents
        # **************** Tournament Selection **************** #
        candidates: typ.List[Hypothesis] = copy.deepcopy(p.candidateHypotheses)  # copy to avoid overwriting
        first = None  # the tournament winner
        score = 0     # the winning score
        for i in range(0, TOURNEY):  # compare TOURNEY number of random hypothesis
            randomIndex = random.randint(0, (len(candidates)-1))  # get a random index value
            candidate: Hypothesis = candidates.pop(randomIndex)   # get the hypothesis at the random index
            # we pop here to avoid getting duplicates. The index uses candidates current size so it will be in range
            fitness = candidate.getFitness()                      # get that hypothesis's fitness score

            if first is None:      # if first has not been set,
                first = candidate  # then  set it

            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it
                score = fitness    # then update the score to higher fitness
        return first
        # ************ End of Tournament Selection ************* #

    # ******************* Evolution ******************* #
    newPopulation = Population([], population.generation+1)  # create a new population with no hypotheses
    
    # while the new population has fewer hypotheses than the max pop size
    while (len(newPopulation.candidateHypotheses)-1) < POPULATION_SIZE:
        
        probability = random.uniform(0, 1)  # get a random number between 0 & 1
        
        # **************** Mutate **************** #
        if probability < MUTATION_RATE:  # if probability is less than mutation rate, mutate
            
            # parent is from a copy of population made by tournament, NOT original pop
            parent: Hypothesis = __tournament(population)  # get parent hypothesis using tournament
            
            # get a random feature from the hypothesis, where M is the number of constructed features
            featureIndex = random.randint(0, (M-1))
            # get the indexes of the terminal values (for the feature)
            terminal: typ.List[int] = parent.features[featureIndex].relevantFeatures
            # get the tree (for the feature)
            feature: Tree = parent.features[featureIndex].tree

            # randomly select a subtree in feature
            while True:  # walk the tree & find a random subtree
                
                decide = random.choice(["left", "right", "choose"])  # make a random decision

                if decide == "choose" or feature.data in terminal:
                    break
                
                elif decide == "left":     # go left
                    feature = feature.left

                elif decide == "right":  # go right
                    feature = feature.right
                    
            decideGrow = random.choice([True, False])  # randomly decide which method to use to construct the new tree
            
            # randomly generate subtree
            if decideGrow:  # use grow
                t, size = __grow(terminal)  # build the subtree

            else:  # use full
                t, size = __full(terminal)  # build the subtree
                
            cl = parent.features[featureIndex].className                     # get the className of the feature
            parent.features[featureIndex] = ConstructedFeature(cl, t, size)  # replace the parent with the mutated child
            newPopulation.candidateHypotheses.append(parent)                # add the parent to the new pop
            # appending is needed because parent is a copy made by tournament NOT a reference from the original pop
        # ************* End of Mutation ************* #

        # **************** Crossover **************** #
        else:
            
            # parent1 & parent2 are from a copy of population made by tournament, NOT original pop
            # because of this they should not be viewed as references
            parent1 = __tournament(population)
            parent2 = __tournament(population)
            
            #  check that each parent is unique
            # if they are the same they should reference the same object & so 'is' is used instead of ==
            while parent1 is parent2:
                parent2 = __tournament(population)

            # get a random feature from each parent, where M is the number of constructed features
            featureIndex = random.randint(0, (M-1))

            # feature 1
            # get the indexes of the terminal values (for the feature)
            terminals1: typ.List[int] = parent1.features[featureIndex].relevantFeatures
            # get the tree (for the feature)
            feature1: Tree = parent1.features[featureIndex].tree
            
            # feature 2
            # get the indexes of the terminal values (for the feature)
            terminals2: typ.List[int] = parent2.features[featureIndex].relevantFeatures
            # get the tree (for the feature)
            feature2: Tree = parent2.features[featureIndex].tree

            while True:  # walk the tree & find a random subtree for feature 1
                
                # make a random decision
                decide = random.choice(["left", "right", "choose"])
                
                if decide == "choose" or feature1.data in terminals1:
                    break
    
                elif decide == "left":   # go left
                    feature1 = feature1.left
                    
                elif decide == "right":  # go right
                    feature1 = feature1.right
                    
            while True:  # walk the tree & find a random subtree for feature 2
                
                # make a random decision
                decide = random.choice(["left", "right", "choose"])
                
                if decide == "choose" or feature2.data in terminals2:
                    break

                elif decide == "left":   # go left
                    parent2 = feature2
                    feature2 = feature2.left

                elif decide == "right":  # go right
                    feature2 = feature2.right

            # swap the two subtrees, this should be done in place as they are both references
            # these are not used as the don't need to be. We merely want to swap their pointers
            feature1, feature2 = feature2, feature1

            # get the size of the new constructed features by walking the trees
            parent1.setSize()
            parent2.setSize()

            # parent 1 & 2 are both hypotheses and should have been changed in place,
            # but they refer to a copy made in tournament so add them to the new pop
            newPopulation.candidateHypotheses.append(parent1)
            newPopulation.candidateHypotheses.append(parent2)
            # **************** End of Crossover **************** #
    
            # handle elitism
            newHypothFitness = newPopulation.candidateHypotheses[-1].getFitness()
            if newHypothFitness > elite.getFitness():
                elite = newPopulation.candidateHypotheses[-1]
                
    return newPopulation, elite


def cdfc(train: np.ndarray) -> Hypothesis:
    # Class Dependent Feature Construction

    # makes sure we're using global variables
    global FEATURE_NUMBER
    global POPULATION_SIZE
    global INSTANCES_NUMBER
    global LABEL_NUMBER
    global M
    global rows
    global row
    global ENTROPY_OF_S  # * used in entropy calculation * #

    classes = []       # this will hold classIds and how often they occur
    classSet = set()   # this will hold how many classes there are
    classToOccur = {}  # maps a classId to the number of times it occurs
    ids = []           # this will be a list of all labels/ids with no repeats

    # set global variables using the now transformed data
    # each line in train will be an instances
    for line in tqdm(train, desc="Setting Global Variables"):
        
        # parse the file
        # ? I am correctly setting the classId/name so that it works with using FEATURE_NUMBER?
        rows.append(row(line[0], line[1:]))  # reader[0] = classId, reader[1:] = attribute values
        classes.append(line[0])
        classSet.add(line[0])
        INSTANCES_NUMBER += 1

        # track how many different IDs there are
        if line[0] in ids:
            continue
        else:
            ids.append(line[0])

        # ********* The Code Below is Used to Calculated Entropy  ********* #
        # this will count the number of times a class occurs in the provided data
        # dictionary[classId] = counter of times that class is found
        
        if classToOccur.get(line[0]):   # if we have encountered the class before
            classToOccur[line[0]] += 1  # increment
        else:  # if this is the first time we've encountered the class
            classToOccur[line[0]] = 1   # set to 1
        # ****************************************************************** #

    FEATURE_NUMBER = len(rows[0].attributes)  # get the number of features in the data set
    POPULATION_SIZE = FEATURE_NUMBER * BETA  # set the pop size
    LABEL_NUMBER = len(ids)                  # get the number of classes in the data set
    M = R * LABEL_NUMBER                     # get the number of constructed features

    # ********* The Code Below is Used to Calculated Entropy  ********* #
    # loop over all classes
    for i in classToOccur.keys():
        pi = classToOccur[i] / INSTANCES_NUMBER  # compute p_i
        ENTROPY_OF_S -= pi * math.log(pi, 2)     # calculation entropy summation
    # ***************************************************************** #

    # *********************** Run the Algorithm *********************** #
    currentPopulation = createInitialPopulation()     # create initial population
    elite = currentPopulation.candidateHypotheses[0]  # init elitism

    # loop, evolving each generation. This is where most of the work is done
    log.debug('Starting generations stage...')
    for __ in trange(GENERATIONS, desc="CDFC Generations", unit="gen"):
        newPopulation, elite = evolve(currentPopulation, elite)  # generate a new population by evolving the old one
        # update currentPopulation to hold the new population
        # this is done in two steps to avoid potential namespace issues
        currentPopulation = newPopulation
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
    fitBar = tqdm(currentPopulation.candidateHypotheses, desc='Finding most fit hypothesis', unit='hyp')
    bestHypothesis = max(fitBar, key=lambda x: x.fitness)
    log.debug('Found best hypothesis, returning...')
    return bestHypothesis  # return the best hypothesis generated
