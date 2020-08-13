import copy
import math
import random
import numpy as np
import typing as typ
import collections as collect
from scipy import stats

# ! Next Steps
# TODO check feature1, feature2 issue in crossover

# TODO write code for the if function in OPS
# TODO add citation in entropy calc
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
# ************************ End of Constants/Globals ************************ #

# ********************** Namespaces/Structs & Objects *********************** #
row = collect.namedtuple('row', ['className', 'attributes'])  # a single line in the csv, representing a record/instance
rows: typ.List[row] = []  # this will store all of the records read in (the training dat) as a list of rows


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
        self.left = left    # Will a another tree, or None if this is a terminal
        self.right = right  # Will a another tree, or None if this is a terminal
        self.data = data    # must either be a function or a terminal (if a terminal it should be it's index)

    # running a tree should return a single value
    # featureValues -- the values of the relevant features keyed by their index in the original data
    def runTree(self, featureValues: typ.Dict[int, np.float64]) -> typ.Optional[np.float64]:
        return self.__runNode(featureValues)

    def __runNode(self, featureValues):
        # if this tree's node is a valid operation, then execute it
        if self.data in OPS:
            # find out which operation it is & return it's value
            if self.data == 'add':
                return self.left.runTree(featureValues) + self.right.runTree(featureValues)
    
            elif self.data == 'subtract':
                return self.left.runTree(featureValues) - self.right.runTree(featureValues)
    
            elif self.data == 'times':
                return self.left.runTree(featureValues) * self.left.runTree(featureValues)
    
            elif self.data == 'min':
                return min(self.left.runTree(featureValues), self.right.runTree(featureValues))
            
        # if the node is not an operation, then it is a terminal index so return the value
        else:
            return featureValues[self.data]

    def getSize(self, counter):

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
        return values      # values should now hold the indexes of the tree's terminals

    def transform(self, instance: row) -> np.float64:
    
        # this will hold the values of relevant features
        relevantValues: typ.Dict[int, typ.Optional[np.float64]] = {}

        # loop over the indexes of the relevant features, and take the values of the relevant features out
        # & store them in a dictionary keyed by their index in the original data
        for i in self.relevantFeatures:
            relevantValues[i] = instance.attributes[i]

        # transform the data for a single feature
        # this should return a single value (a value transformed by the tree)
        return self.tree.runTree(relevantValues)

    def setSize(self):
        return self.tree.getSize(0)  # call getSize on the root of the tree


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

        def __Czekanowski(Vi, Vj):
            minSum = 0
            addSum = 0

            for d in range(1, len(self.features)):  # loop over the number of features
                minSum += min(Vi[d], Vj[d])         # the top of the fraction
                addSum += Vi[d] + Vj[d]             # the bottom of the fraction
            return 1 - ((2*minSum) / addSum)        # calculate it & return

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
            partition: typ.Dict[np.float64, typ.List[row]] = {}
            
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
            # BUG maxInfoGain is None type. This is because MaxInfo gain is never set
            # !   find a way to set it for each feature
            if self.maxInfoGain < f.infoGain:
                self.maxInfoGain = f.infoGain

        # calculate the average info gain using formula 3
        # TODO create more correct citation later #
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

    def transform(self, data=None):

        instance = collect.namedtuple(
            'instance', ['className', 'values'])

        # if data is None then we are transforming as part of evolution/training
        # so we should use rows (the provided training data)
        if data is None:
            
            transformed = []  # this will hold the transformed values
            
            for r in rows:   # for each instance
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
            
            transformed = []  # this will hold the transformed values
            
            for d in data:   # for each instance
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
    # BUG for some reason the canidateHypotheses are not getting intilized
    def __init__(self, candidates: typ.List[typ.Type[Hypothesis]], generationNumber: int) -> None:
        self.candidateHypotheses = candidates  # a list of all the candidate hypotheses
        self.generation = generationNumber     # this is the number of this generation
# ***************** End of Namespaces/Structs & Objects ******************* #


def terminals(classId: int) -> typ.List[int]:
    """terminals creates the list of relevant terminals for a given class.

    Arguments:
        classId {String} -- classId is the identifier for the class for
                             which we want a terminal set

    Returns:
        list -- terminals returns the highest scoring features as a list.
                The list will have a length of FEATURE_NUMBER/2, and will
                hold the indexes of the features.
    """

    Score = collect.namedtuple('Score', ['Attribute', 'Relevancy'])
    scores = []

    for i in range(1, FEATURE_NUMBER):
        inClass, notIn = valuesInClass(classId, i)        # find the values of attribute i in/not in class classId
        # ? This currently uses a two-tailed test. Because of this p-value can be below 0, and
        # ?     this can cause divide by 0 errors during the else case of relevancy calculation.
        # ?     Should it be a one-tailed test instead?
        tValue, pValue = stats.ttest_ind(inClass, notIn)  # get the t-test & p-value for the feature

        # calculate relevancy for a single feature
        if pValue >= 0.05:  # if p-value is less than 0.05
            relevancy = 0   # set relevancy score to 0
            scores.append(Score(i, relevancy))  # add relevancy score to the list of scores
        # otherwise
        else:
            # NOTE: this causes a divided by zero error if p-Value comes from a two-tailed test
            relevancy = abs(tValue)/pValue      # set relevancy using t-value/p-value
            scores.append(Score(i, relevancy))  # add relevancy score to the list of scores

    sortedScores = sorted(scores, key=lambda s: s.Attribute)  # sort the features by relevancy scores

    terminalSet = []                     # this will hold relevant terminals
    top = len(sortedScores)              # find the halfway point
    relevantScores = sortedScores[:top]  # slice top half
    
    for i in relevantScores:             # loop over relevant scores
        terminalSet.append(i.Attribute)  # add the attribute number to the terminal set

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

    inClass = []     # attribute values that appear in the class
    notInClass = []  # attribute values that do not appear in the class

    for value in rows:  # loop over all the rows, where value is the row at the current index
        
        if value.className == classId:                   # if the class is the same as the class given, then
            inClass.append(value.attributes[attribute])  # add the feature's value to in
            
        else:                                               # if the class is not the same as the class given, then
            notInClass.append(value.attributes[attribute])  # add the feature's value to not in

    return inClass, notInClass  # return inClass & notInClass


def createInitialPopulation() -> Population:

    def __grow(classId: int) -> typ.Tuple[Tree, int]:
        # This function uses the grow method to generate an initial population
        
        def assign(level: int, counter: int) -> typ.Tuple[Tree, int]:

            counter += 1  # used to compute individual size
            
            if level != MAX_DEPTH:  # recursively assign tree values
                spam = ls[random.randint(0, (len(ls)-1))]  # get the random value
                
                if spam in terminal:            # if the item is a terminal
                    return Tree(spam), counter  # just return, stopping recursion

                node = Tree(spam)
                node.left, cLeft = assign(level + 1, counter)
                node.right, cRight = assign(level + 1, counter)
                # add the number of nodes from the left subtree, to the number of nodes from the right subtree
                counter = cLeft + cRight
                return node, counter

            else:  # stop recursion; max depth has been reached
                spam = terminal[random.randint(0, (len(terminal)-1))]  # add a terminal to the leaf
                return Tree(spam), counter                         # return

        # pick a random function & put it in the root
        ls = OPS[:]
        random.shuffle(ls)
        rootData = ls[random.randint(0, (len(ls)-1))]
        tree = Tree(rootData)  # make a new tree

        terminal = terminals(classId)  # get the list of terminal characters
        ls += terminals(classId)  # add the terminal values to the list of functions & reorder
        random.shuffle(ls)

        # create the tree
        tree.left, counterLeft = assign(1, counter=1)
        tree.right, counterRight = assign(1, counter=1)
        
        size = counterRight + counterLeft  # add the number of nodes together to get the individual size

        return tree, size

    def __full(classId: int) -> typ.Tuple[Tree, int]:
        # This function uses the full method to generate an initial population
        
        def assign(level: int, counter: int) -> typ.Tuple[Tree, int]:
            
            counter += 1
            
            if level != MAX_DEPTH:  # recursively assign tree values
                node = Tree(ls[random.randint(0, len(ls)-1)])  # get a random function & add it to the tree
                # call for branches
                node.left, cLeft = assign(level + 1, counter)
                node.right, cRight = assign(level + 1, counter)
                # add the number of nodes from the left subtree, to the number of nodes from the right subtree
                counter = cLeft + cRight
                return node, counter

            else:  # stop recursion; max depth has been reached, so add a terminal to the leaf & return
                return Tree(terminal[random.randint(0, (len(terminal)-1))]), counter

        # pick a random function & put it in the root
        ls = OPS[:]
        rootData = ls[random.randint(0, (len(ls)-1))]
        tree = Tree(rootData)  # make a new tree

        terminal = terminals(classId)  # get the list of terminal characters
        ls += terminal  # add the terminal values to the list of functions & reorder
        random.shuffle(ls)

        # create the tree
        tree.left, counterLeft = assign(1, counter=1)
        tree.right, counterRight = assign(1, counter=1)

        size = counterRight + counterLeft  # add the number of nodes together to get the individual size

        return tree, size

    def createHypothesis() -> typ.Type[Hypothesis]:
        # given a list of trees, create a hypothesis
        # NOTE this will make 1 tree for each feature, and 1 CF for each class

        # get a list of all classIds
        classIds = list(range(1, LABEL_NUMBER+1))
        random.shuffle(classIds)

        ftrs: typ.List[ConstructedFeature] = []
        HypSize = 0
        
        for nll in range(LABEL_NUMBER):
            # randomly decide if grow or full should be used.
            # Also randomly assign the class ID then remove that ID
            # so each ID may only be used once
            
            if random.choice([True, False]):
                name = classIds.pop(0)        # get a random id
                tree, size = __grow(name)     # create tree
                ftrs.append(ConstructedFeature(name, tree, size))
            else:
                name = classIds.pop(0)        # get a random id
                tree, size = __full(name)     # create tree
                ftrs.append(ConstructedFeature(name, tree, size))

            HypSize += size

        h = Hypothesis
        h.features = ftrs
        h.size = HypSize
        return h

    hypothesis: typ.List[typ.Type[Hypothesis]] = []

    for nl in range(POPULATION_SIZE):
        hypothesis.append(createHypothesis())

    return Population(hypothesis, 0)


def evolve(population: Population, elite: Hypothesis) -> typ.Tuple[Population, Hypothesis]:
    # NOTE pop should be a list of hypotheses

    def __tournament(pop: Population) -> Hypothesis:
        # used by evolve to selection the parents
        
        # **************** Tournament Selection **************** #
        candidates = copy.deepcopy(pop.candidateHypotheses)
        first = pop.candidateHypotheses[0]
        score = 0
        for i in range(0, TOURNEY):  # compare TOURNEY number of random hypothesis
            
            candidate = candidates.pop(random.randint(0, (len(candidates)-1)))  # get a random hypothesis
            fitness = candidate.getFitness()                              # get that hypothesis's fitness score

            if first is None:      # if first has not been set,
                first = candidate  # then  set it

            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it
                score = fitness
        # ************ End of Tournament Selection ************* #

        return first

    def __generateTree(node, terminalValues, values, depth, max_depth, counter=0):
        
        # **************** Tree Generation **************** #
        counter += 1  # increment size counter

        # if this node contains a terminal return
        if node.data in terminalValues:
            return counter

        choice = random.choice(["left", "right", "both"])  # make a random choice about which way to grow

        if choice == "left":  # grow left

            if depth == max_depth:  # check to see if we are at the max depth
                index = terminalValues[random.randint(0, len(terminalValues))]  # get a random terminal
                node.left = Tree(index)                                         # put that terminal in the left branch
                return counter

            index = values[random.randint(0, len(values))]  # pick a random operation or terminal
            node.left = Tree(index)                         # put the operation or terminal in the left node

            __generateTree(node.left, terminalValues, values, depth+1, max_depth, counter)  # generate tree recursively

        elif choice == "right":  # grow right

            if depth == max_depth:  # check to see if we are at the max depth
                index = terminalValues[random.randint(0, len(terminalValues))]  # get a random terminal
                node.left = Tree(index)                                         # put the terminal in the left branch
                return counter

            index = values[random.randint(0, len(values))]  # pick a random operation or terminal
            node.right = Tree(index)                        # put the operation or terminal in the left node

            __generateTree(node.right, terminalValues, values, depth + 1, max_depth)  # call generateTree recursively

        elif choice == "both":

            if depth == max_depth:  # check to see if we are at the max depth
                index = terminalValues[random.randint(0, len(terminalValues))]  # get a random terminal
                node.left = Tree(index)                                         # put the terminal in the left branch
                return

            # left branch
            index = values[random.randint(0, len(values))]  # pick a random operation or terminal
            node.left = Tree(index)                         # put the operation or terminal in the left node

            # right branch
            index = values[random.randint(0, len(values))]  # pick a random operation or terminal
            node.right = Tree(index)                        # put the operation or terminal in the left node

            cLeft = __generateTree(node.left, terminalValues, values, depth + 1, max_depth)  # generate tree recursively
            cRight = __generateTree(node.right, terminalValues, values, depth + 1, max_depth)  # call grow recursively

            counter = cLeft + cRight  # calculate the size (the number of nodes) by adding the size of the subtrees
            return counter
        # ************ End of Tree Generation ************ #

    # ******************* Evolution ******************* #
    # ? Do I need to create a new population or is this all done in place?
    newPopulation = Population([], population.generation+1)  # create a new population with no hypotheses
    
    # while the size of the new population is less than the max pop size
    while len(newPopulation.candidateHypotheses) < POPULATION_SIZE:
        
        probability = random.uniform(0, 1)  # get a random number between 0 & 1
        
        # **************** Mutate **************** #
        if probability < MUTATION_RATE:  # if probability is less than mutation rate, mutate
            
            parent = __tournament(population)  # get parent hypothesis using tournament
            
            # get a random feature from the hypothesis
            featureIndex = random.randint(0, M)
            feature = parent.features[featureIndex]
            terminal = feature.relevantFeatures
            # ? because lists are mutable all the changes happen in place?
            # ? So I don't need to create a new hypoth/pop as there is only ever the one?
            # TODO check
            feature = feature.tree  # get the tree for that feature

            # randomly select a subtree in feature
            while True:  # walk the tree & find a random subtree
                
                decide = random.choice(["left", "right", "choose"])  # make a random decision
                
                if decide == "left":     # go left
                    feature = feature.left

                elif decide == "right":  # go right
                    feature = feature.right

                elif decide == "choose" or feature.data in terminal:
                    break

            decideGrow = random.choice([True, False])  # randomly decide which method to use to construct the new tree
            
            # randomly generate subtree
            if decideGrow:  # use grow
                
                # pick a random function & put it in the root
                ls = OPS[:]
                random.shuffle(ls)
                rootData = ls[random.randint(0, len(ls))]
                
                t = Tree(rootData)  # make a new tree
                
                # build the rest of the subtree
                ls.append(terminal)                           # append the terminals
                random.shuffle(ls)                            # shuffle the operations & terminals
                size = __generateTree(t, terminal, ls, 0, 8)  # set the size of the tree

            else:  # use full
                
                # pick a random function & put it in the root
                ls = OPS[:]
                random.shuffle(ls)
                rootData = ls[random.randint(0, len(ls))]
                
                t = Tree(rootData)                             # make a new tree
                size = __generateTree(t, terminal, OPS, 0, 8)  # build the rest of the subtree

            cl = parent.features[featureIndex].className                     # get the className of the feature
            parent.features[featureIndex] = ConstructedFeature(cl, t, size)  # replace the parent with the mutated child
            newPopulation.candidateHypotheses.append(parent)                # add the parent to the new pop
        # ************* End of Mutation ************* #

        # **************** Crossover **************** #
        else:

            parent1 = __tournament(population)
            parent2 = __tournament(population)
            
            while parent1 is parent2:  # check that each parent is unique
                parent2 = __tournament(population)

            featureIndex = random.randint(0, M)  # get a random feature from each parent
            
            # feature 1
            feature1 = parent1.features[featureIndex]
            terminals1 = feature1.relevantFeatures
            feature1 = feature1.tree
            
            # feature 2
            feature2 = parent2.features[featureIndex]
            terminals2 = feature2.relevantFeatures
            feature2 = feature2.tree

            while True:  # walk the tree & find a random subtree for feature 1
                
                # make a random decision
                decide = random.choice(["left", "right", "choose"])

                if decide == "left":     # go left
                    feature1 = feature1.left
                    
                elif decide == "right":  # go right
                    feature1 = feature1.right
                    
                elif decide == "choose" or feature1.data in terminals1:
                    break

            while True:  # walk the tree & find a random subtree for feature 2
                
                # make a random decision
                decide = random.choice(["left", "right", "choose"])

                if decide == "left":     # go left
                    feature2 = feature2.left

                elif decide == "right":  # go right
                    feature2 = feature2.right

                elif decide == "choose" or feature2.data in terminals2:
                    break

            feature1, feature2 = feature2, feature1  # ? is this done in place? # swap the two subtrees
            # TODO they aren't used after the reassignment. Check this is correct

            # get the size of the new constructed features by walking the trees
            parent1.setSize()
            parent2.setSize()

            # parent 1 & 2 are both hypotheses and should have been changed in place, so add them to the new pop
            newPopulation.candidateHypotheses.append(parent1)
            newPopulation.candidateHypotheses.append(parent2)
        # **************** End of Crossover **************** #

        # handle elitism
        newHypothFitness = newPopulation.candidateHypotheses[-1].getFitness()
        if newHypothFitness > elite.getFitness():
            elite = newPopulation.candidateHypotheses[-1]

    return newPopulation, elite


def cdfc(train: np.ndarray) -> typ.Type[Hypothesis]:
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
    for line in train:  # each line in train will be an instances
        
        # parse the file
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
    
    for i in range(GENERATIONS):  # loop, evolving each generation. This is where most of the work is done
        newPopulation, elite = evolve(currentPopulation, elite)  # generate a new population by evolving the old one
        # update currentPopulation to hold the new population
        # this is done in two steps to avoid potential namespace issues
        currentPopulation = newPopulation
    # ***************************************************************** #

    # ****************** Return the Best Hypothesis ******************* #
    # check to see if the last generation has generated fitness scores
    if currentPopulation.candidateHypotheses[2].fitness is None:
        # if not then generate them
        for i in currentPopulation.candidateHypotheses:
            i.getFitness()

    # now that we know each hypothesis has a fitness score, get the one with the highest fitness
    bestHypothesis = max(currentPopulation.candidateHypotheses, key=lambda x: x.fitness)
    return bestHypothesis  # return the best hypothesis generated
