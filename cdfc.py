
import csv
import math
import random
import numpy as np
import collections as collect
from scipy import stats
from sklearn.preprocessing import StandardScaler


# * Next Steps
# TODO write code for the if function in OPS
# TODO figure out K loop part of cross validation
# TODO debug


# ******************** Constants/Globals ******************** #

# CROSSOVER_RATEis the chance that a candidate will reproduce
CROSSOVER_RATE = 0.8

# GENERATIONS is the number of generations the GP should run for
GENERATIONS = 50

# MUTATION_RATE is the chance that a candidate will be mutated
MUTATION_RATE = 0.2

# ELITISM_RATE is the elitism rate
ELITISM_RATE = 1

# TOURNEY is the tournament size
TOURNEY = 7

# ALPHA is the fitness weight alpha
ALPHA = 0.8

# a constant used to calculate the pop size
BETA = 2

# the number of features in the data set
FEATURE_NUMBER = 0

# the population size
POPULATION_SIZE = 0

# the number of instances in the training data
INSTANCES_NUMBER = 0

MAX_DEPTH = 8

# used for entropy calculation
ENTROPY_OF_S = 0

# *** set values below for every new dataset *** #

# C is the number of classes in the data
C = 3

# R is the ratio of number of constructed features to the number of classes
# (features/classes)
R = 2

# M is the number of constructed features
M = R*C

# FN (feature num) is number of features in the data set
FN = 0

# PS (pop size) is the population size (equal to number of features * beta)
PS = FN * BETA

# ************************ End of Constants/Globals ************************ #

# ********************** Namespaces/Structs & Objects *********************** #

# a single line in the csv, representing a record/instance
row = collect.namedtuple('row', ['className', 'attributes'])

# this will store all of the records read in (list is a list of rows)
# this also makes it the set of all training data
rows = []


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
    left = None
    right = None
    data = None  # must either be a function or a terminal (if a terminal it should be it's index)

    def __init__(self, data, left=None, right=None):
        self.left = left
        self.right = right
        self.data = data

    # running a tree should return a single value
    # rFt -- relevant features
    # rVls -- the values of the relevant features
    def runTree(self, rFt, rVls):
        if self.data in rFt:  # if the root is a terminal
            return self.data  # return value
        else:  # if the root is not a terminal
            # run the tree recursively
            return self.data(self.__runLeft(rFt, rVls), self.__runRight(rFt, rVls))

    def __runLeft(self, relevantFeatures, featureValues):
        # if left node is a terminal
        if self.left.data in relevantFeatures:
            return featureValues[self.left.data]
        else:  # if left node is a function
            return self.left.data()

    def __runRight(self, relevantFeatures, featureValues):
        # if right node is a terminal
        if self.right.data in relevantFeatures:
            return featureValues[self.right.data]
        else:  # if right node is a function
            return self.right.data()

    def getSize(self, counter):

        counter += 1  # increment the counter

        leftCount = 0
        rightCount = 0
        # if the left node is not null
        if self.left:
            # recursive call
            leftCount = self.left.getSize(counter)

        # if the right node is not null
        if self.right:
            # recursive call
            rightCount = self.right.getSize(counter)

        # add together the size of the subtrees & return them up the call stack
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

    tree = None  # the root node of the constructed feature
    className = None  # the name of the class this tree is meant to distinguish
    infoGain = None  # the info gain for this feature
    relevantFeatures = None  # holds the indexes of the relevant features
    usedFeatures = None  # the relevant features the cf actually uses
    size = 0  # the individual size
    # the values data after they have been transformed by the tree
    transformedValues = None

    def __init__(self, className, tree, size=0):
        self.className = className
        self.tree = tree
        self.size = size
        # call terminals to create the terminal set
        self.relevantFeatures = terminals(className)

    def getUsedFeatures(self):
        values = []  # will hold the indexes found at each terminal node

        def __walk(node):  # given a node, walk the tree

            if node.left and node.right is None:  # if there are no children
                # we have reached a terminal. Get it's index/ID
                values.append(node.data)
                # there are no more children down this branch
                return

            # if there is no left child, but there is a right child
            elif node.left is None:
                __walk(node.right)  # walk down the right

            # if there is no right child, but there is a left child
            elif node.right is None:
                __walk(node.left)  # walk down the left

            else: # if there are both left & right children
                # walk down both
                __walk(node.right)
                __walk(node.left)

        __walk(self.tree)  # walk the tree starting with the CF's root
        # values should now hold the indexes of the tree's terminals
        return values

    def transform(self, instance):  # instance should be a row object

        relevantValues = {}  # this will hold the values of relevant features
        # loop over the indexes of the relevant features
        for i in self.relevantFeatures:
            # and take the values of the relevant features out
            # & store them in a dict keyed by their index in the original data
            relevantValues[i] = instance.attributes[i]

        # transform the data for a single feature
        # this should return a single value (a value transformed by the tree)
        return self.tree.runTree(self.relevantFeatures, relevantValues)

    def setSize(self):
        # call getSize on the root of the tree
        return self.tree.getSize(0)


class Hypothesis:
    # a single hypothesis(a GP individual)
    features = []  # a list of all the constructed features
    size = 0  # the number of nodes in all the cfs
    fitness = None  # the fitness score
    distance = 0  # the distance function score
    averageInfoGain = None  # the average info gain of the hypothesis
    maxInfoGain = None  # the max info gain in the hypothesis

    def getFitness(self):

        def __Czekanowski(Vi, Vj):
            minSum = 0
            addSum = 0
            # loop over the number of features
            for d in range(1, len(self.features)):
                minSum += min(Vi[d], Vj[d])  # the top of the fraction
                addSum += Vi[d] + Vj[d]  # the bottom of the fraction
            return 1 - ((2*minSum) / addSum)

        def Distance(values):

            # ********** Compute Vi & Vj ********** #
            Db = 0  # the sum of mins
            Dw = 0  # the sum of maxs
            # loop over all the transformed values
            for i, vi in enumerate(values):

                minSum = 2  # * must be high enough that it is always reset
                maxSum = 0

                for j, vj in enumerate(values):

                    if i == j:  # if i equals j ignore this case
                        # continue / skip / return go to next element
                        pass

                    dist = __Czekanowski(vi, vj)  # compute the distance

                    # if vi & vj are in the same class (Dw)
                    if vi.inClass == vj.inClass:
                        # replace the max if the current value is higher
                        if dist > maxSum:
                            maxSum = dist
                    else:  # if vi & vj are not in the same class (Db)
                        # replace the min if the current value is smaller
                        if dist < minSum:
                            minSum = dist

                # update the running totals
                Db += minSum
                Dw += maxSum

            # perform the final distance calculations
            t1 = Db / len(values)
            t2 = Dw / len(values)
            return 1 / (1 + math.pow(math.e, -5*(t1 - t2)))

        def __entropy(partition):

            # p[classId] = number of instances in the class in the partition sv
            p = {}
            # for instance i in a partition sv
            for i in partition:
                # if we have already found the class once,
                # increment the counter
                if p[i.className]:
                    p[i.className] += 1
                # if we have not yet encountered the class
                # set the counter to 1
                else:
                    p[i.className] = 1

            spam = 0
            # for class in the list of classes in the partition sv
            for c in p.keys():
                # perform entropy calculation
                pi = p[c] / len(partition)
                spam -= pi * math.log(pi, 2)

            return spam

        def __conditionalEntropy(feature):

            # this is a feature struct that will be used to store feature values
            # with their indexes/IDs in CFs
            ft = collect.namedtuple('ft', ['id', 'value'])
            partition = {}  # key = CF(Values), Entry = instance in training data
            s = 0  # used to sum CF's conditional entropy
            used = feature.getUsedFeatures()  # get the indexes of the used features
            v = []  # this will hold the used features ids & values
            for i in rows:  # loop over all instances

                # get CF(v) for this instance (i is a row struct which is what transform needs)
                cfv = feature.transform(i)  # needs the values for an instance

                # get the values in this instance i of the used feature
                for u in used:  # loop over all the indexes of used features
                    # create a ft where the id is u (a used index) &
                    # where the value is from the instance
                    v.append(ft(u, i.attributes[u]))

                if partition[cfv]:  # if the partition exists
                    partition[cfv].append(i)  # add the instance to it
                else:  # if the partition doesn't exist
                    partition[cfv] = [i]  # create it

            for e in partition.keys():
                s += (len(partition[e])/INSTANCES_NUMBER) * __entropy(partition[e])

            # s holds the conditional entropy value
            return s

        # loop over all features & get their info gain
        gainSum = 0  # the info gain of the hypothesis
        for f in self.features:

            # ********* Entropy calculation ********* #
            # find the conditional entropy
            condEntropy = __conditionalEntropy(f)

            # ******** Info Gain calculation ******* #
            # H(class) - H(class|f)
            f.infoGain = ENTROPY_OF_S - condEntropy
            gainSum += f.infoGain  # update the info sum

            # updates the max info gain of the hypothesis if needed
            if self.maxInfoGain < f.infoGain:
                self.maxInfoGain = f.infoGain

        # calculate the average info gain using formula 3
        # TODO create more correct citation later #
        term1 = gainSum+self.maxInfoGain
        term2 = (M+1)*(math.log(C, 2))
        self.averageInfoGain += term1 / term2

        # set size
        # * this must be based off the number of nodes a tree has because
        # * the depth will be the same for all of them

        # *********  Distance Calculation ********* #
        # calculate the distance using the transformed values
        self.distance = Distance(self.transform())

        # ********* Final Calculation ********* #
        term1 = ALPHA*self.averageInfoGain
        term2 = (1-ALPHA)*self.distance
        term3 = (math.pow(10, -7)*self.size)
        return term1 + term2 - term3

    def transform(self):

        instance = collect.namedtuple(
            'instance', ['inClass', 'values'])
        # ? should this be an array or a dictionary?
        transformed = []  # this will hold the transformed values
        for r in rows:  # for each instance
            # this will hold the calculated values for all
            # the constructed features
            values = []
            # transform the original input using each constructed feature
            for f in self.features:
                # append the transformed values for a single
                # constructed feature to values
                values.append(f.transform(r))
            # each instance will hold the new values for
            # an instance & className. Transformed will hold
            # all the instances for a hypothesis
            transformed.append(instance(r.className, values))
            # ? how to make class name a bool? Does it need to be?
        return transformed  # return the list of all instances


class Population:
    # this will be the population of hypotheses
    candidateHypotheses = []  # a list of all the candidate hypotheses
    generation = None  # this is the number of this generation

    def __init__(self, candidates, generationNumber):
        self.candidateHypotheses = candidates
        self.generation = generationNumber


# ***************** End of Namespaces/Structs & Objects ******************* #

# ********************** Valid Operations Within Tree ********************** #
# all functions for the tree must be of the form x(y,z)

# OPS is the list of valid operations on the tree
OPS = ['add', 'subtract', 'times', 'max', 'isTrue']


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def times(a, b):
    return a * b


def isTrue(a, b):
    pass  # ? how to deal with if function?

# min is built in

# max is built in

# *************************** End of Operations *************************** #


def terminals(classId):
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
        # find the values of attribute i in/not in class classId
        inClass, notIn = valuesInClass(classId, i)

        # get the t-test & p-value for the feature
        tValue, pValue = stats.ttest_ind(inClass, notIn)

        # calculate relevancy for a single feature
        relevancy = 0  # this will hold the relevancy score for this feature
        if pValue >= 0.05:  # if p-value is less than 0.05
            relevancy = 0  # set relevancy score to 0
        else:  # otherwise
            # set relevancy using t-value/p-value
            relevancy = abs(tValue)/pValue

        scores.append(Score(i, relevancy))

    # sort the features by relevancy scores
    sortedScores = sorted(scores, key=lambda Score: Score.Attribute)

    terminalSet = []  # this will hold relevant terminals
    top = len(sortedScores)  # find the halfway point
    relevantScores = sortedScores[:top]  # slice top half
    for i in relevantScores:  # loop over relevant scores
        # add the attribute number to the terminal set
        terminalSet.append(i.Attribute)

    return terminalSet


def valuesInClass(classId, attribute):
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

    inClass = []  # attribute values that appear in the class
    notInClass = []  # attribute values that do not appear in the class

    # loop over all the rows, where value is the row at the current index
    for value in rows:
        # if the class is the same as the class given
        if value.className == classId:
            # add the feature's value to in
            inClass.append(value.attributes[attribute])
        else:  # if the class is not the same as the class given
            # add the feature's value to not in
            notInClass.append(value.attributes[attribute])
    # return inClass & notInClass
    return inClass, notInClass


def createInitialPopulation():

    def __grow(classId):
        # This function uses the grow method to generate an initial population
        def assign(level, counter):

            # used to compute individual size
            counter += 1
            # recursively assign tree values
            if level != MAX_DEPTH:
                # get the random value
                spam = ls[random.randint(0, len(ls))]
                if spam in terminal:  # if the item is a terminal
                    return Tree(spam), counter  # just return, stopping recursion

                tree = Tree(spam)
                tree.left, cLeft = assign(level + 1, counter)
                tree.right, cRight = assign(level + 1, counter)
                # add the number of nodes from the left subtree,
                # to the number of nodes from the right subtree
                counter = cLeft + cRight
                return tree, counter

            else:
                # stop recursion; max depth has been reached
                # add a terminal to the leaf
                spam = terminal[random.randint(0, len(terminal))]
                # return
                return Tree(spam), counter

        # pick a random function & put it in the root
        ls = random.shuffle(OPS)
        rootData = ls[random.randint(0, len(ls))]
        tree = Tree(rootData)  # make a new tree

        # get the list of terminal characters
        terminal = terminals(classId)
        # add the terminal values to the list of functions & reorder
        ls = random.shuffle(ls.append(terminal))

        # create the tree
        tree.left, counterLeft = assign(1, counter=1)
        tree.right, counterRight = assign(1, counter=1)

        # add the number of nodes together to get the individual size
        size = counterRight + counterLeft

        return tree, size

    def __full(classId):
        # This function uses the full method to generate an initial population
        def assign(level, counter):
            counter += 1
            # recursively assign tree values
            if level != MAX_DEPTH:
                # get a random function & add it to the tree
                tree = Tree(ls[random.randint(0, len(ls))])
                # call for branches
                tree.left, cLeft = assign(level + 1, counter)
                tree.right, cRight = assign(level + 1, counter)
                # add the number of nodes from the left subtree,
                # to the number of nodes from the right subtree
                counter = cLeft + cRight
                return tree, counter

            else:  # stop recursion; max depth has been reached
                # add a terminal to the leaf & return
                return Tree(terminal[random.randint(0, len(terminal))]), counter

        # pick a random function & put it in the root
        ls = random.shuffle(OPS)
        rootData = ls[random.randint(0, len(ls))]
        tree = Tree(rootData)  # make a new tree

        # get the list of terminal characters
        terminal = terminals(classId)
        # add the terminal values to the list of functions & reorder
        ls = random.shuffle(ls.append(terminal))

        # create the tree
        tree.left, counterLeft = assign(1, counter=1)
        tree.right, counterRight = assign(1, counter=1)

        # add the number of nodes together to get the individual size
        size = counterRight + counterLeft

        return tree, size

    def createHypothesis():
        # given a list of trees, create a hypothesis

        # get a list of all classIds
        classIds = random.shuffle(range(1, C))

        ftrs = []
        HypSize = 0
        # assumes one tree per feature, and creates 1 tree for
        # each class
        for __ in range(C):

            # randomly decide if grow or full should be used.
            # Also randomly assign the class ID then remove that ID
            # so each ID may only be used once
            if random.choice([True, False]):
                name = classIds.pop(0)  # get a random id
                tree, size = __grow(name)     # create tree
                ftrs.append(ConstructedFeature(name, tree, size))
            else:
                name = classIds.pop(0)  # get a random id
                tree, size = __full(name)     # create tree
                ftrs.append(ConstructedFeature(name, tree, size))

            HypSize += size

        h = Hypothesis
        h.features = ftrs
        h.size = HypSize
        return h

    hypothesis = []

    for __ in range(POPULATION_SIZE):
        hypothesis.append(createHypothesis())

    return Population(hypothesis, 0)


def evolve(population, elite):  # pop should be a list of hypotheses

    def __tournament(candidates):
        # used by evolve to selection the parents
        # ************* Tournament Selection ************* #

        first = None
        score = 0
        # compare TOURNEY number of random hypothesis
        for i in range(0, TOURNEY):

            # get a random hypothesis
            candidate = candidates.pop(random.randint(0, len(candidates)))
            # get that hypothesis's fitness score
            fitness = candidate.getFitness()

            # if first has not been set, set it
            if first is None:
                first = candidate
            # if first is set, but knight is more fit, update it
            elif score < fitness:
                first = candidate
                score = fitness

        return first

    # ************ Tree Generation ************ #
    def __generateTree(node, terminalValues, values, depth, max_depth, counter=0):

        counter += 1  # increment size counter

        # if this node contains a terminal return
        if node.data in terminalValues:
            return counter

        # make a random choice about which way to grow
        choice = random.choice(["left", "right", "both"])

        # grow left
        if choice == "left":

            # check to see if we are at the max depth
            if depth == max_depth:
                # get a random terminal
                index = terminalValues[random.randint(0, len(terminalValues))]
                # put the terminal in the left branch
                node.left = Tree(index)
                return counter

            # pick a random operation or terminal
            index = values[random.randint(0, len(values))]
            # put the operation or terminal in the left node
            node.left = Tree(index)

            # call generateTree recursively
            __generateTree(node.left, terminalValues, values, depth+1, max_depth, counter)

        # grow right
        elif choice == "right":

            # check to see if we are at the max depth
            if depth == max_depth:
                # get a random terminal
                index = terminalValues[random.randint(0, len(terminalValues))]
                # put the terminal in the left branch
                node.left = Tree(index)
                return counter

            # pick a random operation or terminal
            index = values[random.randint(0, len(values))]
            # put the operation or terminal in the left node
            node.right = Tree(index)

            # cal__generateTree recursively
            __generateTree(node.right, terminalValues, values, depth + 1, max_depth)

        elif choice == "both":

            # check to see if we are at the max depth
            if depth == max_depth:
                # get a random terminal
                index = terminalValues[random.randint(0, len(terminalValues))]
                # put the terminal in the left branch
                node.left = Tree(index)
                return

            # left branch
            # pick a random operation or terminal
            index = values[random.randint(0, len(values))]
            # put the operation or terminal in the left node
            node.left = Tree(index)

            # right branch
            # pick a random operation or terminal
            index = values[random.randint(0, len(values))]
            # put the operation or terminal in the left node
            node.right = Tree(index)

            # cal__generateTree recursively
            cLeft = __generateTree(node.left, terminalValues, values, depth + 1, max_depth)
            # call grow recursively
            cRight = __generateTree(node.right, terminalValues, values, depth + 1, max_depth)

            # calculate the size (the number of nodes) by adding the size of the subtrees
            counter = cLeft + cRight
            return counter

    # ************ Evolution ************ #

    # ? Do I need to create a new population or is this all done in place?
    # create a new population with no hypotheses
    newPopulation = Population([], population.generation+1)
    # while the size of the new population is less than the max pop size
    while len(newPopulation.candidateHypotheses) < POPULATION_SIZE:
        # get a random number between 0 & 1
        probability = random.uniform(0,1)
        # if probability is less than mutation rate, mutate
        if probability < MUTATION_RATE:  # ****** mutate ****** #
            # get parent hypothesis using tournament
            parent = __tournament(population)
            # get a random feature from the hypothesis
            featureIndex = random.randint(0, M)
            feature = parent.feature[featureIndex]
            terminal = feature.relevantFeatures
            # ? because lists are mutable all the changes happen in place?
            # ? So I don't need to create a new hypoth/pop as there is only ever the one?
            feature = feature.tree  # get the tree for that feature

            # randomly select a subtree in feature
            while True:  # walk the tree & find a random subtree
                # make a random decision
                decide = random.choice(["left", "right", "choose"])

                if decide == "left":  # go left
                    feature = feature.left

                elif decide == "right":  # go right
                    feature = feature.right

                elif decide == "choose" or feature.data in terminal:
                    break

            # randomly decide which method to use to construct the new tree
            decideGrow = random.choice([True, False])
            # the individual's size
            size = 0
            # randomly generate subtree
            if decideGrow:  # use grow
                # pick a random function & put it in the root
                ls = random.shuffle(OPS)
                rootData = ls[random.randint(0, len(ls))]
                # make a new tree
                t = Tree(rootData)
                # build the rest of the subtree
                size = __generateTree(t, terminal, ls[random.shuffle(OPS.append(terminal))], 0, 8)
                # set the size of the tree

            else:  # use full
                # pick a random function & put it in the root
                ls = random.shuffle(OPS)
                rootData = ls[random.randint(0, len(ls))]
                # make a new tree
                t = Tree(rootData)
                # build the rest of the subtree
                size = __generateTree(t, terminal, OPS, 0, 8)

            # get the className of the feature
            cl = parent.feature[featureIndex].className
            # add the replace the the parent with the mutated child
            parent.feature[featureIndex] = ConstructedFeature(cl, t, size)
            # add the parent to the new pop
            newPopulation.candidateHypotheses.append(parent)

        else:  # ************ crossover ************ #

            parent1 = __tournament(population)
            parent2 = __tournament(population)
            # check that each parent is unique
            while parent1 is parent2:
                parent2 = __tournament(population)

            # get a random feature from each parent
            featureIndex = random.randint(0, M)
            # feature 1
            feature1 = parent1.feature[featureIndex]
            terminals1 = feature1.relevantFeatures
            feature1 = feature1.tree
            # feature 2
            feature2 = parent2.feature[featureIndex]
            terminals2 = feature2.relevantFeatures
            feature2 = feature2.tree

            while True:  # walk the tree & find a random subtree for feature 1
                # make a random decision
                decide = random.choice(["left", "right", "choose"])

                if decide == "left":  # go left
                    feature1 = feature1.left

                elif decide == "right":  # go right
                    feature1 = feature1.right

                elif decide == "choose" or feature1.data in terminals1:
                    break

            while True:  # walk the tree & find a random subtree for feature 2
                # make a random decision
                decide = random.choice(["left", "right", "choose"])

                if decide == "left":  # go left
                    feature2 = feature2.left

                elif decide == "right":  # go right
                    feature2 = feature2.right

                elif decide == "choose" or feature2.data in terminals2:
                    break

            # swap the two subtrees
            feature1, feature2 = feature2, feature1  # ? is this done in place?

            # get the size of the new constructed features by walking the trees
            parent1.setSize()
            parent2.setSize()

            # parent 1 & 2 are both hypotheses and should have been changed in place, so add them to the new pop
            newPopulation.candidateHypotheses.append(parent1, parent2)

        # handle elitism
        newHypothFitness = newPopulation.candidateHypotheses[-1].getFitness()
        if newHypothFitness > elite.getFitness:
            elite = newPopulation.candidateHypotheses[-1]

    return newPopulation, elite


def discretization(data):

    for index, item in np.ndenumerate(data):

        if item < -0.5:
            data[index] = -1

        elif item > 0.5:
            data[index] = 1

        else:
            data[index] = 0

    return data


def normalize(entries, scalar=None):

    # entries will be a list of all the training data in the form of
    # [ list[[some instance, some instance, ...]],
    #   list[[some instance, some instance, ...]], ...]
    # (where each index is a list of instances)
    # OR it will be a list of all the testing data in the form of
    # [some instance, some instance, ...]
    # (where an instance is a list of feature values)

    clss = entries[:, :1]  # remove the class ids

    # set stdScalar either using parameter in the case of testing data,
    # or making it in the case of training data

    # if we are dealing with training data, not testing data
    if scalar is None:
        # transform the data
        stdScalar = StandardScaler().fit_transform(clss)  # z-score
    # if we are dealing with testing data
    else:
        stdScalar = scalar  # discrete transformation

    # ? does this work for the training data (when entries is a list of lists)
    normalizedData = discretization(stdScalar)  # discrete transformation

    finalData = np.empty()
    finalData = np.append(finalData, np.array(entries[:, 0]), axis=1)
    finalData = np.append(finalData, np.array(normalizedData), axis=0)

    # stdScalar - Standard Scalar, will used to on test & training data
    return finalData, stdScalar


def createTrainingData(entries, K=10):

    # *** connect a class to every instance that occurs in it ***

    # this will map a classId to a 2nd dictionary that will hold the number of instances in that class &
    # the instances
    # classToInstances[classId] = list[counter, instance1Values, instance2Values, ...]
    classToInstances = {}

    for e in entries:

        # get the class id from the entry
        cId = e[0]

        # if we already have an entry for that classId, append to it
        if classToInstances[cId]:
            # get the current list for this class
            spam = classToInstances[cId]
            # increment instance counter
            spam[0] += 1
            # add the new instance at the new index, removing the classId from it
            spam[spam[0]] = e[1:]

        # if this is the first time we've seen the class, create a new list for it
        else:
            # initialize the counter, add the new instance to the new list at index 1
            # after removing the classId from it
            spam = [1, e[1:]]
            # add the new list to the "master" dictionary
            classToInstances[cId] = spam

    # *** Now create a random permutation of the instances in a class, & put them in buckets ***

    buckets = [None] * K  # this list will be what we "deal" our "deck" of instances to
    index = 0  # this will be the index of the bucket we are "dealing" to

    for cId in classToInstances.keys():

        # *** get a permutation of the instances that are in the class given by classId ***

        # get the list for the class
        # list is of the form [counter, instance1Values, instance2Values, ...]
        # where instanceNValues is a lists of a attribute values
        instances = classToInstances[cId]

        # get a permutation of the instances for the class, but first remove the counter
        # permutation is of the form [instance?Values, instance?Values, ...]
        # where instanceNValues is a list of a attribute values
        permutation = np.random.permutation(instances[1:])

        # *** assign instances to buckets ***

        # loop over every instance in the class cId, in what is now a random order
        # p should be an instance?Value, which is a list of a attribute values
        for p in permutation:
            buckets[index] = p  # add the random instance to the bucket at index p
            index = (index+1) % K  # increment index in a round robin style

    # *** The buckets are now full & together should contain every instance ***

    # get a randomly ordered range of numbers up to K
    # this will be used to give a random index in the loop below
    randomIndex = list(range(K))
    random.shuffle(randomIndex)

    # these will hold the testing & training data. Each index will
    # hold the values from 1 of the K loops, So testing will hold
    # 1 value at any index while train will hold a list of them
    testing = []
    train = []

    # *** Divide them into training & test data K times ***

    # loop over all the random index values
    for r in randomIndex:  # len(r) = K so this will be done K times
        # take only the Rth bucket
        testing.append(buckets[r])
        # trim the Rth bucket so train doesn't contain the testing instance
        trimd = buckets  # needed since pop works in place
        # take all the buckets except R
        train = trimd.pop(r)

    # train will be a list of all the training data and so is
    # of len(k) & each index is a list of instances
    # testing will be a list of all the testing data and so is
    # of len(K) & each index is a single instance
    # (where an instance is a list of feature values)
    return train, testing


def cdfc(path):
    # Class Dependent Feature Construction

    # makes sure we're using global variables
    global FEATURE_NUMBER
    global POPULATION_SIZE
    global INSTANCES_NUMBER
    global rows
    global row
    # *** used in entropy calculation *** #
    global ENTROPY_OF_S

    classes = []  # this will hold classIds and how often they occur
    classSet = set()  # this will hold how many classes there are

    classToOccur = {}  # maps a classId to the number of times it occurs

    with open(path) as filePath:  # open selected file
        # create a reader using the file
        reader = csv.reader(filePath, delimiter=',')

        # *** Parse the file into a numpy 2d array *** #
        # this will hold all of the instances in our data
        entries = np.array()
        for line in reader:  # read the file
            entries.append(line)  # add the line to the list of entries

        # now that the data has been parsed into a numpy array use it
        # to create both the training data & the testing data
        train, testing = createTrainingData(entries)
        # now normalize the training & testing data, and keep the scalar used
        train, scalar = normalize(train)

        # set global variables using the now transformed data
        # TODO check the below
        # ? Do I want to calculate this using the training data,
        # ?  or the training data & the testing data?
        for s in train:
            # each s in train will be a set of instances, and then
            # each line in s will be a single instance
            for line in s:
                # parse the file
                # reader[0] = classId, reader[1:] = attribute values
                rows.append(row(line[0], line[1:]))
                classes.append(line[0])
                classSet.add(line[0])
                INSTANCES_NUMBER += 1

                # ********* The Code Below is Used to Calculated Entropy  ********* #

                # this will count the number of times a class occurs in the provided data
                # dictionary[classId] = counter of times that class is found
                if classToOccur.get(line[0]):  # if we have encountered the class before
                    classToOccur[line[0]] += 1  # increment
                else:  # if this is the first time we've encountered the class
                    classToOccur[line[0]] = 1  # set to 1

                # ****************************************************************** #

        # get the number of features in the data set
        FEATURE_NUMBER = len(rows[0].attribute)
        POPULATION_SIZE = FEATURE_NUMBER * BETA  # set the pop size

    filePath.close()  # close the file

    # ********* The Code Below is Used to Calculated Entropy  ********* #

    # loop over all classes
    for i in classToOccur.keys():
        # compute p_i
        pi = classToOccur[i] / INSTANCES_NUMBER
        # calculation entropy summation
        ENTROPY_OF_S -= pi * math.log(pi, 2)

    # ***************************************************************** #

    # *********************** Run the Algorithm *********************** #

    # TODO run K times using the different training data
    # ? how do I run this K different times?
    # create initial population
    currentPopulation = createInitialPopulation()
    elite = currentPopulation.candidateHypotheses[0]  # init elitism
    # loop, evolving each generation. This is where most of the work is done
    for i in range(GENERATIONS):
        # generate a new population by evolving the old one
        newPopulation, elite = evolve(currentPopulation, elite)
        # update currentPopulation to hold the new population
        # this is done in two steps to avoid potential namespace issues
        currentPopulation = newPopulation

    # TODO use testing data to report accuracy