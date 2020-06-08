
import csv
import math
import random
import collections as collect
from scipy import stats
from tkinter import Tk
from tkinter.filedialog import askFile

# * Next Steps
# TODO create a print function for population
# TODO finish docstrings
# TODO rework mutation to use parallelism
# TODO optimize & make modular

# * Sanity Checking / debugging
# TODO check entropy function code in Hypothesis
# TODO check transform function code in Constructed Feature
# TODO check transform function code in Tree
# TODO check logic for initial population generation
# TODO debug mutation & crossover


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

# ALPHA is the fitess weight alpha
ALPHA = 0.8

# a constant used to calculate the pop size
BETA = 2

# the number of features in the dataset
FEATURE_NUMBER = 0

# the population size
POPULATION_SIZE = 0

# the number of instances in the training data
INSTANCES_NUMBER = 0

MAX_DEPTH = 8

# *** the next 3 variables are used to compute entropy *** #
# this will store the number of times a class occurs in the training data in
# a dictionary keyed by it's classId
Occurences = {}

# stores the number of times a value occurs in the training data
# (occurences keyed value)
Values = {}

# the number of times a value occurs keyed by class
fGivenC = {}

# *** set values below for every new dataset *** #

# C is the number of classes in the data
C = 3

# R is the ratio of number of constructed features to the number of classes
# (features/classes)
R = 2

# M is the number of constructed features
M = R*C

# FN (feature num) is number of features in the dataset
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
    data = None  # must either be a function or a terminal

    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    # running a tree should return a single value
    def runTree(self, rFt):
        if self.data in rFt:  # if the root is a terminal
            return self.data  # return value
        else:  # if the root is not a terminal
            # run the tree recursively
            return self.data(self.__runLeft(rFt), self.__runRight(rFt))

    def __runLeft(self, relevantFeatures, featureValues):
        # if left node is a terminal
        if self.left in relevantFeatures:
            return featureValues[self.data]
        else:  # if left node is a function
            return self.left()

    def __runRight(self, relevantFeatures, featureValues):
        # if right node is a terminal
        if self.right in relevantFeatures:
            return featureValues[self.data]
        else:  # if right node is a function
            return self.right()


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

    # TODO does this need to be a list because we might have multiple trees?
    tree = None  # the root node of the constructed feature
    className = None  # the name of the class this tree is meant to distinguish
    infoGain = None  # the info gain for this feature
    relevantFeatures = None  # the set of relevant features
    # the values data after they have been transformed by the tree
    transformedValues = None

    def __init__(self, className, tree):
        self.className = className
        self.tree = tree
        # call terminals to create the terminal set
        self.relevantFeatures = terminals(className)

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
            term1 = Db / len(values)
            term2 = Dw / len(values)
            return 1 / (1 + math.pow(math.e, -5*(term1 - term2)))

        def __entropy(pos, neg):
            return -pos*math.log(pos, 2)-neg*math.log(neg, 2)

        # loop over all features & get their info gain
        gainSum = 0  # the info gain of the hypothesis
        for f in self.features:

            # ********* Entropy calculation ********* #
            # find the +/- probabilities of a class
            pPos = Occurences.get(f.className)
            pNeg = INSTANCES_NUMBER - pPos
            entClass = __entropy(pPos, pNeg)

            # find the +/- probabilites of a feature given a class
            pPos = None  # TODO use Baye's Theorem to compute
            pNeg = None  # TODO use Baye's Theorem to compute
            entFeature = __entropy(pPos, pNeg)

            # ******** Info Gain calculation ******* #
            # H(class) - H(class|f)
            f.infoGain = entClass - entFeature
            gainSum += f.infoGain  # update the info sum

            # updates the max info gain of the hypothesis if needed
            if self.maxInfoGain < f.infoGain:
                self.maxInfoGain = f.infoGain

        # calculate the average info gain using formula 3
        # * create more correct citation later * #
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
            transformed.append(instance(r.className, f.transform(r)))
            # ? how to make class name a bool? Does it need to be?
        return transformed  # return the list of all instances


class Population:
    # this will be the population of hypotheses
    candidateHypotheses = []  # a list of all the candidate hyps
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
    scores = None

    for i in range(1, FEATURE_NUMBER):
        # find the values of attribute i in/not in class classId
        inClass, notIn = valuesInClass(classId, i)

        # get the t-test & p-value for the feature
        tValue, pValue = stats.ttest_ind(inClass, notIn)

        # calculate relevancy for a single feature
        relevancy = None  # this will hold the relevancy score for this feature

        if pValue >= 0.05:  # if p-value is less than 0.05
            relevancy = 0  # set relevancy score to 0
        else:  # otherwise
            # set relevancy using t-value/p-value
            relevancy = abs(tValue)/pValue
        scores.append(Score(i, relevancy))

    # sort the features by relevancy scores
    sortedScores = sorted(scores, key=lambda Score: Score.Attribute)

    terminalSet = None  # this will hold relevant terminals
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

    inClass = None  # attribute values that appear in the class
    notInClass = None  # attribute values that do not appear in the class

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

    def __grow(size):
        # This function uses the grow method to generate an initial population
        # TODO double check this logic
        def assign(level):
            # ? should this be changed to guarantee every terminal is used?
            # recursively assign tree values
            if level != MAX_DEPTH:
                # get the random value
                spam = ls[random.randint(0, len(ls))]
                if spam in terminal:  # if the item is a terminal
                    return Tree(spam)  # just return, stopping recursion

                tree = Tree(spam)
                tree.left = assign(level + 1)
                tree.right = assign(level + 1)
                return tree

            else:
                # stop recursion; max depth has been reached
                # add a terminal to the leaf
                spam = terminal[random.randint(0, len(terminal))]
                # return
                return Tree(spam)

        classId = None
        # create a list of classIds
        classIds = random.shuffle(range(1, FEATURE_NUMBER))

        for i in range(size):

            # ******** assign a random class id ******** #
            # if the list is empty of class ids get new ones
            if classIds:
                classId = classIds.pop()
            else:
                classIds = random.shuffle(range(1, FEATURE_NUMBER))
                classId = classIds.pop()

            # pick a random function & put it in the root
            ls = random.shuffle(OPS)
            rootData = ls[random.randint(0, len(ls))]
            tree = Tree(rootData)  # make a new tree

            # get the list of terminal characters
            terminal = terminals(classId)
            # add the terminal values to the list of functions & reorder
            ls = random.shuffle(ls.append(terminal))

            # create the tree
            tree.left = assign(0)
            tree.right = assign(0)

    def __full(size):
        # This function uses the full method to generate an initial population
        # TODO double check this logic
        def assign(level):
            # ? should this be changed to guarantee every terminal is used?
            # recursively assign tree values
            if level != MAX_DEPTH:
                # get a random function & add it to the tree
                tree = Tree(ls[random.randint(0, len(ls))])
                # call for branches
                tree.left = assign(level + 1)
                tree.right = assign(level + 1)
                return tree

            else:  # stop recursion; max depth has been reached
                # add a terminal to the leaf & return
                return Tree(terminal[random.randint(0, len(terminal))])

        for i in range(size):
            classId = None  # ? how do I know that class that a tree is for?
            # pick a random function & put it in the root
            ls = random.shuffle(OPS)
            rootData = ls[random.randint(0, len(ls))]
            tree = Tree(rootData)  # make a new tree
            # TODO somehow save & return the root of this tree
            # ? maybe this needs to make a constructed feature...
            # get the list of terminal characters
            terminal = terminals(classId)

            # create the tree
            tree.left = assign(0)
            tree.right = assign(0)

    halfPopulation = POPULATION_SIZE//2

    pop = []  # this will hold the initial population
    # TODO these should be lists of constructed features
    # create half of pop via grow
    pop.append(__grow(halfPopulation))
    # create half of pop via full
    pop.append(__full(halfPopulation))
    # create a constructed feature for each item in the list
    # TODO check this logic
    features = [ConstructedFeature('className', i) for i in pop]
    # create a list of empty hypotheses
    hypList = [Hypothesis() for i in range(POPULATION_SIZE)]
    for c in range(FEATURE_NUMBER):
        spam = filter(lambda x: x.className == c, features)
        # Loop over hList and add one feature to each hypotheses
        for i in range(hypList):
            i.tree.append(spam.pop())
            # TODO set size somehow
    # return the population
    return pop


def evolve(pop, elitism=True):  # pop should be a list of hypotheses

    def __tournament(pop):  # used by evolve to selection the parents
        # ************* Tournament Selection ************* #
        for j in range(0, 1):
            # randomly select hypotheses from the population to create
            # a list of all possible parents (we want to do this twice)

            # this will hold the list of random potential parents
            parents = (None, None)

            for i in range(TOURNEY):
                # this loop create 1 parent & is repeated twice
                # ? is the append adding by value or reference?
                # ? If it's adding by reference this will
                # ? overwrite the added items...
                possible = []

                # get a random index integer
                spam = random.randint(0, len(pop))

                while possible.contains(pop[spam]):
                    # if we have already selected the value as a parent,
                    # get a new random value
                    spam = random.randint(0, len(pop))

                # add the potential parents to the list
                possible.append(pop[spam])

            # find the candidate with the max fitness & make it a parent
            parents[j] = max(possible, key=lambda i: possible.fitness)

        # parents now holds the two parents for a new hypothesis; return it
        return parents  # ? is this several parent sets or just one?

    # ********** Mutation & Crossover ********** #
    def __crossover(mother, father):  # mother & father should be two trees
        # ? is the crossover point selected randomly?
        # this wil be the crossover point in the tree
        # (must be before a terminal node)
        motherCrossPoint = random.randint(0, MAX_DEPTH-1)
        fatherCrossPoint = random.randint(0, MAX_DEPTH-1)

        child = mother  # this will be the child we return

        # ! BUG - what if the tree isn't full and we can't reach the depth?
        # we will walk through the child's copy of mother to prevent
        # the mother from being overwritten
        childCrossPoint = None
        for i in range(motherCrossPoint):
            # randomly walk the tree until we reach the crossover point
            if random.randint(1, 2) == 1:
                childCrossPoint = child.left()
            else:
                childCrossPoint = child.right()

        fatherCrossNode = None  # the node at the crossover point
        for i in range(fatherCrossPoint):
            # randomly walk the tree until we reach the crossover point
            if random.randint(1, 2) == 1:
                fatherCrossNode = father.left()
            else:
                fatherCrossNode = father.right()

        # take the sub-tree from father & add it to child at the
        # crossover point for the mother
        childCrossPoint.left = fatherCrossNode.left
        childCrossPoint.right = fatherCrossNode.right

        return child

    def __mutate(candidate):
        # ? how many nodes are mutated? How are they selected? Randomly?

        # mutate a random number of random nodes in random ways

        # get the number of nodes to mutate
        numberToMutate = random.randint(1, POPULATION_SIZE)

        # get the depth of nodes to mutate (root node would be 0)
        depthsToMutate = [random.randint(1, MAX_DEPTH-1)
                          for i in range(numberToMutate)]
        # sort the depths so they are in numerical order. This will save us
        # time when walking the tree
        depthsToMutate.sort()

        # this will be the node we are currently at in our tree
        currentNode = candidate
        # this will hold our current depth
        currentDepth = 0
        # ! BUG - what if the tree isn't full and we can't reach the depth?
        # loop over the list of nodes to mutate & walk to them
        for i in depthsToMutate:

            # randomly walk the tree until we reach a node
            # of the depth we want
            while currentDepth != i:
                if random.randint(1, 2) == 1:
                    currentNode = currentNode.left
                    currentDepth += 1
                else:
                    currentNode = currentNode.left
                    currentDepth += 1

            # Since we have reached a node of the random depth
            # assign it a new random value
            currentNode.data = OPS[random.randint(0, len(OPS))]
        # ? Do I need to return candidate or are the changes in place
        # ? (passed by reference or by value)?
        return

    # TODO check parameters of mutate & crossover
    if pop.generation >= GENERATIONS:
        return  # if we have reached our generation max, exit

    mostFit = __tournament(pop)  # collect the possible parents
    newCandidates = []

    for c in mostFit:
        # loop over all mostFit hypotheses *(c should be a hypothesis object)
        # if the random number is greater than the mutation rate (the lower of
        # the two), evolve using crossover
        if random.uniform(0, 1) > MUTATION_RATE:
            newCandidates.append(__crossover(pop))
        else:  # otherwis use mutation
            newCandidates.append(__mutate(pop))
    # create & return a new population
    return Population(newCandidates, pop.generation+1)

    # *********** Calculate fitness & Handle Elitism *********** #
    if elitism:
        # if we are using elitism, make the structures we'll use to track it
        elite = collect.namedtuple('elite', ['fitness', 'hypothesis'])
        elites = []

    for h in pop:  # for every hypothesis, set it's fitness
        h.getFitness()

        # if we are using elitism we'll also need a list of all hypotheses
        if elitism:
            elites.append(elite(h.fitess, h))

    if elitism:
        # collect the best hypotheses & store them
        spam = sorted(elites, key=lambda elite: elite.fitness)
        elites = spam[:ELITISM_RATE]


def main():
    Tk().withdraw()  # prevent root window caused by Tkinter
    path = askFile()  # prompt user for file path

    # makes sure we're using global variables
    global FEATURE_NUMBER
    global POPULATION_SIZE
    global INSTANCES_NUMBER
    global rows
    global row

    classes = []  # this will hold classIds and how often they occur
    classSet = set()  # this will hold how many classes there are
    vals = []  # holds all the that occur in the training data
    valuesSet = set()  # same as values but with no repeated values

    with open(path) as filePath:  # open selected file
        # create a reader using the file
        reader = csv.reader(filePath, delimiter=',')
        counter = 0  # this is our line counter
        for line in reader:  # read the file
            if counter:  # if we are not reading the column headers,
                # parse the file
                # reader[0] = classId, reader[1:] = attribute values
                rows.append(row(line[0], line[1:]))
                classes.append(line[0])
                classSet.add(line[0])
                INSTANCES_NUMBER += 1
            else:  # if we are reading the column headers,
                counter += 1  # skip

    # get the number of features in the dataset
    FEATURE_NUMBER = len(rows[0].attribute)
    POPULATION_SIZE = FEATURE_NUMBER * BETA  # set the pop size

    # ********* The Code Below is Used to Calculated Entropy  ********* #
    for v in valuesSet:
        # find out how many times a value occurred and store it
        # in a dictionary keyed by value
        Values[v] = vals.count(v)
        if Values[v] > 1:  # if the value occurs more than once
            for r in rows:  # loop over rows
                # if the value appears in this instance
                if r.attributes.contains(v):
                    # update the dictionary's amount of occurences
                    # !BUG this won't work: dictionaries can't reuse keys
                    fGivenC[r.className] += 1
    # loop over the class set - each classId will be id only once
    for id in classSet:
        # finds out how many times a class occurs in data and add to dictionary
        Occurences[id] = classes.count(id)
    # ****************************************************************** #

    # ********************* Run the Algorithm ********************* #
    # create initial population
    currentPopulation = createInitialPopulation()
    # loop, evolving each generation. This is where most of the work is done
    for i in range(GENERATIONS):
        # generate a new population by evolving the old one
        newPopulation = evolve(currentPopulation, elitism=True)
        # update currentPopulation to hold the new population
        # this is done in two steps to avoid potential namespace issues
        currentPopulation = newPopulation

    # report the information about the final population
    print(currentPopulation)


if __name__ == "__main__":
    main()
