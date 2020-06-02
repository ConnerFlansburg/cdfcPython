
import csv
import math
import collections as collect
from scipy import stats
from tkinter import Tk
from tkinter.filedialog import askFile


# * conversion workload
# TODO write fitness function

# * still to do
# TODO handle evolution (try to leverage parallelism)
# TODO create initial population
# TODO change appends so they use pointers
# TODO write main

# ******************** Constants/Globals ******************** #

# CROSSOVER_RATE is the crossover rate
CROSSOVER_RATE = 0.8

# GENERATIONS is the number of generations the GP should run for
GENERATIONS = 50

# MUTATION_RATE is the mutation rate
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
S = []


class Tree:

    left = None
    right = None
    data = None  # must either be a function or a terminal

    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insertLeft(self, data):
        self.left = Tree(data)

    def insertRight(self, data):
        self.right = Tree(data)

    # running a tree should return a single value
    def runTree(self, rFt):
        if self.data in rFt:  # if the root is a terminal
            return self.data  # return value
        else:  # if the root is not a terminal
            # run the tree recursively
            return self.data(self.__runLeft(rFt), self.__runRight(rFt))

    def __runLeft(self, relevantFeatures):
        # if left node is a terminal
        if self.left in relevantFeatures:
            # TODO change so this uses the value for an instance
            return self.left
        else:  # if left node is a function
            return self.left()

    def __runRight(self, relevantFeatures):
        # if right node is a terminal
        if self.right in relevantFeatures:
            return self.right
        else:  # if right node is a function
            return self.right()

    def PrintTree(self):
        print(self.data)


class ConstructedFeature:

    tree = None  # the root node of the constructed feature
    className = None  # the name of the class this tree is meant to distinguish
    infoGain = None  # the info gain for this feature
    relevantFeatures = None  # the set of relevant features
    # the values data after they have been transformed by the tree
    transformedValues = None

    def __init__(self, className):
        self.className = className
        # call terminals to create the terminal set
        self.relevantFeatures = terminals(className)

    def getInfoGain(self):
        if self.infoGain is None:
            pass  # TODO write getInfoGain
        else:
            return self.infoGain

    def transform(self):  # ? pass the instance here somehow?
        # transform the data for a single feature
        # this should return a single value (a value transformed by the tree)
        return self.tree.runTree(self.relevantFeatures)


class Hypothesis:
    # a single hypothesis(a GP individual)
    features = []  # a list of all the constructed features
    size = 0  # the number of nodes in all the cfs
    maxInfoGain = None  # the max info gain in the hypothesis
    averageInfoGain = None  # the average info gain of the hypothesis
    distance = 0  # the distance function score
    fitness = None  # the fitness score

    def transform(self):  # TODO need to pass information on instance somehow
        transformed = []  # this will hold the transformed values
        # transform the original input using each constructed feature
        for f in self.features:
            transformed.append(f.transform())

        return transformed


# ***************** End of Namespaces/Structs & Objects ******************* #

# ********************** Valid Operations Within Tree ********************** #
# all functions for the tree must be of the form x(y,z)

# *************************** End of Operations *************************** #


def main():
    Tk().withdraw()  # prevent root window caused by Tkinter
    path = askFile()  # prompt user for file path

    # makes sure we're using global variables
    global FEATURE_NUMBER
    global POPULATION_SIZE
    global INSTANCES_NUMBER
    global S
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
            if counter == 0:  # if we are reading the column headers,
                counter += 1  # skip
            else:  # otherwise parse file
                # reader[0] = classId, reader[1:] = attribute values
                S.append(row(line[0], line[1:]))  # parse file
                classes.append(line[0])
                classSet.add(line[0])
                INSTANCES_NUMBER += 1

    # get the number of features in the dataset
    FEATURE_NUMBER = len(S[0].attribute)
    POPULATION_SIZE = FEATURE_NUMBER * BETA  # set the pop size

    # ********* The Code Below is Used to Calculated Entropy  ********* #
    for v in valuesSet:
        # find out how many times a value occurred and store it
        # in a dictionary keyed by value
        Values[v] = vals.count(v)
        if Values[v] > 1:  # if the value occurs more than once
            for r in S:  # loop over S
                # if the value appears in this instance
                if r.attributes.contains(v):
                    # update the dictionary's amount of occurences
                    # !BUG this won't work: dictionaries can't reuse keys
                    fGivenC[r.className] += 1
    # loop over the class set - each classId will be id only once
    for id in classSet:
        # finds out how many times a class occurs in data and add to dictionary
        Occurences[id] = classes.count(id)


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
                            namedtuple in the S list

    Returns:
        inClass -- This holds the values in the class.
        notInClass -- This holds the values not in the class.
    """

    inClass = None  # attribute values that appear in the class
    notInClass = None  # attribute values that do not appear in the class

    # loop over all the S, where value is the row at the current index
    for value in S:
        # if the class is the same as the class given
        if value.className == classId:
            # add the feature's value to in
            inClass.append(value.attributes[attribute])
        else:  # if the class is not the same as the class given
            # add the feature's value to not in
            notInClass.append(value.attributes[attribute])
    # return inClass & notInClass
    return inClass, notInClass


def fitness(h):

    def entropy(pos, neg):
        return -pos*math.log(pos, 2)-neg*math.log(neg, 2)

    # loop over all features & get their info gain
    gainSum = 0  # the info gain of the hypothesis
    for f in h.features:

        # find the +/- probabilities of a class
        pPos = Occurences.get(f.className)
        pNeg = INSTANCES_NUMBER - pPos
        entClass = entropy(pPos, pNeg)

        # find the +/- probabilites of a feature given a class
        pPos = None  # TODO use Baye's Theorem to compute
        pNeg = None  # TODO use Baye's Theorem to compute
        entFeature = entropy(pPos, pNeg)

        # H(class) - H(class|f)
        f.infoGain = entClass - entFeature
        gainSum += f.infoGain  # update the info sum

        # updates the max info gain of the hypothesis if needed
        if h.maxInfoGain < f.infoGain:
            h.maxInfoGain = f.infoGain

        # ? this computes the distance for each constructed feature &
        # ? adds it to the distance of all the other features. Is this correct?
        h.distance += Distance(f.className)

    # calculate the average info gain using formula 3
    # * create more correct citation later * #
    h.averageInfoGain += (gainSum + h.maxInfoGain)/((M+1)*(math.log(C, 2)))

    # set size
    # * this must be based off the number of nodes a tree has because
    # * the depth will be the same for all of them


def Distance():
    return 1 / (1 + math.pow(math.e, -5*(Db() - Dw())))


def Db(classId):
    DbSum = 0
    # loop over all the training instances where the class occurs
    for i in S:
        if i.className == classId:
            Vi, Vj = valuesInClass(classId, i.attributes)
            DbSum += min(Czekanowski(Vi, Vj))
    return (1/len(S))*DbSum


def Dw(classId):
    DbSum = 0
    # loop over all the training instances where the class occurs
    for i in S:
        if i.className == classId:
            Vi, Vj = valuesInClass(classId, i.attributes)
            DbSum += max(Czekanowski(Vi, Vj))
    return (1/len(S))*DbSum


def Czekanowski(Vi, Vj):
    pass  # TODO write Czekanowski function


if __name__ == "__main__":
    main()
