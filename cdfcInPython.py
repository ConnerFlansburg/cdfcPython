
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

##################### Constants/Globals ######################

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
FEATURE_NUMBER = None

# the population size
POPULATION_SIZE = None

# the number of instances in the training data
INSTANCES_NUMBER = 0

# *** the next 3 variables are used to compute entropy *** #
# this will store the number of times a class occurs in the training data in a dictionary keyed by it's classId
Occurences = {}

# stores the number of times a value occurs in the training data (occurences keyed value)
Values = {}

# the number of times a value occurs keyed by class
fGivenC = {}

# *** set values below for every new dataset *** #

# C is the number of classes in the data
C = 3

# R is the ratio of number of constructed features to the number of classes  (features/classes)
R = 2

# M is the number of constructed features
M = R*C

# FN (feature num) is number of features in the dataset
FN = None

# PS (pop size) is the population size (equal to number of features * beta)
PS = None

######################### End of Constants/Globals ###############################

####################### Namespaces/Structs & Objects #########################

# a single line in the csv, representing a record/instance
row = collect.namedtuple('row', ['className', 'attributes'])

# this will store all of the records read in (list is a list of rows)
rows = []


class Tree:

    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insertLeft(self, data):
        self.left = Tree(data)

    def insertRight(self, data):
        self.right = Tree(data)

    def PrintTree(self):
        print(self.data)


class ConstructedFeature:

    tree = None  # the tree representation of the constructed feature
    className = None  # the name of the class this tree is meant to distinguish
    infoGain = None  # the info gain for this feature

    def __init__(self, tree, className):
        self.tree = tree
        self.className = className

    def getInfoGain(self):
        if self.infoGain is None:
            pass  # TODO write getInfoGain
        else:
            return self.infoGain


class Hypothesis:
    # a single hypothesis(a GP individual)
    features = None  # a list of all the constructed features
    size = 0  # the number of nodes in all the cfs
    maxInfoGain = None  # the max info gain in the hypothesis
    averageInfoGain = None  # the average info gain of the hypothesis
    distance = None  # the distance function score
    fitness = None  # the fitness score


#################### End of Namespaces/Structs & Objects #######################

def main():
    Tk().withdraw()  # prevent root window caused by Tkinter
    path = askFile()  # prompt user for file path

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
                rows.append(row(line[0], line[1:]))  # parse file
                classes.append(line[0])
                classSet.add(line[0])
                INSTANCES_NUMBER += 1

    # get the number of features in the dataset
    FEATURE_NUMBER = len(rows[0].attribute)
    POPULATION_SIZE = FEATURE_NUMBER * BETA  # set the pop size

    ######### The Code Below is Used to Calculated Entropy  ##########
    for v in valuesSet:
        # find out how many times a value occurred and store it in a dictionary keyed by value
        Values[v] = vals.count(v)
        if Values[v] > 1:  # if the value occurs more than once
            for r in rows:  # loop over rows
                # if the value appears in this instance
                if r.attributes.contains(v):
                    # update the dictionary's amount of occurences
                    #! BUG this won't work because dictionaries can't reuse keys
                    fGivenC[r.className] += 1
    # loop over the class set - each classId will be id only once because classSet is a set
    for id in classSet:
        # finds out how many times a class occurs in training data and add to dictionary
        Occurences[id] = classes.count(id)


def valuesInClass(classId, attribute):
    """valuesInClass determines what values of an attribute occur in a class and what values do not

    Arguments:
        classId {String or int} -- This is the identifier for the class that should be examined
        attribute {int} -- this is the attribute to be investigated. It should be the index of the attribute in the row namedtuple in the rows list

    Returns:
        inClass -- This holds the values in the class.
        notInClass -- This holds the values not in the class.
    """

    inClass = None  # attribute values that appear in the class
    notInClass = None  # attribute values that do not appear in the class

    # loop over all the rows, where value is the row at the current index
    for value in rows:
        if value.className == classId:  # if the class is the same as the class given
            # add the feature's value to in
            inClass.append(value.attributes[attribute])
        else:  # if the class is not the same as the class given
            # add the feature's value to not in
            notInClass.append(value.attributes[attribute])
    # return inClass & notInClass
    return inClass, notInClass


def terminals(classId):
    """terminals creates the list of relevant terminals for a given class.

    Arguments:
        classId {String} -- classId is the identifier for the class for which we want a terminal set

    Returns:
        list -- terminals returns the highest scoring features as a list. The list will have a length of FEATURE_NUMBER/2, and will hold the indexes of the features. 
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
        # TODO use Baye's Theorem to compute
        pPos = None
        # TODO use Baye's Theorem to compute
        pNeg = None
        entFeature = entropy(pPos, pNeg)

        # H(class) - H(class|f)
        f.infoGain = entClass - entFeature
        gainSum += f.infoGain  # update the info sum

        # updates the max info gain of the hypothesis if needed
        if h.maxInfoGain < f.infoGain:
            h.maxInfoGain = f.infoGain

    # calculate the average info gain using formula 3
    #* create more correct citation later *#
    h.averageInfoGain = (gainSum + h.maxInfoGain)/((M+1)*(math.log(C, 2)))

    # set size

    # exit loop

    # get average info gain

    # get the distance


if __name__ == "__main__":
    main()
