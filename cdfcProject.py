import random
import copy
import typing as typ
import numpy as np
import collections as collect
import tkinter as tk
from tkinter import filedialog
# from cdfc import cdfc
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# *** Only one of these imports will be used at a time, which one depends on the model being used *** #
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# * Next Steps
# TODO figure out why model is giving the same value everytime -- maybe it because our data is suited to regression?
# TODO get CDFC working
from sklearn.tree import DecisionTreeClassifier


def discretization(data: np.ndarray) -> np.ndarray:
    
    for index, item in np.ndenumerate(data):

        if item < -0.5:
            data[index] = -1

        elif item > 0.5:
            data[index] = 1

        else:
            data[index] = 0

    return data


def normalize(entries: np.ndarray, scalar: typ.Union[None, StandardScaler, ]) -> typ.Tuple[np.ndarray, StandardScaler]:

    # remove the class IDs so the don't get normalized
    noIds = np.array(entries[:, 1:])

    # noIds is a list of instances without IDs of the form:
    # [ [value, value, value, ...],
    #   [value, value, value, ...], ...]

    # set stdScalar either using parameter in the case of testing data,
    # or making it in the case of training data
    # if we are dealing with training data, not testing data
    if scalar is None:

        # *** transform the data *** #
        stdScalar = StandardScaler()        # create a scalar object
        stdScalar.fit(noIds)                # fit the scalar using the data
        tData = stdScalar.transform(noIds)  # perform the transformation
        tData = discretization(tData)       # discrete transformation

    # if we are dealing with testing data
    else:

        # *** transform the data *** #
        stdScalar = scalar                  # used the passed scalar object
        tData = stdScalar.transform(noIds)  # transform the scalar using passed fit
        tData = discretization(tData)       # discrete transformation

    # *** add the IDs back on *** #
    entries[:, 1:] = tData  # this overwrites everything in entries BUT the ids

    # stdScalar - Standard Scalar, will used to on test & training data
    return entries, stdScalar


def fillBuckets(entries: np.ndarray, K: int) -> typ.List[typ.List[np.ndarray]]:

    # *** connect a class to every instance that occurs in it ***
    # this will map a classId to a 2nd dictionary that will hold the number of instances in that class &
    # the instances
    # classToInstances[classId] = list[counter, instance1Values, instance2Values, ...]
    classToInstances = {}

    for e in entries:

        # get the class id from the entry
        cId = e[0]
        
        spam = []          # create an empty list
        spam.insert(0, 0)  # initialize counter to 0

        # if we already have an entry for that classId, append to it
        if classToInstances.get(cId):
            spam = classToInstances.get(cId)  # get the current list for this class
            spam[0] += 1                      # increment instance counter
            index = spam[0]                   # get the index that the counter says is next
            spam.insert(index, e[:])          # add the new instance at the new index

        # if this is the first time we've seen the class, create a new list for it
        else:
            spam = [1, e[:]]  # initialize the counter, add the new instance to the new list at index 1
            classToInstances[cId] = spam  # add the new list to the "master" dictionary

    # *** Now create a random permutation of the instances in a class, & put them in buckets ***
    buckets = [[]] * K  # create a list of empty lists that we will "deal" our "deck" of instances to
    index = 0         # this will be the index of the bucket we are "dealing" to

    for cId in classToInstances.keys():

        # *** for each class use cId to get it's instances ***
        # list is of the form [counter, instance1, instance2, ...]
        # where instanceN is a numpy array and where instanceN[0] is the classId & instanceN[1:] are the values
        instances = classToInstances[cId]

        # *** create a permutation of a class's instances *** #
        # permutation is of the form [instance?, instance?, ...]
        # where instance? is a list and where instance?[0] is the classId & instance?[1:] are the values
        permutation = np.random.permutation(instances[1:])  # but first remove the counter

        # *** assign instances to buckets *** #
        # loop over every instance in the class cId, in what is now a random order
        # p should be an instance?, which is a row from the input data
        for p in permutation:
            buckets[index].append(p)  # add the random instance to the bucket at index
            index = (index+1) % K     # increment index in a round robin style

    # *** The buckets are now full & together should contain every instance *** #
    return buckets


def main() -> None:

    tk.Tk().withdraw()  # prevent root window caused by Tkinter
    path = filedialog.askopenfilename()  # prompt user for file path

    # *** Parse the file into a numpy 2d array *** #
    entries = np.genfromtxt(path, delimiter=',', skip_header=1)
    
    # *** create a set of K buckets filled with our instances *** #
    K = 10  # set the K for k fold cross validation
    buckets = fillBuckets(entries, K)  # using the parsed data, fill the k buckets

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #
    # this will be used to select random indexes to serve as the testing/training data
    randomIndex = list(range(0, K-1))  # get a randomly ordered range of numbers up to K
    random.shuffle(randomIndex)        # this will be used to give a random index in the loop below

    accuracy = []  # this will store the details about the accuracy of our hypotheses

    oldR = 0                        # used to remember previous r in loop
    testingList = None                  # used to keep value for testing after a loop
    trainList = copy.deepcopy(buckets)  # make a copy of buckets so we don't override it

    # *** Divide them into training & test data K times ***
    # loop over all the random index values
    for r in randomIndex:  # len(r) = K so this will be done K times
    
        # *** Get the Training & Testing Data *** #
        
        # the Rth bucket becomes our testing data, everything else becomes training data
        # NOTE: the below is done in order to prevent accidental overwrites
        if testingList is None:             # if this is not the first time though the loop
            testingList = trainList.pop(r)  # then set train & testing
            oldR = r                        # save the current r value for then next loop
    
        else:                                    # if we've already been through the loop at least once
            trainList.insert(oldR, testingList)  # add testing back into train
            testingList = trainList.pop(r)       # then set train & testing
            oldR = r                             # save the current r value for then next loop
    
        # *** Flatten the Training Data *** #
        train = []              # currently training is a list of lists of lists because of the buckets.
        for lst in trainList:   # we can now remove the buckets by concatenating the lists of instance
            train += lst        # into one list of instances, flattening our data, & making it easier to work with

        # transform the training & testing data into numpy arrays & free the List vars to be reused
        train = np.array(train)          # turn training data into a numpy array
        testing = np.array(testingList)  # turn testing data into a numpy array
    
        # *** Normalize the Training Data *** #
        train, scalar = normalize(train, None)  # now normalize the training, and keep the scalar used
    
        # *** Train the CDFC Model *** #
        # CDFC_Hypothesis = cdfc(train)  # now that we have our train & test data create our hypothesis
    
        # *** Train the Learning Algorithm *** #
        # transform data using the CDFC model
        # train = CDFC_Hypothesis.transform(train)
    
        # format data for SciKit Learn
        # TODO change the below to use transformedData instead of train
        # create the label array Y (the target of our training)
        flat = np.ravel(train[:, :1])  # flatten the label list
        labels = np.array(flat)        # convert the label list to a numpy array
        # create the feature matrix X ()
        ftrs = np.array(train[:, 1:])  # get everything BUT the labels/ids
    
        # now that the data is formatted, run the learning algorithm
        # ? what's a good value for n_neighbors?
        # + model = KNeighborsClassifier(n_neighbors=3)     # Kth Nearest Neighbor Classifier
        # + model = DecisionTreeClassifier(random_state=0)  # Decision Tree Classifier
        model = GaussianNB()                            # Create a Gaussian Classifier (Naive Baye's)
        model.fit(ftrs, labels)                         # Train the model
    
        # *** Normalize the Testing Data *** #
        testing, scalar = normalize(testing, scalar)
    
        # *** Reduce the Testing Data Using the CDFC Model *** #
        # + testing = CDFC_Hypothesis.transform(testing)  # use the cdfc model to reduce the data's size
    
        # format data for SciKit Learn
        # create the label array Y (the target of our training)
        flat = np.ravel(testing[:, :1])  # flatten the label list
        trueLabels = np.array(flat)      # convert the label list to a numpy array
        # create the feature matrix X ()
        ftrs = np.array(testing[:, 1:])  # get everything BUT the labels/ids
    
        # compute accuracy
        labelPrediction = model.predict(ftrs)  # use model to predict labels
        # compute the accuracy score by comparing the actual labels with those predicted
        # ? currently the accuracy generated is the same every time, is this a bug?
        # + Kth Nearest Neighbor Classifier Gives --- 0.9350
        # + Decision Tree Gives --------------------- 0.9262
        # + Naive Baye's Gives ---------------------- 0.5660
        accuracy.append(accuracy_score(trueLabels, labelPrediction))
    
    # *** Report Accuracy *** #
    results = collect.namedtuple('results', ['standard_deviation', 'mean', 'median', 'max', 'min'])
    r = results(np.std(accuracy), np.mean(accuracy), np.median(accuracy), max(accuracy), min(accuracy))
    
    print(f'The standard deviation: {r.standard_deviation}',
          f'The mean: {r.mean}',
          f'The median: {r.median}',
          f'The max: {r.max}',
          f'The min: {r.min}', sep='\n')
    
    pprint(accuracy)


if __name__ == "__main__":

    main()
