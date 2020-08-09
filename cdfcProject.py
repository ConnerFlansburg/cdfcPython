import random
import numpy as np
import tkinter as tk
from tkinter import filedialog
from cdfc import cdfc
from sklearn.preprocessing import StandardScaler


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
        # TODO use scalar to transform entries

    # ? does this work for the training data (when entries is a list of lists)
    normalizedData = discretization(entries)  # discrete transformation

    # TODO create an empty numpy array so we can append
    finalData = np.empty()
    finalData = np.append(finalData, np.array(entries[:, 0]), axis=1)
    finalData = np.append(finalData, np.array(normalizedData), axis=0)

    # stdScalar - Standard Scalar, will used to on test & training data
    return finalData, stdScalar


def fillBuckets(entries, K):

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
            # add the new instance at the new index
            spam[spam[0]] = e[:]

        # if this is the first time we've seen the class, create a new list for it
        else:
            # initialize the counter, add the new instance to the new list at index 1
            spam = [1, e[:]]
            # add the new list to the "master" dictionary
            classToInstances[cId] = spam

    # *** Now create a random permutation of the instances in a class, & put them in buckets ***

    buckets = [None] * K  # this list will be what we "deal" our "deck" of instances to
    index = 0  # this will be the index of the bucket we are "dealing" to

    for cId in classToInstances.keys():

        # *** get a permutation of the instances that are in the class given by classId ***

        # get the list for the class
        # list is of the form [counter, instance1, instance2, ...]
        # where instanceN is a list where instanceN[0] is the classId & instanceN[1:] are the values
        instances = classToInstances[cId]

        # get a permutation of the instances for the class, but first remove the counter
        # permutation is of the form [instance?, instance?, ...]
        # where instance? is a list where instance?[0] is the classId & instance?[1:] are the values
        permutation = np.random.permutation(instances[1:])

        # *** assign instances to buckets ***

        # loop over every instance in the class cId, in what is now a random order
        # p should be an instance?, which is a row from the input data
        for p in permutation:
            buckets[index] = p  # add the random instance to the bucket at index p
            index = (index+1) % K  # increment index in a round robin style

    # *** The buckets are now full & together should contain every instance ***
    return buckets


def main():

    # prevent root window caused by Tkinter
    root = tk.Tk()
    root.withdraw()

    # prompt user for file path
    path = filedialog.askopenfilename()

    # *** Parse the file into a numpy 2d array *** #
    entries = np.genfromtxt(path, delimiter=',')

    # *** create a set of K buckets filled with our instances *** #
    K = 10  # set the K for k fold cross validation
    # using the parsed data, fill the k buckets
    buckets = fillBuckets(entries, K)

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #

    # this will be used to select random indexes to serve as the testing/training data
    randomIndex = list(range(K))  # get a randomly ordered range of numbers up to K
    random.shuffle(randomIndex)  # this will be used to give a random index in the loop below

    # TODO what data structure/form is best for accuracy
    # this will store the details about the accuracy of our hypotheses
    accuracy = []

    # *** Divide them into training & test data K times ***

    # loop over all the random index values
    for r in randomIndex:  # len(r) = K so this will be done K times

        # take only the Rth bucket, this will be our testing data
        testing = buckets[r]

        # trim the Rth bucket so train doesn't contain the testing instance
        trimmed = buckets  # needed since pop works in place
        # take all the buckets except R
        train = trimmed.pop(r)

        # now normalize the training, and keep the scalar used
        train, scalar = normalize(train)

        # now that we have our train & test data create our hypothesis
        hypothesis = cdfc(train)

        testing = normalize(testing, scalar)
        # now that we have created our hypothesis, test it
        accuracy.append(hypothesis.test(testing))

    # TODO report accuracy's average, standard deviation, & extremes
    # TODO implement other models from scikit


if __name__ == "__main__":

    main()
