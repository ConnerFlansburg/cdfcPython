import copy
import typing as typ
import numpy as np
import tkinter as tk
import pandas as pd
from tkinter import filedialog
# from cdfc import cdfc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# * Next Steps
# TODO get CDFC working & use it to reduce data


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
    # classToInstances[classId] = list[counter, instance1[], instance2[], ...]
    classToInstances: typ.Dict[int, typ.List[typ.Union[int, typ.List[typ.Union[int, float]]]]] = {}

    for e in entries:

        label = e[0]  # get the class id from the entry

        # if we already have an entry for that classId, append to it
        if classToInstances.get(label):
            classToInstances.get(label)[0] += 1            # increment instance counter
            idx = classToInstances.get(label)[0]           # get the index that the counter says is next
            classToInstances.get(label).insert(idx, e[:])  # at that index insert a list representing the instance
            
        # if this is the first time we've seen the class, create a new list for it
        else:
            # add a list, at index 0 put the counter, and at index 1 put a list containing the instance (values & label)
            classToInstances[label] = [1, e[:]]

    # *** Now create a random permutation of the instances in a class, & put them in buckets ***
    buckets = [[] for _ in range(K)]  # create a list of empty lists that we will "deal" our "deck" of instances to
    index = 0                         # this will be the index of the bucket we are "dealing" to

    for classId in classToInstances.keys():

        # *** for each class use the class id to get it's instances ***
        # instances is of the form [instance1, instance2, ...]  (the counter has been removed)
        # where instanceN is a numpy array and where instanceN[0] is the classId & instanceN[1:] are the values
        instances: typ.List[typ.List[typ.Union[int, float]]] = classToInstances[classId][1:]

        # *** create a permutation of a class's instances *** #
        # permutation is a 2D numpy array, where each line is an instance
        permutation = np.random.permutation(instances)  # this will reorder the instances but not their values

        # *** assign instances to buckets *** #
        # loop over every instance in the class classId, in what is now a random order
        # p will be a single row from permutation & so will be a 1D numpy array representing an instance
        for p in permutation:         # for every row in permutation
            buckets[index].append(p)  # add the random instance to the bucket at index
            index = (index+1) % K     # increment index in a round robin style

    # *** The buckets are now full & together should contain every instance *** #
    return buckets


def buildModel(entries, model) -> typ.List[float]:

    # *** create a set of K buckets filled with our instances *** #
    K = 10  # set the K for k fold cross validation
    buckets = fillBuckets(entries, K)  # using the parsed data, fill the k buckets

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #
    oldR = 0                            # used to remember previous r in loop
    testingList = None                  # used to keep value for testing after a loop
    trainList = copy.deepcopy(buckets)  # make a copy of buckets so we don't override it
    accuracy = []                       # this will store the details about the accuracy of our hypotheses

    # *** Divide them into training & test data K times ***
    # loop over all the random index values
    for r in range(0, K-1):  # len(r) = K so this will be done K times
    
        # *** Get the Training & Testing Data *** #
        # the Rth bucket becomes our testing data, everything else becomes training data
        # this is done in order to prevent accidental overwrites
        if testingList is None:                  # if this is not the first time though the loop
            testingList = trainList.pop(r)       # then set train & testing
            oldR = r                             # save the current r value for then next loop
            
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
        testing = np.array(testingList)  # turn testing data into a numpy array, testing doesn't need to be flattened
    
        # *** 3A Normalize the Training Data *** #
        train, scalar = normalize(train, None)  # now normalize the training, and keep the scalar used
    
        # *** 3B Train the CDFC Model & Transform the Training Data using It *** #
        # CDFC_Hypothesis = cdfc(train)  # now that we have our train & test data create our hypothesis
        # train = CDFC_Hypothesis.transform(train)  # transform data using the CDFC model

        # *** 3C Train the Learning Algorithm *** #
        # format data for SciKit Learn
        # TODO change the below to use transformedData instead of train
        # create the label array Y (the target of our training)
        flat = np.ravel(train[:, :1])  # get a list of all the labels as a list of lists & then flatten it
        labels = np.array(flat)        # convert the label list to a numpy array
        # create the feature matrix X ()
        ftrs = np.array(train[:, 1:])  # get everything BUT the labels/ids
    
        # now that the data is formatted, run the learning algorithm
        model.fit(ftrs, labels)                         # Train the model
    
        # *** 3D.1 Normalize the Testing Data *** #
        testing, scalar = normalize(testing, scalar)
    
        # *** 3D.2 Reduce the Testing Data Using the CDFC Model *** #
        # + testing = CDFC_Hypothesis.transform(testing)  # use the cdfc model to reduce the data's size
    
        # format data for SciKit Learn
        # create the label array Y (the target of our training)
        flat = np.ravel(testing[:, :1])  # get a list of all the labels as a list of lists & then flatten it
        trueLabels = np.array(flat)      # convert the label list to a numpy array
        # create the feature matrix X ()
        ftrs = np.array(testing[:, 1:])  # get everything BUT the labels/ids
    
        # *** 3D.3 Feed the Training Data into the Model & get Accuracy *** #
        labelPrediction = model.predict(ftrs)  # use model to predict labels
        # compute the accuracy score by comparing the actual labels with those predicted
        accuracy.append(accuracy_score(trueLabels, labelPrediction))
    
    # *** Return Accuracy *** #
    return accuracy


def main() -> None:
    
    tk.Tk().withdraw()                   # prevent root window caused by Tkinter
    path = filedialog.askopenfilename()  # prompt user for file path
    
    # *** Parse the file into a numpy 2d array *** #
    entries = np.genfromtxt(path, delimiter=',', skip_header=1)  # + this line is used to read .csv files

    # accuracy is a float list, each value is the accuracy for a single run
    knnAccuracy = buildModel(entries, KNeighborsClassifier(n_neighbors=3))    # Kth Nearest Neighbor Classifier
    dtAccuracy = buildModel(entries, DecisionTreeClassifier(random_state=0))  # Decision Tree Classifier
    nbAccuracy = buildModel(entries, GaussianNB())                            # Create a Gaussian Classifier (Naive Baye's)
    
    # use accuracy to create a dataframe
    accuracyList = {'KNN': knnAccuracy, 'Decision Tree': dtAccuracy, 'Naive Bayes': nbAccuracy}
    df = pd.DataFrame(accuracyList, columns=['KNN', 'Decision Tree', 'Naive Bayes'])
    
    # export the data frame to a csv
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    df.to_csv(export_file_path, index=False, encoding='utf-8')


if __name__ == "__main__":

    main()
