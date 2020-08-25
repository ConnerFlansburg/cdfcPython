import copy
import sys
import tkinter as tk
import typing as typ
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfiglet import Figlet
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# from cdfc import cdfc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

'''
                                       CSVs should be of the form

           |  label/id   |   attribute 1   |   attribute 2   |   attribute 3   |   attribute 4   | ... |   attribute k   |
--------------------------------------------------------------------------------------------------------------------------
instance 1 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
instance 2 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
instance 3 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
instance 4 | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------
    ...    |    ...      |      ...        |       ...       |       ...       |       ...       | ... |       ...       |
--------------------------------------------------------------------------------------------------------------------------
instance n | class value | attribute value | attribute value | attribute value | attribute value | ... | attribute value |
--------------------------------------------------------------------------------------------------------------------------

'''


# * Next Steps
# TODO get CDFC working & use it to reduce data

K: typ.Final[int] = 10  # set the K for k fold cross validation


def createPlot(df):
    # *** Create the Plot *** #
    outlierSymbol = dict(markerfacecolor='tab:red', marker='D')  # change the outliers to be red diamonds
    medianSymbol = dict(linewidth=2.5, color='tab:green')        # change the medians to be green
    meanlineSymbol = dict(linewidth=2.5, color='tab:blue')       # change the means to be blue
    fig1 = df.boxplot(showmeans=True,                            # create boxplot, store it in fig1, & show the means
                      meanline=True,                             # show mean as a mean line
                      flierprops=outlierSymbol,                  # set the outlier properties
                      medianprops=medianSymbol,                  # set the median properties
                      meanprops=meanlineSymbol)                  # set the mean properties
    fig1.set_title("Accuracy")                                   # set the title of the plot
    fig1.set_xlabel("Accuracy Ratio")                            # set the label of the x-axis
    fig1.set_ylabel("Model Type")                                # set the label of the y-axis
    # *** Save the Plot as an Image *** #
    # create a list of file formats that the plot may be saved as
    images = [('Image Files', ['.jpeg', '.jpg', '.png', '.tiff', '.tif', '.bmp'])]
    # ask the user where they want to save the plot
    out = filedialog.asksaveasfilename(defaultextension='.png', filetypes=images)
    # save the plot to the location provided by the user
    plt.savefig(out)


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


def __mapInstanceToClass(entries: np.ndarray):
    # *** connect a class to every instance that occurs in it ***
    # this will map a classId to a 2nd dictionary that will hold the number of instances in that class &
    # the instances
    # classToInstances[classId] = list[counter, instance1[], instance2[], ...]
    classToInstances: typ.Dict[int, typ.List[typ.Union[int, typ.List[typ.Union[int, float]]]]] = {}
    
    for e in entries:
        
        label = e[0]  # get the class id from the entry
        
        # if we already have an entry for that classId, append to it
        if classToInstances.get(label):
            classToInstances.get(label)[0] += 1  # increment instance counter
            idx = classToInstances.get(label)[0]  # get the index that the counter says is next
            classToInstances.get(label).insert(idx, e[:])  # at that index insert a list representing the instance
        
        # if this is the first time we've seen the class, create a new list for it
        else:
            # add a list, at index 0 put the counter, and at index 1 put a list containing the instance (values & label)
            classToInstances[label] = [1, e[:]]
    
    return classToInstances


def dealToBuckets(classToInstances):
    # *** Now create a random permutation of the instances in a class, & put them in buckets ***
    buckets = [[] for _ in range(K)]  # create a list of empty lists that we will "deal" our "deck" of instances to
    index = 0  # this will be the index of the bucket we are "dealing" to
    
    for classId in classToInstances.keys():
        
        # *** for each class use the class id to get it's instances ***
        # instances is of the form [instance1, instance2, ...]  (the counter has been removed)
        # where instanceN is a numpy array and where instanceN[0] is the classId & instanceN[1:] are the values
        instances: typ.List[typ.List[typ.Union[int, float]]] = classToInstances[classId][1:]
        
        # *** create a permutation of a class's instances *** #
        # permutation is a 2D numpy array, where each line is an instance
        # seed = None
        seed = 498
        rand = np.random.default_rng(seed)  # create a numpy random generator with/without seed
        permutation = rand.permutation(instances)  # shuffle the instances
        
        # *** assign instances to buckets *** #
        # loop over every instance in the class classId, in what is now a random order
        # p will be a single row from permutation & so will be a 1D numpy array representing an instance
        for p in permutation:  # for every row in permutation
            buckets[index].append(p)  # add the random instance to the bucket at index
            index = (index + 1) % K  # increment index in a round robin style
    
    # *** The buckets are now full & together should contain every instance *** #
    return buckets


def fillBuckets(entries: np.ndarray) -> typ.List[typ.List[np.ndarray]]:
    
    classToInstances = __mapInstanceToClass(entries)
    buckets = dealToBuckets(classToInstances)
    
    return buckets


def buildModel(entries, model) -> typ.List[float]:
    
    # *** create a set of K buckets filled with our instances *** #
    buckets = fillBuckets(entries)  # using the parsed data, fill the k buckets

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #
    oldR = 0                            # used to remember previous r in loop
    testingList = None                  # used to keep value for testing after a loop
    trainList = copy.deepcopy(buckets)  # make a copy of buckets so we don't override it
    accuracy = []                       # this will store the details about the accuracy of our hypotheses

    # *** Divide them into training & test data K times ***
    # loop over all the random index values
    for r in range(0, K):  # len(r) = K so this will be done K times
    
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
    
    # formatting strings
    hdr = '*' * 6
    success = u' \u2713\n'
    overWrite = '\r' + hdr

    sysOut = sys.stdout
    sysOut.write(Figlet(font='larry3d').renderText('C D F C'))
    sysOut.write("Program initialized " + success)
    
    tk.Tk().withdraw()                                          # prevent root window caused by Tkinter
    try:
        inPath = Path(filedialog.askopenfilename())             # prompt user for file path
    except PermissionError:
        sys.stderr.write("Permission Denied\nExiting...")       # exit gracefully
        sys.exit("Could not access file/No file was selected")
    
    # *** Parse the file into a numpy 2d array *** #
    entries = np.genfromtxt(inPath, delimiter=',', skip_header=1)  # + this line is used to read .csv files
    sysOut.write('Parsed .csv file ' + success)
    
    # accuracy is a float list, each value is the accuracy for a single run
    sysOut.write("\nBuilding models...\n")
    sysOut.write(hdr + ' KNN model starting ......')                          # print \tab * KNN model starting......
    sysOut.flush()                                                            # since there's no newline push buffer to console
    knnAccuracy = buildModel(entries, KNeighborsClassifier(n_neighbors=3))    # * Kth Nearest Neighbor Classifier
    sysOut.write(overWrite +                                                  # replace starting with complete
                 ' KNN model completed '.ljust(50, '-')+success)
    
    sysOut.write(hdr + ' Decision Tree model starting ......')                # print \tab * dt model starting......
    sysOut.flush()                                                            # since there's no newline push buffer to console
    dtAccuracy = buildModel(entries, DecisionTreeClassifier(random_state=0))  # * Decision Tree Classifier
    sysOut.write(overWrite +                                                  # replace starting with complete
                 ' Decision Tree model built '.ljust(50, '-')+success)
    
    sysOut.write(hdr + ' Naive Bayes model starting ......')                  # print \tab * nb model starting......
    sysOut.flush()                                                            # since there's no newline push buffer to console
    nbAccuracy = buildModel(entries, GaussianNB())                            # * Gaussian Classifier (Naive Baye's)
    sysOut.write(overWrite +                                                  # replace starting with complete
                 ' Naive Bayes model built '.ljust(50, '-')+success)
    sysOut.write("Models built\n\n")
    
    # *** Create a Dataframe that Combines the Accuracy of all the Models *** #
    sysOut.write("Creating accuracy dataframe...")
    sysOut.flush()
    
    # use accuracy data to create a dictionary. This will become the frame
    accuracyList = {'KNN': knnAccuracy, 'Decision Tree': dtAccuracy, 'Naive Bayes': nbAccuracy}
    # this will create labels for each instance ("fold") of the model, of the form "Fold i"
    rowList = [["Fold {}".format(i) for i in range(1, K+1)]]
    # create the dataframe. It will be used both by the latex & the plot exporter
    df = pd.DataFrame(accuracyList, columns=['KNN', 'Decision Tree', 'Naive Bayes'], index=rowList)
    
    sysOut.write('\rAccuracy Dataframe created without error'+success)

    # *** Modify the Dataframe to Match our LaTeX File *** #
    
    sysOut.write("\nConverting frame to LaTeX...\n")
    
    sysOut.write(hdr + ' Transposing dataframe')
    sysOut.flush()
    latexFrame = df.transpose()  # transposing passes a copy, so as to avoid issues with plot (should we want it)
    sysOut.write(overWrite + ' Dataframe transposed '.ljust(50, '-')+success)

    sysOut.write(hdr + ' Calculating statistics...')
    sysOut.flush()
    mn = [min(knnAccuracy), min(dtAccuracy), min(nbAccuracy)]                        # create the new min,
    median = [np.median(knnAccuracy), np.median(dtAccuracy), np.median(nbAccuracy)]  # median,
    mean = [np.mean(knnAccuracy), np.mean(dtAccuracy), np.mean(nbAccuracy)]          # mean,
    mx = [max(knnAccuracy), max(dtAccuracy), max(nbAccuracy)]                        # & max columns
    sysOut.write(overWrite + ' Statistics calculated '.ljust(50, '-')+success)

    sysOut.write(hdr + ' Adding statistics to dataframe...')
    sysOut.flush()
    latexFrame['min'] = mn             # add the min,
    latexFrame['median'] = median      # median,
    latexFrame['mean'] = mean          # mean,
    latexFrame['max'] = mx             # & max columns to the dataframe
    sysOut.write(overWrite + ' Statistics added to dataframe '.ljust(50, '-')+success)
    
    sysOut.write(hdr + ' Converting dataframe to percentages...')
    sysOut.flush()
    latexFrame *= 100                  # turn the decimal into a percent
    latexFrame = latexFrame.round(1)   # round to 1 decimal place
    sysOut.write(overWrite + ' Converted dataframe values to percentages '.ljust(50, '-')+success)

    sysOut.write("Dataframe converted to LaTeX\n")

    # *** Export the Dataframe as a LaTeX File *** #
    sysOut.write('\nExporting LaTeX dataframe...')
    
    # ? BUG for some reason filedialog will not open. Path works however
    # outPath = filedialog.asksaveasfilename(defaultextension='.tex')  # ask the user where they want to save the latex output
    # save the .tex file under the same name as the input but in the outputs folder
    outPath = Path.cwd() / 'data' / 'outputs' / (inPath.stem + '.tex')

    with open(outPath, "w") as texFile:             # open the selected file
        print(latexFrame.to_latex(), file=texFile)  # & write dataframe to it, converting it to latex


if __name__ == "__main__":

    main()
