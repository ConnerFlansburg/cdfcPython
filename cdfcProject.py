import copy
import sys
import tkinter as tk
import typing as typ
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfiglet import Figlet
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from cdfc import cdfc
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
# TODO add doc strings
# TODO add more unit tests

K: typ.Final[int] = 10  # set the K for k fold cross validation
ModelList = typ.Tuple[typ.List[float], typ.List[float], typ.List[float]]          # type hinting alias
ModelTypes = typ.Union[KNeighborsClassifier, GaussianNB, DecisionTreeClassifier]  # type hinting alias
ScalarsOut = typ.Tuple[np.ndarray, StandardScaler]
ScalarsIn = typ.Union[None, StandardScaler]

# console formatting strings
HDR = '*' * 6
SUCCESS = u' \u2713\n'
OVERWRITE = '\r' + HDR
SYSOUT = sys.stdout


# def __createPlot(df):
#     # *** Create the Plot *** #
#     outlierSymbol = dict(markerfacecolor='tab:red', marker='D')  # change the outliers to be red diamonds
#     medianSymbol = dict(linewidth=2.5, color='tab:green')        # change the medians to be green
#     meanSymbol = dict(linewidth=2.5, color='tab:blue')           # change the means to be blue
#     fig1 = df.boxplot(showmeans=True,                            # create boxplot, store it in fig1, & show the means
#                       meanline=True,                             # show mean as a mean line
#                       flierprops=outlierSymbol,                  # set the outlier properties
#                       medianprops=medianSymbol,                  # set the median properties
#                       meanprops=meanSymbol)                      # set the mean properties
#     fig1.set_title("Accuracy")                                   # set the title of the plot
#     fig1.set_xlabel("Accuracy Ratio")                            # set the label of the x-axis
#     fig1.set_ylabel("Model Type")                                # set the label of the y-axis
#     # *** Save the Plot as an Image *** #
#     # create a list of file formats that the plot may be saved as
#     images = [('Image Files', ['.jpeg', '.jpg', '.png', '.tiff', '.tif', '.bmp'])]
#     # ask the user where they want to save the plot
#     out = filedialog.asksaveasfilename(defaultextension='.png', filetypes=images)
#     # save the plot to the location provided by the user
#     plt.savefig(out)


def __discretization(data: np.ndarray) -> np.ndarray:
    
    for index, item in np.ndenumerate(data):

        if item < -0.5:
            data[index] = -1

        elif item > 0.5:
            data[index] = 1

        else:
            data[index] = 0

    return data


def __transform(entries: np.ndarray, scalarPassed: ScalarsIn) -> ScalarsOut:
    # NOTE: this should only ever be called if useNormalize is true!

    # remove the class IDs so they don't get normalized
    noIds = np.array(entries[:, 1:])

    # noIds is a list of instances without IDs of the form:
    # [ [value, value, value, ...],
    #   [value, value, value, ...], ...]

    # set stdScalar either using parameter in the case of testing data,
    # or making it in the case of training data
    # if we are dealing with training data, not testing data
    if scalarPassed is None:

        # *** transform the data *** #
        scalar = StandardScaler()        # create a standard scalar object (this will make the mean 0 & the sd 1)
        scalar.fit(noIds)                # fit the scalar's distribution using the data
        tData = scalar.transform(noIds)  # transform the data to fit the scalar's distribution
        
        # since we are using normalize, perform a discrete transformation on the data
        tData = __discretization(tData)

    # if we are dealing with testing data, we don't want to fit the scalar; just transform the fit
    else:

        # *** transform the data *** #
        scalar = scalarPassed            # used the passed scalar object
        tData = scalar.transform(noIds)  # transform the scalar using passed fit

        # since we are using normalize, perform a discrete transformation on the data
        tData = __discretization(tData)

    # *** add the IDs back on *** #
    entries[:, 1:] = tData  # this overwrites everything in entries BUT the ids

    # stdScalar - Standard Scalar, will used to on test & training data
    return entries, scalar


def __mapInstanceToClass(entries: np.ndarray) -> typ.Dict[int, typ.List[typ.Union[int, typ.List[float]]]]:
    # *** connect a class to every instance that occurs in it ***
    # this will map a classId to a 2nd dictionary that will hold the number of instances in that class &
    # the instances
    # classToInstances[classId] = list[counter, instance1[], instance2[], ...]
    classToInstances: typ.Dict[int, typ.List[typ.Union[int, typ.List[float]]]] = {}
    
    for e in entries:
        label = e[0]  # get the class id
        dictPosition: typ.List[typ.Union[int, typ.List[float]]] = classToInstances.get(label)  # get the position for the class in the dictionary
        
        # if we already have an entry for that classId, append to it
        if dictPosition:
            dictPosition[0] += 1            # increment instance counter
            idx = dictPosition[0]           # get the index that the counter says is next
            eggs: typ.List[typ.List[float]] = e.tolist()  # get the instances from the entry
            # noinspection PyTypeChecker
            dictPosition.insert(idx, eggs)  # at that index insert a list representing the instance
        
        # if this is the first time we've seen the class, create a new list for it
        else:
            # add a list, at index 0 put the counter, and at index 1 put a list containing the instance (values & label)
            classToInstances[label] = [1, e.tolist()]

    return classToInstances


def __getPermutation(instances: typ.List[typ.List[typ.Union[int, float]]], seed: int = None):
    # *** create a permutation of a class's instances *** #
    # permutation is a 2D numpy array, where each line is an instance
    rand = np.random.default_rng(seed)  # create a numpy random generator with/without seed
    permutation = rand.permutation(instances)  # shuffle the instances
    return permutation


def __dealToBuckets(classToInstances: typ.Dict[int, typ.List[typ.Union[int, typ.List[float]]]]) -> typ.List[typ.List[np.ndarray]]:
    # *** Now create a random permutation of the instances in a class, & put them in buckets ***
    buckets = [[] for _ in range(K)]  # create a list of empty lists that we will "deal" our "deck" of instances to
    index = 0  # this will be the index of the bucket we are "dealing" to
    seed = 498
    
    # classId is a int
    for classId in classToInstances.keys():
        
        # *** for each class use the class id to get it's instances ***
        # instances is of the form [instance1, instance2, ...]  (the counter has been removed)
        # where instanceN is a numpy array and where instanceN[0] is the classId & instanceN[1:] are the values
        permutation = __getPermutation(classToInstances[classId][1:], seed)

        # *** assign instances to buckets *** #
        # loop over every instance in the class classId, in what is now a random order
        # p will be a single row from permutation & so will be a 1D numpy array representing an instance
        for p in permutation:  # for every row in permutation
            buckets[index].append(p)  # add the random instance to the bucket at index
            index = (index + 1) % K  # increment index in a round robin style
    
    # *** The buckets are now full & together should contain every instance *** #
    return buckets


def __fillBuckets(entries: np.ndarray) -> typ.List[typ.List[np.ndarray]]:
    
    classToInstances = __mapInstanceToClass(entries)
    buckets = __dealToBuckets(classToInstances)

    return buckets


def __flattenTrainingData(trainList: typ.List[typ.List[np.ndarray]]) -> np.ndarray:
    train = []             # currently training is a list of lists of lists because of the buckets.
    for lst in trainList:  # we can now remove the buckets by concatenating the lists of instance
        train += lst       # into one list of instances, flattening our data, & making it easier to work with
    
    # transform the training & testing data into numpy arrays & free the List vars to be reused
    train = np.array(train)  # turn training data into a numpy array

    return train


def __formatForSciKit(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # create the label array Y (the target of our training)
    # from all rows, pick the 0th column
    flat = np.ravel(data[:, :1])  # get a list of all the labels as a list of lists & then flatten it
    labels = np.array(flat)       # convert the label list to a numpy array
    
    # create the feature matrix X ()
    ftrs = np.array(data[:, 1:])  # get everything BUT the labels/ids

    return ftrs, labels


def __buildModel(entries: np.ndarray, model: ModelTypes, useNormalize) -> typ.List[float]:
    
    # *** create a set of K buckets filled with our instances *** #
    buckets = __fillBuckets(entries)  # using the parsed data, fill the k buckets

    # *** Loop over our buckets K times, each time running creating a new hypothesis *** #
    oldR = 0                            # used to remember previous r in loop
    testingList = None                  # used to keep value for testing after a loop
    trainList = copy.deepcopy(buckets)  # make a copy of buckets so we don't override it
    accuracy = []                       # this will store the details about the accuracy of our hypotheses

    # *** Divide them into training & test data K times ***
    # loop over all the random index values
    for r in range(0, K):  # len(r) = K so this will be done K times
    
        # ********** Get the Training & Testing Data ********** #
        # the Rth bucket becomes our testing data, everything else becomes training data
        # this is done in order to prevent accidental overwrites
        if testingList is None:                  # if this is not the first time though the loop
            testingList = trainList.pop(r)       # then set train & testing
            oldR = r                             # save the current r value for then next loop
            
        else:                                    # if we've already been through the loop at least once
            trainList.insert(oldR, testingList)  # add testing back into train
            testingList = trainList.pop(r)       # then set train & testing
            oldR = r                             # save the current r value for then next loop
    
        # ********** Flatten the Training Data ********** #
        train = __flattenTrainingData(trainList)  # remove buckets to create a single pool

        testing = np.array(testingList)  # turn testing data into a numpy array, testing doesn't need to be flattened

        # ********** 3A Normalize the Training Data (if useNormalize is True) ********** #
        # this is also fitting the data to a distribution
        if useNormalize:
            train, scalar = __transform(train, None)  # now normalize the training, and keep the scalar used
        else:  # this case is not strictly needed, but helps the editor/IDE not throw errors
            scalar = None
    
        # ********** 3B Train the CDFC Model & Transform the Training Data using It ********** #
        SYSOUT.write(HDR + ' Training CDFC ......')                          # print \tab * Training CDFC ....
        SYSOUT.flush()
        CDFC_Hypothesis = cdfc(train)        # now that we have our train & test data create our hypothesis
        SYSOUT.write(OVERWRITE + ' CDFC Trained '.ljust(50, '-') + SUCCESS)  # replace starting with complete
        
        SYSOUT.write(HDR + ' Transforming Training Data ......')                 # print
        SYSOUT.flush()
        train = CDFC_Hypothesis.transform(train)                                 # transform data using the CDFC model
        SYSOUT.write(OVERWRITE + ' Data Transformed '.ljust(50, '-') + SUCCESS)  # replace starting with complete

        # ********** 3C Train the Learning Algorithm ********** #
        # format data for SciKit Learn
        ftrs, labels = __formatForSciKit(train)
        
        # now that the data is formatted, run the learning algorithm.
        # if useNormalize is True the data has been transformed to fit
        # a scalar & gone through discretization. If False, it has not
        model.fit(ftrs, labels)
    
        # ********** 3D.1 Normalize the Testing Data ********** #
        # if use
        if useNormalize:
            testing, scalar = __transform(testing, scalar)
    
        # ********** 3D.2 Reduce the Testing Data Using the CDFC Model ********** #
        # + testing = CDFC_Hypothesis.transform(testing)  # use the cdfc model to reduce the data's size
    
        # format testing data for SciKit Learn
        ftrs, trueLabels = __formatForSciKit(testing)
    
        # ********** 3D.3 Feed the Training Data into the Model & get Accuracy ********** #
        labelPrediction = model.predict(ftrs)  # use model to predict labels
        # compute the accuracy score by comparing the actual labels with those predicted
        accuracy.append(accuracy_score(trueLabels, labelPrediction))
    
    # *** Return Accuracy *** #
    return accuracy


def __runSciKitModels(entries: np.ndarray, useNormalize: bool) -> ModelList:
    # NOTE adding more models requires updating the Models type at top of file
    # hdr, overWrite, & success are just used to format string for the console
    # accuracy is a float list, each value is the accuracy for a single run

    SYSOUT.write("\nBuilding models...\n")  # update user
    
    # ************ Kth Nearest Neighbor Classifier ************ #
    SYSOUT.write(HDR + ' KNN model starting ......')                        # print \tab * KNN model starting......
    SYSOUT.flush()                                                          # since there's no newline push buffer to console
    knnAccuracy: typ.List[float] = __buildModel(entries, KNeighborsClassifier(n_neighbors=3), useNormalize)  # build the model
    SYSOUT.write(OVERWRITE+' KNN model completed '.ljust(50, '-')+SUCCESS)  # replace starting with complete

    # ************ Decision Tree Classifier ************ #
    SYSOUT.write(HDR + ' Decision Tree model starting ......')              # print \tab * dt model starting......
    SYSOUT.flush()                                                          # since there's no newline push buffer to console
    dtAccuracy: typ.List[float] = __buildModel(entries, DecisionTreeClassifier(random_state=0), useNormalize)  # build the model
    SYSOUT.write(OVERWRITE +                                                # replace starting with complete
                 ' Decision Tree model built '.ljust(50, '-') + SUCCESS)

    # ************ Gaussian Classifier (Naive Bayes) ************ #
    SYSOUT.write(HDR + ' Naive Bayes model starting ......')                # print \tab * nb model starting......
    SYSOUT.flush()                                                          # since there's no newline push buffer to console
    nbAccuracy: typ.List[float] = __buildModel(entries, GaussianNB(), useNormalize)       # build the model
    SYSOUT.write(OVERWRITE +                                                # replace starting with complete
                 ' Naive Bayes model built '.ljust(50, '-') + SUCCESS)
    
    SYSOUT.write("Models built\n\n")  # update user

    return knnAccuracy, dtAccuracy, nbAccuracy


def __buildAccuracyFrame(modelsTuple: ModelList) -> pd.DataFrame:

    SYSOUT.write("Creating accuracy dataframe...\n")  # update user
    
    # get the models from the tuple
    knnAccuracy = modelsTuple[0]
    dtAccuracy = modelsTuple[1]
    nbAccuracy = modelsTuple[2]

    SYSOUT.write(HDR + ' Creating dictionary ......')
    # use accuracy data to create a dictionary. This will become the frame
    accuracyList = {'KNN': knnAccuracy, 'Decision Tree': dtAccuracy, 'Naive Bayes': nbAccuracy}
    SYSOUT.write(OVERWRITE + ' Dictionary created '.ljust(50, '-') + SUCCESS)
    
    SYSOUT.write(HDR + ' Creating labels ......')
    # this will create labels for each instance ("fold") of the model, of the form "Fold i"
    rowList = [["Fold {}".format(i) for i in range(1, K + 1)]]
    SYSOUT.write(OVERWRITE + ' Labels created successfully '.ljust(50, '-') + SUCCESS)
    
    SYSOUT.write(HDR + ' Creating dataframe ......')
    # create the dataframe. It will be used both by the latex & the plot exporter
    df = pd.DataFrame(accuracyList, columns=['KNN', 'Decision Tree', 'Naive Bayes'], index=rowList)
    SYSOUT.write(OVERWRITE + ' Dataframe created successfully '.ljust(50, '-') + SUCCESS)

    SYSOUT.write('Accuracy Dataframe created without error\n')  # update user
    
    return df


def __accuracyFrameToLatex(modelsTuple: typ.Tuple[typ.List[float], typ.List[float], typ.List[float]], df: pd.DataFrame) -> pd.DataFrame:
    
    SYSOUT.write("\nConverting frame to LaTeX...\n")  # update user
    SYSOUT.write(HDR + ' Transposing dataframe')      # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    # transposing passes a copy, so as to avoid issues with plot (should we want it)
    frame = df.transpose()  # update user
    
    SYSOUT.write(OVERWRITE + ' Dataframe transposed '.ljust(50, '-') + SUCCESS)
    
    # get the models from the tuple
    knnAccuracy = modelsTuple[0]
    dtAccuracy = modelsTuple[1]
    nbAccuracy = modelsTuple[2]
    
    SYSOUT.write(HDR + ' Calculating statistics...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    mn = [min(knnAccuracy), min(dtAccuracy), min(nbAccuracy)]                        # create the new min,
    median = [np.median(knnAccuracy), np.median(dtAccuracy), np.median(nbAccuracy)]  # median,
    mean = [np.mean(knnAccuracy), np.mean(dtAccuracy), np.mean(nbAccuracy)]          # mean,
    mx = [max(knnAccuracy), max(dtAccuracy), max(nbAccuracy)]                        # & max columns
    
    SYSOUT.write(OVERWRITE + ' Statistics calculated '.ljust(50, '-') + SUCCESS)     # update user
    
    SYSOUT.write(HDR + ' Adding statistics to dataframe...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    frame['min'] = mn         # add the min,
    frame['median'] = median  # median,
    frame['mean'] = mean      # mean,
    frame['max'] = mx         # & max columns to the dataframe
    
    SYSOUT.write(OVERWRITE + ' Statistics added to dataframe '.ljust(50, '-') + SUCCESS)  # update user
    SYSOUT.write(HDR + ' Converting dataframe to percentages...')                         # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    frame *= 100  # turn the decimal into a percent
    frame = frame.round(1)  # round to 1 decimal place
    
    SYSOUT.write(OVERWRITE + ' Converted dataframe values to percentages '.ljust(50, '-') + SUCCESS)  # update user
    
    return frame


def main() -> None:
    SYSOUT.write(Figlet(font='larry3d').renderText('C D F C'))  # formatted start up message
    SYSOUT.write("Program Initialized Successfully\n")
    
    parent = tk.Tk()            # prevent root window caused by Tkinter
    parent.overrideredirect(1)  # Avoid it appearing and then disappearing quickly
    parent.withdraw()           # Hide the window

    SYSOUT.write('\nGetting File...\n')
    try:
        inPath = Path(filedialog.askopenfilename(parent=parent))  # prompt user for file path
        SYSOUT.write(f"{HDR} File {inPath.name} selected......")
    except PermissionError:
        sys.stderr.write(f"\n{HDR} Permission Denied, or No File was Selected\nExiting......")  # exit gracefully
        sys.exit("Could not access file/No file was selected")

    SYSOUT.write(f"{OVERWRITE} File {inPath.name} found ".ljust(57, '-') + SUCCESS)

    useNormalize = messagebox.askyesno('CDFC - Transformations', 'Do you want to transform the data before using it?', parent=parent)  # Yes / No
    
    # *** Parse the file into a numpy 2d array *** #
    SYSOUT.write(HDR + ' Parsing .csv file...')  # update user
    entries = np.genfromtxt(inPath, delimiter=',', skip_header=1)  # + this line is used to read .csv files
    SYSOUT.write(OVERWRITE + ' .csv file parsed successfully '.ljust(50, '-') + SUCCESS)  # update user
    SYSOUT.write('\rFile Found & Loaded Successfully\n')  # update user
    
    # *** Build the Models *** #
    modelsTuple = __runSciKitModels(entries, useNormalize)  # knnAccuracy, dtAccuracy, nbAccuracy

    # *** Create a Dataframe that Combines the Accuracy of all the Models *** #
    accuracyFrame = __buildAccuracyFrame(modelsTuple)

    # *** Modify the Dataframe to Match our LaTeX File *** #
    latexFrame = __accuracyFrameToLatex(modelsTuple, accuracyFrame)
    
    # *** Export the Dataframe as a LaTeX File *** #
    SYSOUT.write(HDR + ' Exporting LaTeX dataframe...')
    
    # set the output file path
    if useNormalize:                       # if we are using the transformations
        stm = inPath.stem + 'Transformed'  # add 'Transformed' to the file name
    
    else:                          # if we are not transforming the data
        stm = inPath.stem + 'Raw'  # add 'Raw' to the file name
    
    outPath = Path.cwd() / 'data' / 'outputs' / (stm + '.tex')  # create the file path

    with open(outPath, "w") as texFile:             # open the selected file
        print(latexFrame.to_latex(), file=texFile)  # & write dataframe to it, converting it to latex
    texFile.close()                                 # close the file
    
    SYSOUT.write(OVERWRITE + ' Export Successful '.ljust(50, '-') + SUCCESS)
    SYSOUT.write('Dataframe converted to LaTeX & Exported\n')
    
    # *** Exit *** #
    SYSOUT.write('\nExiting')
    sys.exit(0)  # close program
    

if __name__ == "__main__":

    main()
