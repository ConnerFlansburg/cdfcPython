import sys
import typing as typ
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# console formatting strings
HDR = '*' * 6
SUCCESS = u' \u2713\n'
OVERWRITE = '\r' + HDR
SYSOUT = sys.stdout

ModelList = typ.Tuple[typ.List[float], typ.List[float], typ.List[float]]  # type hinting alias


def __createPlot(df):
    # *** Create the Plot *** #
    outlierSymbol = dict(markerfacecolor='tab:red', marker='D')  # change the outliers to be red diamonds
    medianSymbol = dict(linewidth=2.5, color='tab:green')        # change the medians to be green
    meanSymbol = dict(linewidth=2.5, color='tab:blue')           # change the means to be blue
    fig1 = df.boxplot(showmeans=True,                            # create boxplot, store it in fig1, & show the means
                      meanline=True,                             # show mean as a mean line
                      flierprops=outlierSymbol,                  # set the outlier properties
                      medianprops=medianSymbol,                  # set the median properties
                      meanprops=meanSymbol)                      # set the mean properties
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


def __flattenTrainingData(trainList: typ.List[typ.List[np.ndarray]]) -> np.ndarray:
    train = []  # currently training is a list of lists of lists because of the buckets.
    for lst in trainList:  # we can now remove the buckets by concatenating the lists of instance
        train += lst  # into one list of instances, flattening our data, & making it easier to work with
    
    # transform the training & testing data into numpy arrays & free the List vars to be reused
    train = np.array(train)  # turn training data into a numpy array
    
    return train


def __formatForSciKit(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # create the label array Y (the target of our training)
    # from all rows, pick the 0th column
    flat = np.ravel(data[:, :1])  # get a list of all the labels as a list of lists & then flatten it
    labels = np.array(flat)  # convert the label list to a numpy array
    
    # create the feature matrix X ()
    ftrs = np.array(data[:, 1:])  # get everything BUT the labels/ids
    
    return ftrs, labels


def __buildAccuracyFrame(modelsTuple: ModelList, K) -> pd.DataFrame:
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


def __accuracyFrameToLatex(modelsTuple: typ.Tuple[typ.List[float], typ.List[float], typ.List[float]],
                           df: pd.DataFrame) -> pd.DataFrame:
    SYSOUT.write("\nConverting frame to LaTeX...\n")  # update user
    SYSOUT.write(HDR + ' Transposing dataframe')  # update user
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
    
    mn = [min(knnAccuracy), min(dtAccuracy), min(nbAccuracy)]  # create the new min,
    median = [np.median(knnAccuracy), np.median(dtAccuracy), np.median(nbAccuracy)]  # median,
    mean = [np.mean(knnAccuracy), np.mean(dtAccuracy), np.mean(nbAccuracy)]  # mean,
    mx = [max(knnAccuracy), max(dtAccuracy), max(nbAccuracy)]  # & max columns
    
    SYSOUT.write(OVERWRITE + ' Statistics calculated '.ljust(50, '-') + SUCCESS)  # update user
    
    SYSOUT.write(HDR + ' Adding statistics to dataframe...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    frame['min'] = mn  # add the min,
    frame['median'] = median  # median,
    frame['mean'] = mean  # mean,
    frame['max'] = mx  # & max columns to the dataframe
    
    SYSOUT.write(OVERWRITE + ' Statistics added to dataframe '.ljust(50, '-') + SUCCESS)  # update user
    SYSOUT.write(HDR + ' Converting dataframe to percentages...')  # update user
    SYSOUT.flush()  # no newline, so buffer must be flushed to console
    
    frame *= 100  # turn the decimal into a percent
    frame = frame.round(1)  # round to 1 decimal place
    
    SYSOUT.write(OVERWRITE + ' Converted dataframe values to percentages '.ljust(50, '-') + SUCCESS)  # update user
    
    return frame
