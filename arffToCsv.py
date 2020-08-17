import tkinter as tk
import pandas as pd
from tkinter import filedialog
from scipy.io import arff

# the class column is the label column
name = 'CLASS'  # + It should always be class, but check here for errors

# *** Read in the .arff File as a Dataframe *** #
tk.Tk().withdraw()                   # prevent root window caused by Tkinter
path = filedialog.askopenfilename()  # prompt user for file path
entries, __ = arff.loadarff(path)    # read in the .arff as a record array
df = pd.DataFrame(entries)           # convert the record array into a dataframe

# *** Convert the CLASS Column if Needed *** #
# ! label column will often need to be converted from an object BUT
# ! what it will need to be converted to will change from dataset to dataset
df["CLASS"] = df["CLASS"].astype(int)  # convert class id column to int from object
# df["CLASS"] = df["CLASS"].astype(str)  # convert class id column to string from object

# *** Move the CLASS Column to be Where the CDFC Expects it to be  *** #
first_col = df.pop(name)       # pop the class id column
df.insert(0, name, first_col)  # insert the column back in at position 0

# *** Export the Dataframe as CSV *** #
export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
df.to_csv(export_file_path, index=False, encoding='utf-8')
