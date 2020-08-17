import tkinter as tk
import pandas as pd
from tkinter import filedialog
from scipy.io import arff

# get the file path from a file browser
tk.Tk().withdraw()  # prevent root window caused by Tkinter
path = filedialog.askopenfilename()  # prompt user for file path

# read in the .arff as a record array
entries, __ = arff.loadarff(path)

# * used for debugging
# pprint(meta)

# convert the record array into a dataframe
df = pd.DataFrame(entries)

# convert class id column to int from object
df["CLASS"] = df["CLASS"].astype(int)

# pop the class id column
name = 'CLASS'
first_col = df.pop(name)

# insert the column back in at position 0
df.insert(0, name, first_col)


# export the dataframe as csv
export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
df.to_csv(export_file_path, index=False, encoding='utf-8')
