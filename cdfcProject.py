import tkinter as tk
from tkinter import filedialog
from cdfc import cdfc


def main():

    # prevent root window caused by Tkinter
    root = tk.Tk()
    root.withdraw()

    # prompt user for file path
    path = filedialog.askopenfilename()

    cdfc(path)  # run class dependent feature construction
    # TODO implement other models


if __name__ == "__main__":

    main()
