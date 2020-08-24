# install.packages(farff)  # install if needed
library(farff)             # load in the libraries
library(dplyr) 

path <- choose.files(default = getwd(), caption = "Load File", multi = FALSE)   # get the file's path
df <- readARFF(path)                                                            # read the arff file into a data frame

# convert the string id to an integer id   !! change for each file !!
recode(df$Class, "Cancer" = 1, "Normal" = 2)                                    # replace B-cell with a 1, T-cell with a 2, & AML with a 3
class(df$Class) <- "Numeric"                                                    # set the data type of the id

# move the id column to first place
df <- df %>% dplyr::relocate(ncol(df))                                          # make it the first column

out <- choose.files(default = getwd(), caption = "Save File", multi = FALSE)    # choose an location for the output (must add extension)
write_csv(df, path = out)                                                       # write it to a csv

