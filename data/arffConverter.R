# install.packages(farff)          # install if needed

library(farff)                                                                # load in the library
library(dplyr) 

path = choose.files(default = getwd(), caption = "Load File", multi = FALSE)  # get the file's path

df = readARFF(path)                                                           # read the arff file into a data frame

# convert the string id to an integer id
# !! change for each file !!
df$class = ifelse(df$class == "Tumor", 1, 0)  # replace tumor with 1 & normal with 0
class(df$class) = "Numeric"                   # set the data type of the id

# move the id column to first place
df <- df %>% dplyr::relocate(ncol(df))  # make it the first column

# check for Nans & Nas
# sum(apply(df, 1, is.na))
# sum(apply(df,2,is.nan))

out = choose.files(default = getwd(), caption = "Save File", multi = FALSE)   # choose an location for the output (must add extension)

write_csv2(df, path = out)                                                    # write it to a csv

max(df)
