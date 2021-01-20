"""
main.py acts as a wrapper around cdfcProject, which itself is a wrapper around cdfc.py. The primary purpose of this
file is to allow the program to parse the command line. Currently (Dec. 20th, 2020) the only valid flag are --function,
--stats & --order, which determine what distance function to use (--function), if cProfile should be run (--stats) &
how the cProfile report should be sorted (--order). All the flags are optional, and the valid values for --function are
correlation, czekanowski, and euclidean. If no flag is provided, --function defaults to euclidean.  The valid flags for
--order are ncalls (number of calls made), tottime (total time), percall (average time per call), or  cumtime
(total time, including time spent by called functions). If no value for --order is given, the default value, ncalls, is
chosen.

# example call: $ python main.py --function czekanowski --stats --order ncalls
# example call: $ python main.py --function czekanowski
# example call: $ python main.py --stats --order ncalls
# example call: $ python main.py -f czekanowski -s -o ncalls
# example call: $ python main.py -s -o tottime

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git

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

"""

import argparse
import cProfile
import pstats
import time as time
from pathlib import Path
import sys
import cdfcProject as cdfc

# ******************************************** Constants used by Profiler ******************************************** #
profiler = cProfile.Profile()                       # create a profiler to profile cdfc during testing
statsPath = str(Path.cwd() / 'logs' / 'stats.log')  # set the file path that the profiled info will be stored at
timeFormat = '%H:%M:%S'
# ******************************************** Parsing Command Line Flags ******************************************** #
argumentParser = argparse.ArgumentParser()  # create the argument parser

# Distance Function Flag
argumentParser.add_argument("-f", "--function", required=False, help="the distance function that should be used",
                            choices=['correlation', 'czekanowski', 'euclidean'], default='euclidean', type=str)

# cProfile Flag
argumentParser.add_argument("-s", "--stats", required=False, help="run the program with cProfile",
                            action='store_true', default=False)  # if found set to True, otherwise set False

# cProfile Ordering Flag
argumentParser.add_argument("-o", "--order", required=False, help="how the cProfile report should be sorted",
                            choices=['ncalls', 'tottime', 'percall', 'cumtime'], default='ncalls', type=str)

# Learning Model Type Flag
argumentParser.add_argument("-m", "--model", required=False, help="what learning model should be used/tested",
                            choices=['KNN', 'NB', 'DT'], default='KNN', type=str)
# ******************************************************************************************************************** #

if __name__ == "__main__":
    
    # parse the arguments into a namespace
    provided = argumentParser.parse_args()
    
    if provided.stats:  # if the stats flag was set
        
        print('Starting Profiler')
        
        # * Run the Profiler * #
        profiler.enable()    # start collecting profiling info
        start = time.time()  # get the start time
        
        cdfc.run(provided.function, provided.model)  # run cdfc

        end = time.time() - start    # get the elapsed
        print(f'Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(end))}')  # print the elapsed time
        
        # * Sort & Export the Report * #
        profiler.create_stats()
        print('Profiler Finished')
        with open(statsPath, 'w') as file:                   # open file to write to
            stats = pstats.Stats(profiler, stream=file)      # create the report streamer
            stats.sort_stats(provided.order)                 # sort the report
            stats.print_stats()                              # print the report to the file
            print(f'Profiler Exported to {str(statsPath)}')  # print the stats file location

        # *** Exit *** #
        sys.stdout.write('\nExiting')
        sys.exit(0)                  # close program

    else:                            # otherwise just run cdfc
        start = time.time()          # get the start time
        cdfc.run(provided.function, provided.model)
        end = time.time() - start    # get the elapsed
        print(f'Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(end))}')  # print the elapsed time
