"""
main.py acts as a wrapper around cdfcProject, which itself is a wrapper around cdfc.py. The primary purpose of this
file is to allow the program to parse the command line. Currently (Nov. 20th, 2020) the only valid flag are --stats &
--order, which determine if cProfile should be run (--stats) & how it's report should be sorted (--order). Both flags
are optional, and the valid values for --order are ncalls (number of calls made), tottime (total time), percall
(average time per call), or cumtime (total time, including time spent by called functions). If no value for --order is
given, the default value, ncalls, is chosen.

# example call: $ python main.py --stats --order ncalls
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
from pathlib import Path

import cdfcProject as cdfc

# ******************************************** Constants used by Profiler ******************************************** #
profiler = cProfile.Profile()                       # create a profiler to profile cdfc during testing
statsPath = str(Path.cwd() / 'logs' / 'stats.log')  # set the file path that the profiled info will be stored at
# ******************************************** Parsing Command Line Flags ******************************************** #
argumentParser = argparse.ArgumentParser()  # create the argument parser
argumentParser.add_argument("-s", "--stats", required=False, help="run the program with cProfile",
                            action='store_true', default=False)  # if found set to True, otherwise set False
argumentParser.add_argument("-o", "--order", required=False, help="how the cProfile report should be sorted",
                            choices=['ncalls', 'tottime', 'percall', 'cumtime'], default='ncalls')
# ******************************************************************************************************************** #

if __name__ == "main":
    
    # parse the arguments into a dictionary keyed by name of flag
    args = vars(argumentParser.parse_args())
    
    if args.get('stats'):                                         # if the stats flag was set
        
        # * Run the Profiler * #
        profiler.enable()                                         # start collecting profiling info
        cdfc.main()                                               # run cdfc
        profiler.disable()                                        # stop collecting profiling info
        
        # * Sort & Export the Report * #
        stats = pstats.Stats(profiler).sort_stats(args['order'])  # sort the report
        stats.dump_stats(statsPath)                               # save report to file
        
    else:                                                         # otherwise just run cdfc
        cdfc.main()
