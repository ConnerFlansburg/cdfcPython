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
from pathlib import Path
import sys

import cdfcProject as cdfc

# ******************************************** Constants used by Profiler ******************************************** #
profiler = cProfile.Profile()                       # create a profiler to profile cdfc during testing
statsPath = str(Path.cwd() / 'logs' / 'stats.log')  # set the file path that the profiled info will be stored at
# ******************************************** Parsing Command Line Flags ******************************************** #
argumentParser = argparse.ArgumentParser()  # create the argument parser
argumentParser.add_argument("-f", "--function", required=False, help="the distance function that should be used",
                            choices=['Correlation', 'Czekanowski', 'Euclidean'], default='Euclidean')
argumentParser.add_argument("-s", "--stats", required=False, help="run the program with cProfile",
                            action='store_true', default=False)  # if found set to True, otherwise set False
argumentParser.add_argument("-o", "--order", required=False, help="how the cProfile report should be sorted",
                            choices=['ncalls', 'tottime', 'percall', 'cumtime'], default='ncalls')
# ******************************************************************************************************************** #

if __name__ == "__main__":
    
    # parse the arguments into a dictionary keyed by name of flag
    args = vars(argumentParser.parse_args())
    
    if args.get('stats'):                                         # if the stats flag was set
        print('Starting Profiler')
        # * Run the Profiler * #
        profiler.enable()                                         # start collecting profiling info
        cdfc.run(args.get('function'))                            # run cdfc
        profiler.disable()                                        # stop collecting profiling info
        print('Profiler Finished')
        # * Sort & Export the Report * #
        stats = pstats.Stats(profiler).sort_stats(args['order'])  # sort the report
        print(stats)
        # BUG: this is printing byte code to a file/is not human-readable
        stats.dump_stats(statsPath)                               # save report to file
        print('Profiler Exported to log/stats.log')

        # *** Exit *** #
        sys.stdout.write('\nExiting')
        sys.exit(0)  # close program
    else:                                                         # otherwise just run cdfc
        cdfc.run(args.get('function'))
