"""
cdfc.py creates, and evolves a genetic program using Class Dependent Feature Select.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import collections as collect
import copy
import logging as log
import math
import random
import sys
import typing as typ
import warnings
from pathlib import Path
import traceback
from pprint import pprint
from treelib.exceptions import NodeIDAbsentError
from treelib import Tree as libTree
from treelib import Node as Node
from collections import defaultdict
from cdfcFmt import printError
import numpy as np
from alive_progress import alive_bar, config_handler

# ! Next Steps
# TODO performance enhancements

# TODO check copyright on imported packages
# TODO add testing functions

# **************************** Constants/Globals **************************** #
ALPHA: typ.Final = 0.8                        # ALPHA is the fitness weight alpha
BARCOLS = 25                                  # BARCOLS is the number of columns for the progress bar to print
CROSSOVER_RATE: typ.Final = 0.8               # CROSSOVER_RATE is the chance that a candidate will reproduce
ELITISM_RATE: typ.Final = 1                   # ELITISM_RATE is the elitism rate
# GENERATIONS: typ.Final = 50                   # GENERATIONS is the number of generations the GP should run for
GENERATIONS: typ.Final = 5                   # ! value for testing/debugging to increase speed
# MAX_DEPTH: typ.Final = 8                      # MAX_DEPTH is the max depth trees are allowed to be & is used in grow/full
MAX_DEPTH: typ.Final = 4                      # ! value for testing/debugging to make trees more readable
MUTATION_RATE: typ.Final = 0.2                # MUTATION_RATE is the chance that a candidate will be mutated
# ! changes here must also be made in the tree object, and the grow & full functions ! #
OPS: typ.Final = ['add', 'subtract',          # OPS is the list of valid operations on the tree
                  'times', 'max', 'if']
NUM_TERMINALS = {'add': 2, 'subtract': 2,     # NUM_TERMINALS is a dict that, when given an OP as a key, give the number of terminals it needs
                 'times': 2, 'max': 2, 'if': 3}
# ! set the value of R for every new dataset, it is NOT set automatically ! #
TERMINALS: typ.Dict[int, typ.List[int]] = {}  # TERMINALS is a dictionary that maps class ids to their relevant features
TOURNEY: typ.Final = 7                        # TOURNEY is the tournament size
ENTROPY_OF_S = 0                              # ENTROPY_OF_S is used for entropy calculation
FEATURE_NUMBER = 0                            # FEATURE_NUMBER is the number of features in the data set
LABEL_NUMBER = 0                              # LABEL_NUMBER is the number of classes/labels in the data
CLASS_IDS: typ.List[int] = []                 # CLASS_IDS is a list of all the unique class ids
INSTANCES_NUMBER = 0                          # INSTANCES_NUMBER is  the number of instances in the training data
M = 0                                         # M is the number of constructed features
POPULATION_SIZE = 0                           # POPULATION_SIZE is the population size
CL_DICTION = typ.Dict[int, typ.Dict[int, typ.List[float]]]
CLASS_DICTS: CL_DICTION = {}                  # CLASS_DICTS is a list of dicts (indexed by classId) mapping attribute values to classes
# ++++++++ console formatting strings ++++++++ #
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'     # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32m' + HDR  # overwrite previous text & set the text color to green
SYSOUT = sys.stdout
# ++++++++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++ configurations & file paths +++++++++++++++ #
sys.setrecursionlimit(10000)                                 # set the recursion limit for the program

np.seterr(divide='ignore')                                   # suppress divide by zero warnings from numpy
suppressMessage = 'invalid value encountered in true_divide'  # suppress the divide by zero error from Python
warnings.filterwarnings('ignore', message=suppressMessage)

config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', title_length=0, length=20)  # the global config for the loading bars
# config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', force_tty=True, title_length=0, length=10)  # the global config for the loading bars

logPath = str(Path.cwd() / 'logs' / 'cdfc.log')              # create the file path for the log file & configure the logger
log.basicConfig(level=log.DEBUG, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')
# log.basicConfig(level=log.ERROR, filename=logPath, filemode='w', format='%(levelname)s - %(lineno)d: %(message)s')

# profiler = cProfile.Profile()                               # create a profiler to profile cdfc during testing
# statsPath = str(Path.cwd() / 'logs' / 'stats.log')          # set the file path that the profiled info will be stored at
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# ************************ End of Constants/Globals ************************ #

# ********************** Namespaces/Structs & Objects ***********************


def countNodes() -> typ.Generator:
    """Used to generate unique node IDs."""
    
    num = 0
    while True:
        yield num
        num += 1


NodeCount = countNodes()  # create a generator for node count


def countTrees() -> typ.Generator:
    """Used to generate unique root IDs."""
    num = 1
    while True:
        yield num
        num += 1

        
TreeCount = countTrees()  # create a generator for tree count


class Instance:
    """Instance is an instance, or row, within our data set. It represents a single
       record of whatever we are measuring. This replaces the earlier row struct.
       
    Variables:
        className (int): The class id of the class that the record is in (this class ids start at 0).
        attribute ({key: int, value: float}): This dictionary stores the features values in the
                                                  instance, keyed by index (these start at 0)
        vList ([float]): The list of the values stored in the dictionary (used to speed up iteration)
    """

    def __init__(self, className: int, values: typ.Dict[int, float], vlist: typ.Optional[typ.List[float]] = None):
        # this stores the name of the class that the instance is in
        self.className: int = className
        # this stores the values of the features, keyed by index, in the instance
        self.attributes: typ.Dict[int, float] = values
        # this creates a list of the values stored in the dictionary for when iteration is wanted
        if vlist is None:
            self.vList: typ.List[float] = list(self.attributes.values())  # ? is this to expensive?
        else:
            self.vList: typ.List[float] = vlist
        
        try:                                  # check that all the feature values are valid
            if None in self.attributes.values():  # if a None is in the list of feature values
                raise Exception('Tried to create an Instance obj with a None feature value')
        except Exception as err:
            log.error(str(err))
            printError(str(err))
            traceback.print_stack()           # print stack trace so we know how None is reaching Instance
            sys.exit(-1)                      # exit on error; recovery not possible
    
    def __array__(self) -> np.array:
        """Converts an Instance to an Numpy array."""
        return np.array([float(self.className)] + self.vList)


rows: typ.List[Instance] = []  # this will store all of the records read in (the training dat) as a list of rows


class Tree(libTree):
    """Tree is a binary tree data structure that is used to represent a
       constructed feature

    Variables:
        left: Left child of the tree. This will be a tree object.
        right: Right child of the tree. This will be a tree object.
        data: Will either be a terminal character or a function name.

    Methods:
        addLeft: Creates a left child.
        addRight: Creates a right child.
        addMiddle: Creates a middle child.
        getDepth: Gets the depth of a node.
        getRoot: Gets the root of the tree.
        getRandomNode: Gets a random node from the tree.
        getLeft: Getter for left.
        getRight: Getter for right.
        getMiddle: Getter for middle.
        runTree: Wrapper function for __runNode, and is used to __transform data using the tree.
        __runNode: Helper function for runTree.
    """
    # typing alias for tree nodes. They will either have a terminal index (int) or an OP (str)
    TREE_DATA = typ.Union[int, str]
    
    def __init__(self, *args, **kwargs) -> None:
        """Constructor for Tree object."""
        libTree.__init__(self, *args, **kwargs)  # call parent constructor
        # used to get children quickly Dict[key=parentId, value=Dict[key=branch, value=childId]]
        self.BRANCHES: typ.Dict[int, typ.Dict[str, str]] = defaultdict(dict)
        
    def remove_subtree(self, nid, identifier=None) -> "Tree":
        """This overrides the remove_subtree method so the new tree's BRANCHES dictionary is set."""
        # NOTE: we don't need to delete dictionary values from the old tree as they will be overwritten during swap
        subTree: Tree = super().remove_subtree(nid=nid)  # create a subtree
        subTree.BRANCHES = self.BRANCHES.copy()            # copy the dict from the original tree into the subtree
        return subTree                                     # return the subtree
    
    def printNodes(self) -> None:
        """Prints all the nodes in a tree & then the tree."""
        # pprint(self.all_nodes())                # print all the nodes using pretty print
        
        for n in self.all_nodes_itr():          # loop over every node
            if self.BRANCHES[n.identifier] == {}:  # if the n is a terminal, skip
                continue
            print(n)                            # print the node
            print(self.BRANCHES[n.identifier])  # print the dictionary for that node
        
        self.show(idhidden=False)               # print the tree
        out = Path.cwd() / 'logs' / 'tree.log'  # create the file path
        self.save2file(out)                     # save to tree.log
        
    def checkTree(self) -> None:
        """Checks the tree dictionary"""
        # self.printNodes()           # print the tree being checked
        for n in self.all_nodes():  # loop over every node
            try:
                if not self.contains(n.identifier):  # check that the node is in the tree
                    log.error(f'Tree found node that was not in tree: {n}')
                    print(f'Tree found node that was not in tree: {n}')
                
                if n.data not in OPS:                # if this is a terminal node
                    continue                         # skip this iteration
                
                lft: Node = self.get_node(self.BRANCHES[n.identifier]['left'])  # all nodes should have a left
                r: Node = self.get_node(self.BRANCHES[n.identifier]['right'])   # all nodes should have a right
                
                if lft is None or r is None:                # if getting left or right key got a None, raise exception
                    pprint(self.children(n.identifier))
                    print(f'left found: {lft}\nright found: {r}')
                    raise AssertionError('Getting left and/or right key failed')
                if lft not in self.children(n.identifier):  # check that the left child is valid
                    pprint(self.children(n.identifier))
                    print(lft)
                    raise AssertionError(f'Left child was set incorrectly parent = {n}')
                if r not in self.children(n.identifier):    # check that the right child is valid
                    pprint(self.children(n.identifier))
                    print(r)
                    raise AssertionError(f'Right child was set incorrectly parent = {n}')
                
                if n.data == 'if':                   # if the node stores an IF OP
                    m = self.get_node(self.BRANCHES[n.identifier]['middle'])
                    if m is None:                    # if getting middle key failed, throw exception
                        raise AssertionError('Getting middle key failed')
                    if not (m in self.children(n.identifier)):  # check that the middle child is valid
                        raise AssertionError('Left child was set incorrectly')
            # NOTE: KeyError indicates that the key (left, right, middle) doesn't exist,
            # +     AssertionError means it does exist & stores a None
            except KeyError:
                lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
                log.error(f'CheckTree raised a KeyError, on line {lineNm}')  # log the error
                printError(f'CheckTree raised a KeyError, dict = {self.BRANCHES[n.identifier].items()}, on line {lineNm}')  # print message
                traceback.print_stack()  # print stack trace
                sys.exit(-1)  # exit on error; recovery not possible
                
            except AssertionError as err:
                lineNm = sys.exc_info()[-1].tb_lineno        # get the line number of error
                printError(str(err))
                log.error(f'{str(err)}, on line {lineNm}')   # log the error
                printError(f'CheckTree raised a AssertionError, dict = {self.BRANCHES[n.identifier].items()}, on line {lineNm}')  # print message
                traceback.print_stack()                      # print stack trace
                sys.exit(-1)                                 # exit on error; recovery not possible
    
    def sendToStdOut(self) -> None:
        """Used to print a Tree."""
        self.show(idhidden=False)  # ascii
        # self.show('ascii-ex')   # ascii-ex FAILS
        # self.show('ascii-exr')  # ascii-exr FAILS
        # self.show('ascii-em')   # ascii-em FAILS
        # self.show('ascii-emv')  # ascii-emv FAILS
        # self.show('ascii-emh')  # ascii-emh FAILS
        out = Path.cwd() / 'logs' / 'tree.log'  # create the file path
        self.save2file(out)
    
    def addRoot(self) -> Node:
        """Adds a root node to the tree"""
        op = random.choice(OPS)
        root = self.create_node(tag=f'root: {op}', identifier=f'Tree {next(TreeCount)}', data=op)  # create a root node for the tree
        return root
     
    def addLeft(self, parent: Node, data: TREE_DATA) -> Node:
        """Sets the left child of a tree."""
        # create the node & add it to the tree
        new: Node = self.create_node(tag=str(data), parent=parent.identifier,
                                     data=data, identifier=f'{str(data)}({next(NodeCount)})')

        # update dictionary used by getLeft(), getRight(), & getMiddle()
        # store new Node ID at ParentId, 'left' (overwriting any old values)
        self.BRANCHES[parent.identifier]['left'] = new.identifier
        
        '''try:  # ! For Testing Only !! - attempt to access created entry
            self.BRANCHES[parent.identifier]['left']
        except KeyError:
            msg: str = f'Tree encountered an error in addLeft(), {parent}'
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Error on line {lineNm}')
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible'''
        
        return new
    
    def addRight(self, parent: Node, data: TREE_DATA) -> Node:
        """Sets the right child of a tree."""

        # create the node & add it to the tree
        new: Node = self.create_node(tag=str(data), parent=parent.identifier,
                                     data=data, identifier=f'{str(data)}({next(NodeCount)})')
        
        # update dictionary used by getLeft(), getRight(), & getMiddle()
        # store new Node ID at ParentId, 'left' (overwriting any old values)
        self.BRANCHES[parent.identifier]['right'] = new.identifier
        
        '''try:  # ! For Testing Only !! - attempt to access created entry
            self.BRANCHES[parent.identifier]['right']
        except KeyError:
            msg: str = f'Tree encountered an error in addRight(), {parent}'
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Error on line {lineNm}')
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible'''
        
        return new
    
    def addMiddle(self, parent: Node, data: TREE_DATA) -> Node:
        """Sets the middle child of a tree."""

        # create the node & add it to the tree
        new: Node = self.create_node(tag=str(data), parent=parent.identifier,
                                     data=data, identifier=f'{str(data)}({next(NodeCount)})')
        
        # update dictionary used by getLeft(), getRight(), & getMiddle()
        # store new Node ID at ParentId, 'left' (overwriting any old values)
        self.BRANCHES[parent.identifier]['middle'] = new.identifier
        '''try:  # ! For Testing Only !! - attempt to access created entry
            self.BRANCHES[parent.identifier]['middle']
        except KeyError:
            msg: str = f'Tree encountered an error in addMiddle(), {parent}'
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Error on line {lineNm}')
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible'''
        
        return new
    
    def addSubTree(self, parent: Node, branch: str, subtree: "Tree"):
        """Adds a subtree as a child of parent. This will also update the BRANCHES
           dictionary & should overwrite any old values. This shouldn't delete
           values from the old tree's dictionary as subtree is a copy"""

        try:
            self.BRANCHES[parent.identifier][branch] = subtree.root  # add the subtree root to the dictionary
            
            # this shouldn't delete values from the old tree's dictionary as subtree is a copy
            for nid in subtree.nodes.keys():                         # loop over all the nids in the subtree
                # ? this makes a shallow copy. Is that okay or should it be deep?
                self.BRANCHES[nid] = subtree.BRANCHES[nid].copy()    # copy the sub-dictionary over
            del subtree.BRANCHES                                     # delete the old dictionary
            
            # add the subtree to the original tree as a child of parent
            # deep=False because the subtree has been deleted from the old tree so can't affect it anymore.
            # However is subtree is used after this it will need to be True
            self.paste(parent.identifier, subtree)  # ? should deep be true or false?
    
        except NodeIDAbsentError as err:  # catch error thrown by deep paste
            print('adding to:')
            self.sendToStdOut()                          # print the tree
            print('subTree:')
            subtree.sendToStdOut()
            msg: str = f'addSubTree failed, {str(err)}'  # create the message
            log.error(msg)                               # log the error
            printError(msg)                              # print message
            printError(f'Parent = {parent}')
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
        
        except Exception as err:
            msg: str = f'addSubTree failed, {str(err)}'  # create the message
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Error on line {lineNm}')
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
        
    def getDepth(self, node: Node) -> int:
        """Given a node, return the node's depth"""
        return self.depth(node)
    
    def getRoot(self) -> Node:
        """Returns the trees root & prevent root from being overwritten"""
        return self.get_node(self.root)
    
    def getRandomNode(self) -> Node:
        """Get a random node obj from the tree."""
        n: Node = random.choice(self.all_nodes())      # pick a random node
        
        if (n is self.getRoot()) or not(self.contains(n.identifier)):
            nodeList = self.all_nodes()  # get all the node
            random.shuffle(nodeList)     # shuffle them
        
            # if we picked the root or a node not in the tree
            while (n is self.getRoot()) or not(self.contains(n.identifier)):
                if len(nodeList) == 0:  # if nodeList is empty
                    raise Exception('getRandomNode could not find a node in the tree')
                n = nodeList.pop(0)  # pop the top of the list
        
        return n
    
    def getLeft(self, parent: Node) -> typ.Optional[Node]:
        """Gets the left child of a tree."""
        try:
            dct = self.BRANCHES[parent.identifier]
            nid: str = dct['left']                 # get the nodes Id
            return self.get_node(nid)              # get the node & return NOTE: if the nid not in tree this will trigger error
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f'getLeft() could not find left for {parent}, on line {lineNm}'
            log.error(msg)           # log the error
            printError(msg)          # print message
            printError(f'Node\'s children: {self.children(parent.identifier)}')
            # printError(f'Error on line {lineNm}')
            # traceback.print_stack()  # print stack trace
            sys.exit(-1)
            # return None              # attempt recovery
        
    def getRight(self, parent: Node) -> typ.Optional[Node]:
        """Gets the right child of a tree."""
        try:
            nid: str = self.BRANCHES[parent.identifier]['right']  # get the nodes Id
            return self.get_node(nid)                             # get the node & return
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno        # get the line number of error
            msg: str = f'getRight() could not find right for {parent}, on line {lineNm}'
            log.error(msg)           # log the error
            printError(msg)          # print message
            printError(f'Node\'s children: {self.children(parent.identifier)}')
            # printError(f'Error on line {lineNm}')
            # traceback.print_stack()  # print stack trace
            sys.exit(-1)
            # return None              # attempt recovery
    
    def getMiddle(self, parent: Node) -> typ.Optional[Node]:
        """Gets the middle child of a tree."""
        try:
            nid: str = self.BRANCHES[parent.identifier]['middle']  # get the nodes Id
            return self.get_node(nid)                              # get the node & return
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno        # get the line number of error
            msg: str = f'getMiddle() could not find middle for {parent}, on line {lineNm}'
            log.error(msg)           # log the error
            printError(msg)          # print message
            printError(f'Node\'s children: {self.children(parent.identifier)}')
            # printError(f'Error on line {lineNm}')
            # traceback.print_stack()  # print stack trace
            sys.exit(-1)
            # return None              # attempt recovery
            
    def getBranch(self, child: Node) -> typ.Tuple[str, Node]:
        """Given a child, this returns what branch of it's parent it was on."""
        parent: Node = self.parent(child.identifier)                     # get the parents Id
        children: typ.Dict[str, str] = self.BRANCHES[parent.identifier]  # get all of the parents children
        
        for branch in children.keys():                                        # loop over all children
            if self.BRANCHES[parent.identifier][branch] == child.identifier:  # if this branch maps to the child
                return branch, parent                                         # return the branch
        raise Exception('getBranch was not able to find a connection between the child & parent')

    # running a tree should return a single value
    # featureValues -- the values of the relevant features keyed by their index in the original data
    def runTree(self, featureValues: typ.Dict[int, float], classId: int) -> float:
        """runTree is a wrapper for runNode & is used to __transform provided data
           by walking the decision tree

        Parameters:
            featureValues ({key: int, value: float}): The dictionary mapping feature ids to
                                                      their values (in the current instance).
            classId (int): Class ID the tree is meant to classify.

        Returns:
            (float): The final value that the decision tree creates given the provided data.
        """

        return self.__runNode(featureValues, self.getRoot(), classId)
    
    def __runNode(self, featureValues: typ.Dict[int, float], node: Node, classId: int) -> typ.Union[int, float]:
        """runTree is a wrapper for runNode & is used to __transform provided data
           by walking the decision tree

        Parameters:
            featureValues ({key: int, value: float}): The dictionary mapping feature ids to
                                                      their values (in the current instance).

        Returns:
            (float): The value of a terminal, or the value computed by one or more operations.
        """

        try:
            
            if node.data in OPS:  # if the node is an OP
                # *************************** Error Checking *************************** #
                lftNone: bool = self.getLeft(node) is None                          # is left None?
                rgtNone: bool = self.getRight(node) is None                         # is right None?
                xor: bool = (lftNone and not rgtNone) or (not lftNone and rgtNone)  # exclusive or
                if xor:                                             # if one child is None, but not both
                    raise AssertionError(f'runNode found a node in OPS with 1 \'None\' child,\n\t node = {node}')
                if lftNone and rgtNone:                             # if both children are None
                    raise AssertionError(f'runNode found a node in OPS with 2 \'None\' children,\n\t node = {node}')
                if node.data == 'if' and self.getMiddle(node) is None:  # if the OP is IF and it has no middle
                    raise AssertionError('runNode found a node with a IF OP and no middle node')
                # ************ Determine Which OP is Stored & Run Recursion ************ #
                left: Node = self.getLeft(node)  # get the left child (all OPS wil have a left)
                right: Node = self.getRight(node)  # get the right child (all OPS wil have a right)
                
                if node.data == 'add':                                      # if the OP was add
                    vl = (self.__runNode(featureValues, left, classId) +    # left + right
                          self.__runNode(featureValues, right, classId))
                    return vl
                
                elif node.data == 'subtract':                                # if the OP was subtract
                    vl = (self.__runNode(featureValues, left, classId) -     # left - right
                          self.__runNode(featureValues, right, classId))
                    return vl
                
                elif node.data == 'times':                                   # if the OP was multiplication
                    vl = (self.__runNode(featureValues, left, classId) *     # left * right
                          self.__runNode(featureValues, right, classId))
                    return vl
                
                elif node.data == 'max':                                     # if the OP was max
                    vl = max(self.__runNode(featureValues, left, classId),   # max(left, right)
                             self.__runNode(featureValues, right, classId))
                    return vl
                
                elif node.data == 'if':                                      # if the OP was if
                    if self.__runNode(featureValues, left, classId) >= 0:    # if the left value is positive,
                        vl = self.__runNode(featureValues, right, classId)   # return the right node
                    else:                                                    # if the left value is negative,
                        middle: Node = self.getMiddle(node)          # get the middle child
                        vl = self.__runNode(featureValues, middle, classId)  # return the middle node
                    return vl
                # ********************************************************************* #

            elif node.data in TERMINALS[classId]:     # if the node is a terminal
                # *************************** Error Checking *************************** #
                if math.isnan(node.data):             # if the value stored is a NaN
                    msg: str = f'NaN stored in tree. Expected a class ID, OPS value, or number, got {node.data}'
                    raise TypeError(f'ERROR: {msg}')  # raise TypeError

                if featureValues[node.data] is None:  # if the value stored is a None
                    raise TypeError(f'featureValues contained a None at index {node.data}')
                # ************************ Return Terminal Value ************************ #
                return featureValues[node.data]       # if the terminal is valid, return it
                # *********************************************************************** #
                
            else:                                     # if the node is not a terminal or a OP
                raise TypeError(f'runNode could not parse data in tree, data ={node.data}')
        
        except (TypeError, AssertionError) as err:                      # catch any exceptions
            self.sendToStdOut()                                         # record the tree
            lineNm = sys.exc_info()[-1].tb_lineno                       # get the line number of error
            log.error(f'line = {lineNm}, {str(err)}'
                      f'\ndict = {self.BRANCHES[node.identifier]}'
                      f'\n{self.all_nodes()}')                        # log the error
            printError(f'line = {lineNm}, {str(err)}')                  # print message
            printError(f'dict = {self.BRANCHES[node.identifier]}')
            traceback.print_stack()                                     # print stack trace
            sys.exit(-1)                                                # exit on error; recovery not possible


class ConstructedFeature:
    """Constructed Feature is used to represent a single constructed feature in
       a hypothesis. It contains the tree representation of the feature along
       with additional information about the feature.

    Variables:
        className (int): Class id of the class that the feature is meant to distinguish.
        tree (Tree): Constructed feature's binary decision tree.
        size (int): Number of nodes in the tree
        relevantFeatures ([int]): List of terminal characters relevant to the feature's class.

    Methods:
        __transform: Takes the original data & transforms it using the feature's decision tree.
    """

    def __init__(self, className: int, tree: Tree) -> None:
        self.className = className                    # the name of the class this tree is meant to distinguish
        self.tree = tree                              # the root node of the constructed feature
        self.size = tree.size                         # the individual size (the size of the tree)
        self.relevantFeatures = TERMINALS[className]  # holds the indexes of the relevant features
        # sanityCheckCF(self)  # ! testing purposes only!
    
    '''def __str__(self):
        """Used to print a Constructed Feature."""
        print(f'Class ID: {self.className}, \n{self.tree}')'''

    def transform(self, instance: Instance) -> float:
        """Takes an instance, transforms it using the decision tree, and return the value computed."""
        
        # Send the tree a list of all the attribute values in a single instance
        featureValues: typ.Dict[int, float] = instance.attributes
        return self.tree.runTree(featureValues, self.className)


class Hypothesis:
    """Hypothesis is a single hypothesis (a GP individual), and will contain a list of constructed features. It
       should have the same number of constructed features for every class id, and should have at least one for
       each class id.
       
        Variables:
            features ([ConstructedFeature]): List of the constructed features for this hypothesis.
            size (int): Sum of the constructed feature's sizes.
            fitness (int or float): Calculated fitness score.
            distance (int or float): Calculated distance value.
            averageInfoGain (int or float): Average of every features info gain.
            maxInfoGain (int or float): Largest info gain for any feature.
            
        Methods:
            getFitness: Get the fitness score of the Hypothesis
            __transform: Transforms a dataset using the trees in the constructed features.

    """

    _fitness: typ.Union[None, int, float] = None  # the fitness score
    distance: typ.Union[float, int] = 0          # the distance function score
    averageInfoGain: typ.Union[float, int] = -1  # the average info gain of the hypothesis
    maxInfoGain: typ.Union[float, int] = -1      # the max info gain in the hypothesis
    idsToFeatures: typ.Dict[int, ConstructedFeature] = {}  # key all the CFs by their class ids
    # + averageInfoGain & maxInfoGain must be low enough that they will always be overwritten + #
    
    def __init__(self, features: typ.List[ConstructedFeature], size: int) -> None:
        self.features: typ.List[ConstructedFeature] = features      # a list of all the constructed features
        self.size: int = size                                       # the number of nodes in all the cfs
        self.idsToFeatures: typ.Dict[int, ConstructedFeature] = {}  # key all the CFs by their class ids
        for feature in features:
            k = feature.className
            self.idsToFeatures[k] = feature

    @property
    def fitness(self) -> typ.Union[int, float]:
        """Getter for fitness."""
        if self._fitness is None:  # if fitness isn't set
            self.getFitness()      # run fitness to set it
        return self._fitness       # either way return fitness
    
    def getFitness(self) -> float:
        """getFitness uses several helper functions to calculate the fitness of a Hypothesis"""
    
        # log.debug('Starting getFitness() method')
        
        def __Czekanowski(Vi: typ.List[float], Vj: typ.List[float]) -> float:
            
            # log.debug('Starting Czekanowski() method')

            # ************************** Error checking ************************** #
            try:
                if len(Vi) != len(Vj):
                    log.error(f'In Czekanowski Vi[d] & Vi[d] are not equal Vi = {Vi}, Vj = {Vj}')
                    raise Exception(f'ERROR: In Czekanowski Vi[d] & Vi[d] are not equal Vi = {Vi}, Vj = {Vj}')
                if None in Vi:
                    log.error(f'In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                    raise Exception(f'ERROR: In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                if None in Vj:
                    log.error(f'In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
                    raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
            except Exception as err:
                lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                printError(str(err) + f', line = {lineNm}')
                sys.exit(-1)  # recovery impossible, exit with an error
            # ******************************************************************** #

            minSum: typ.Union[int, float] = 0
            addSum: typ.Union[int, float] = 0
            
            # + range(len(self.features)) loops over the number of features the hypothesis has.
            # + Vi & Vj are lists of the instances from the original data, that have been transformed
            # + by the hypothesis.
            try:

                for i, j in zip(Vi, Vj):                    # zip Vi & Vj so that we can iterate in parallel
                    
                    # **************************************** Error Checking **************************************** #
                    if Vi == [None, None] and Vj == [None, None]:
                        log.error(f'In Czekanowski Vj ({Vj}) & Vi ({Vi}) was found to be a \'None type\'')
                        raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) & Vi ({Vi}) was found to be a \'None type\'')
                    
                    elif Vj == [None, None]:
                        log.error(f'In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
                        raise Exception(f'ERROR: In Czekanowski Vj ({Vj}) was found to be a \'None type\'')
                    
                    elif Vi == [None, None]:
                        log.error(f'In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                        raise Exception(f'ERROR: In Czekanowski Vi ({Vi}) was found to be a \'None type\'')
                    # ************************************************************************************************ #
                    
                    top: typ.Union[int, float] = min(i, j)  # get the top of the fraction
                    bottom: typ.Union[int, float] = i + j   # get the bottom of the fraction
                    
                    minSum += top                           # the top of the fraction
                    addSum += bottom                        # the bottom of the fraction
            
                # resolve 0/0 case
                if addSum and minSum == 0:  # if the numerator & the denominator both 0, set to zero
                    value = 0
                
                # resolve the n/0 case
                elif addSum == 0:                        # if only the denominator is 0, set it to a small number
                    addSum = -30                         # + Experiment with this value
                    # pow(10, -6) causes overflow in line ~765
                    value = 1 - ((2 * minSum) / addSum)  # create the return value
                
                # the normal case
                else:
                    value = 1 - ((2 * minSum) / addSum)  # create the return value
            
            except Exception as err2:
                lineNm = sys.exc_info()[-1].tb_lineno       # print line number error occurred on
                log.error(str(err2))
                printError(str(err2) + f', line = {lineNm}')
                sys.exit(-1)                                # recovery impossible, exit with error

            # log.debug('Finished Czekanowski() method')
            
            return value

        def Distance(values: typ.List[Instance]):
            """"Distance calculates the distance value of the Hypothesis"""
    
            # log.debug('Starting Distance() method')
            
            Db: typ.Union[int, float] = 2  # this will hold the lowest distance Czekanowski found
            Dw: typ.Union[int, float] = 0  # this will hold the highest distance Czekanowski found
    
            # ********** Compute Vi & Vj ********** #
            # the reason for these two loops is to allow us to compare vi with every other instance (vj)
            for vi in values:                                   # loop over all the training examples
                for vj in values:                               # loop over all the training examples

                    dist = __Czekanowski(vi.vList, vj.vList)  # compute the distance using the values
            
                    if vi.className == vj.className:            # if the instances vi & vj are from the same class, skip
                        continue
            
                    elif vi.attributes == vj.attributes:    # if vi & vj are not in the same class (Db), skip
                        continue
            
                    else:                                   # if vi & vj are valid
                        if dist > Dw:                       # replace the max if the current value is higher
                            Dw = dist
                            
                        if dist < Db:                       # replace the min if the current value is smaller
                            Db = dist

            # perform the final distance calculations
            Db *= (1 / len(values))  # multiply by 1/|S|
            Dw *= (1 / len(values))  # multiply by 1/|S|

            # log.debug('Finished Distance() method')
            
            return 1 / (1 + math.pow(math.e, -5*(Db - Dw)))

        def __entropy(partition: typ.List[Instance]) -> float:
            """Calculates the entropy of a Hypothesis"""
            
            # log.debug('Starting entropy() method')
            
            p: typ.Dict[int, int] = {}   # p[classId] = number of instances in the class in the partition sv
            for i in partition:          # for instance i in a partition sv
                if i.className in p:       # if we have already found the class once,
                    p[i.className] += 1  # increment the counter
                    
                else:                   # if we have not yet encountered the class
                    p[i.className] = 1  # set the counter to 1

            calc = 0
            for c in p.keys():  # for class in the list of classes in the partition sv
                # perform entropy calculation
                pi = p[c] / len(partition)
                calc -= pi * math.log(pi, 2)

            # log.debug('Finished entropy() method')

            return calc

        def __conditionalEntropy(feature: ConstructedFeature) -> float:
            """Calculates the entropy of a Hypothesis"""
    
            # log.debug('Starting conditionalEntropy() method')

            # this is a feature struct that will be used to store feature values
            # with their indexes/IDs in CFs
            ft = collect.namedtuple('ft', ['id', 'value'])
            
            # key = CF(Values), Entry = instance in training data
            partition: typ.Dict[float, typ.List[Instance]] = {}
            
            s = 0                                # used to sum CF's conditional entropy
            used = TERMINALS[feature.className]  # get the indexes of the used features
            v = []                               # this will hold the used features ids & values
            for i in rows:                       # loop over all instances

                # get CF(v) for this instance (i is a Instance struct which is what __transform needs)
                cfv = feature.transform(i)  # needs the values for an instance

                # get the values in this instance i of the used feature
                for u in used:  # loop over all the indexes of used features
                    # create a ft where the id is u (a used index) &
                    # where the value is from the instance
                    v.append(ft(u, i.attributes[u]))

                if cfv in partition:            # if the partition exists
                    partition[cfv].append(i)  # add the instance to it
                else:                     # if the partition doesn't exist
                    partition[cfv] = [i]  # create it

            for e in partition.keys():
                s += (len(partition[e])/INSTANCES_NUMBER) * __entropy(partition[e])

            # log.debug('Finished conditionalEntropy() method')

            return s  # s holds the conditional entropy value

        gainSum = 0  # the info gain of the hypothesis
        for f in self.features:  # loop over all features & get their info gain

            # ********* Entropy calculation ********* #
            condEntropy = __conditionalEntropy(f)  # find the conditional entropy

            # ******** Info Gain calculation ******* #
            f.infoGain = ENTROPY_OF_S - condEntropy  # H(class) - H(class|f)
            gainSum += f.infoGain                    # update the info sum

            # updates the max info gain of the hypothesis if needed
            if self.maxInfoGain < f.infoGain:
                self.maxInfoGain = f.infoGain

        # calculate the average info gain using formula 3
        term1 = gainSum+self.maxInfoGain
        term2 = (M+1)*(math.log(LABEL_NUMBER, 2))
        self.averageInfoGain += term1 / term2

        # set size
        # * this must be based off the number of nodes a tree has because
        # * the depth will be the same for all of them

        # *********  Distance Calculation ********* #
        self.distance = Distance(self.__transform())  # calculate the distance using the transformed values

        # ********* Final Calculation ********* #
        term1 = ALPHA*self.averageInfoGain
        term2 = (1-ALPHA)*self.distance
        term3 = (math.pow(10, -7)*self.size)
        final = term1 + term2 - term3
        # ********* Finish Calculation ********* #

        # log.debug('Finished getFitness() method')
        self._fitness = final
        return final

    def runCDFC(self, data: np.array) -> np.array:
        """runCDFC transforms a dataset using the trees in the constructed features, and is use by cdfcProject
           to convert (reduce) data using class dependent feature construction.
        
        Parameter:
            data (np.array): A dataset to be converted by the CDFC algorithm.
        
        Return:
            transformedData (np.array): A dataset that has been converted by the algorithm.
        """
        # this is a type hint alias for the values list where: [classID(int), value(float), value(float), ...]
        valueList = typ.List[typ.Union[int, float]]
        # a list of values lists
        transformedData: typ.List[valueList] = []  # this will hold the new transformed dataset after as it's built

        # loop over each row/instance in data and transform each row using each constructed feature.
        # We want to transform a row once, FOR EACH CF in a Hypothesis.
        for d in data:
            # This will hold the transformed values for each constructed feature until we have all of them.
            values: valueList = [d[0]]  # values[0] = class name(int), values[0:] = transformed values (float)

            # NOTE: here we want to create a np array version of an Instance object of the form
            # +     (classID, values[]), for each row/instance
            # for each row, convert that row using each constructed feature (where f is a constructed feature)
            for f in self.features:
                # convert the numpy array to an instance & transform it
                currentLine: float = f.transform(Instance(d[0], dict(zip(range(len(d[1:])), d[1:])), d[1:]))
                # add the value of the transformation to the values list
                values.append(currentLine)
            
            # NOTE: now we want to add the np array Instance to the array of all the transformed Instances to create
            # a new data set
            transformedData.append(values)

        # convert the data set from a list of lists to a numpy array
        return np.array([np.array(x) for x in transformedData])
        
    def __transform(self) -> typ.List[Instance]:
        """__transform transforms a dataset using the trees in the constructed features. This is used internally
           during training, and will be done over the Rows constant. This is produces data of a different format
           then runCDFC.
            
            Return:
                transformed (np.array): A new dataset, created by transforming the original one.
        """
    
        # log.debug('Starting __transform() method')
        
        transformed: typ.List[Instance] = []  # this will hold the transformed values
        
        # if data is None then we are transforming as part of the distance calculation
        # so we should use rows (the provided training data)
    
        for r in rows:    # for each Instance
            values = []   # this will hold the calculated values for all the constructed features

            for f in self.features:            # __transform the original input using each constructed feature
                values.append(f.transform(r))  # append the transformed values for a single CF to values
            
            # each Instance will hold the new values for an Instance & className, and
            # transformed will hold all the instances for a hypothesis
            vls = dict(zip(range(len(values)), values))  # create a dict of the values keyed by their index
            transformed.append(Instance(r.className, vls, values))

        # log.debug('Finished __transform() method')
        
        return transformed  # return the list of all instances

    # ! testing purposes only!
    def sanityCheck(self):
        """Used in debugging to check a Hypothesis"""
        log.debug('Starting Hypothesis Sanity Check...')
        self.__transform()
        log.debug('Population Hypothesis Check Passed')


class Population:
    """Population is a list of Hypothesis, and a generation number. It is largely
       just a namespace.
    
        Variables:
            candidateHypotheses ([Hypothesis]): list of hypotheses
            generationNumber (int): current generation number.
    """

    def __init__(self, candidates: typ.List[Hypothesis], generationNumber: int) -> None:
        self.candidateHypotheses: typ.List[Hypothesis] = candidates  # a list of all the candidate hypotheses
        self.generation = generationNumber     # this is the number of this generation

    
# ***************** End of Namespaces/Structs & Objects ******************* #


def __grow(classId: int, node: Node, tree: Tree) -> Node:
    """Grow creates a tree or sub-tree starting at the Node node, and using the Grow method.
       If node is a root Node, grow will build a tree, otherwise grow will build a sub-tree
       starting at node. Grow assumes that node's data has already been set & makes all
       changes in place.

       NOTE: During testing whatever calls grow should use this sanity check after building
             it to check for errors: sanityCheckTree(newTree)
    """
    
    coin = random.choice(['OP', 'TERM']) == 'TERM'  # flip a coin & decide OP or TERM
    
    # *************************** A Terminal was Chosen *************************** #
    # NOTE: check depth-1 because we will create children
    if coin == 'TERM' or (tree.getDepth(node) == MAX_DEPTH - 1):  # if we need to add terminals

        # pick the needed amount of terminals
        terms: typ.List[int] = random.choices(TERMINALS[classId], k=NUM_TERMINALS[node.data])
        
        if NUM_TERMINALS[node.data] == 2:                  # if the OP needs 2 children
            tree.addLeft(parent=node, data=terms.pop(0))   # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))  # create a new left node & add it
            
            return tree.getRoot()                          # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:                 # if the OP needs 3 children
            tree.addLeft(parent=node, data=terms.pop(0))    # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))   # create a new right node & add it
            tree.addMiddle(parent=node, data=terms.pop(0))  # create a new middle node & add it
            
            return tree.getRoot()                           # return the root node of the tree
        
        else:                                               # if NUM_TERMINALS was not 2 or 3
            raise IndexError("Grow could not find the number of terminals need")
    
    # *************************** A Operation was Chosen *************************** #
    else:  # if we chose to add an operation
        
        if NUM_TERMINALS[node.data] == 2:                              # if the number of terminals needed by node is two
            ops: typ.List[str] = random.choices(OPS, k=2)              # pick the needed amount of OPs
            
            left: Node = tree.addLeft(parent=node, data=ops.pop(0))    # add the new left node
            right: Node = tree.addRight(parent=node, data=ops.pop(0))  # add the new right node
            
            __grow(classId, left, tree)                                # call grow on left to set it's children
            __grow(classId, right, tree)                               # call grow on right to set it's children
            return tree.getRoot()                                      # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:                              # if the number of terminals needed by node is three
            ops: typ.List[str] = random.choices(OPS, k=3)                # pick the needed amount of OPs
            
            left: Node = tree.addLeft(parent=node, data=ops.pop(0))      # create & add the new left node to the tree
            right: Node = tree.addRight(parent=node, data=ops.pop(0))    # create & add the new right node to the tree
            middle: Node = tree.addMiddle(parent=node, data=ops.pop(0))  # create & add the new middle node to the tree
            
            __grow(classId, left, tree)                                  # call grow on left to set it's children
            __grow(classId, right, tree)                                 # call grow on right to set it's children
            __grow(classId, middle, tree)                                # call grow on middle to set it's children
            return tree.getRoot()                                        # return the root node of the tree
        
        else:  # if NUM_TERMINALS was not 1 or 2
            raise IndexError("Grow could not find the number of terminals need")


def __full(classId: int, node: Node, tree: Tree):
    """Full creates a tree or sub-tree starting at the Node node, and using the Full method.
       If node is a root Node, full will build a tree, otherwise full will build a sub-tree
       starting at node. Full assumes that node's data has already been set & makes all
       changes in place.
      
       NOTE: During testing whatever calls full should use this sanity check after building
             it to check for errors: sanityCheckTree(newTree)
    """
    
    # *************************** Max Depth Reached *************************** #
    if tree.getDepth(node) == MAX_DEPTH - 1:
        
        # pick the needed amount of terminals
        terms: typ.List[int] = random.choices(TERMINALS[classId], k=NUM_TERMINALS[node.data])
        
        if NUM_TERMINALS[node.data] == 2:      # if the OP needs 2 children
            tree.addLeft(parent=node, data=terms.pop(0))   # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))  # create a right left node & add it
            return tree.getRoot()              # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:     # if the OP needs 3 children
            tree.addLeft(parent=node, data=terms.pop(0))    # create a new left node & add it
            tree.addRight(parent=node, data=terms.pop(0))   # create a new right node & add it
            tree.addMiddle(parent=node, data=terms.pop(0))  # create a new middle node & add it
            return tree.getRoot()               # return the root node of the tree
        
        else:  # if NUM_TERMINALS was not 1 or 2
            raise IndexError("Grow could not find the number of terminals need")
    
    # *************************** If Not at Max Depth *************************** #
    else:  # if we haven't reached the max depth, add operations
        
        if NUM_TERMINALS[node.data] == 2:                              # if the number of terminals needed by node is two
            ops: typ.List[str] = random.choices(OPS, k=2)              # pick the needed amount of OPs

            left: Node = tree.addLeft(parent=node, data=ops.pop(0))    # add the new left node
            right: Node = tree.addRight(parent=node, data=ops.pop(0))  # add the new right node
            
            __full(classId, left, tree)                                # call grow on left to set it's children
            __full(classId, right, tree)                               # call grow on right to set it's children
            return tree.getRoot()                                      # return the root node of the tree
        
        elif NUM_TERMINALS[node.data] == 3:                              # if the number of terminals needed by node is three
            ops: typ.List[str] = random.choices(OPS, k=3)                # pick the needed amount of OPs
            
            left: Node = tree.addLeft(parent=node, data=ops.pop(0))      # create & add the new left node to the tree
            right: Node = tree.addRight(parent=node, data=ops.pop(0))    # create & add the new right node to the tree
            middle: Node = tree.addMiddle(parent=node, data=ops.pop(0))  # create & add the new middle node to the tree
            
            __full(classId, left, tree)                                   # call grow on left to set it's children
            __full(classId, right, tree)                                  # call grow on right to set it's children
            __full(classId, middle, tree)                                 # call grow on middle to set it's children
            return tree.getRoot()                                         # return the root node of the tree
        
        else:  # if NUM_TERMINALS was not 1 or 2
            raise IndexError("Grow could not find the number of terminals need")


def createInitialPopulation() -> Population:
    """Creates the initial population by calling createHypothesis() the needed number of times"""
    
    def createHypothesis() -> Hypothesis:
        """Helper function that creates a single hypothesis"""
        # given a list of trees, create a hypothesis
        # NOTE this will make 1 tree for each feature, and 1 CF for each class

        classIds: typ.List[int] = copy.copy(CLASS_IDS)  # create a copy of all the unique class ids

        ftrs: typ.List[ConstructedFeature] = []
        size = 0

        # ? should this be LABEL_NUMBER or FEATURE_NUMBER
        for _ in CLASS_IDS:  # create a CF for each class
            # randomly decide if grow or full should be used.
            # Also randomly assign the class ID then remove that ID
            # so each ID may only be used once
            
            try:
                name = classIds.pop(0)     # get a random id
                if name not in CLASS_IDS:  # make sure name is a valid class id
                    raise Exception(f'createHypothesis got an invalid name ({name}) from classIds')
            
            except IndexError:                         # if classIds.pop() tried to pop an empty list, log error & exit
                lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                log.error(f'Index error encountered in createInitialPopulation (popped from an empty list), line {lineNm}')
                print(f'ERROR: Index error encountered in createInitialPopulation (popped from an empty list), line {lineNm}')
                sys.exit(-1)                           # exit on error; recovery not possible
            except Exception as err:                   # if class ids some how gave an invalid name
                lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
                log.error(str(err))
                printError(f'ERROR: {str(err)}, line {lineNm}')
                sys.exit(-1)                           # exit on error; recovery not possible
            # DEBUG if we pass this point we know that name is valid
            
            tree = Tree()   # create an empty tree
            tree.addRoot()  # create a root node for the tree
            
            '''# !!!!!!!!!!!!!!!!!!!!! Used for Testing Only !!!!!!!!!!!!!!!!!!!!! #
            root = tree.addRoot()  # create a root node for the tree
            if not (root.is_root()):  # + This does pass so root doesn't have parents
                raise Exception('Root is not tree root')
            if root is not tree.getRoot():
                raise Exception('Root is not equal to tree root')
            # __grow(name, root, tree)    # create tree using grow
            __full(name, root, tree)      # create tree using full
            tree.checkTree()
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #'''
            
            if random.choice([True, False]):         # *** use grow *** #
                __grow(name, tree.getRoot(), tree)   # create tree using grow
            else:                                    # *** use full *** #
                __full(name, tree.getRoot(), tree)   # create tree using full

            cf = ConstructedFeature(name, tree)                   # create constructed feature
            ftrs.append(cf)                                       # add the feature to the list of features

            size += size
            
        if classIds:  # if we didn't pop everything from the classIds list, raise an exception
            log.error(f'creatInitialPopulation didn\'t use all of classIds, classIds = {classIds}')
            raise Exception(f'ERROR: creatInitialPopulation didn\'t use all of classIds, classIds = {classIds}')
            
        # create a hypothesis & return it
        return Hypothesis(ftrs, size)

    hypothesis: typ.List[Hypothesis] = []

    # creat a number hypotheses equal to pop size
    with alive_bar(POPULATION_SIZE, title="Initial Hypotheses") as bar:
        for __ in range(POPULATION_SIZE):  # iterate as usual
            hyp = createHypothesis()       # create a Hypothesis
            # sanityCheckHyp(hyp)            # ! testing purposes only!
            hypothesis.append(hyp)         # add the new hypothesis to the list
            bar()
    
    # sanityCheckPop(hypothesis)  # ! testing purposes only!
    return Population(hypothesis, 0)


# ********** Sanity Check Functions used for Debugging ********** #
# ! testing purposes only!
def sanityCheckPop(hypothesis: typ.List[Hypothesis]):
    """Used in debugging to check a Population"""
    log.debug('Starting Population Sanity Check...')
    for h in hypothesis:
        h.sanityCheck()
    log.debug('Population Sanity Check Passed')


# ! testing purposes only!
def sanityCheckCF(cf: ConstructedFeature):
    """Used in debugging to check a Constructed Feature"""
    log.debug('Starting Constructed Feature Sanity Check...')
    cf.transform(rows[0])
    log.debug('Constructed Feature Sanity Check Passed')


# ! testing purposes only!
def sanityCheckTree(tree: Tree, classId):
    """Used in debugging to check a Tree"""
    log.debug('Starting Tree Sanity Check...')
    tree.checkTree()
    tree.runTree(rows[0].attributes, classId)
    log.debug('Tree Sanity Check Passed')
# *************************************************************** #


def evolve(population: Population, elite: Hypothesis) -> typ.Tuple[Population, Hypothesis]:
    """evolve is used by CDFC during the evolution step to create the next generation of the algorithm.
       This is done by randomly choosing between mutation & crossover based on the mutation rate.
    
        Parameter:
            population (Population): Population to be evolved.
            elite (Hypothesis): Highest scoring Hypothesis created so far.
            
        Functions:
            tournament: Finds Constructed Features to be mutated/crossover-ed.
            mutate: Performs the mutation operation.
            crossover: Performs the crossover operation.
    """

    def __tournament(p: Population) -> Hypothesis:
        """Used by evolution to selection the parent(s)"""

        # **************** Tournament Selection **************** #
        candidates: typ.List[Hypothesis] = copy.deepcopy(p.candidateHypotheses)  # copy to avoid overwriting
        first = None  # the tournament winner
        score = 0     # the winning score
        for i in range(TOURNEY):  # compare TOURNEY number of random hypothesis
            randomIndex = random.choice(range(len(candidates)))   # get a random index value
            candidate: Hypothesis = candidates.pop(randomIndex)   # get the hypothesis at the random index
            # we pop here to avoid getting duplicates. The index uses candidates current size so it will be in range
            # log.debug('Making getFitness method call in Tournament')
            fitness = candidate.getFitness()                      # get that hypothesis's fitness score
            # log.debug('Finished getFitness method call in Tournament')

            if first is None:      # if first has not been set,
                first = candidate  # then  set it

            elif score < fitness:  # if first is set, but knight is more fit,
                first = candidate  # then update it
                score = fitness    # then update the score to higher fitness
                
        try:
            if first is None:
                raise Exception(f'ERROR: Tournament could not set first correctly, first = {first}')
        except Exception as err2:
            lineNm2 = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            log.error(f'Tournament could not set first correctly, first = {first}, line number = {lineNm2}')
            print(f'{str(err2)}, line = {lineNm2}')
            sys.exit(-1)                            # exit on error; recovery not possible
        
        # log.debug('Finished Tournament method')
        # bar.text('found parent')  # ! for debugging
        return first
        # ************ End of Tournament Selection ************* #

    # ******************* Evolution ******************* #
    # create a new population with no hypotheses (made here crossover & mutate can access it)
    newPopulation = Population([], population.generation+1)

    def mutate() -> None:
        """Finds a random node and builds a new sub-tree starting at it. Currently mutate
           uses the same grow & full methods as the initial population generation without
           an offset. This means that mutate trees will still obey the max depth rule.
        """
        
        log.debug('Starting mutation')
        # ******************* Fetch Values Needed ******************* #
        parent: Hypothesis = __tournament(population)                # get copy of a parent Hypothesis using tournament
        randIndex: int = random.choice(range(len(parent.features)))  # get a random index
        randCF: ConstructedFeature = parent.features[randIndex]      # get a random Constructed Feature
        terminals = randCF.relevantFeatures                          # save the indexes of the relevant features
        tree: Tree = randCF.tree                                     # get the tree from the CF
        # tree.checkTree()     # ! For Testing purposes only !!
        # tree.sendToStdOut()  # ! For Testing purposes only !!
        node: Node = randCF.tree.getRandomNode()                     # get a random node from the CF's tree
        # *********************************************************** #
        
        # ************* Remove the Children of the Node ************* #
        children: typ.List[Node] = tree.children(node.identifier)   # get all the children
        [tree.remove_node(child.identifier) for child in children]  # delete all the children
        # *********************************************************** #
    
        # ************************* Mutate ************************* #
        if random.choice(['OPS', 'TERM']) == 'TERM' or tree.depth(node.identifier) == MAX_DEPTH:
            node.data = random.choice(terminals)  # if we are at max depth or choose TERM,
    
        else:  # if we choose to add an OP
            node.data = random.choice(OPS)  # give the node a random OP
        
            # randomly decide which method to use to construct the new tree (grow or full)
            if random.choice(['Grow', 'Full']) == 'Grow':  # * Grow * #
                __grow(randCF.className, node, tree)       # tree is changed in place starting with node
            else:                                          # * Full * #
                __full(randCF.className, node, tree)       # tree is changed in place starting with node
        # *********************************************************** #

        # tree.checkTree()     # ! For Testing purposes only !!
        # tree.sendToStdOut()  # ! For Testing purposes only !!
        
        # overwrite old CF with the new one
        parent.features[randIndex] = ConstructedFeature(randCF.className, randCF.tree)
        # add the mutated parent to the new pop (appending is needed because parent is a copy NOT a reference)
        newPopulation.candidateHypotheses.append(parent)
    
    def crossover() -> None:
        """Performs the crossover operation on two trees"""
        
        log.debug('Starting Crossover')
        
        # * Find Random Parents * #
        parent1: Hypothesis = __tournament(population)  # parent1 & parent2 are from a copy of population made by tournament, NOT original pop
        parent2: Hypothesis = __tournament(population)  # because of this they should not be viewed as references
        
        #  check that each parent is unique
        # if they are the same they should reference the same object & so 'is' is used instead of ==
        while parent1 is parent2:
            parent2 = __tournament(population)

        # * Get CFs from the Same Class * #
        # Feature 1
        feature1: ConstructedFeature = random.choice(parent1.features)  # get a random feature from the parent
        tree1: Tree = feature1.tree                                     # get the tree
        # tree1.checkTree()        # ! For Testing Only !!
        # tree1.sendToStdOut()     # ! For Testing Only !!

        # Feature 2
        feature2: ConstructedFeature = parent2.idsToFeatures[feature1.className]  # makes sure CFs are from/for the same class
        tree2: Tree = feature2.tree                                               # get the tree
        # tree2.checkTree()        # ! For Testing Only !!
        # tree2.sendToStdOut()     # ! For Testing Only !!
    
        # *************** Find the Two Sub-Trees **************** #
        node1: Node = tree1.getRandomNode()           # get a random node
        branch1, p1 = tree1.getBranch(node1)          # get the branch string
        subTree1: Tree = tree1.remove_subtree(node1.identifier)  # get a sub-tree with node1 as root
        # subTree1.sendToStdOut()  # ! For Testing Only !!

        node2: Node = tree2.getRandomNode()           # get a random node
        branch2, p2 = tree2.getBranch(node2)          # get the branch string
        subTree2: Tree = tree2.remove_subtree(node2.identifier)  # get a sub-tree with node2 as root
        # subTree2.sendToStdOut()  # ! For Testing Only !!
        # ******************************************************* #
    
        # ************************** swap the two subtrees ************************** #
        tree1.addSubTree(parent=p1, branch=branch1, subtree=subTree1)  # update the first parent tree
        tree2.addSubTree(parent=p2, branch=branch2, subtree=subTree2)  # update the second parent tree
        # **************************************************************************** #

        # !!!!!!!!!!!!!! For Testing Only !!!!!!!!!!!!!! #
        # Print the trees to see if crossover broke them/performed correctly
        # print('Crossover Finished')
        # print('Parent Tree 1:')
        # print('1st Swapped Tree:')
        # tree1.sendToStdOut()
        # tree1.checkTree()
        # print('Parent Tree 2:')
        # print('2nd Swapped Tree:')
        # tree2.sendToStdOut()
        # tree2.checkTree()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    
        # parent 1 & 2 are both hypotheses and should have been changed in place,
        # but they refer to a copy made in tournament so add them to the new pop
        newPopulation.candidateHypotheses.append(parent1)
        newPopulation.candidateHypotheses.append(parent2)

    # each iteration evolves 1 new candidate hypothesis, and we want to do this until
    # range(newPopulation.candidateHypotheses) = POPULATION_SIZE so loop over pop size
    with alive_bar(POPULATION_SIZE, title="Evolving") as bar:  # declare your expected total
        for pop in range(POPULATION_SIZE):
            # print(f' evolving {pop}/{POPULATION_SIZE}...')  # update user on progress
            probability = random.uniform(0, 1)              # get a random number between 0 & 1
        
            # mutate()     # ! For Testing Only
            # crossover()  # ! For Testing Only
            # bar()        # ! For Testing Only
        
            # ***************** Mutate ***************** #
            if probability < MUTATION_RATE:  # if probability is less than mutation rate, mutate
                bar.text('mutating...')      # update user
                mutate()                     # perform mutation
                bar()                        # update bar now that a candidate is finished
            # ************* End of Mutation ************* #

            # **************** Crossover **************** #
            else:                        # if probability is greater than mutation rate, use crossover
                bar.text('crossing...')  # update user
                crossover()              # perform crossover operation
                bar()                    # update bar now that a candidate is finished
            # ************* End of Crossover ************* #
            
            # ****************** Elitism ****************** #
            # check that if latest hypothesis has a higher fitness than our current elite
            newHypothFitness = newPopulation.candidateHypotheses[-1].getFitness()
            if newHypothFitness > elite.getFitness():
                elite = newPopulation.candidateHypotheses[-1]
            # ************** End of Elitism *************** #

    return newPopulation, elite


def cdfc(dataIn) -> Hypothesis:
    """cdfc is the 'main' of cdfc.py. It is called by cdfcProject which passes dataIn.
       It then creates an initial population & evolves several hypotheses. After going
       through a set amount ofr generations it returns a Hypothesis object.
       
       Parameters:
           dataIn (tuple): Index 0 contains the values of the global constants that cdfc needs,
                           and index 1 contains the TERMINALS dictionary.
           
       Return:
           bestHypothesis: Hypothesis with the highest fitness score.
    """

    values = dataIn[0]
    
    # makes sure we're using global variables
    global FEATURE_NUMBER
    global CLASS_IDS
    global POPULATION_SIZE
    global INSTANCES_NUMBER
    global LABEL_NUMBER
    global M
    global rows
    global ENTROPY_OF_S  # * used in entropy calculation * #
    global CLASS_DICTS
    global TERMINALS
    
    # Read the values in the dictionary into the constants
    FEATURE_NUMBER = values['FEATURE_NUMBER']
    CLASS_IDS = values['CLASS_IDS']
    # POPULATION_SIZE = values['POPULATION_SIZE']
    POPULATION_SIZE = 10
    INSTANCES_NUMBER = values['INSTANCES_NUMBER']
    LABEL_NUMBER = values['LABEL_NUMBER']
    M = values['M']
    rows = values['rows']
    ENTROPY_OF_S = values['ENTROPY_OF_S']
    CLASS_DICTS = values['CLASS_DICTS']
    TERMINALS = dataIn[1]
    
    # *********************** Run the Algorithm *********************** #

    currentPopulation = createInitialPopulation()     # run initialPop/create the initial population
    SYSOUT.write(HDR + ' Initial population generated '.ljust(50, '-') + SUCCESS)
    elite = currentPopulation.candidateHypotheses[0]  # init elitism

    # loop, evolving each generation. This is where most of the work is done
    SYSOUT.write('\nStarting generations stage...\n')  # update user
    
    for gen in range(GENERATIONS):  # iterate as usual
        print(f'\n{HDR} Starting Generation {gen}/{GENERATIONS}')
        newPopulation, elite = evolve(currentPopulation, elite)  # generate a new population by evolving the old one
        # update currentPopulation to hold the new population
        # this is done in two steps to avoid potential namespace issues
        currentPopulation = newPopulation

    SYSOUT.write(HDR + ' Final Generation Reached \n'.ljust(50, '-') + SUCCESS)  # update user
    # ***************************************************************** #

    # ****************** Return the Best Hypothesis ******************* #
    log.debug('Finding best hypothesis')

    bestHypothesis = max(currentPopulation.candidateHypotheses, key=lambda x: x.fitness)
    log.debug('Found best hypothesis, returning...')
    return bestHypothesis  # return the best hypothesis generated
