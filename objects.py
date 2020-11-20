"""
objects.py contains several of the data structures used by both cdfcProject & cdfc. It exists primarily for
organizational purposes, rather than structural necessity.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo: https://github.com/brom94/cdfcPython.git
"""

import logging as log
import pprint
import random
import sys
import traceback
import typing as typ
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
from treelib import Node as Node
from treelib import Tree as libTree
from treelib.exceptions import NodeIDAbsentError

from formatting import printError

# OPS is the list of valid operations on the tree
OPS: typ.Final = ['add', 'subtract', 'times', 'max', 'if']


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


class WrapperInstance:
    """Instance is an instance, or row, within our data set. It represents a single
       record of whatever we are measuring. This replaces the earlier row struct.

    Variables:
        className (int): The class id of the class that the record is in (this class ids start at 0).
        attribute ({key: int, value: float}): This dictionary stores the features values in the
                                                  instance, keyed by index (these start at 0)
        vList ([float]): The list of the values stored in the dictionary (used to speed up iteration)
    """
    
    def __init__(self, className: int, values: np.ndarray):
        # this stores the name of the class that the instance is in
        self.className: int = className
        # this stores the values of the features, keyed by index, in the instance
        self.attributes: typ.Dict[int, float] = dict(zip(range(values.size), values))
        # this creates a list of the values stored in the dictionary for when iteration is wanted
        self.vList: typ.List[float] = values.tolist()  # + this will be a list[float], ignore warnings
        
        # ! For Debugging Only
        # try:  # check that all the feature values are valid
        #     if None in self.vList:  # if a None is in the list of feature values
        #         raise Exception('Tried to create an Instance obj with a None feature value')
        # except Exception as err:
        #     log.error(str(err))
        #     tqdm.write(str(err))
        #     traceback.print_stack()
        #     sys.exit(-1)  # exit on error; recovery not possible
    
    def __array__(self) -> np.array:
        """Converts an Instance to an Numpy array."""
        return np.array([float(self.className)] + self.vList)


class cdfcInstance:
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
        
        try:  # check that all the feature values are valid
            if None in self.attributes.values():  # if a None is in the list of feature values
                raise Exception('Tried to create an Instance obj with a None feature value')
        except Exception as err:
            log.error(str(err))
            printError(str(err))
            traceback.print_stack()  # print stack trace so we know how None is reaching Instance
            sys.exit(-1)  # exit on error; recovery not possible
    
    def __array__(self) -> np.array:
        """Converts an Instance to an Numpy array."""
        return np.array([float(self.className)] + self.vList)


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
        subTree.BRANCHES = self.BRANCHES.copy()  # copy the dict from the original tree into the subtree
        return subTree  # return the subtree
    
    def printNodes(self) -> None:
        """Prints all the nodes in a tree & then the tree."""
        # pprint(self.all_nodes())                # print all the nodes using pretty print
        
        for n in self.all_nodes_itr():  # loop over every node
            if self.BRANCHES[n.identifier] == {}:  # if the n is a terminal, skip
                continue
            print(n)  # print the node
            print(self.BRANCHES[n.identifier])  # print the dictionary for that node
        
        self.show(idhidden=False)  # print the tree
        out = Path.cwd() / 'logs' / 'tree.log'  # create the file path
        self.save2file(out)  # save to tree.log
    
    def checkTree(self) -> None:
        """Checks the tree dictionary"""
        # self.printNodes()           # print the tree being checked
        for n in self.all_nodes():  # loop over every node
            try:
                if not self.contains(n.identifier):  # check that the node is in the tree
                    log.error(f'Tree found node that was not in tree: {n}')
                    print(f'Tree found node that was not in tree: {n}')
                
                if n.data not in OPS:  # if this is a terminal node
                    continue  # skip this iteration
                
                lft: Node = self.get_node(self.BRANCHES[n.identifier]['left'])  # all nodes should have a left
                r: Node = self.get_node(self.BRANCHES[n.identifier]['right'])  # all nodes should have a right
                
                if lft is None or r is None:  # if getting left or right key got a None, raise exception
                    pprint(self.children(n.identifier))
                    print(f'left found: {lft}\nright found: {r}')
                    raise AssertionError('Getting left and/or right key failed')
                if lft not in self.children(n.identifier):  # check that the left child is valid
                    pprint(self.children(n.identifier))
                    print(lft)
                    raise AssertionError(f'Left child was set incorrectly parent = {n}')
                if r not in self.children(n.identifier):  # check that the right child is valid
                    pprint(self.children(n.identifier))
                    print(r)
                    raise AssertionError(f'Right child was set incorrectly parent = {n}')
                
                if n.data == 'if':  # if the node stores an IF OP
                    m = self.get_node(self.BRANCHES[n.identifier]['middle'])
                    if m is None:  # if getting middle key failed, throw exception
                        raise AssertionError('Getting middle key failed')
                    if not (m in self.children(n.identifier)):  # check that the middle child is valid
                        raise AssertionError('Left child was set incorrectly')
            # NOTE: KeyError indicates that the key (left, right, middle) doesn't exist,
            # +     AssertionError means it does exist & stores a None
            except KeyError:
                lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
                log.error(f'CheckTree raised a KeyError, on line {lineNm}')  # log the error
                printError(
                    f'CheckTree raised a KeyError, dict = {self.BRANCHES[n.identifier].items()}, on line {lineNm}')  # print message
                traceback.print_stack()  # print stack trace
                sys.exit(-1)  # exit on error; recovery not possible
            
            except AssertionError as err:
                lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
                printError(str(err))
                log.error(f'{str(err)}, on line {lineNm}')  # log the error
                printError(
                    f'CheckTree raised a AssertionError, dict = {self.BRANCHES[n.identifier].items()}, on line {lineNm}')  # print message
                traceback.print_stack()  # print stack trace
                sys.exit(-1)  # exit on error; recovery not possible
    
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
        root = self.create_node(tag=f'root: {op}', identifier=f'Tree {next(TreeCount)}',
                                data=op)  # create a root node for the tree
        return root
    
    def addLeft(self, parent: Node, data: TREE_DATA) -> Node:
        """Sets the left child of a tree."""
        # create the node & add it to the tree
        new: Node = self.create_node(tag=str(data), parent=parent.identifier,
                                     data=data, identifier=f'{str(data)}({next(NodeCount)})')
        
        # update dictionary used by getLeft(), getRight(), & getMiddle()
        # store new Node ID at ParentId, 'left' (overwriting any old values)
        self.BRANCHES[parent.identifier]['left'] = new.identifier
        
        # ! For Testing Only !! - attempt to access created entry
        # try:
        #     self.BRANCHES[parent.identifier]['left']
        # except KeyError:
        #     msg: str = f'Tree encountered an error in addLeft(), {parent}'
        #     lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        #     log.error(msg)  # log the error
        #     printError(msg)  # print message
        #     printError(f'Error on line {lineNm}')
        #     traceback.print_stack()  # print stack trace
        #     sys.exit(-1)  # exit on error; recovery not possible
        
        return new
    
    def addRight(self, parent: Node, data: TREE_DATA) -> Node:
        """Sets the right child of a tree."""
        
        # create the node & add it to the tree
        new: Node = self.create_node(tag=str(data), parent=parent.identifier,
                                     data=data, identifier=f'{str(data)}({next(NodeCount)})')
        
        # update dictionary used by getLeft(), getRight(), & getMiddle()
        # store new Node ID at ParentId, 'left' (overwriting any old values)
        self.BRANCHES[parent.identifier]['right'] = new.identifier
        
        # ! For Testing Only !! - attempt to access created entry
        # try:
        #     self.BRANCHES[parent.identifier]['right']
        # except KeyError:
        #     msg: str = f'Tree encountered an error in addRight(), {parent}'
        #     lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        #     log.error(msg)  # log the error
        #     printError(msg)  # print message
        #     printError(f'Error on line {lineNm}')
        #     traceback.print_stack()  # print stack trace
        #     sys.exit(-1)  # exit on error; recovery not possible
        
        return new
    
    def addMiddle(self, parent: Node, data: TREE_DATA) -> Node:
        """Sets the middle child of a tree."""
        
        # create the node & add it to the tree
        new: Node = self.create_node(tag=str(data), parent=parent.identifier,
                                     data=data, identifier=f'{str(data)}({next(NodeCount)})')
        
        # update dictionary used by getLeft(), getRight(), & getMiddle()
        # store new Node ID at ParentId, 'left' (overwriting any old values)
        self.BRANCHES[parent.identifier]['middle'] = new.identifier
        # ! For Testing Only !! - attempt to access created entry
        # try:
        #     self.BRANCHES[parent.identifier]['middle']
        # except KeyError:
        #     msg: str = f'Tree encountered an error in addMiddle(), {parent}'
        #     lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        #     log.error(msg)  # log the error
        #     printError(msg)  # print message
        #     printError(f'Error on line {lineNm}')
        #     traceback.print_stack()  # print stack trace
        #     sys.exit(-1)  # exit on error; recovery not possible
        
        return new
    
    def addSubTree(self, parent: Node, branch: str, subtree: "Tree"):
        """Adds a subtree as a child of parent. This will also update the BRANCHES
           dictionary & should overwrite any old values. This shouldn't delete
           values from the old tree's dictionary as subtree is a copy"""
        
        try:
            self.BRANCHES[parent.identifier][branch] = subtree.root  # add the subtree root to the dictionary
            
            # this shouldn't delete values from the old tree's dictionary as subtree is a copy
            for nid in subtree.nodes.keys():  # loop over all the nids in the subtree
                # ? this makes a shallow copy. Is that okay or should it be deep?
                self.BRANCHES[nid] = subtree.BRANCHES[nid].copy()  # copy the sub-dictionary over
            del subtree.BRANCHES  # delete the old dictionary
            
            # add the subtree to the original tree as a child of parent
            # deep=False because the subtree has been deleted from the old tree so can't affect it anymore.
            # However is subtree is used after this it will need to be True
            self.paste(parent.identifier, subtree)  # ? should deep be true or false?
        
        except NodeIDAbsentError as err:  # catch error thrown by deep paste
            print('adding to:')
            self.sendToStdOut()  # print the tree
            print('subTree:')
            subtree.sendToStdOut()
            msg: str = f'addSubTree failed, {str(err)}'  # create the message
            log.error(msg)  # log the error
            printError(msg)  # print message
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
        n: Node = random.choice(self.all_nodes())  # pick a random node
        
        if (n is self.getRoot()) or not (self.contains(n.identifier)):
            nodeList = self.all_nodes()  # get all the node
            random.shuffle(nodeList)  # shuffle them
            
            # if we picked the root or a node not in the tree
            while (n is self.getRoot()) or not (self.contains(n.identifier)):
                if len(nodeList) == 0:  # if nodeList is empty
                    raise Exception('getRandomNode could not find a node in the tree')
                n = nodeList.pop(0)  # pop the top of the list
        
        return n
    
    def getLeft(self, parent: Node) -> typ.Optional[Node]:
        """Gets the left child of a tree."""
        try:
            dct = self.BRANCHES[parent.identifier]
            nid: str = dct['left']  # get the nodes Id
            return self.get_node(nid)  # get the node & return NOTE: if the nid not in tree this will trigger error
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f'getLeft() could not find left for {parent}, on line {lineNm}'
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Node\'s children: {self.children(parent.identifier)}')
            # printError(f'Error on line {lineNm}')
            # traceback.print_stack()  # print stack trace
            sys.exit(-1)
            # return None              # attempt recovery
    
    def getRight(self, parent: Node) -> typ.Optional[Node]:
        """Gets the right child of a tree."""
        try:
            nid: str = self.BRANCHES[parent.identifier]['right']  # get the nodes Id
            return self.get_node(nid)  # get the node & return
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f'getRight() could not find right for {parent}, on line {lineNm}'
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Node\'s children: {self.children(parent.identifier)}')
            # printError(f'Error on line {lineNm}')
            # traceback.print_stack()  # print stack trace
            sys.exit(-1)
            # return None              # attempt recovery
    
    def getMiddle(self, parent: Node) -> typ.Optional[Node]:
        """Gets the middle child of a tree."""
        try:
            nid: str = self.BRANCHES[parent.identifier]['middle']  # get the nodes Id
            return self.get_node(nid)  # get the node & return
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f'getMiddle() could not find middle for {parent}, on line {lineNm}'
            log.error(msg)  # log the error
            printError(msg)  # print message
            printError(f'Node\'s children: {self.children(parent.identifier)}')
            # printError(f'Error on line {lineNm}')
            # traceback.print_stack()  # print stack trace
            sys.exit(-1)
            # return None              # attempt recovery
    
    def getBranch(self, child: Node) -> typ.Tuple[str, Node]:
        """Given a child, this returns what branch of it's parent it was on."""
        parent: Node = self.parent(child.identifier)  # get the parents Id
        children: typ.Dict[str, str] = self.BRANCHES[parent.identifier]  # get all of the parents children
        
        for branch in children.keys():  # loop over all children
            if self.BRANCHES[parent.identifier][branch] == child.identifier:  # if this branch maps to the child
                return branch, parent  # return the branch
        raise Exception('getBranch was not able to find a connection between the child & parent')
    
    # running a tree should return a single value
    # featureValues -- the values of the relevant features keyed by their index in the original data
    def runTree(self, featureValues: typ.Dict[int, float], classId: int,
                terminals: typ.Dict[int, typ.List[int]]) -> float:
        """runTree is a wrapper for runNode & is used to __transform provided data
           by walking the decision tree

        Parameters:
            featureValues ({key: int, value: float}): The dictionary mapping feature ids to
                                                      their values (in the current instance).
            classId (int): Class ID the tree is meant to classify.
            terminals (typ.Dict[int, typ.List[int]]): The dictionary that maps class ids to their relevant features.

        Returns:
            (float): The final value that the decision tree creates given the provided data.
        """
        
        return self.__runNode(featureValues, self.getRoot(), classId, terminals)
    
    def __runNode(self, featureValues: typ.Dict[int, float], node: Node,
                  classId: int, terminals: typ.Dict[int, typ.List[int]]) -> typ.Union[int, float]:
        """runTree is a wrapper for runNode & is used to __transform provided data
           by walking the decision tree

        Parameters:
            featureValues ({key: int, value: float}): The dictionary mapping feature ids to
                                                      their values (in the current instance).
            terminals (typ.Dict[int, typ.List[int]]): The dictionary that maps class ids to their relevant features.

        Returns:
            (float): The value of a terminal, or the value computed by one or more operations.
        """
        
        try:
            
            if node.data in OPS:  # if the node is an OP
                # *************************** Error Checking *************************** #
                # ! For Debugging Only
                # lftNone: bool = self.getLeft(node) is None                          # is left None?
                # rgtNone: bool = self.getRight(node) is None                         # is right None?
                # xor: bool = (lftNone and not rgtNone) or (not lftNone and rgtNone)  # exclusive or
                # if xor:                                             # if one child is None, but not both
                #     raise AssertionError(f'runNode found a node in OPS with 1 \'None\' child,\n\t node = {node}')
                # if lftNone and rgtNone:                             # if both children are None
                #     raise AssertionError(f'runNode found a node in OPS with 2 \'None\' children,\n\t node = {node}')
                # if node.data == 'if' and self.getMiddle(node) is None:  # if the OP is IF and it has no middle
                #     raise AssertionError('runNode found a node with a IF OP and no middle node')
                # ************ Determine Which OP is Stored & Run Recursion ************ #
                left: Node = self.getLeft(node)  # get the left child (all OPS wil have a left)
                right: Node = self.getRight(node)  # get the right child (all OPS wil have a right)
                
                if node.data == 'add':  # if the OP was add
                    vl = (self.__runNode(featureValues, left, classId, terminals) +  # left + right
                          self.__runNode(featureValues, right, classId, terminals))
                    return vl
                
                elif node.data == 'subtract':  # if the OP was subtract
                    vl = (self.__runNode(featureValues, left, classId, terminals) -  # left - right
                          self.__runNode(featureValues, right, classId, terminals))
                    return vl
                
                elif node.data == 'times':  # if the OP was multiplication
                    vl = (self.__runNode(featureValues, left, classId, terminals) *  # left * right
                          self.__runNode(featureValues, right, classId, terminals))
                    return vl
                
                elif node.data == 'max':  # if the OP was max
                    vl = max(self.__runNode(featureValues, left, classId, terminals),  # max(left, right)
                             self.__runNode(featureValues, right, classId, terminals))
                    return vl
                
                elif node.data == 'if':  # if the OP was if
                    if self.__runNode(featureValues, left, classId, terminals) >= 0:  # if the left value is positive,
                        vl = self.__runNode(featureValues, right, classId, terminals)  # return the right node
                    else:  # if the left value is negative,
                        middle: Node = self.getMiddle(node)  # get the middle child
                        vl = self.__runNode(featureValues, middle, classId, terminals)  # return the middle node
                    return vl
                # ********************************************************************* #
            
            elif node.data in terminals[classId]:  # if the node is a terminal
                # *************************** Error Checking *************************** #
                # ! For Debugging Only
                # if math.isnan(node.data):             # if the value stored is a NaN
                #     msg: str = f'NaN stored in tree. Expected a class ID, OPS value, or number, got {node.data}'
                #     raise TypeError(f'ERROR: {msg}')  # raise TypeError
                #
                # if featureValues[node.data] is None:  # if the value stored is a None
                #     raise TypeError(f'featureValues contained a None at index {node.data}')
                # ************************ Return Terminal Value ************************ #
                return featureValues[node.data]  # if the terminal is valid, return it
                # *********************************************************************** #
            
            else:  # if the node is not a terminal or a OP
                raise TypeError(f'runNode could not parse data in tree, data ={node.data}')
        
        except (TypeError, AssertionError) as err:  # catch any exceptions
            self.sendToStdOut()  # record the tree
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(f'line = {lineNm}, {str(err)}'
                      f'\ndict = {self.BRANCHES[node.identifier]}'
                      f'\n{self.all_nodes()}')  # log the error
            printError(f'line = {lineNm}, {str(err)}')  # print message
            printError(f'dict = {self.BRANCHES[node.identifier]}')
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
