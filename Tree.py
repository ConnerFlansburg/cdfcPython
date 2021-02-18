
import typing as typ
from Node import Node
import random
import sys
import uuid
import logging as log
import traceback
from formatting import printError
import pprint
from io import StringIO
from contextlib import redirect_stdout
# from copy import copy as cpy

TREE_DATA = typ.Union[int, str]  # typing alias for tree nodes. They will either have a terminal index (int) or an OP (str)

OPS: typ.Final = ['add', 'subtract', 'times', 'max', 'if']  # OPS is the list of valid operations on the tree
NUM_TERMINALS = {'add': 2, 'subtract': 2,                   # NUM_TERMINALS is a dict that, when given an OP as a key, give the number of terminals it needs
                 'times': 2, 'max': 2, 'if': 3}
sys.setrecursionlimit(10000)                                # set the recursion limit for the program

# TODO: add documentation to classes & methods


class Tree:
    
    def __init__(self, root: Node = None, nodes: typ.Dict[str, Node] = None, ID: str = None):
        """ Trees are almost always made empty & then built by adding to them."""
        if nodes is None:
            nodes = {}
        
        if ID is None:
            self._ID: str = str(uuid.uuid4())
        else:
            self._ID: str = ID

        # this dictionary will hold the nodes in the tree, keyed by their id value
        self._nodes: typ.Dict[str, Node] = nodes
        
        # * Set the Root Node * #
        if root is None:  # create a root if none was provided
            self.__addRoot()
        else:
            self._root = root
        
        # the values below are used by methods & should not be set
        # this dictionary is used when copying a subtree 8 should never be used elsewhere
        self._copyDictionary: typ.Dict[str, Node] = {}

    def __eq__(self, tree2: "Tree"):
        """
        Allows two trees to be compared using ==. This is done
        by comparing the ID values of each tree.
        """
        
        if self._ID == tree2.ID:
            return True
        else:
            return False

    def __neq__(self, tree2: "Tree"):
        """
        Allows two trees to be compared using !=. This is done
        by comparing the ID values of each tree.
        """
    
        if self._ID != tree2.ID:
            return True
        else:
            return False

    def __str__(self):
        out: str
        try:  # attempt to use better print method
            # call recursive print starting with root
            out = self.__print_tree()
        # if we aren't able to use the nicer print, use simple
        except Exception as err:
            printError(f'Encountered an error while printing tree: {str(err)}')
            printError('Switching to simple print...')
            # call the simpler print, overriding anything in out
            out = self.__print_tree_simple()
        return out
    
    def __repr__(self):
        return self.__str__()
    
    def __print_tree_simple(self) -> str:
        keys = list(self._nodes.keys())
        vls = list(self._nodes.values())
        out: str = f'Tree ID {self.ID}\nKeys:\n'
        for k in keys:  # loop over the keys
            out += f'\t{k}\n'
        out += 'Nodes:\n'
        for n in vls:  # loop over the nodes
            out += f'\t{n}\n'
        return out
    
    def __print_tree(self) -> str:
        out = StringIO()  # creat the string variable
        with redirect_stdout(out):  # redirect standard out
            print(f'Tree {self.ID}')  # print the tree's id
            self.__rPrint(self.root.ID, "", True)
            print('\n')  # print a new line
        # turn it into a string & return
        return out.getvalue()

    def __rPrint(self, nodeID: str, indent: str, isLast: bool):
    
        node: Node = self._nodes.get(nodeID)
        
        # This should print an error if nodeID was invalid
        if node is None:
            thisNode = '\u2612'
            isLeaf: bool = True
        else:
            thisNode: str = str(node)
            isLeaf: bool = node.isLeaf()
    
        if nodeID == self.root.ID:  # if this is the root of a tree
            print(f'{indent}{thisNode}')  # print this node
            indent += "   "
            print(f"{indent}\u2503")
        elif isLast:  # if this is the last child of a node
            print(f'{indent}\u2517\u2501{thisNode}')  # print this node
            indent += "   "
            if isLeaf:  # if it is a leaf, don't print the extra bar
                print(f"{indent}")
            else:
                print(f"{indent}\u2503")
        else:  # if this is not the last child
            print(f'{indent}\u2523\u2501{thisNode}')  # print this node
            indent += "\u2503   "
            if isLeaf:  # if it is a leaf, don't print the extra bar
                print(f"{indent}")
            else:
                print(f"{indent}\u2503")

        children = ('left', 'middle', 'right')
    
        for child in children:
            if child == 'left' and (node.left is not None):
                self.__rPrint(node.left, indent, False)
            elif child == 'middle' and (node.middle is not None):
                self.__rPrint(node.middle, indent, False)
            elif child == 'right' and (node.right is not None):
                self.__rPrint(node.right, indent, True)
        return
    
    # *** ID *** #
    @property
    def ID(self):
        return self._ID
    
    @ID.setter
    def ID(self, newID):
        self._ID = newID
        
    # TODO: check this & generateNewNodeIDs
    def generateNewIDs(self):
        
        # * Update the Tree's ID
        self._ID = str(uuid.uuid4())
        
        # * Update the IDs of the Nodes * #
        self._generateNewNodeIDs(self.root.ID)
        
    def _generateNewNodeIDs(self, currentID: str):
    
        # get the node needing to be updated
        current: Node = self._nodes.get(currentID)
        
        if current is None:
            raise AssertionError
        
        # generate a new ID for current node
        current.ID = str(uuid.uuid4())
        newID: str = current.ID
        
        # get the children
        left: Node = self._nodes.get(current.left)
        middle: Node = self._nodes.get(current.middle)
        right: Node = self._nodes.get(current.right)
        
        # if the child exists, update it's parent ID and
        # call generateNewNodeID on it
        if left:
            left.parent = newID
            self._generateNewNodeIDs(current.left)
            
        if middle:
            middle.parent = newID
            self._generateNewNodeIDs(current.middle)

        if right:
            right.parent = newID
            self._generateNewNodeIDs(current.right)

    # *** Copy Dictionary *** #
    @property
    def copyDictionary(self):
        return self._copyDictionary
    
    # *** Root *** #
    @property
    def root(self):
        # if the root hasn't been set
        if self._root is None:
            raise RootNotSetError
        return self._root

    @root.setter
    def root(self, newRoot):
        self._root = newRoot
    
    def __addRoot(self) -> str:
        """Adds a root node to the tree"""
        op = random.choice(OPS)
        new: Node = Node(data=op)  # create a root node for the tree
        self._nodes[new.ID] = new                     # add to the node dictionary
        self._root = new                              # set the root
        return self._root.ID
    
    def overrideRoot(self, new_root: Node):
        """ This should only be used to test the Tree object """
        self._nodes[new_root.ID] = new_root  # add the new root to dict
        self._root = new_root                # update the root property
    
    # *** Size *** #
    @property
    def size(self):
        return len(self._nodes.values())

    # *** Methods *** #
    
    # *** Add Children *** #
    # + whenever we add using one of these we create a new node
    # +  so no ids should ever conflict
    def addLeft(self, parentID: str, data: TREE_DATA) -> str:

        # create the new node (id will be made by node's init)
        new = Node(data=data, parent=parentID, branch='left')
        
        # make the parent aware of it
        self._nodes[parentID].left = new.ID  # type: str
        
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
        
        return new.ID
    
    def addRight(self, parentID: str, data: TREE_DATA) -> str:
        
        # create the new node (id will be made by node's init)
        new = Node(data=data, parent=parentID, branch='right')

        # make the parent aware of it
        self._nodes[parentID].right = new.ID  # type: str
    
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
    
        return new.ID
    
    def addMiddle(self, parentID: str, data: TREE_DATA) -> str:
        
        # create the new node (id will be made by node's init)
        new = Node(data=data, parent=parentID, branch='middle')

        # make the parent aware of it
        self._nodes[parentID].middle = new.ID  # type: str
    
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
    
        return new.ID

    # *** Get Children *** #
    
    def getNode(self, targetID) -> Node:
        if self._nodes.get(targetID):
            return self._nodes[targetID]
        else:
            raise NotInTreeError(targetID)
    
    def getLeft(self, parentID: str) -> typ.Optional[Node]:
        """ Used to get the left child of a node """
        hasLeft = self._nodes.get(self._nodes[parentID].left)
        validParent = self._nodes.get(parentID)

        if validParent and hasLeft:
            return self._nodes[self._nodes[parentID].left]
        else:
            return None

    def getRight(self, parentID: str) -> typ.Optional[Node]:
        """ Used to get the right child of a node """
        hasRight = self._nodes.get(self._nodes[parentID].right)
        validParent = self._nodes.get(parentID)

        if validParent and hasRight:
            return self._nodes[self._nodes[parentID].right]
        else:
            return None

    def getMiddle(self, parentID: str) -> typ.Optional[Node]:
        """ Used to get the middle child of a node """
        hasMiddle = self._nodes.get(self._nodes[parentID].middle)
        validParent = self._nodes.get(parentID)
        
        if validParent and hasMiddle:
            return self._nodes[self._nodes[parentID].middle]
        else:
            return None
    
    def children(self, parentID: str) -> typ.Optional[typ.List[Node]]:
        parent: Node = self._nodes[parentID]  # get the parent node
        children = []  # this will hold the children as we find them

        # if the node has children, add them to children
        if parent.hasChildren:
            children.append(parent.left)
            children.append(parent.right)
            if parent.hasMiddle:
                children.append(parent.middle)
            return children  # return the children
        else:  # if the node has no children, return None
            return None

    # *** Values *** #
    def getDepth(self, targetID: str, currentID: str, depth=0) -> int:
    
        # * If the Current Node is Root Return Depth * #
        if currentID == self.root.ID:
            return depth
        
        # * Get the Current Node * #
        try:
            if currentID is None:  # if the ID is None
                raise AssertionError('getDepth was given a currentID of None')
            
            current: Node = self._nodes[currentID]  # this might raise a key error
            
            if current is None:  # if the Node could not be indexed
                raise NullNodeError(currentID)
            
        except AssertionError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f"geDepth was passed a currentID of None: line {lineNm}"
            log.error(msg)
            printError(msg)
            printError(f'ID is of Type: {type(currentID)}')
            print(f"\n{self}")
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible

        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f"geDepth was passed a currentID not in the Tree: line {lineNm}\nID: {currentID}"
            log.error(msg)
            printError(msg)
            printError(f'ID is of Type: {type(currentID)}')
            print(f"\n{self}")
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible

        except NotInTreeError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            msg: str = f'getDepth found that the Node with ID {currentID} was None: line {lineNm}'
            log.error(msg)
            printError(msg)
            printError(f'ID is of Type: {type(currentID)}')
            print(f"\n{self}")
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible

        # * Get the Parent of the Current Node * #
        parent = self._nodes.get(current.parent)
        if parent is None:  # if the Node could not be indexed
            raise NotInTreeError(f'newSearch could not find parent of Node with ID {currentID}')
        
        # * Step Up the Tree Increasing Depth * #
        return self.getDepth(targetID, parent.ID, depth+1)

    def getBranch(self, childID: str) -> str:
        return self._nodes[childID].branch
    
    # *** Operations *** #
    def getRandomNode(self) -> str:
        """ Get a random node from the tree (leaves are allowed)"""
        options: typ.List[str] = list(self._nodes.keys())
        options.remove(self._root.ID)
        
        return random.choice(options)
    
    def removeChildren(self, nodeID: str):
        """
        Used to remove a all the children of a node from the tree.
        This will delete the children of a node, but not the node
        itself.
        """
        # if the key isn't in the tree, raise an error
        if not (self._nodes.get(nodeID)):
            raise NotInTreeError(nodeID)
        
        if nodeID is None:
            raise NotInTreeError(nodeID)
        
        # delete the children & any of their children
        for child in self._nodes[nodeID].children:
            self.__rDelete(child, rootID=child)

        # remove the references to the now deleted children
        self._nodes[nodeID].left = None
        self._nodes[nodeID].right = None
        self._nodes[nodeID].middle = None
        
        return
    
    def testDelete(self, currentID: str, rootID: str, copy: bool = False):
        self.__rDelete(currentID, rootID, copy)
    
    def __rDelete(self, currentID: str, rootID: str, makeCopy: bool = False):
        """
        This will delete a subtree from the original tree
        (storing it in copyDictionary if requested).
        
        NOTE: __rDelete has been tested & works as expected
        """
        current: Node = self._nodes.get(currentID)  # get the current Node
        
        if current is None:  # if get failed
            return
        
        # * If This is the Subtree's Root, Deal with Parents Still in Tree * #
        if currentID == rootID:  # if we are looking at the root of the subtree
            branch: str = current.branch  # get what branch of parent current is on
            parentID: str = current.parent            # get the parents ID
            parent: Node = self._nodes.get(parentID)  # get the parent Node

            # if parent IS None they this is root so don't mess with parents
            if parent is not None:
                
                current.branch = None  # * Root is Not on a Branch so Set to Null * #
                
                # * Deal with Parent's Left/Right/Middle Value * #
                if branch == 'left':
                    parent.left = None
                elif branch == 'right':
                    parent.right = None
                elif branch == 'middle':
                    parent.middle = None
        
        if self._nodes.get(currentID):  # if the current node is in the tree
    
            # *** Recursion *** #
            # delete the left node
            self.__rDelete(self._nodes[currentID].left, rootID, makeCopy)
            # delete the right node
            self.__rDelete(self._nodes[currentID].right, rootID, makeCopy)
            # delete the middle node
            self.__rDelete(self._nodes[currentID].middle, rootID, makeCopy)
            # *** End of Recursion *** #
    
            # after we have reached the leaves of the tree, return up
            # the stack, deleting/copying as we go
    
            # *** Copy *** #
            if makeCopy:  # if we are creating a subtree
                # this should not raise a key error because of the earlier IF statement
                # NOTE: don't use copy as that will generate new nodes & change node IDs
                self._copyDictionary[currentID] = self._nodes[currentID]
            # *** End of Copy *** #

            # *** Delete Current Node from Original Tree *** #
            del self._nodes[currentID]

        # if we have hit the bottom of the tree, or node didn't have child
        elif currentID is None:
            return
        else:  # if the node is not in the tree, raise an error
            try:
                raise NotInTreeError(currentID)  # BUG: this is getting raised
            except NotInTreeError:
                lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
                printError(f'NotInTreeError encountered on line {lineNm} of Tree.py')
                printError(f'Node with ID {currentID} could not be found in tree by rDelete()')
                print(self)  # print the tree
                print('\n')
                printError(''.join(traceback.format_stack()))  # print stack trace
                sys.exit(-1)  # exit on error; recovery not possible

    def removeSubtree(self, newRootID: str) -> ("Tree", str, str):
        # NOTE: removeSubtree has been tested & works
        
        # if the node is in the tree
        if self._nodes.get(newRootID):
    
            # get the parents id & branch
            parentOfSubtreeID: str = self._nodes[newRootID].parent
            orphanBranch: str = self._nodes[newRootID].branch
            if orphanBranch is None:
                printError(f'Found None branch on Node with ID {newRootID}')
                print(self)
            
            if parentOfSubtreeID is None:  # see if the parent is None
                printError('Parent Stored in Node was stored as None')
                raise MissingNodeError(role='Parent', ID=newRootID)
            
            # *** Create/Get the New Root *** #
            rt = self._nodes[newRootID]  # get the root of the subtree
            
            # *** Create a Copy of the Tree Below the Root, Starting with Root *** #
            self._copyDictionary = {}    # make sure the copy dictionary is empty
            self.__rDelete(newRootID, rootID=newRootID, makeCopy=True)  # copy the subtree & delete it from original
            
            # *** Build a new Subtree Using the Copy *** #
            # NOTE: we set ID to self so we can check that it isn't added back to the same tree
            subtree: Tree = Tree(root=rt, nodes=self._copyDictionary, ID=self.ID)
            self._copyDictionary = {}  # empty copyDictionary

            return subtree, parentOfSubtreeID, orphanBranch
    
        else:  # if the key is bad, raise an error
            try:
                raise NotInTreeError(newRootID)
            except NotInTreeError:
                lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
                printError(f'NotInTreeError encountered by removeSubtree on line {lineNm} of Tree.py')
                printError(f'Node with ID {newRootID} could not be found in tree by removeSubtree')
                print(self)  # print the tree
                print('\n')
                printError(''.join(traceback.format_stack()))  # print stack trace
                sys.exit(-1)  # exit on error; recovery not possible
                
    # ! call this after crossover `& use it to hut down what's causing the duplicates
    def checkForDuplicateKeys(self, otherTree: "Tree"):
        """ Given two trees, check them for duplicate keys """

        # ! duplicate nodes are being found after initial pop generation
        # check that there aren't any duplicate keys
        duplicates = []
        for key1 in otherTree._nodes.keys():  # for every key in subtree,
            if key1 in self._nodes.keys():  # if that key is also in this tree,
                duplicates.append(key1)  # add the key to the list of copies

        try:
            if duplicates:  # if duplicates were found, raise an error
                raise DuplicateNodeError(keyList=duplicates)
        except DuplicateNodeError as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(f'line = {lineNm}, {str(err)}')  # log the error
            printError(''.join(traceback.format_stack()))  # print stack trace
            printError(f'On line {lineNm} DuplicateNodeError encountered')  # print message
            
            print('Duplicate(s):')
            pprint.pprint(duplicates)  # print the list of duplicate nodes
            print('Subtree:')
            pprint.pprint(list(otherTree._nodes.keys()))
            
            for k in duplicates:  # loop over the duplicates
                # print the data
                same = self._nodes.get(k) is otherTree._nodes.get(k)
                print(f'For Key: {k}')
                print(f'Do they share a memory address? {same}')
                print(f'Data in Original Node: {self._nodes.get(k).data}')
                print(f'Data in Subtree Node: {otherTree._nodes.get(k).data}')
                print(f'Children of Original Node: {self._nodes.get(k).children}')
                print(f'Children of Subtree Node: {otherTree._nodes.get(k).children}\n')
                
            sys.exit(-1)  # exit on error; recovery not possible

    def addSubtree(self, subtree: "Tree", newParent: str, orphanBranch: str):
        
        # *** Error Checking *** #
        # check that parent id is valid
        if newParent is None:
            print('addSubtree was given a None (root?) newParent')
        if self._nodes.get(newParent) is None:
            raise MissingNodeError(msg=f'addSubtree could not find it\'s new parent')
        # check that we aren't adding the subtree back onto it's original tree
        if self.ID == subtree.ID:
            printError(f'AddSubtree attempted to add itself back to it\'s original tree')
            raise AssertionError
        # *** End of Error Checking *** #
        
        # set the adopted parents to point to the subtree
        if orphanBranch == 'left':
            self._nodes[newParent].left = subtree._root.ID
        elif orphanBranch == 'right':
            self._nodes[newParent].right = subtree._root.ID
        elif orphanBranch == 'middle':
            self._nodes[newParent].middle = subtree._root.ID
        else:
            message: str = f"addSubtree encountered an invalid branch\nParent: {newParent}\nBranch: {orphanBranch}\n Subtree:\n{subtree}"
            raise InvalidBranchError(message)
    
        # set the subtree root to point to adopted parents
        subtree._root.parent = newParent
        # set the subtree's branch
        subtree._root.branch = orphanBranch
        
        # check for duplicate nodes
        # self.checkForDuplicateKeys(subtree)
        
        # it is now safe to add subtree to dictionary of nodes
        self._nodes.update(subtree._nodes)
        
        # delete the subtree from memory now that it's been copied
        del subtree
        
        return
    
    # TODO covert old code to new tree structure
    def runTree(self, featureValues: typ.Dict[int, float], classId: int,
                terminals: typ.Dict[int, typ.List[int]]) -> float:
        """
        runTree is a wrapper for runNode & is used to __transform provided data
        by walking the decision tree

        :param featureValues: The dictionary mapping feature ids to their values (in the current instance).
        :param classId: The class the tree is meant to identify (this is used to find the terminal values).
        :param terminals: The dictionary that maps class ids to their relevant features.

        :type featureValues: dict
        :type classId: The value of a terminal, or the value computed by one or more operations.
        :type terminals: dict

        :returns: The final value that the decision tree creates given the provided data.
        :rtype: float
        """
        try:
            value = self.__runNode(featureValues, self._root, classId, terminals)
        except Exception as err:
            lineNm = sys.exc_info()[-1].tb_lineno
            printError(f'runTree found an error on line {lineNm}: {str(err)}')
            print('')
            print(self)
            sys.exit(-1)
            
        return value

    def __runNode(self, featureValues: typ.Dict[int, float], node: Node,
                  classId: int, terminals: typ.Dict[int, typ.List[int]]) -> typ.Union[int, float]:
        """
        __runNode is used to transform provided data by walking the decision tree.

        :param featureValues: The dictionary mapping feature ids to their values (in the current instance).
        :param node: The node being examined (this is used during recursion).
        :param classId: The class the tree is meant to identify (this is used to find the terminal values).
        :param terminals: The dictionary that maps class ids to their relevant features.

        :type featureValues: dict
        :type node: Node
        :type classId: The value of a terminal, or the value computed by one or more operations.
        :type terminals: dict

        :returns: The transformed value.
        :rtype: float
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
                left: Node = self.getLeft(node.ID)  # get the left child (all OPS wil have a left)
                right: Node = self.getRight(node.ID)  # get the right child (all OPS wil have a right)
            
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
                        middle: Node = self.getMiddle(node.ID)  # get the middle child
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
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            log.error(f'line = {lineNm}, {str(err)}')  # log the error
            printError(f'line = {lineNm}, {str(err)}')  # print message
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
        except Exception as err:
            lineNm = sys.exc_info()[-1].tb_lineno
            printError(f'runNode found an error on line {lineNm}: {str(err)}')
            print('')
            print(self)
            sys.exit(-1)

    # ******************* Growing the Tree ******************* #
    def grow(self, classId: int, nodeID: str, MAX_DEPTH: int, TERMINALS: typ.Dict[int, typ.List[int]], depth: int):
        """
        Grow creates a tree or sub-tree starting at the Node node, and using the Grow method.
        If node is a root Node, grow will build a tree, otherwise grow will build a sub-tree
        starting at node. Grow assumes that node's data has already been set & makes all
        changes in place.

        NOTE:
        During testing whatever calls grow should use the sanity check sanityCheckTree(newTree)

        :param classId: ID of the class that the tree should identify.
        :param nodeID: The root node of the subtree __grow will create.
        :param MAX_DEPTH: Max tree depth allowed
        :param TERMINALS: Terminals for the class

        :type classId: int
        :type nodeID: str
        :type MAX_DEPTH: int
        """
        try:
            node: Node = self._nodes.get(nodeID)
    
            coin = random.choice(['OP', 'TERM']) == 'TERM'  # flip a coin & decide OP or TERM
        
            # *************************** A Terminal was Chosen *************************** #
            # NOTE: check depth-1 because we will create children
            # print(f'Grow Node ID: {node.ID}, ID Passed {nodeID}')  # ! debugging
            if coin == 'TERM' or (depth == MAX_DEPTH - 1):  # if we need to add terminals
            
                # pick the needed amount of terminals
                terms: typ.List[int] = random.choices(TERMINALS[classId], k=NUM_TERMINALS[node.data])
            
                if NUM_TERMINALS[node.data] == 2:  # if the OP needs 2 children
                    self.addLeft(parentID=node.ID, data=terms.pop(0))  # create a new left node & add it
                    self.addRight(parentID=node.ID, data=terms.pop(0))  # create a new left node & add it
                
                    return
            
                elif NUM_TERMINALS[node.data] == 3:  # if the OP needs 3 children
                    self.addLeft(parentID=node.ID, data=terms.pop(0))  # create a new left node & add it
                    self.addRight(parentID=node.ID, data=terms.pop(0))  # create a new right node & add it
                    self.addMiddle(parentID=node.ID, data=terms.pop(0))  # create a new middle node & add it
                
                    return
            
                else:  # if NUM_TERMINALS was not 2 or 3
                    raise IndexError("Grow could not find the number of terminals need")
        
            # *************************** A Operation was Chosen *************************** #
            else:  # if we chose to add an operation
            
                if NUM_TERMINALS[node.data] == 2:  # if the number of terminals needed by node is two
                    ops: typ.List[str] = random.choices(OPS, k=2)  # pick the needed amount of OPs
                
                    leftID: str = self.addLeft(parentID=node.ID, data=ops.pop(0))  # add the new left node
                    rightID: str = self.addRight(parentID=node.ID, data=ops.pop(0))  # add the new right node
                
                    self.grow(classId, leftID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call grow on left to set it's children
                    self.grow(classId, rightID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call grow on right to set it's children
                    return
                
                elif NUM_TERMINALS[node.data] == 3:  # if the number of terminals needed by node is three
                    ops: typ.List[str] = random.choices(OPS, k=3)  # pick the needed amount of OPs
                
                    left: str = self.addLeft(parentID=node.ID, data=ops.pop(0))  # create & add the new left node to the tree
                    right: str = self.addRight(parentID=node.ID, data=ops.pop(0))  # create & add the new right node to the tree
                    middle: str = self.addMiddle(parentID=node.ID, data=ops.pop(0))  # create & add the new middle node to the tree
                
                    self.grow(classId, left, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call grow on left to set it's children
                    self.grow(classId, right, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call grow on right to set it's children
                    self.grow(classId, middle, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call grow on middle to set it's children
                    return
            
                else:  # if NUM_TERMINALS was not 1 or 2
                    raise IndexError("Grow could not find the number of terminals need")
        except RecursionError as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            print(f'{str(err)}, line = {lineNm}')
            print(f'Node ID: {nodeID},\nDepth: {self.getDepth(nodeID, self.root.ID)}')
            # self.__print_tree_simple()
            sys.exit(-1)  # exit on error; recovery not possible

    def full(self, classId: int, nodeID: str, MAX_DEPTH: int, TERMINALS: typ.Dict[int, typ.List[int]], depth: int):
        """
        Full creates a tree or sub-tree starting at the Node node, and using the Full method.
        If node is a root Node, full will build a tree, otherwise full will build a sub-tree
        starting at node. Full assumes that node's data has already been set & makes all
        changes in place.

        NOTE:
        During testing whatever calls full should use the sanity check sanityCheckTree(newTree)
 
        :param depth: How deep full is in the current tree
        :param classId: ID of the class that the tree should identify.
        :param nodeID: The root node of the subtree __full will create.
        :param MAX_DEPTH: Max depth of trees
        :param TERMINALS: Terminals values

        :type depth: int
        :type classId: int
        :type nodeID: str
        :type MAX_DEPTH: int
        :type TERMINALS: typ.Dict[int, typ.List[int]]
        """
        try:
            node: Node = self._nodes[nodeID]
            
            # *************************** Max Depth Reached *************************** #
            if depth == (MAX_DEPTH - 1):

                # pick the needed amount of terminals
                terms: typ.List[int] = random.choices(TERMINALS[classId], k=NUM_TERMINALS[node.data])
            
                if NUM_TERMINALS[node.data] == 2:  # if the OP needs 2 children
                    self.addLeft(parentID=node.ID, data=terms.pop(0))  # create a new left node & add it
                    self.addRight(parentID=node.ID, data=terms.pop(0))  # create a right left node & add it
                    return
            
                elif NUM_TERMINALS[node.data] == 3:  # if the OP needs 3 children
                    self.addLeft(parentID=node.ID, data=terms.pop(0))  # create a new left node & add it
                    self.addRight(parentID=node.ID, data=terms.pop(0))  # create a new right node & add it
                    self.addMiddle(parentID=node.ID, data=terms.pop(0))  # create a new middle node & add it
                    return
            
                else:  # if NUM_TERMINALS was not 1 or 2
                    raise IndexError("Full could not find the number of terminals need")
        
            # *************************** If Not at Max Depth *************************** #
            else:  # if we haven't reached the max depth, add operations
            
                if NUM_TERMINALS[node.data] == 2:  # if the number of terminals needed by node is two
                    ops: typ.List[str] = random.choices(OPS, k=2)  # pick the needed amount of OPs
                
                    leftID: str = self.addLeft(parentID=node.ID, data=ops.pop(0))  # add the new left node
                    rightID: str = self.addRight(parentID=node.ID, data=ops.pop(0))  # add the new right node
                
                    self.full(classId, leftID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call full on left to set it's children
                    self.full(classId, rightID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call full on right to set it's children
                    return
                
                elif NUM_TERMINALS[node.data] == 3:  # if the number of terminals needed by node is three
                    ops: typ.List[str] = random.choices(OPS, k=3)  # pick the needed amount of OPs
                
                    leftID: str = self.addLeft(parentID=node.ID, data=ops.pop(0))  # create & add the new left node to the tree
                    rightID: str = self.addRight(parentID=node.ID, data=ops.pop(0))  # create & add the new right node to the tree
                    middleID: str = self.addMiddle(parentID=node.ID, data=ops.pop(0))  # create & add the new middle node to the tree
    
                    self.full(classId, leftID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call full on left to set it's children
                    self.full(classId, rightID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call full on right to set it's children
                    self.full(classId, middleID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS, depth=depth+1)  # call full on middle to set it's children
                    return
            
                else:  # if NUM_TERMINALS was not 1 or 2
                    raise IndexError("Full could not find the number of terminals need")
        except RecursionError as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            print(f'{str(err)}, line = {lineNm}')
            print(f'Node ID: {nodeID},\nDepth: {self.getDepth(nodeID, self.root.ID)}')
            # self.__print_tree()
            sys.exit(-1)  # exit on error; recovery not possible


class DuplicateNodeError(Exception):
    """ Thrown if a node with the same id is already in the tree """
    def __init__(self, key=None, keyList=None):
        self.key = key
        
        if keyList:
            out: str = pprint.pformat(keyList)
            self.message = f'Duplicate keys were found:\n{out}'
        elif key:
            self.message = f'Node with ID {self.key} is already in the tree'
        else:
            self.message = f'A duplicate node was found'
        
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class MissingNodeError(Exception):
    """
    Thrown if a node ID should exist (like a leaf's parent), but is not in the tree.
    This often means that the tree has been corrupted in some way.
    """
    
    def __init__(self, role=None, ID=None, msg=None):
        self.role = role
        self.ID = ID
        if role and ID:
            self.message = f'{role} Node for Node ID {ID} is missing from the tree'
        elif role:
            self.message = f'{role} Node is missing from the tree'
        elif ID:
            self.message = f'Node ID {ID} is missing necessary values'
        elif msg:
            self.message = msg
        else:
            self.message = f'Node is missing necessary values'
        
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class InvalidBranchError(Exception):
    
    def __init__(self, msg: str):
        self.message: str = msg
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class NullNodeError(Exception):
    """ Thrown if a node id is in the tree, but that node is None """

    def __init__(self, key: typ.Optional[str]):
        self.key: typ.Optional[str] = key
        self.message: str = f'NullNodeError: ID {key} valid, but stores a None instead of a Node'
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class NotInTreeError(Exception):
    """
    Thrown if a node id is simply not in the tree.
    This often means that the node passed was incorrect
    """
    
    def __init__(self, key: typ.Optional[str]):
        self.key: typ.Optional[str] = key
        self.message: str = f'Node with ID {key} is not in the tree'
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class RootNotSetError(Exception):
    
    def __str__(self):
        return 'Root accessed before it was set'
