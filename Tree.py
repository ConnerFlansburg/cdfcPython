
import typing as typ
from Node import Node
import random
import sys
import uuid
import logging as log
import traceback
from formatting import printError
import pprint
from copy import deepcopy
# from copy import copy as cpy

TREE_DATA = typ.Union[int, str]  # typing alias for tree nodes. They will either have a terminal index (int) or an OP (str)

OPS: typ.Final = ['add', 'subtract', 'times', 'max', 'if']  # OPS is the list of valid operations on the tree
NUM_TERMINALS = {'add': 2, 'subtract': 2,                   # NUM_TERMINALS is a dict that, when given an OP as a key, give the number of terminals it needs
                 'times': 2, 'max': 2, 'if': 3}
sys.setrecursionlimit(10000)                                # set the recursion limit for the program


class Tree:
    
    def __init__(self, root: typ.Optional[Node] = None, nodes: typ.Optional[typ.Dict[str, Node]] = None, ID: str = None):
        """ Trees are almost always made empty & then built by adding to them."""
        if nodes is None:
            nodes = {}
        
        if root is None:  # create a root if none was provided
            op = random.choice(OPS)
            root: Node = Node(tag=f'root: {op}', data=op)  # create a root node for the tree
            nodes[root.ID] = root  # add to the node dictionary
        
        if ID is None:
            self._ID: str = str(uuid.uuid4())
        else:
            self._ID: str = ID
        
        self._root: Node = root  # set the root as root

        # this dictionary will hold the nodes in the tree, keyed by their id value
        self._nodes: typ.Dict[str, Node] = nodes
        
        # the values below are used by methods & should not be set
        # this dictionary is used when copying a subtree 8 should never be used elsewhere
        self._copyDictionary: typ.Dict[str, Node] = {}
        
        self._rValue = None  # this will be used by recursive search

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
        # call recursive print starting with root
        return self.__print_tree(self._root.ID, 0)
    
    def __repr__(self):
        return self.__str__()
        
    def __print_tree(self, currentID: str, level: int) -> str:
    
        # add the current node using the Node str() method
        out: str = "\t"*level + str(self._nodes[currentID])+'\n'
        
        # if there are children, print them
        if self._nodes[currentID].hasChildren:
            
            # print the left child
            if self._nodes.get(currentID).left:
                out += self.__print_tree(self._nodes.get(currentID).left, level+1)
            else:
                out += '\t'*(level+1)+'\033[91;1m[\u2718 -- Left Child Missing]\033[00m'
            
            # print the middle child (if one exists)
            if self._nodes.get(currentID).hasMiddle:
                out += self.__print_tree(self._nodes[currentID].middle, level+1)
            
            # print the right child
            if self._nodes.get(currentID).right:
                out += self.__print_tree(self._nodes.get(currentID).right, level+1)
            else:
                out += '\t' * (level + 1) + '\033[91;1m[\u2718 -- Right Child Missing]\033[00m'
        return out

    # *** ID *** #
    @property
    def ID(self):
        return self._ID

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
        new: Node = Node(tag=f'root: {op}', data=op)  # create a root node for the tree
        self._nodes[new.ID] = new  # add to the node dictionary
        self._root = new  # set the root
        return self._root.ID
      
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
        new = Node(tag=f'{data}', data=data, parent=parentID, branch='left')
        
        # make the parent aware of it
        self._nodes[parentID].left = new.ID  # type: str
        
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
        
        return new.ID
    
    def addRight(self, parentID: str, data: TREE_DATA) -> str:
        
        # create the new node (id will be made by node's init)
        new = Node(tag=f'{data}', data=data, parent=parentID, branch='right')

        # make the parent aware of it
        self._nodes[parentID].right = new.ID  # type: str
    
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
    
        return new.ID
    
    def addMiddle(self, parentID: str, data: TREE_DATA) -> str:
        
        # create the new node (id will be made by node's init)
        new = Node(tag=f'{data}', data=data, parent=parentID, branch='middle')

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
    def getDepth(self, ID: str) -> int:
        
        try:
            # if the node isn't in the tree at all
            if ID not in self._nodes.keys():
                raise NotInTreeError(ID)
            
            # if the root has not been set first
            if self._root is None:
                print('Root was not set before calling getDepth')
                raise Exception
            
            # if the id is valid
            elif self._nodes.get(ID):
            
                # run rSearch
                self.__search(ID, self._root.ID, 0)  # ! There's a bug thrown here
                
                # if rValue is none, then the node could not be found
                if self._rValue is None:
                    raise NotInTreeError(ID)
                # if rValue is not None, then it was found so return depth
                else:
                    return self._rValue
    
            else:  # if the key is bad, raise an error
                raise NotInTreeError(ID)
        except NotInTreeError as err:
            lineNm = sys.exc_info()[-1].tb_lineno       # get the line number of error
            log.error(f'line = {lineNm}, {str(err)}')   # log the error
            printError(f'line = {lineNm}, {str(err)}')  # print message
            print(f'Root: {self._root},')
            pprint.pprint(self)
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible

    # BUG: this throws a NodeNotInTree error we getting depth
    def __search(self, targetID: str, currentID: str, depth: int):
        
        # *** check for a bad ID *** #
        try:
            
            if targetID is None:
                printError('search target was None')
                sys.exit(-1)
            
            if currentID is None:
                printError('search was passed a None ID')
                sys.exit(-1)
            
            # is the target ID in the tree?
            if targetID not in list(self._nodes.keys()):
                printError('target ID of search was not in the tree')
                raise NotInTreeError(targetID)
            
            # is the node ID in the list of valid keys for the tree?
            if currentID not in list(self._nodes.keys()):
                raise NotInTreeError(currentID)
        
            # does get() work on the node ID?
            if self._nodes.get(currentID) is None:
                # is it storing a None, or does the key not exist?
                # if the key does not exist this will raise a key error
                if self._nodes[currentID] is None:
                    # if the key is valid, then a None is being stored here
                    raise NullNodeError(currentID)
        
        except NotInTreeError as err:  # catch errors that check the key is valid
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            printError(f'NotInTreeError encountered on line {lineNm}')
            if err.key == targetID:  # if the target ID threw the error
                printError(f'The target of search() is not in the tree')
                printError(f'Invalid ID = {targetID}')
            else:                    # if the search discovered an invalid ID
                printError(f'search found an invalid ID while searching for target')
                printError(f'Invalid ID = {currentID}')
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
        
        except KeyError:  # catch any error from indexing key
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            printError(f'KeyError encountered on line {lineNm}')
            printError(f'get() works for ID {currentID}, but index notation fails')
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
    
        except NullNodeError:  # this will be reached if the tree is storing Nones
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            printError(f'NullNodeError encountered on line {lineNm}')
            printError(f'ID {currentID} is a valid key value, but the object it keys to is None')
            printError(''.join(traceback.format_stack()))  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
        
        # *** Do the Search *** #
        # if we have reached the target node, return depth
        if currentID == targetID:
            return depth
        
        # if this is not the target depth then call search on the children
        else:
            depthLeft: int = 0
            depthRight: int = 0
            depthMiddle: int = 0
            
            # if this node is not an operation, but a terminal
            if not (self._nodes[currentID].data in OPS):
                #  we have reached a leaf node, so return
                return 0
            
            # if this node is an operation, it will have children; search them
            else:
                if self._nodes[currentID].left:  # if there is a left, search it
                    depthLeft: int = self.__search(targetID, self._nodes[currentID].left, depth+1)
                    
                if self._nodes[currentID].right:  # if there is a right, search it
                    depthRight: int = self.__search(targetID, self._nodes[currentID].right, depth+1)
                    
                if self._nodes[currentID].middle:  # if there is a middle, search it
                    depthMiddle: int = self.__search(targetID, self._nodes[currentID].middle, depth+1)
        # everything but the actual depth should return 0, so add them together
        return depthLeft + depthRight + depthMiddle

    def getBranch(self, childID: str) -> str:
        return self._nodes[childID].branch
    
    # *** Operations *** #
    def getRandomNode(self) -> str:
        """ Get a random node from the tree (leaves are allowed)"""
        options: list[str] = list(self._nodes.keys())
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
            print('removeChildren was passed a node ID of None ')
            raise NotInTreeError(nodeID)
        
        # delete the children & any of their children
        for child in self._nodes[nodeID].children:
            self.__rDelete(child)

        # !!! debugging only !!! #
        # self.checkForDuplicateKeys(subtree)  # ! debugging
        # print('removeChildren is checking for missing IDs...')
        # self.checkForMissingKeys()
        # print('No missing keys detected\n')
        # !!! debugging only !!! #
        
        return
    
    def __rDelete(self, currentID: str, copy: bool = False):
        """ This can be used to delete a subtree or copy it  """
        
        if self._nodes.get(currentID):  # if the current node is in the tree
    
            # *** Recursion *** #
            # delete the left node
            self.__rDelete(self._nodes[currentID].left)
            # delete the right node
            self.__rDelete(self._nodes[currentID].right)
            # delete the middle node
            self.__rDelete(self._nodes[currentID].middle)
            # *** End of Recursion *** #
    
            # after we have reached the leaves of the tree, return up
            # the stack, deleting/copying as we go
    
            # *** Copy *** #
            if copy:  # if we are creating a subtree
                # this should not raise a key error because of the earlier IF statement
                # TODO: experiment with these 3 options & decide on one
                # + So dictionaries copying a dictionary would require using copy, but we
                # +     aren't copying a dictionary, but a Node. Does a Node need a copy call?
                # +     Custom objects typically need copy calls. Deepcopy is needed for nest objects,
                # +     so a tree would require deepcopy, but a node just needs copy?
                # self._copyDictionary[currentID] = self._nodes[currentID]
                self._copyDictionary[currentID] = deepcopy(self._nodes[currentID])
                # self._copyDictionary[currentID] = cpy(self._nodes[currentID])
            # *** End of Copy *** #
    
            # *** Delete *** #
            # NOTE: we don't want to set the parent to point to None.
            # This is because we want to copy their structure into subtree
            
            # check that children have been deleted, if not raise error
            # if self._nodes[currentID].hasChildren:
            #     printError('Recursive delete attempt to delete a parent who\'s children were not None')
            #     raise Exception
            
            del self._nodes[currentID]  # delete the current node from the original tree
            # *** End of Delete *** #

        # if we have hit the bottom of the tree, or node didn't have child
        elif currentID is None:
            return
        else:  # if the node is not in the tree, raise an error
            raise NotInTreeError(currentID)
    
    # BUG: currently this is returning a parentID of none.
    def removeSubtree(self, newRootID: str) -> ("Tree", str, str):
        
        # if the node is in the tree
        if self._nodes.get(newRootID):
    
            # get the parents id & branch
            parentOfSubtreeID: str = self._nodes[newRootID].parent
            orphanBranch: str = self._nodes[newRootID].branch
            
            if parentOfSubtreeID is None:  # see if the parent is None
                printError('Parent Stored in Node was stored as None')
                raise MissingNodeError(role='Parent', ID=newRootID)
            
            # set the new root to be a root
            self._nodes[newRootID].parent = None
            self._nodes[newRootID].branch = None
            
            # we don't need to set the parents to none as adding the next
            # subtree will do this for us. This is done for debugging
            # if orphanBranch == 'left':
            #     self._nodes[parentOfSubtreeID].left = None
            # elif orphanBranch == 'right':
            #     self._nodes[parentOfSubtreeID].right = None
            # elif orphanBranch == 'middle':
            #     self._nodes[parentOfSubtreeID].middle = None
            # else:
            #     raise InvalidBranchError(f'removeSubtree found an invalid branch: {orphanBranch}')
            
            self._copyDictionary = {}             # make sure the copy dictionary is empty
            rt = self._nodes[newRootID]           # get the root of the subtree
            self.__rDelete(newRootID, copy=True)  # copy the subtree & delete it from original
            # copyDictionary now contains the subtree, so build
            # the new subtree using a copy of copyDictionary
            subtree: Tree = Tree(root=rt, nodes=deepcopy(self._copyDictionary))
            self._copyDictionary = {}             # reset the value of copyDictionary
            
            # !!! debugging only !!! #
            # self.checkForDuplicateKeys(subtree)  # ! debugging
            # print('removeSubtree is checking for missing IDs...')  # !!!!! THIS IS GETTING RAISED !!!!! #
            # self.checkForMissingKeys(originalDict=self._nodes, subtreeDict=subtree._nodes)
            # print('No missing keys detected\n')
            # if parentOfSubtreeID is None:  # see if the parent is None before returning it
            #     printError('Parent ID was invalidate by removeSubtree')
            #     raise MissingNodeError(role='Parent', ID=newRootID)
            # !!! debugging only !!! #
            
            # BUG: parentOfSubtreeID is None
            return subtree, parentOfSubtreeID, orphanBranch
    
        else:  # if the key is bad, raise an error
            raise NotInTreeError(newRootID)
    
    # ! this should be used in debugging only !
    # ? is there a way to if the subtree is getting extra IDs from the original that it doesn't need?
    def checkForMissingKeys(self, originalDict=None, subtreeDict=None):
        keys = list(self._nodes.keys())
        childIDs = []
        nodeID: typ.Optional[str] = None
        try:
            # + if the error is thrown here than tree not every ID is a key (the nodes exist, they just can't be found)
            for nodeID in keys:  # loop over every node id in the tree
                # try to find every node in the tree
                self.__search(targetID=nodeID, currentID=self.root.ID, depth=0)
                # get the ids of all the children
                childIDs += self._nodes[nodeID].children
        
        except KeyError:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            if nodeID:  # if for some reason we can't add to childIDs
                printError(f'KeyError encountered line {lineNm}')
                printError(f'Node ID {nodeID} could not be accessed while fetching children')
            else:  # some unexpected KeyError
                printError(f'Unpredicted KeyError thrown on line {lineNm}')
        
        except NotInTreeError as err:
            lineNm = sys.exc_info()[-1].tb_lineno       # get the line number of error
            printError(''.join(traceback.format_stack()))  # print stack trace
            log.error(f'line = {lineNm}, {str(err)}')   # log the error
            
            printError(f'NotInTree error encountered on line {lineNm}, for ID {err.key}')
            printError(f'Not every valid key is in the tree')
            
            # print the two dictionaries
            if originalDict and subtreeDict:
                print('Original Tree:')
                pprint.pprint(list(originalDict.keys()))
                print('Generated Subtree:')
                pprint.pprint(list(subtreeDict.keys()))
            
            sys.exit(-1)  # exit on error; recovery not possible
        
        try:
            # + if the error is triggered here that nodes have children that are not in the tree
            for nodeID in childIDs:  # now loop over the collected child ids
                self.__search(nodeID, self.root.ID, 0)  # try to find every child
        except NotInTreeError as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
            printError(''.join(traceback.format_stack()))  # print stack trace
            log.error(f'line = {lineNm}, {str(err)}')  # log the error
    
            printError(f'NotInTree error encountered on line {lineNm}, for ID {err.key}')
            printError(f'Nodes have children that are not in the tree')

            # print the two dictionaries
            if originalDict and subtreeDict:
                print('Original Tree:')
                pprint.pprint(list(originalDict.keys()))
                print('Generated Subtree:')
                pprint.pprint(list(subtreeDict.keys()))
    
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
        # check that parent id is valid
        if newParent is None:
            print('addSubtree was given a None (root?) newParent')
        if not (self._nodes.get(newParent)):  # ! this is being raised because the nodeID is None
            raise MissingNodeError(msg=f'addSubtree could not find it\'s new parent')
        
        # set the adopted parents to point to the subtree
        if orphanBranch == 'left':
            self._nodes[newParent].left = subtree._root.ID
        elif orphanBranch == 'right':
            self._nodes[newParent].right = subtree._root.ID
        elif orphanBranch == 'middle':
            self._nodes[newParent].middle = subtree._root.ID
        else:
            raise InvalidBranchError(f'addSubtree encountered an invalid branch: {orphanBranch}')
    
        # set the subtree root to point to adopted parents
        subtree._root.parent = newParent
        # set the subtree's branch
        subtree.branch = orphanBranch
        
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
    
        return self.__runNode(featureValues, self._root, classId, terminals)

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

    # ******************* Growing the Tree ******************* #
    def grow(self, classId: int, nodeID: str, MAX_DEPTH: int, TERMINALS: typ.Dict[int, typ.List[int]]):
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
        
        node: Node = self._nodes[nodeID]

        coin = random.choice(['OP', 'TERM']) == 'TERM'  # flip a coin & decide OP or TERM
    
        # *************************** A Terminal was Chosen *************************** #
        # NOTE: check depth-1 because we will create children
        # print(f'Grow Node ID: {node.ID}, ID Passed {nodeID}')  # ! debugging
        if coin == 'TERM' or (self.getDepth(node.ID) == MAX_DEPTH - 1):  # if we need to add terminals
        
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
            
                self.grow(classId, leftID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call grow on left to set it's children
                self.grow(classId, rightID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call grow on right to set it's children
                return
            
            elif NUM_TERMINALS[node.data] == 3:  # if the number of terminals needed by node is three
                ops: typ.List[str] = random.choices(OPS, k=3)  # pick the needed amount of OPs
            
                left: str = self.addLeft(parentID=node.ID, data=ops.pop(0))  # create & add the new left node to the tree
                right: str = self.addRight(parentID=node.ID, data=ops.pop(0))  # create & add the new right node to the tree
                middle: str = self.addMiddle(parentID=node.ID, data=ops.pop(0))  # create & add the new middle node to the tree
            
                self.grow(classId, left, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call grow on left to set it's children
                self.grow(classId, right, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call grow on right to set it's children
                self.grow(classId, middle, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call grow on middle to set it's children
                return
        
            else:  # if NUM_TERMINALS was not 1 or 2
                raise IndexError("Grow could not find the number of terminals need")

    def full(self, classId: int, nodeID: str, MAX_DEPTH: int, TERMINALS: typ.Dict[int, typ.List[int]]):
        """
        Full creates a tree or sub-tree starting at the Node node, and using the Full method.
        If node is a root Node, full will build a tree, otherwise full will build a sub-tree
        starting at node. Full assumes that node's data has already been set & makes all
        changes in place.

        NOTE:
        During testing whatever calls full should use the sanity check sanityCheckTree(newTree)
 
        :param classId: ID of the class that the tree should identify.
        :param nodeID: The root node of the subtree __full will create.
        :param MAX_DEPTH: Max depth of trees
        :param TERMINALS: Terminals values

        :type classId: int
        :type nodeID: str
        :type MAX_DEPTH: int
        :type TERMINALS: typ.Dict[int, typ.List[int]]
        """
        try:
            node: Node = self._nodes[nodeID]
            
            # *************************** Max Depth Reached *************************** #
            if self.getDepth(node.ID) == (MAX_DEPTH - 1):

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
                
                    self.full(classId, leftID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call full on left to set it's children
                    self.full(classId, rightID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call full on right to set it's children
                    return
                
                elif NUM_TERMINALS[node.data] == 3:  # if the number of terminals needed by node is three
                    ops: typ.List[str] = random.choices(OPS, k=3)  # pick the needed amount of OPs
                
                    leftID: str = self.addLeft(parentID=node.ID, data=ops.pop(0))  # create & add the new left node to the tree
                    rightID: str = self.addRight(parentID=node.ID, data=ops.pop(0))  # create & add the new right node to the tree
                    middleID: str = self.addMiddle(parentID=node.ID, data=ops.pop(0))  # create & add the new middle node to the tree
    
                    self.full(classId, leftID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call full on left to set it's children
                    self.full(classId, rightID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call full on right to set it's children
                    self.full(classId, middleID, MAX_DEPTH=MAX_DEPTH, TERMINALS=TERMINALS)  # call full on middle to set it's children
                    return
            
                else:  # if NUM_TERMINALS was not 1 or 2
                    raise IndexError("Full could not find the number of terminals need")
        except RecursionError as err:
            lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
            print(f'{str(err)}, line = {lineNm}')
            print(f'Node ID: {nodeID},\nDepth: {self.getDepth(nodeID)}')
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
