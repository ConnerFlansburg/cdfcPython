
import typing as typ
from Node import Node
import random
import sys
import logging as log
import traceback
from formatting import printError

# typing alias for tree nodes. They will either have a terminal index (int) or an OP (str)
TREE_DATA = typ.Union[int, str]
# OPS is the list of valid operations on the tree
OPS: typ.Final = ['add', 'subtract', 'times', 'max', 'if']


class Tree:
    
    def __init__(self, root: typ.Optional[Node] = None, nodes: typ.Optional[typ.Dict[str, Node]] = None):
        """ Trees are almost always made empty & then built by adding to them."""
        if nodes is None:
            nodes = {}
        
        self._root: typ.Optional[Node] = root

        # this dictionary will hold the nodes in the tree, keyed by their id value
        self._nodes = nodes
        
        # this dictionary is used when copying a subtree 8 should never be used elsewhere
        self._copyDictionary = {}

    # *** Root *** #
    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, newRoot):
        self._root = newRoot

    @root.deleter
    def root(self):
        del self._root
    
    def addRoot(self) -> Node:
        """Adds a root node to the tree"""
        op = random.choice(OPS)
        new: Node = Node(tag=f'root: {op}', data=op)  # create a root node for the tree
        self._nodes[new.ID] = new  # add to the node dictionary
        self._root = new  # set the root
        return self._root
      
    # *** Size *** #
    @property
    def size(self):
        return len(self._nodes.values())

    # *** Methods *** #
    
    # *** Add Children *** #
    # + whenever we add using one of these we create a new node
    # +  so no ids should ever conflict
    def addLeft(self, parentID: str, data: TREE_DATA) -> Node:
        
        # create the new node (id will be made by node's init)
        new = Node(tag=f'{data}', data=data, parent=parentID, branch='left')
        
        # make the parent aware of it
        self._nodes[parentID].left = new
        
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
        
        return new
    
    def addRight(self, parentID: str, data: TREE_DATA) -> Node:
        
        # create the new node (id will be made by node's init)
        new = Node(tag=f'{data}', data=data, parent=parentID, branch='right')

        # make the parent aware of it
        self._nodes[parentID].right = new
    
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
    
        return new
    
    def addMiddle(self, parentID: str, data: TREE_DATA) -> Node:
        
        # create the new node (id will be made by node's init)
        new = Node(tag=f'{data}', data=data, parent=parentID, branch='middle')

        # make the parent aware of it
        self._nodes[parentID].middle = new
    
        # add the new node to the dictionary of nodes
        self._nodes[new.ID] = new
    
        return new

    # *** Get Children *** #
    def getLeft(self, parentID: str) -> typ.Optional[Node]:
        """ Used to get the left child of a node """
        return self._nodes[parentID].left

    def getRight(self, parentID: str) -> typ.Optional[Node]:
        """ Used to get the right child of a node """
        return self._nodes[parentID].right

    def getMiddle(self, parentID: str) -> typ.Optional[Node]:
        """ Used to get the middle child of a node """
        return self._nodes[parentID].middle
    
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
    def getDepth(self, ident: str) -> int:
        
        if self._nodes.get(ident):
            
            # grab the target node
            current: Node = self._nodes[ident]
            
            depth: int = 0
            
            # walk back up the tree until we reach the root
            while current is not self._root:
                depth += 1  # increment root
                current = self._nodes[current.parent]
            
            return depth
        
        else:  # if the key is bad, raise an error
            raise BadIdERROR(ident)
    
    def getBranch(self, childID: str) -> str:
        return self._nodes[childID].branch
    
    # *** Operations *** #
    def createRoot(self):
        """ Used to create a root node for the Tree """
        
        op = random.choice(OPS)  # choose an random operation
        self._root = Node(tag=f'root: {op}', data=op)  # set the root
        return self._root
    
    def getRandomNode(self) -> Node:
        """ Get a random node from the tree (root and leaves are allowed)"""
        return random.choice(list(self._nodes.values()))
    
    def removeChildren(self, nodeID: str):
        """ Used to remove a all the children of a node from the tree """
        # if the key isn't in the tree, raise an error
        if not (self._nodes.get(nodeID)):
            raise BadIdERROR(nodeID)
        
        # calling remove subTree should delete any
        # children the children have as well
        for child in self._nodes[nodeID].children:
            self.removeSubtree(child)
        return
    
    def _copy(self, current: Node):
        
        # check the the node is in the tree
        if self._nodes.get(current.ID):
            
            # TODO add/copy current
            # add the current node to the copy dictionary
            self._copyDictionary[current.ID] = self._nodes[current.ID]
            
            # call copy on each of the children (if any)
            if current.hasChildren:
                # call copy on the left & right child
                self._copy(self.getLeft(current.ID))
                self._copy(self.getRight(current.ID))
                # if there's a middle child copy it too
                if current.hasMiddle:
                    self._copy(self.getMiddle(current.ID))
            
            # after we have copied the children (or they're aren't any)
            # delete the subtree's node from this tree
            del self._nodes[current.ID]
            # return up the stack
            return
        
        else:  # if the key is bad, raise an error
            raise BadIdERROR(current.ID)
    
    def removeSubtree(self, newRootID: str) -> ("Tree", str, str):
        # TODO make the parent of newRootId point to Null
        
        # if the node is in the tree
        if self._nodes.get(newRootID):
    
            # get the parents id
            parentOfSubtreeID = self._nodes[newRootID].parent
            orphanBranch = self._nodes[newRootID].branch
            
            # set the new root to be a root
            self._nodes[newRootID].parent = None
            self._nodes[newRootID].branch = None
            
            # make sure the copy dictionary is empty
            self._copyDictionary = {}

            # get the root of the subtree
            rt = self._nodes[newRootID]
            
            self._copy(rt)  # call copy (recursive)
            
            # copyDictionary now contains the subtree, and
            # it's members have been removed from this tree
            
            # so build the new subtree using copyDictionary
            subtree = Tree(root=rt, nodes=self._copyDictionary)
            
            # reset the value of copDictionary
            self._copyDictionary = {}
            
            return subtree, parentOfSubtreeID, orphanBranch
    
        else:  # if the key is bad, raise an error
            raise BadIdERROR(newRootID)

    def addSubtree(self, subtree: "Tree", newParent: str, orphanBranch: str):
        # check that parent id is valid
        if not (self._nodes.get(newParent)):
            raise BadIdERROR(newParent)
        
        # set the adopted parents to point to the subtree
        if orphanBranch == 'left':
            self._nodes[newParent].left = subtree.root.ID
        elif orphanBranch == 'right':
            self._nodes[newParent].right = subtree.root.ID
        elif orphanBranch == 'middle':
            self._nodes[newParent].middle = subtree.root.ID
    
        # set the subtree root to point to adopted parents
        subtree.root.parent = newParent
        # set the subtree's branch
        subtree.branch = orphanBranch
        
        # check that there aren't any duplicate keys
        if any(i in self._nodes.keys() for i in subtree._nodes.keys()):
            raise DuplicateNodeError()
        
        # add subtree to dictionary of nodes
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
    
        return self.__runNode(featureValues, self.root, classId, terminals)

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
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible


class DuplicateNodeError(Exception):
    """ Thrown if a node with the same id is already in the tree """
    def __init__(self, key=None):
        self.key = key
        
        if key is None:
            self.message = f'A duplicate node was found'
        else:
            self.message = f'Node with ID {self.key} is already in the tree'
        
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class BadIdERROR(Exception):
    """ Thrown if a node id is not in the tree """
    
    def __init__(self, key):
        self.key = key
        self.message = f'Node with ID {self.key} is not in the tree'
        super().__init__(self.message)
    
    def __str__(self):
        return self.message
