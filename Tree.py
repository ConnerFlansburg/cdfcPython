
import typing as typ
from Node import Node
import random
import sys
import logging as log
import traceback
from formatting import printError
import pprint

# typing alias for tree nodes. They will either have a terminal index (int) or an OP (str)
TREE_DATA = typ.Union[int, str]
# OPS is the list of valid operations on the tree
OPS: typ.Final = ['add', 'subtract', 'times', 'max', 'if']

NUM_TERMINALS = {'add': 2, 'subtract': 2,     # NUM_TERMINALS is a dict that, when given an OP as a key, give the number of terminals it needs
                 'times': 2, 'max': 2, 'if': 3}
sys.setrecursionlimit(10000)                                  # set the recursion limit for the program


class Tree:
    
    def __init__(self, root: typ.Optional[Node] = None, nodes: typ.Optional[typ.Dict[str, Node]] = None):
        """ Trees are almost always made empty & then built by adding to them."""
        if nodes is None:
            nodes = {}
        
        if root is None:  # create a root if none was provided
            op = random.choice(OPS)
            root: Node = Node(tag=f'root: {op}', data=op)  # create a root node for the tree
            nodes[root.ID] = root  # add to the node dictionary
            
        self._root: Node = root  # set the root as root

        # this dictionary will hold the nodes in the tree, keyed by their id value
        self._nodes: typ.Dict[str, Node] = nodes
        
        # the values below are used by methods & should not be set
        # this dictionary is used when copying a subtree 8 should never be used elsewhere
        self._copyDictionary = {}
        
        self._rValue = None  # this will be used by recursive search

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

    @root.deleter
    def root(self):
        del self._root
    
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
    def getDepth2(self, ID: str) -> int:
        
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
                self._rSearch(ID, self._root.ID, 0)  # ! There's a bug thrown here
                
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
            pprint.pprint(self._nodes)
            traceback.print_stack()  # print stack trace
            sys.exit(-1)  # exit on error; recovery not possible
            
    def _rSearch(self, targetID: str, currentID: str, depth: int):
        
        # if the key is valid
        if self._nodes.get(currentID):
        
            if currentID != targetID:
        
                depth += 1  # increment depth
        
                # if the current node has children run recursive search
                if self._nodes[currentID].hasChildren:
                    
                    if self._nodes[currentID].left:  # if there is a left child
                        self._rSearch(targetID, self._nodes[currentID].left, depth)
                    if self._nodes[currentID].right:  # if there is a right child
                        self._rSearch(targetID, self._nodes[currentID].right, depth)
                    if self._nodes[currentID].hasMiddle:
                        self._rSearch(targetID, self._nodes[currentID].middle, depth)
        
                # if this is not the node, and it doesn't have children, return up stack
                else:
                    return
            
            else:  # if we have found the node, set rValue
                self._rValue = depth
                return
        # if the key is invalid
        else:
            print(f'Node key:{currentID}, Node:{self._nodes.get(currentID)}')
            raise NotInTreeError(currentID)  # ! for some reason this is getting a node
        
    def getDepth(self, ident: str) -> int:
        
        # if the root has not been set first
        if self._root is None:
            print('Root was not set before calling getDepth')
            raise Exception
        # if the id is valid
        elif self._nodes.get(ident):
            
            depth: int = 0  # start depth off at zero
            nKey: str = ident  # stores node id key in a variable
            
            while True:
            
                # if this is the root, break
                if 'root' in self._nodes[nKey].tag:
                    break
                if nKey == self._root.ID:
                    break
                # increment the depth since we aren't at the root yet
                depth += 1
                
                # grab the target node
                current: Node = self._nodes[nKey]
                # update node id key to hold parent's id
                nKey = self._nodes[current.parent].ID
            
            print('getDepth returned (while)')  # ! debugging
            return depth
        
        else:  # if the key is bad, raise an error
            raise NotInTreeError(ident)
    
    def getBranch(self, childID: str) -> str:
        return self._nodes[childID].branch
    
    # *** Operations *** #
    def getRandomNode(self) -> str:
        """ Get a random node from the tree (root and leaves are allowed)"""
        return random.choice(list(self._nodes.keys()))
    
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
            raise Exception
        
        # delete the children & any of their children
        for child in self._nodes[nodeID].children:
            self.__delete(child)
        return
    
    def __delete(self, ID):
        
        # if the id is valid
        if self._nodes.get(ID):
            node: Node = self._nodes[ID]
            # if there's a left child
            if node.left:
                self.__delete(node.left)
            # if there's a right child
            if node.right:
                self.__delete(node.right)
            # if there's a middle child
            if node.middle:
                self.__delete(node.middle)
            # after the children have been deleted,
            # delete this node from the dictionary
            del self._nodes[ID]
        
        # if we've hit the bottom of the tree
        elif ID is None:
            return

        else:
            raise NotInTreeError(ID)
    
    def _copy(self, cid: str):
        
        # check the the node is in the tree
        if self._nodes.get(cid):
            
            current: Node = self._nodes[cid]
            # call copy on each of the children (if any)
            if current.left is not None:
                self._copy(current.left)
            if current.right is not None:
                self._copy(current.right)
            if current.middle is not None:
                self._copy(current.middle)
            
            # after we have reached the leaves of the subtree,
            # return back up the stack, copying as we go
            self._copyDictionary[cid] = self._nodes.pop(cid)
            return
        
        else:  # if the key is bad, raise an error
            raise NotInTreeError(cid)
    
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
            
            self._copy(newRootID)  # call copy (recursive)
            
            # copyDictionary now contains the subtree, and
            # it's members have been removed from this tree
            
            # so build the new subtree using copyDictionary
            subtree = Tree(root=rt, nodes=self._copyDictionary)
            
            # reset the value of copDictionary
            self._copyDictionary = {}
            
            return subtree, parentOfSubtreeID, orphanBranch
    
        else:  # if the key is bad, raise an error
            raise NotInTreeError(newRootID)  # ! this is being hit when called by removeChildren

    def addSubtree(self, subtree: "Tree", newParent: str, orphanBranch: str):
        # check that parent id is valid
        if not (self._nodes.get(newParent)):
            raise NotInTreeError(newParent)
        
        # set the adopted parents to point to the subtree
        if orphanBranch == 'left':
            self._nodes[newParent].left = subtree._root.ID
        elif orphanBranch == 'right':
            self._nodes[newParent].right = subtree._root.ID
        elif orphanBranch == 'middle':
            self._nodes[newParent].middle = subtree._root.ID
    
        # set the subtree root to point to adopted parents
        subtree._root.parent = newParent
        # set the subtree's branch
        subtree.branch = orphanBranch
        
        # ! duplicate nodes are being found after initial pop generation
        # check that there aren't any duplicate keys
        duplicates = []
        for key1 in subtree._nodes.keys():  # for every key in subtree,
            if key1 in self._nodes.keys():  # if that key is also in this tree,
                duplicates.append(key1)     # add the key to the list of copies
                
        try:
            if duplicates:                  # if duplicates were found, raise an error
                raise DuplicateNodeError(keyList=duplicates)
        except DuplicateNodeError as err:
            lineNm = sys.exc_info()[-1].tb_lineno       # get the line number of error
            log.error(f'line = {lineNm}, {str(err)}')   # log the error
            traceback.print_stack()                     # print stack trace
            printError(f'On line {lineNm} DuplicateNodeError encountered')  # print message
            print('Duplicate(s):')
            pprint.pprint(duplicates)  # print the list of duplicate nodes
            print('Subtree:')
            pprint.pprint(list(subtree._nodes.keys()))
            
            sys.exit(-1)  # exit on error; recovery not possible
        
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
            traceback.print_stack()  # print stack trace
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
        if coin == 'TERM' or (self.getDepth2(node.ID) == MAX_DEPTH - 1):  # if we need to add terminals
        
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
            if self.getDepth2(node.ID) == (MAX_DEPTH - 1):

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
            print(f'Node ID: {nodeID},\nDepth: {self.getDepth2(nodeID)}')
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


class NotInTreeError(Exception):
    """ Thrown if a node id is not in the tree """
    
    def __init__(self, key: str):
        self.key: str = key
        self.message: str = f'Node with ID {key} is not in the tree'
        super().__init__(self.message)
    
    def __str__(self):
        return self.message


class RootNotSetError(Exception):
    
    def __str__(self):
        return 'Root accessed before it was set'
