
import typing as typ
import uuid


class Node:
    
    def __init__(self, data, parent: typ.Optional[str] = None, branch: typ.Optional[str] = None,
                 left: typ.Optional[str] = None,  right: typ.Optional[str] = None, middle: typ.Optional[str] = None):
        
        self._ID: str = str(uuid.uuid4())
        # self._ID = next(countNodes())  # create a unique ID
        self._data: str = data  # the operation or terminal to be stored in this node
        
        self._parent: typ.Optional[str] = parent  # the id of the parent node
        self._branch: typ.Optional[str] = branch  # which branch from the parent this child is on
        self._left: typ.Optional[str] = left       # the id string of left child
        self._right: typ.Optional[str] = right     # the id string of right child
        self._middle: typ.Optional[str] = middle   # the id string of middle child
        
        return
    
    def __str__(self):
        # + verbose v2
        if self.hasChildren:
            out: str = f'[{self.data} | {self._ID} | L:{self._left}, M:{self.middle}, R:{self.right}]'
        else:
            out: str = f'[{self.data} | {self._ID}]'
        # + verbose v1
        # out: str = f'[ID: {self._ID} | Data: {self.data}]'
        # + condensed
        # out: str = f'[{self.data}]'
        return out
    
    def __repr__(self):
        return self.__str__()
        
    # *** ID *** #
    @property
    def ID(self):
        return self._ID

    # *** Data *** #
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, newData):
        self._data = newData
    
    # *** Parent *** #
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, newParent: str):
        self._parent = newParent

    # *** Branch *** #
    @property
    def branch(self):
        return self._branch

    @branch.setter
    def branch(self, newBranch):
        self._branch = newBranch

    # *** Left *** #
    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, newLeft: str):
        self._left = newLeft

    # *** Right *** #
    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, newRight: str):
        self._right = newRight

    # *** Middle *** #
    @property
    def middle(self):
        return self._middle

    @middle.setter
    def middle(self, newMiddle: str):
        self._middle = newMiddle

    # *** Children *** #
    @property
    def children(self):
        return self._left, self._right, self._middle

    @property
    def hasChildren(self):
        """ This will return True is the node has a left & right child """
        if self._left and self._right:
            return True
        else:
            return False
    
    @property
    def hasMiddle(self):
        """ This will return True is the node has a middle child """
        if self._middle:
            return True
        else:
            return False

    # *** Methods *** #
    def isRoot(self):
        if self._parent is None:
            return True
        else:
            return False
    
    def isLeaf(self):
        if self.hasChildren is False:
            return True
        else:
            return False
