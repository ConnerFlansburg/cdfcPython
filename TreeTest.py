from Tree import Tree, NotInTreeError
from Node import Node
import sys
import typing as typ
import traceback
from formatting import printError


# TODO: test tree search methods
# TODO: test getDepth
# TODO: Convert to better proper test class

MAX_DEPTH: int = 4

TERMINAL_NODES1: typ.List[str] = []
TERMINAL_NODES2: typ.List[str] = []


def create_tree1() -> Tree:
    """ Creates a tree of a predetermined structure """

    # * Create a New Tree Object * #
    test_tree: Tree = Tree()
    test_tree.ID = 'TREE_1'

    # * Create a New Root Node * #
    root: Node = Node(tag='root: add', data='add')  # create a root node for the tree
    rootID = root.ID  # get the root ID  (this will be tested later)
    
    # * Override the Old Root in the Tree * #
    test_tree.overrideRoot(root)
    
    # * Test RootID * #
    if test_tree.root.ID != rootID:  # print error if there's a problem with root's ID
        printError(f'Root ID {rootID} & {test_tree.root.ID} do not match')
        rootID = test_tree.root.ID  # update rootID to avoid issues
    
    # * Root is ADD so create two children * #
    test_tree.addLeft(parentID=rootID, data='subtract')  # create a SUBTRACT node
    test_tree.addRight(parentID=rootID, data='max')  # create a MAX node
    # get the IDs of both children
    root_left: str = test_tree.getLeft(rootID).ID
    root_right: str = test_tree.getRight(rootID).ID
    
    # * Root -> Left is SUBTRACT so add two children * #
    test_tree.addLeft(parentID=root_left, data='max')  # create a MAX node
    test_tree.addRight(parentID=root_left, data='times')  # create a TIMES node
    # get the IDs of both children
    root_left_left: str = test_tree.getLeft(root_left).ID
    root_left_right: str = test_tree.getRight(root_left).ID
    
    # * Root -> Right is MAX so add two children * #
    test_tree.addLeft(parentID=root_right, data='if')  # create a IF node
    test_tree.addRight(parentID=root_right, data='add')  # create a ADD node
    # get the IDs of both children
    root_right_left: str = test_tree.getLeft(root_right).ID
    root_right_right: str = test_tree.getRight(root_right).ID
    
    # * Root -> Left -> Left is MAX so add two children * #
    test_tree.addLeft(parentID=root_left_left, data=3)  # create a TERMINAL node
    test_tree.addRight(parentID=root_left_left, data=5)  # create a TERMINAL node
    # get the IDs of both children
    root_left_left_left: str = test_tree.getLeft(root_left_left).ID
    root_left_left_right: str = test_tree.getRight(root_left_left).ID

    # * Root -> Left -> Right is TIMES so add two children * #
    test_tree.addLeft(parentID=root_left_right, data=12)  # create a TERMINAL node
    test_tree.addRight(parentID=root_left_right, data='add')  # create a ADD node
    # get the IDs of both children
    root_left_right_left: str = test_tree.getLeft(root_left_right).ID
    root_left_right_right: str = test_tree.getRight(root_left_right).ID

    # * Root -> Left -> Right -> Right is ADD so add two children * #
    test_tree.addLeft(parentID=root_left_right_right, data=1)  # create a TERMINAL node
    test_tree.addRight(parentID=root_left_right_right, data=8)  # create a TERMINAL node
    # get the IDs of both children
    root_left_right_right_left: str = test_tree.getLeft(root_left_right_right).ID
    root_left_right_right_Right: str = test_tree.getRight(root_left_right_right).ID

    # * Root -> Right -> Right is ADD so add two children * #
    test_tree.addLeft(parentID=root_right_right, data=4)  # create a TERMINAL node
    test_tree.addRight(parentID=root_right_right, data=9)  # create a TERMINAL node
    # get the IDs of both children
    root_right_right_right: str = test_tree.getLeft(root_right_right).ID
    root_right_right_left: str = test_tree.getRight(root_right_right).ID

    # * Root -> Right -> Left is IF so add three children * #
    test_tree.addLeft(parentID=root_right_left, data=15)  # create a TERMINAL node
    test_tree.addMiddle(parentID=root_right_left, data=1)  # create a TERMINAL node
    test_tree.addRight(parentID=root_right_left, data=7)  # create a TERMINAL node
    # get the IDs of both children
    root_right_left_left: str = test_tree.getLeft(root_right_left).ID
    root_right_left_middle: str = test_tree.getMiddle(root_right_left).ID
    root_right_left_right: str = test_tree.getRight(root_right_left).ID
    
    # * Create a list of all the Terminal Node IDS (these are the tree's leaves) * #
    global TERMINAL_NODES1
    TERMINAL_NODES1 = [root_left_left_left, root_left_left_right, root_left_right_left,
                       root_left_right_right_left, root_left_right_right_Right,
                       root_right_right_right, root_right_right_left, root_right_left_left,
                       root_right_left_middle, root_right_left_right]
    print('Tree 1 Created:')
    print_init(test_tree)  # print the constructed tree
    
    return test_tree


def create_tree2() -> Tree:

    # * Create a New Tree Object * #
    test_tree: Tree = Tree()
    test_tree.ID = 'TREE_2'
    
    # * Create a New Root Node * #
    root: Node = Node(tag='root: add', data='add')  # create a root node for the tree
    rootID = root.ID  # get the root ID  (this will be tested later)
    
    # * Override the Old Root in the Tree * #
    test_tree.overrideRoot(root)
    
    # * Test RootID * #
    if test_tree.root.ID != rootID:  # print error if there's a problem with root's ID
        printError(f'Root ID {rootID} & {test_tree.root.ID} do not match')
        rootID = test_tree.root.ID  # update rootID to avoid issues
    
    # * Root is MAX so create two children * #
    test_tree.addLeft(parentID=rootID, data='times')  # create a TIMES node
    test_tree.addRight(parentID=rootID, data='if')    # create a IF node
    # get the IDs of both children
    root_left: str = test_tree.getLeft(rootID).ID    # TIMES
    root_right: str = test_tree.getRight(rootID).ID  # IF

    # * Root -> Left is TIMES so create two children * #
    test_tree.addLeft(parentID=root_left, data='add')  # create a TIMES node
    test_tree.addRight(parentID=root_left, data=41)    # create a TERMINAL node
    # get the IDs of both children
    root_left_left: str = test_tree.getLeft(root_left).ID    # ADD
    root_left_right: str = test_tree.getRight(root_left).ID  # TERMINAL

    # * Root -> Left -> Left is ADD so create two children * #
    test_tree.addLeft(parentID=root_left_left, data=75)   # create a TIMES node
    test_tree.addRight(parentID=root_left_left, data=76)  # create a TERMINAL node
    # get the IDs of both children
    root_left_left_left: str = test_tree.getLeft(root_left_left).ID    # TERMINAL
    root_left_left_right: str = test_tree.getRight(root_left_left).ID  # TERMINAL

    # * Root -> Right is IF so create three children * #
    test_tree.addLeft(parentID=root_right, data=16)           # create a TERMINAL node
    test_tree.addMiddle(parentID=root_right, data=20)         # create a TERMINAL node
    test_tree.addRight(parentID=root_right, data='subtract')  # create a SUBTRACT node
    # get the IDs of both children
    root_right_left: str = test_tree.getLeft(root_right).ID      # TERMINAL
    root_right_middle: str = test_tree.getMiddle(root_right).ID  # TERMINAL
    root_right_right: str = test_tree.getRight(root_right).ID    # SUBTRACT
    
    # * Root -> Right -> Right is SUBTRACT so create three children * #
    test_tree.addLeft(parentID=root_right_right, data=30)   # create a TERMINAL node
    test_tree.addRight(parentID=root_right_right, data=10)  # create a TERMINAL node
    # get the IDs of both children
    root_right_right_left: str = test_tree.getLeft(root_right_right).ID    # TERMINAL
    root_right_right_right: str = test_tree.getRight(root_right_right).ID  # TERMINAL
    
    global TERMINAL_NODES2
    TERMINAL_NODES2 = [root_left_right, root_left_left_left, root_left_left_right,
                       root_right_left, root_right_middle, root_right_right,
                       root_right_right_left, root_right_right_right]

    print('Tree 2 Created:')
    print_init(test_tree)  # print the constructed tree
    
    return test_tree


# ********************* Remove ********************* #
def check_remove_from_tree(test_tree: Tree, test_node: Node = None) -> Tree:
    """ Removes a subtree from the passed tree & returns it """

    # * Get the Root Node of the Subtree
    if test_node is None:  # if a node was not provided use a default
        # Do Root, Left, Left to get the node TIMES -Left-> 12; -Right-> ADD
        test_node: Node = test_tree.getLeft(test_tree.root.ID)
        test_node = test_tree.getRight(test_node.ID)
    
    # * Perform the Removal * #
    subtree: Tree
    subtree, _, _ = test_tree.removeSubtree(test_node.ID)  # ! BUG: NotInTreeError thrown here (ID is None)
    
    # * Report the Results to the Console * #
    print(f'Removed Node ID: {test_node.ID}')
    print('Tree after subtree removal:')
    print_init(test_tree)
    print('Subtree Removed:')
    print_init(subtree)
    
    return subtree


def check_rDelete(test_tree: Tree):
    # * Get the Node to delete
    # Do Root, Left, Left to get the node TIMES -Left-> 12; -Right-> ADD
    node: Node = test_tree.getLeft(test_tree.root.ID)
    node = test_tree.getRight(node.ID)
    
    # * Preform the Recursive Delete * #
    test_tree.testDelete(node.ID, rootID=node.ID, copy=True)
    print(f'Tree after rDelete:\n{test_tree}')
    
    # * Create Subtree * #
    subtree: Tree = Tree(root=node, nodes=test_tree.copyDictionary, ID=test_tree.ID)
    print(f'Subtree Created:\n{subtree}')
# *************************************************** #


# ******************** Crossover ******************** #
def cross_tree(test_tree: Tree, test_subtree: Tree, parent: str, branch: str):
    """ Performs the Crossover operation on the tree """
    test_tree.addSubtree(test_subtree, parent, branch)
    
    print('Tree after crossover')
    print_init(test_tree)
    
    pass


def check_cross(test_tree1: Tree, test_tree2: Tree):
    
    # * Get Two Nodes to Test * #
    # tree1_node should be ADD -Left-> 4, -Right-> 9
    parent_of_node1: Node = test_tree1.getRight(test_tree1.root.ID)  # go left
    tree1_node: Node = test_tree1.getRight(parent_of_node1.ID)       # go left
    # print(f'Tree 1, parent: {parent_of_node1}')
    # print(f'Tree 1, node 1: {tree1_node}')
    
    # tree2_node should be TERMINAL 16
    parent_of_node2: Node = test_tree2.getLeft(test_tree2.root.ID)  # go left
    tree2_node: Node = test_tree2.getRight(parent_of_node2.ID)      # go right
    # print(f'Tree 2, parent: {parent_of_node2}')
    # print(f'Tree 2, node 2: {tree2_node}')
    
    # * Remove two Subtrees * #
    subtree_of_tree1: Tree = check_remove_from_tree(test_tree1, tree1_node)
    subtree_of_tree2: Tree = check_remove_from_tree(test_tree2, tree2_node)
    
    # * Perform Swap * #
    cross_tree(test_tree1, subtree_of_tree2, parent_of_node1.ID, 'right')
    cross_tree(test_tree2, subtree_of_tree1, parent_of_node2.ID, 'right')
# *************************************************** #


# ******************** Get Depth ******************** #
def test_getDepth(test_tree: Tree):
    
    # * Get the Node to delete
    # Do Root, Left, Left to get the node TIMES -Left-> 12; -Right-> ADD
    node: Node = test_tree.getLeft(test_tree.root.ID)
    node = test_tree.getRight(node.ID)
    
    # print the tree before getting depth
    print(f'Getting Depth of Node: \n{node}\n In Tree:\n{test_tree}')
    test_tree.getNode(node.ID)  # get the depth of the Node


def test_search(test_tree: Tree):
    
    # * Get the Node to delete
    # Do Root, Left, Left to get the node TIMES -Left-> 12; -Right-> ADD
    node: Node = test_tree.getLeft(test_tree.root.ID)
    test_node = test_tree.getRight(node.ID)
    
    # print the tree before searching it
    print(f'Searching for Node: \n{test_node}\n In Tree:\n{test_tree}')
    # test_tree.test_search(test_node.ID)
    depth: int = test_tree.getDepth(test_node.ID, test_node.ID, depth=0)
    print(f'Node was found at depth {depth}')
# *************************************************** #


def print_init(tree: Tree):
    # printError(f'Print Init Root ID = {tree.root.ID}')
    print(tree.ID)
    print_tree(tree, tree.root.ID, "", True)
    print('\n')  # print a new line
    
    return


def print_tree(tree: Tree, nodeID: str, indent: str, isLast: bool):
    
    if nodeID is None:
        return
    
    try:
        
        node: Node = tree.getNode(nodeID)
    
    except NotInTreeError:
        lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        message: str = ''.join(traceback.format_stack())  # traceback to message
        message += f'\nNotInTreeError encountered on line {lineNm} in TreeTest.py'
        message += f'\nKey = {nodeID}\n{tree}'  # print the tree & node
        printError(message)  # print message
        sys.exit(-1)  # exit on error; recovery not possible
    
    isLeaf: bool = node.isLeaf()
    
    if nodeID == tree.root.ID:  # if this is the root of a tree
        print(f'{indent}{str(node)}')  # print this node
        indent += "    "
        print(f"{indent}\u2503")
    elif isLast:  # if this is the last child of a node
        print(f'{indent}\u2517\u2501\u2501{str(node)}')  # print this node
        indent += "     "
        if isLeaf:  # if it is a leaf, don't print the extra bar
            print(f"{indent}")
        else:
            print(f"{indent}\u2503")
    else:  # if this is not the last child
        print(f'{indent}\u2523\u2501\u2501{str(node)}')  # print this node
        indent += "\u2503     "
        if isLeaf:  # if it is a leaf, don't print the extra bar
            print(f"{indent}")
        else:
            print(f"{indent}\u2503")
    
    children = ('left', 'middle', 'right')
    
    for child in children:
        if child == 'left' and (node.left is not None):
            print_tree(tree, node.left, indent, False)
        elif child == 'middle' and (node.left is not None):
            print_tree(tree, node.middle, indent, False)
        elif child == 'right' and (node.left is not None):
            print_tree(tree, node.right, indent, True)
    return


def test_main():

    try:
        # * Create the Test Trees * #
        test_tree1 = create_tree1()  # create tree 1
        # test_tree2 = create_tree2()  # create tree 2

        # check_rDelete(test_tree1)  # * Test __rDelete * #
        # check_remove_from_tree(test_tree1)  # * Test removeFromTree * #
        # check_cross(test_tree1, test_tree2)  # * Test Crossover * #
        
        test_search(test_tree1)  # * Test __rSearch * #

    except KeyError as err:
        lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        message: str = ''.join(traceback.format_stack())  # traceback to message
        message += f'\nKeyError encountered on line {lineNm} in TreeTest.py'
        message += f'\n{str(err)}'  # print the message
        printError(message)  # print message
        print('Tree 1')
        # print(test_tree1)
        print('\nTree2')
        # print(test_tree2)
        sys.exit(-1)  # exit on error; recovery not possible


if __name__ == "__main__":
    test_main()
