from Tree import Tree
from Node import Node
from formatting import printError
import typing as typ

MAX_DEPTH: int = 4

# TODO: write methods
# TODO: call them in IF statement at bottom
TERMINAL_NODES: typ.List[str] = []


def create_tree() -> Tree:
    """ Creates a tree of a predetermined structure """
    
    op = 'add'  # the root node will have the add operation
    root: Node = Node(tag=f'root: {op}', data=op)  # create a root node for the tree
    rootID = root.ID  # get the root ID
    test_tree: Tree = Tree(root=root)
    
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
    test_tree.addRight(parentID=root_right, data='times')  # create a ADD node
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
    global TERMINAL_NODES
    TERMINAL_NODES = [root_left_left_left, root_left_left_right, root_left_right_left,
                      root_left_right_right_left, root_left_right_right_Right,
                      root_right_right_right, root_right_right_left, root_right_left_left,
                      root_right_left_middle, root_right_left_right]
    
    print_init(test_tree)  # print the constructed tree
    
    return test_tree


# ********************* Remove ********************* #
def remove_from_tree(test_tree: Tree):
    """ Removes a subtree from the created tree """
    pass


def remove_check():
    """ Checks that the subtree was removed correctly """
    pass
# *************************************************** #


# ******************** Crossover ******************** #
def cross_tree(test_tree: Tree):
    """ Performs the Crossover operation on the tree """
    pass


def cross_check():
    """
    Called by cross_tree & is used to check
    that the operations was performed as expected
    """
    pass
# *************************************************** #


# TODO: test print_tree
def print_init(tree: Tree) -> str:
    return print_tree(tree, tree.root.ID, "", True)


def print_tree(tree: Tree, nodeID: str, indent: str, isLast: bool) -> str:
    out: str = indent  # out will be the output string
    
    node: Node = tree.getNode(nodeID)
    
    if isLast:  # if this is the last child of a node
        out += "\u2517"
        indent += " "
    else:  # if this is not the last child
        out += "\u2523"
        indent += "\u2503 "
    out += str(node)+'\n'  # print this node
    
    children = ('left', 'middle', 'right')
    
    for child in children:
        if child == 'left' and (node.left is not None):
            print_tree(tree, node.left, indent, False)
        elif child == 'middle' and (node.left is not None):
            print_tree(tree, node.middle, indent, False)
        elif child == 'right' and (node.left is not None):
            print_tree(tree, node.right, indent, True)
    return out


def test_main():
    
    test_tree: Tree = create_tree()  # create the tree
    
    cross_tree(test_tree)  # test crossover
    

if __name__ == "__main__":
    test_main()
