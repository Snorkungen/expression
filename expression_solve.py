from copy import copy
from typing import Tuple, Literal
from expression_parser import (
    TT_OOO_MASK,
    TT_Exponent,
    TT_Ident,
    TT_Int,
    TT_Numeric,
    TT_Operation,
    TT_Div,
    parse,
    RESERVED_IDENTITIES,
)
from expression_tree_builder import (
    Atom,
    Int,
    Equals,
    Operation,
    Variable,
    build_tree,
    is_variable_in_tree,
    compare_variable,
)


def get_opposite_operation(token: Tuple[int, str]):
    assert token[0] & TT_Operation, "token must be an operator"
    # the following is really cursed
    for key in RESERVED_IDENTITIES:
        val = RESERVED_IDENTITIES[key]
        if (token[0] & TT_OOO_MASK) == (val & TT_OOO_MASK) and token[0] != val:
            return (val, key)

    print(token)

    raise ValueError("Could not find an opposite operation")


def solve_for(tree: Atom, target: Variable, show_steps=False) -> Equals:
    assert type(tree) == Equals, "Root node must be a Equals atom"
    assert type(target) == Variable, "Target must be Variable"

    assert is_variable_in_tree(tree, target), "Variable must be in expression"

    if is_variable_in_tree(tree.left, target) and is_variable_in_tree(
        tree.right, target
    ):
        assert False, "Target variable must be on only on side of the expression"

    tree = copy(
        tree
    )  # do not know if this should be a deepcopy because i'm modifying information

    # check the side that the varible is on
    # this variable is for assigning and checking something idk what i'm actually doing
    tree_target = "left" if is_variable_in_tree(tree.left, target) else "right"

    # the destination of the moved operations
    tree_destination = "left" if is_variable_in_tree(tree.right, target) else "right"

    def get_node(node: Operation, key: Literal["left", "right"]) -> Operation:
        """this is a workaround becaus this language implements actually *real* classes"""
        if key == "left":
            return node.left
        return node.right

    def set_node(
        node: Operation, key: Literal["left", "right"], atom: Atom
    ) -> Operation:
        """this is a workaround because this language implements actually *real* classes"""
        if key == "left":
            node.left = atom
            return node.left
        node.right = atom
        return node.right

    # loop until variable is the top node on the side of interest
    while not compare_variable(get_node(tree, tree_target), target):
        if show_steps:
            print(tree)

        # this is the root of the side with the variable on
        root = get_node(tree, tree_target)

        assert isinstance(
            root, Operation
        ), f"{tree} side must either be target or an operation"
        assert type(root) != Equals, 'expression cannot contain more than one "=" sign'

        if is_variable_in_tree(root.left, a) and is_variable_in_tree(root.right, a):
            assert False, "There can only be on target variable in an expression"

        # the side of the root that the target is on
        root_target = "right" if is_variable_in_tree(root.right, target) else "left"
        root_destination = "right" if is_variable_in_tree(root.left, target) else "left"

        # just special cause because supporting root operation is not worth it at the moment
        opposite_operator_token: Tuple[int, str]
        opposite_left = get_node(tree, tree_destination)
        opposite_right = get_node(root, root_destination)

        if root.token_type & TT_Exponent:
            # this is the special sauce
            opposite_operator_token = (root.token_type, root.value)
            opposite_right = Operation(
                (RESERVED_IDENTITIES["/"], "/"),
                Int((TT_Int | TT_Numeric, "1")),
                opposite_right,
            )
        elif root.token_type & TT_Div and is_variable_in_tree(root.right, target):
            # now what this dude got to do is multiply the right side with
            set_node(
                tree,
                tree_destination,
                Operation((RESERVED_IDENTITIES["*"], "*"), opposite_left, root.right),
            )
            set_node(tree, tree_target, root.left)

            tmp = tree_destination
            tree_destination = tree_target
            tree_target = tmp
            continue
        else:
            opposite_operator_token = get_opposite_operation(
                (root.token_type, root.value)
            )

        opposite_operation = Operation(
            opposite_operator_token, opposite_left, opposite_right
        )

        set_node(tree, tree_destination, opposite_operation)
        set_node(tree, tree_target, get_node(root, root_target))

    # for sanity swap over solved variable to always be on the left
    if tree_target == "right":
        if show_steps:
            print(tree)

        tmp = tree.right
        tree.right = tree.left
        tree.left = tmp

    return tree


def print_solver_status(
    expression: str, variable: Variable, answer: str, show_steps=False
):
    if __name__ != "__main__":
        return

    node = build_tree(parse(expression))
    solved = solve_for(node, variable, show_steps)

    print(f"[{str(solved) == answer}]", node, "=>", solved)


a = Variable((TT_Ident, "a"))
b = Variable((TT_Ident, "b"))

print_solver_status("10 + a * 20 / 10 = b", a, "a = ((b - 10) * 10) / 20")
print_solver_status("a = ((b - 10) * 10) / 20", b, "(a * 20) / 10 + 10 = b")

print_solver_status(
    "(a + 10 / 2) / 6 = b * 10 + 10", a, "a = (b * 10 + 10) * 6 - 10 / 2"
)
print_solver_status(
    "a = (b * 10 + 10) * 6 - 10 / 2", b, "((a + 10 / 2) / 6 - 10) / 10 = b"
)

# NOTE: there should this program be aware that square root could be positive or negative
print_solver_status("a ^ 2 = b - 1", a, "a = (b - 1) ^ (1 / 2)")
# print_solver_status("a = (b - 1) ^ (1 / 2)", b, "")

print_solver_status("a / b = c", a, "a = c * b")
print_solver_status("a / b = c", b, "b = a / c", True)
print_solver_status("a / (b + 10) = c", b, "b = a / c - 10", True)
