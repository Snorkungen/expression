from expression_parser import (
    RESERVED_IDENTITIES,
    TT_Ident,
    parse,
    TT_Add,
    TT_Int,
    TT_Numeric,
    TT_Numeric_Negative,
    TT_Numeric_Positive,
)
from expression_tree_builder import (
    Atom,
    Int,
    Operation,
    Variable,
    build_tree,
    compare_varible,
)


def simplify_addition(node: Atom) -> Atom:
    assert isinstance(node, Operation)
    assert node.token_type & TT_Add > 0

    # First easy add two integers
    if isinstance(node.left, Int) and isinstance(node.right, Int):
        value = node.left.value + node.right.value
        token_type = TT_Numeric | TT_Int
        # below set the token type to be explicitly Positive or Negative
        if value < 0:
            token_type |= TT_Numeric_Negative
        else:
            token_type |= TT_Numeric_Positive

        return Int((token_type, str(value)))

    # second easy add two same variables
    if (
        isinstance(node.left, Variable)
        and isinstance(node.right, Variable)
        and compare_varible(node.left, node.right)
    ):
        left = Int((TT_Numeric | TT_Int | TT_Numeric_Positive, "2"))  # the value is two
        return Operation((RESERVED_IDENTITIES["*"], "*"), left, node.right)

    # third this requires using the commutattive property and probably some recursion

    # this case is just testing and figuring out how i would do certain operations
    # if isinstance(node.left, Operation) and isinstance(node.right, Variable):
    #     node = Operation((node.token_type, node.value), simplify_addition(node.left), node.right)
    #     node.left = simplify_addition(node.left)
    #     return node

    # assume the the depth is 2
    # and move the variable

    if isinstance(node.left, Operation):
        if not node.left.token_type & TT_Add:
            return node

        # this is where the recursion occurs

        left = simplify_addition(node.left)
        right = node.right

        if isinstance(left, Int):
            node = Operation((node.token_type, node.value), left, node.right)

            if right.token_type & TT_Ident:
                return node

            if isinstance(right, Int):
                return simplify_addition(node)

        if isinstance(left, Operation) and left.token_type == node.token_type:
            # here check that the depth of left is 1
            if not (
                isinstance(left.left, Operation) or isinstance(left.right, Operation)
            ):
                # swap some stuff

                if isinstance(right, Int):
                    # check on what side the identity is on
                    if left.left.token_type & TT_Ident:
                        # the identity is on the left side
                        tmp = left.left
                        right = Operation(
                            (left.token_type, left.value), left.right, right
                        )
                    else:
                        # the identity is on the right side
                        tmp = left.right
                        right = Operation(
                            (left.token_type, left.value), left.left, right
                        )

                    right = simplify_addition(right)
                    left = tmp

                if right.token_type & TT_Ident:
                    if isinstance(left.left, Int):
                        # the int is on the left side
                        tmp = left.left
                        right = Operation(
                            (left.token_type, left.value), left.right, right
                        )
                    else:
                        # the int is on the right side
                        tmp = left.right
                        right = Operation(
                            (left.token_type, left.value), left.left, right
                        )

                    left = simplify_addition(right)
                    right = tmp

        node = Operation((node.token_type, node.value), left, right)

        return node

    return node


def print_simplification_status(node: Atom, expected: str):
    if not expected:
        expected = str(node)

    simplified = simplify_addition(node)
    print(node, "=>", simplified, f"[{str(simplified) == expected}]")


node = build_tree(parse("2 + 2"))
print_simplification_status(node, "4")

node = build_tree(parse("2 + (-5)"))
print_simplification_status(node, "-3")

node = build_tree(parse("(-2) + 5"))
print_simplification_status(node, "3")

node = build_tree(parse("a + a"))
print_simplification_status(node, "2 * a")

# node = build_tree(parse("a + a + a"))
# print_simplification_status(node, "2 * a + a")

node = build_tree(parse("2 + 2 + 3 + 2"))
print_simplification_status(node, "9")

node = build_tree(parse("2 + 2 + a"))
print_simplification_status(node, "4 + a")

node = build_tree(parse("2 + a + 2"))
print_simplification_status(node, "a + 4")

node = build_tree(parse("a + a + 2"))
print_simplification_status(node, "2 * a + 2")

node = build_tree(parse("2 + a + a"))
print_simplification_status(node, "2 * a + 2")  # the reason why is arbitrary

# TODO: simplify this is might be a challenge but what is wanted is something to the tune of
node = build_tree(parse("2 + a + a + 2"))
print_simplification_status(node, "2 * a + 4")
