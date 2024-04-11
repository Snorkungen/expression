from expression_parser import (
    RESERVED_IDENTITIES,
    TT_Ident,
    parse,
    TT_Add,
    TT_Mult,
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

    left = node.left
    right = node.right

    if isinstance(right, Operation) and right.token_type & TT_Add:
        right = simplify_addition(node.right)

        while isinstance(right, Operation) and right.token_type & TT_Add:
            left = Operation((node.token_type, node.value), left, right.left)
            right = right.right
        # i wonder if this is a infinite loop

    node = Operation((node.token_type, node.value), left, right)

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

    # special case 2*a + 2*a = 4*a
    if (
        isinstance(node.left, Operation)
        and node.left.token_type & TT_Mult
        and isinstance(node.right, Operation)
        and node.right.token_type & TT_Mult
    ):
        if (
            isinstance(node.left.right, Variable)
            and isinstance(node.right.right, Variable)
            and compare_varible(node.left.right, node.right.right)
        ):
            return Operation(
                (node.left.token_type, node.left.value),
                simplify_addition(
                    Operation((node.token_type, node.value), left.left, node.right.left)
                ),
                left.right,
            )

        if (
            isinstance(node.left.right, Variable)
            and isinstance(node.right.left, Variable)
            and compare_varible(node.left.right, node.right.left)
        ):
            return Operation(
                (node.left.token_type, node.left.value),
                simplify_addition(
                    Operation(
                        (node.token_type, node.value), left.left, node.right.right
                    )
                ),
                left.right,
            )

        if (
            isinstance(node.left.left, Variable)
            and isinstance(node.right.left, Variable)
            and compare_varible(node.left.left, node.right.left)
        ):
            return Operation(
                (node.left.token_type, node.left.value),
                simplify_addition(
                    Operation(
                        (node.token_type, node.value), left.right, node.right.right
                    )
                ),
                left.left,
            )

        if (
            isinstance(node.left.left, Variable)
            and isinstance(node.right.right, Variable)
            and compare_varible(node.left.left, node.right.right)
        ):
            return Operation(
                (node.left.token_type, node.left.value),
                simplify_addition(
                    Operation(
                        (node.token_type, node.value), left.right, node.right.left
                    )
                ),
                left.left,
            )

        return node

    # assume the the depth is 2 1-based
    # and move the variable

    if isinstance(node.left, Operation):
        # this is some duplication but it is a special case
        if isinstance(right, Variable):
            # check specifically that the right is n * x and left is x
            if node.left.token_type & TT_Add:
                left = simplify_addition(left)

            if left.token_type & TT_Add:
                tmp = right
                if compare_varible(left.right, right):
                    right = left.left
                    left = Operation(
                        (RESERVED_IDENTITIES["+"], "+"), left.right, tmp
                    )
                elif compare_varible(left.left, right):
                    right = left.right
                    left = Operation((RESERVED_IDENTITIES["+"], "+"), left.left, tmp)

                return Operation((node.token_type, node.value), simplify_addition(left),right)


            if left.token_type & TT_Mult:
                if compare_varible(left.right, right) and isinstance(left.left, Int):
                    tmp = Int((TT_Numeric | TT_Int | TT_Numeric_Positive, "1"))
                    left.left = Operation(
                        (RESERVED_IDENTITIES["+"], "+"), left.left, tmp
                    )
                    left.left = simplify_addition(left.left)
                    return left
                if compare_varible(left.left, right) and isinstance(left.right, Int):
                    tmp = Int((TT_Numeric | TT_Int | TT_Numeric_Positive, "1"))
                    left.right = Operation(
                        (RESERVED_IDENTITIES["+"], "+"), left.right, tmp
                    )
                    left.right = simplify_addition(left.left)
                    return left

        if not node.left.token_type & TT_Add:
            return node

        # this is where the recursion occurs

        left = simplify_addition(node.left)

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

        if isinstance(right, Variable):
            tmp = right

            if not (left.right.token_type & TT_Ident):
                right = left.right
                left.right = tmp
            elif not (left.left.token_type & TT_Ident):
                right = left.left
                left.left = tmp

            left = simplify_addition(left)

        if right.token_type & TT_Numeric:
            if isinstance(left, Operation) and left.token_type & TT_Add:

                if left.right.token_type & TT_Numeric:
                    tmp = left.right
                    left = left.left
                    right = Operation((node.token_type, node.value), tmp, right)
                    right = simplify_addition(right)
    elif isinstance(node.right, Operation):
        # swap and try again, I have no clue as to what is going on
        return simplify_addition(
            Operation(
                (node.token_type, node.value),
                Operation(
                    (node.right.token_type, node.right.value),
                    node.right.left,
                    node.right.right,
                ),
                left,
            )
        )

    node = Operation((node.token_type, node.value), left, right)

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

node = build_tree(parse("a + a + a + a"))
print_simplification_status(node, "4 * a")

node = build_tree(parse("1 + a + 1 + a + a + 1"))
print_simplification_status(node, "3 * a + 3")

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

node = build_tree(parse("2 + a + a + 2"))
print_simplification_status(node, "2 * a + 4")

node = build_tree(parse("2 + (2 + 2)"))
print_simplification_status(node, "6")

node = build_tree(parse("2 + 2 + a + (2 + a + (3 *a + 3))"))
print_simplification_status(node, "5 * a + 9")

node = build_tree(parse("2 * a + a *2 "))
print_simplification_status(node, "4 * a")

node = build_tree(parse("1 + 1/2 + 1 + a + a"))
print_simplification_status(node, "2 * a + 1 / 2 + 2")