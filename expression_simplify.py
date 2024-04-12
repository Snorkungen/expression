from typing import Union
from expression_parser import (
    RESERVED_IDENTITIES,
    TT_INFO_MASK,
    TT_Float,
    TT_Ident,
    TT_Operation,
    TT_Operation_Commutative,
    TT_Sub,
    parse,
    TT_Add,
    TT_Mult,
    TT_Int,
    TT_Numeric,
    TT_Numeric_Negative,
    TT_Numeric_Positive,
    TT_OOO_MASK,
)
from expression_tree_builder import (
    Atom,
    Int,
    Operation,
    Variable,
    build_tree,
    compare_varible,
)


def simplify(node: Atom) -> Atom:
    if isinstance(node, Operation):
        if node.token_type & TT_Add:
            return simplify_addition(node)
        if node.token_type & TT_Sub:
            return simplify_subtraction(node)
        if node.token_type & TT_Mult:
            return simplify_multiplication(node)

    return node


def simplify_multiplication(node: Atom) -> Atom:
    assert isinstance(node, Operation)
    assert node.token_type & TT_Mult > 0
    # just for the vibes

    if (node.right.token_type & TT_Numeric and node.right.value == 0) or (
        node.left.token_type & TT_Numeric and node.left.value == 0
    ):
        return Int((TT_Int | TT_Numeric, "0"))

    if node.right.token_type & TT_Numeric and node.right.value == 1:
        return node.left
    if node.left.token_type & TT_Numeric and node.left.value == 1:
        return node.right

    factors = list[Atom]()

    nodes = [node.left, node.right]
    while len(nodes) > 0:
        sub_node = nodes.pop()
        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Mult:
            nodes.extend((sub_node.left, sub_node.right))
        else:
            factors.append(
                # simplify already here at ingestion to save progress
                simplify(sub_node)
            )
    else:
        del sub_node

    # combine similar factors
    i = 0
    while i < len(factors):
        # simplify can coerce Floats to Int if possible
        factor = factors[i]

        # accumulate integers
        if isinstance(factor, Int):
            accumulator = factor.value
            j = i + 1
            while j < len(factors):
                sub_factor = factors[j]  # all factors are simplified at ingestion
                # yet again simplify can work on coercing floats

                if isinstance(sub_factor, Int):
                    accumulator *= sub_factor.value
                    factors.pop(j) # remove to remove the correct index
                else:
                    j += 1

            token_type = TT_Int | TT_Numeric
            token_type |= TT_Numeric_Negative if accumulator < 0 else TT_Numeric_Positive

            factors[i] = Int((token_type, str(accumulator)))

        i += 1

    if len(factors) == 1:
        return factors[0]

    # I can't be bothered to do this better
    # an implementation of bubble sort
    # the below stuff seems generic enough to be reused
    def __swap(i, j):
        tmp = factors[i]
        factors[i] = factors[j]
        factors[j] = tmp

    for i in range(len(factors)):
        for j in range(len(factors) - i - 1):
            next_factor = factors[j + 1]
            factor = factors[j]

            if isinstance(next_factor, Operation):
                if factor.token_type & TT_Operation and (
                    (next_factor.token_type & TT_OOO_MASK)
                    < (factor.token_type & TT_OOO_MASK)
                    or (next_factor.token_type & TT_Operation_Commutative)
                    < (factor.token_type & TT_Operation_Commutative)
                ):
                    continue

                __swap(j, j + 1)
            elif isinstance(factor, Variable) and not isinstance(next_factor, Variable):
                __swap(j, j + 1)
            elif next_factor.token_type & TT_Int and factor.token_type & TT_Float:
                __swap(j, j + 1)

    i = 0
    while i < len(factors) - 1:
        factor = factors[i]
        next_factor = factors[i + 1]
        if factor.token_type & TT_Mult:
            print("this brach should not be touched")
            raise RuntimeError
        
        factors[i] = Operation((RESERVED_IDENTITIES["*"], "*"), factor, next_factor)
        factors.pop(i + 1)

        i += 1
    else:
        if len(factors) == 2:
            factors[0] =  Operation((RESERVED_IDENTITIES["*"], "*"), factors[0], factors[i])

    return factors[0]


def simplify_addition(node: Atom) -> Atom:
    assert isinstance(node, Operation)
    assert node.token_type & TT_Add > 0

    # first collect terms
    """
        +
    2       +
        a       +
            a       2

    2 + (a + (a + 2))
    2 + a + a + 2
    a + a + 2 + 2
    a + a + 4
    2 * a + 4
    """
    # terms can be collected due to law of Associative property
    terms = list[Atom]()  # create an empty list

    # walk the tree and collect terms
    nodes = [node.left, node.right]
    while len(nodes) > 0:
        sub_node = nodes.pop()
        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Add:
            nodes.extend((sub_node.right, sub_node.left))
        else:
            terms.append(sub_node)
        del sub_node
    i = 0
    # loop through the terms and combine terms
    while i < len(terms):
        term = simplify(terms[i])

        if term.token_type & TT_Numeric:
            if not isinstance(term, Int):
                i += 1
                continue
            # in future check if it is possible to coerce the value to an Int or something else

            sum = term.value
            j = i + 1
            # for simplicity just loop throug and find integers
            while j < len(terms):
                sub_term = simplify(terms[j])
                if sub_term.token_type & TT_Numeric and isinstance(sub_term, Int):
                    # in future check if it is possible to coerce the value to an Int or something else
                    sum += sub_term.value
                    terms.pop(j)
                else:
                    j += 1

            token_type = TT_Int | TT_Numeric
            token_type |= TT_Numeric_Negative if sum < 0 else TT_Numeric_Positive

            terms[i] = Int((token_type, str(sum)))

        elif (
            isinstance(term, Variable)
            or isinstance(term, Operation)
            and term.token_type & TT_Mult
            and (isinstance(term.right, Variable) or isinstance(term.left, Variable))
        ):
            if isinstance(term, Variable):
                variable = term
                values = [Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1"))]
            else:
                # NOTE: this might have some unforseen side-effects
                if isinstance(term.right, Variable):
                    variable = term.right
                    values = [term.left]
                elif isinstance(term.left, Variable):
                    variable = term.left
                    values = [term.right]
                else:
                    raise "this should not happen"
            # this allows for checking (a + (2 * a)) or (a * 2) + a
            j = i + 1

            while j < len(terms):
                sub_term = simplify(terms[j])
                # todo add to "k * x"
                if compare_varible(variable, sub_term):
                    values.append(Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1")))
                    terms.pop(j)
                    continue
                elif isinstance(sub_term, Operation) and sub_term.token_type & TT_Mult:
                    # check that one node on the is the variable
                    if isinstance(sub_term.right, Variable) and compare_varible(
                        sub_term.right, variable
                    ):
                        values.append(sub_term.left)
                        terms.pop(j)
                        continue
                    if isinstance(sub_term.left, Variable) and compare_varible(
                        sub_term.left, variable
                    ):
                        values.append(sub_term.right)
                        terms.pop(j)
                        continue
                j += 1

            if len(values) < 2:
                count = values[0]
            else:
                count = Operation((RESERVED_IDENTITIES["+"], "+"), values[0], values[1])
                sub_node = count
                for j in range(2, len(values)):
                    sub_node.left = Operation(
                        (RESERVED_IDENTITIES["+"], "+"), values[j], sub_node.left
                    )
                    sub_node = sub_node.left

                count = simplify(count)

            terms[i] = simplify(
                # the simplyfy call is due is is to do a final check
                Operation((RESERVED_IDENTITIES["*"], "*"), count, variable)
            )

        i += 1

    # use the same strategy as tree builder
    # but in reality i would like to do some sorting of the values so

    if len(terms) == 1:
        return terms[0]

    # I can't be bothered to do this better
    # an implementation of bubble sort
    # the below stuff seems generic enough to be reused
    def __swap(i, j):
        tmp = terms[i]
        terms[i] = terms[j]
        terms[j] = tmp

    for i in range(len(terms)):
        for j in range(len(terms) - i - 1):
            next_term = terms[j + 1]
            term = terms[j]

            if isinstance(next_term, Operation):
                if term.token_type & TT_Operation and (
                    (next_term.token_type & TT_OOO_MASK)
                    < (term.token_type & TT_OOO_MASK)
                    or (next_term.token_type & TT_Operation_Commutative)
                    < (term.token_type & TT_Operation_Commutative)
                ):
                    continue

                __swap(j, j + 1)
            elif isinstance(next_term, Variable) and not isinstance(term, Variable):
                __swap(j, j + 1)
            elif next_term.token_type & TT_Int and term.token_type & TT_Float:
                __swap(j, j + 1)

            # there should be some thinking for functions but can't be bothered right now

    # reconstruct tree using the same strategy as tree_builder
    i = 0
    while i < len(terms) - 1:
        term = terms[i]
        next_term = terms[i + 1]
        if term.token_type & TT_Add:
            print("this branch should not be touched")
            continue  # assume that this is an a node that has been touched already

        terms[i] = Operation((RESERVED_IDENTITIES["+"], "+"), term, next_term)
        terms.pop(i + 1)

        i += 1
    else:
        if len(terms) == 2:
            terms[0] = Operation((RESERVED_IDENTITIES["+"], "+"), terms[0], terms[1])

    return terms[0]


def simplify_subtraction(node: Atom) -> Atom:
    assert isinstance(node, Operation)
    assert node.token_type & TT_Sub > 0
    """
            -
        -       c
    a      b

            +
        *               +
    -1      c       *        a
                -1      b
    """

    left = node.right  # swap left and right

    # multiply left node with -1
    left = simplify_multiplication(
        Operation(
            (RESERVED_IDENTITIES["*"], "*"),
            left,
            Int(((TT_Int | TT_Numeric | TT_Numeric_Negative), "-1")),
        )
    )

    right = simplify(node.left)

    node = Operation((RESERVED_IDENTITIES["+"], "+"), left, right)

    return simplify_addition(node)


def print_simplification_status(node: Atom, expected: str, s=simplify_subtraction):
    if __name__ != "__main__":
        return
    if not expected:
        expected = str(node)

    simplified = s(node)
    print(node, "=>", simplified, f"[{str(simplified) == expected}]")


node = build_tree(parse("2 * 4 * -1"))
print_simplification_status(node, "-8", simplify_multiplication)

node = build_tree(parse("4 * a * -1"))
print_simplification_status(node, "-4 * a", simplify_multiplication)
