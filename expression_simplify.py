from typing import Callable, Iterable, Tuple, Union
from expression_parser import (
    RESERVED_IDENTITIES,
    TT_Div,
    parse,
    TT_Float,
    TT_Operation,
    TT_Operation_Commutative,
    TT_Sub,
    TT_Add,
    TT_Mult,
    TT_Exponent,
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


def __join_nodes_into_tree(node: Operation, nodes: list[Atom]) -> Operation:
    """takes a root node and adds the nodes into the tree"""
    sub_node = node
    for j in range(len(nodes)):
        sub_node.left = Operation(
            (node.token_type, node.value), nodes[j], sub_node.left
        )
        sub_node = sub_node.left

    return node


def compare_atoms(a: Atom, b: Atom) -> bool:
    if type(a) != type(b):
        return False

    if a.token_type & TT_Numeric:
        # NOTE: floats are innaccurate so this might cause a sneaky bug in the future
        return a.value == b.value

    if isinstance(a, Variable):
        return compare_varible(a, b)

    if isinstance(a, Operation):
        # TODO: create a function that compares operation trees,
        # i think the function will assume that the operation is simplified,
        # or this function simplifies the operation first
        return False


def collect_factors(node: Operation, factors: list[Atom] = None, simplify_node=True):
    if factors == None:
        factors = []
    if not isinstance(node, Operation):
        return factors

    nodes = [node.left, node.right]
    while len(nodes) > 0:
        sub_node = nodes.pop()
        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Mult:
            nodes.extend((sub_node.left, sub_node.right))
        else:
            factors.append(
                # simplify already here at ingestion to save progress
                simplify(sub_node)
                if simplify_node
                else sub_node
            )
    else:
        del sub_node

    return factors


def collect_terms(node: Operation, terms: list[Atom] = None, simplify_node=True):
    if None == terms:
        terms = []
    if not isinstance(node, Operation):
        return terms

    nodes = [node.left, node.right]
    while len(nodes) > 0:
        sub_node = nodes.pop()
        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Add:
            nodes.extend((sub_node.right, sub_node.left))
        else:
            potential_term = simplify(sub_node) if simplify_node else sub_node
            # simplify multiplication might return and addition operation
            if isinstance(sub_node, Operation) and sub_node.token_type & TT_Add:
                nodes.extend((potential_term.right, potential_term.left))
            else:
                terms.append(potential_term)

    del sub_node

    return terms


def expand_exponentiation(root: Operation):
    """takes a value like [(a * b) ^ 2] expands the factors and adds [a, a, b, b] to factors list"""
    assert isinstance(root, Operation)
    assert root.right.value > 0, "Negative exponents are not supported"

    left = root.left
    factors: list[Atom] = []
    if not left.token_type & TT_Mult:
        factors.append(left)
    else:
        collect_factors(left, factors)

    internal_factors = []

    for i, factor in enumerate(factors):
        if factor.token_type & TT_Exponent and isinstance(factor.right, Int):
            expand_exponentiation(factor, internal_factors)
            factors.pop(i)

    factors.extend(internal_factors)

    internal_factors = []
    for factor in factors:
        internal_factors.extend([factor] * (root.right.value - 1))

    factors.extend(internal_factors)

    return factors


def __construct_node_from_factors(
    factors: list[Atom], operation_token: Tuple[int, str]
):
    assert len(factors) >= 1

    if len(factors) < 2:
        return factors[0]

    factor = Operation(operation_token, factors[0], factors[1])
    return simplify(__join_nodes_into_tree(factor, factors[2:]))


def simplify(node: Atom) -> Atom:
    if isinstance(node, Operation):
        if node.token_type & TT_Add:
            return simplify_addition(node)
        if node.token_type & TT_Sub:
            return simplify_subtraction(node)
        if node.token_type & TT_Mult:
            return simplify_multiplication(node)
        if node.token_type & TT_Div:
            return simplify_division(node)

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
        return simplify(node.left)
    if node.left.token_type & TT_Numeric and node.left.value == 1:
        return simplify(node.right)

    factors = list[Atom]()

    collect_factors(node, factors)  # collect all factors of interest

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
                    factors.pop(j)  # remove to remove the correct index
                else:
                    j += 1

            token_type = TT_Int | TT_Numeric
            token_type |= (
                TT_Numeric_Negative if accumulator < 0 else TT_Numeric_Positive
            )

            factors[i] = Int((token_type, str(accumulator)))
        elif isinstance(factor, Variable) or (
            isinstance(factor, Operation)
            and factor.token_type & TT_Exponent
            and isinstance(factor.left, Variable)
        ):
            if isinstance(factor, Variable):
                variable = factor
                values = [Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1"))]
            else:
                variable = factor.left
                values = [factor.right]

            j = i + 1
            while j < len(factors):
                sub_factor = factors[j]

                if isinstance(sub_factor, Variable) and compare_varible(
                    factors[j], variable
                ):
                    values.append(Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1")))
                    factors.pop(j)
                    continue
                elif (
                    isinstance(sub_factor, Operation)
                    and sub_factor.token_type & TT_Exponent
                    and compare_varible(sub_factor.left, variable)
                ):
                    print(variable, sub_factor, "E")
                    values.append(sub_factor.right)
                    factors.pop(j)
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

            # because i can't be bothered to simplify exp...
            if isinstance(count, Int) and count.value == 1:
                pass
            else:
                factors[i] = Operation((RESERVED_IDENTITIES["^"], "^"), variable, count)
        elif isinstance(factor, Operation) and factor.token_type & TT_Div:
            # this should be somehow integrated into the first check
            # where 2 * (1 / 2) = 2 / 1 * (1 / 2)

            # I do not know this branch is all consuming and just converts the whole multiplication into a divison operation

            dividends = [factor.left]
            divisors = [factor.right]

            j = 0
            while j < len(factors):
                sub_factor = factors[j]

                if sub_factor == factor:
                    # the memory addresses are the same
                    j += 1
                    continue
                    raise RuntimeError("If this happens there is a bug somewhere")

                # print(factor, sub_factor, list(map(str, factors)))
                if isinstance(sub_factor, Operation) and sub_factor.token_type & TT_Div:
                    dividends.append(sub_factor.left)
                    divisors.append(sub_factor.right)
                else:
                    dividends.append(sub_factor)

                factors.pop(j)

                if j < i:
                    j += 1

            if len(dividends) < 2:
                dividend = dividends[0]
            else:
                dividend = Operation(
                    (RESERVED_IDENTITIES["*"], "*"), dividends[0], dividends[1]
                )

                dividend = __join_nodes_into_tree(dividend, dividends[2:])
                # simplify_division will simplify multiplication

            # print(list(map(str, divisors)), factor)
            if len(divisors) < 2:
                divisor = divisors[0]
            else:
                divisor = Operation(
                    (RESERVED_IDENTITIES["*"], "*"), divisors[0], divisors[1]
                )  # simplify_division will simplify multiplication

                divisor = __join_nodes_into_tree(divisor, divisors[2:])
                # simplify_division will simplify multiplication

            # if there is a division it consumes all factors and spits out some other stuff
            return simplify_division(
                Operation((RESERVED_IDENTITIES["/"], "/"), dividend, divisor)
            )

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

            if isinstance(factor, Operation) and not isinstance(next_factor, Operation):
                __swap(j, j + 1)
            elif isinstance(factor, Variable) and not isinstance(next_factor, Variable):
                __swap(j, j + 1)
            elif (
                isinstance(factor, Variable)
                and isinstance(next_factor, Operation)
                and next_factor.token_type & TT_Exponent
                and isinstance(next_factor.left, Operation)
            ):
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
            factors[0] = Operation(
                (RESERVED_IDENTITIES["*"], "*"), factors[0], factors[1]
            )

    return factors[0]


def simplify_multiplication_distribute(node: Atom) -> Atom:
    if not isinstance(node, Operation) or not node.token_type & TT_Mult:
        return node

    factors = collect_factors(node)
    i = 0
    while i < len(factors):
        factor = factors[i]
        if not factor.token_type & TT_Add:
            i += 1
            continue

        terms = collect_terms(factor, simplify_node=False)
        factors.pop(i)

        remaining_factor = __construct_node_from_factors(
            factors, (RESERVED_IDENTITIES["*"], "*")
        )

        j = 0
        while j < len(terms):
            terms[j] = simplify_multiplication_distribute(
                simplify_multiplication(
                    Operation(
                        (RESERVED_IDENTITIES["*"], "*"), remaining_factor, terms[j]
                    )
                )
            )
            j += 1

        if len(terms) < 2:
            return terms[0]

        node = Operation((RESERVED_IDENTITIES["+"], "+"), terms[0], terms[1])
        (__join_nodes_into_tree(node, terms[2:]))

        return node
    return node


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
    terms = collect_terms(node, terms)
    i = 0

    # loop through the terms and combine terms
    while i < len(terms):
        term = terms[i]

        if term.token_type & TT_Numeric:
            if not isinstance(term, Int):
                i += 1
                continue
            # in future check if it is possible to coerce the value to an Int or something else

            sum = term.value
            j = i + 1
            # for simplicity just loop throug and find integers
            while j < len(terms):
                sub_term = terms[j]
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
        ):
            if isinstance(term, Variable):
                variable = term
                values = [Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1"))]
            else:
                # So the issue was with there are two variables to choose from
                # get factors look at variables, choose the more popular variable with matches

                factors = collect_factors(
                    term
                )  # this is function call doing a redundand simplify
                variables = list(filter(lambda f: isinstance(f, Variable), factors))
                if 0 == len(variables):
                    # this should be a function due to how python works
                    # and therefore this would be a place to return
                    i += 1
                    continue

                # have some type of ranking step that chooses which variable to choose
                scores = [1] * len(variables)

                j = i + 1
                while j < len(terms):
                    sub_term = simplify(terms[j])
                    if isinstance(sub_term, Variable):
                        for k, var in enumerate(variables):
                            if compare_varible(var, sub_term):
                                # MEMORYISCHEAP
                                scores[k] += 1
                    elif (
                        isinstance(sub_term, Operation)
                        and sub_term.token_type & TT_Mult
                    ):
                        # reuse factors
                        for k, var in enumerate(variables):
                            for sub_var in collect_factors(sub_term):
                                if compare_varible(var, sub_var):
                                    scores[k] += 1

                    j += 1

                vidx = 0
                for j in range(1, len(scores)):

                    if scores[vidx] < scores[j]:
                        vidx = j
                    elif scores[vidx] > 1 and scores[vidx] == scores[j]:
                        # !TODO: figure out what to do i a situation like this
                        print(
                            "two variables are equally used",
                            f"vidx: {vidx}",
                            scores,
                            list(map(str, variables)),
                        )

                if 1 == scores[vidx]:
                    # there is no other occurence of the chosen value therefore this should actually return out
                    # but cannot be bothered
                    i += 1
                    continue

                variable = variables[vidx]
                factors.remove(variable)
                # reconstruct values from factors

                values = [
                    __construct_node_from_factors(
                        factors, (RESERVED_IDENTITIES["*"], "*")
                    )
                ]

            j = i + 1
            while j < len(terms):
                sub_term = simplify(terms[j])
                if compare_varible(variable, sub_term):
                    values.append(Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1")))
                    terms.pop(j)
                    continue
                elif isinstance(sub_term, Operation) and sub_term.token_type & TT_Mult:
                    factors = collect_factors(sub_term)
                    # assume that all instances of the variable is a standalone i.e. not [a * a]
                    k = 0
                    pos = -1
                    while k < len(factors):
                        if compare_varible(factors[k], variable):
                            pos = k
                        k += 1
                    if pos >= 0:
                        terms.pop(j)
                        factors.pop(pos)
                        values.append(
                            __construct_node_from_factors(
                                factors, (RESERVED_IDENTITIES["*"], "*")
                            )
                        )
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
    left = simplify_multiplication_distribute(
        simplify_multiplication(
            Operation(
                (RESERVED_IDENTITIES["*"], "*"),
                left,
                Int(((TT_Int | TT_Numeric | TT_Numeric_Negative), "-1")),
            )
        )
    )

    right = simplify(node.left)

    node = Operation((RESERVED_IDENTITIES["+"], "+"), left, right)

    return simplify_addition(node)


def __get_factors(value: int) -> Iterable[Int]:
    values: list[Int] = [
        # Int((TT_Int | TT_Numeric | TT_Numeric_Positive, str(1)))
    ]  # 1 is assumed but also known so it is unnessecery

    if 0 > value:
        # value is negative
        values.append(Int((TT_Int | TT_Numeric | TT_Numeric_Negative, str(-1))))
        value = value * -1

    if 1 == value:
        # I do not know if this is the right way to go but 1 is always a factor
        return values

    # a better more efficent algo... probably exists due to all of this being a solved problem
    divisor = 2
    while divisor <= value:
        if 0 == value % divisor:
            values.append(Int((TT_Int | TT_Numeric, str(divisor))))

            # reset divisor
            value = value / divisor
            divisor = 2
        else:
            divisor += 1

    return values


def simplify_division(node: Atom) -> Atom:
    assert isinstance(node, Operation)
    assert node.token_type & TT_Div

    if node.right.token_type & TT_Numeric and 1 == node.right.value:
        return simplify(node.right)
    if node.right.token_type & TT_Numeric and 0 == node.right.value:
        # this should actually fail
        assert (
            False
        ), "The divisor is zero, do not know if this should silently fail, or just do something else"

    dividends = list[Atom]()
    divisors = list[Atom]()

    # first just collect dividends
    nodes = [node.left]
    while 0 < len(nodes):
        sub_node = nodes.pop()

        if isinstance(sub_node, Operation):
            if sub_node.token_type & TT_Mult:
                nodes.extend((sub_node.left, sub_node.right))
            elif sub_node.token_type & TT_Div:
                nodes.append(sub_node.left)
                # have another step that does logic and factorizes values
                # i'm not sure how the simplification step would work for the divisor
                divisors.append(sub_node.right)
            else:
                dividends.append(sub_node)
        else:
            dividends.append(sub_node)
    else:
        del sub_node

    nodes = [node.right]
    while 0 < len(nodes):
        sub_node = nodes.pop()

        if isinstance(sub_node, Operation):
            if sub_node.token_type & TT_Mult:
                nodes.extend((sub_node.left, sub_node.right))
            elif sub_node.token_type & TT_Div:
                nodes.append(sub_node.left)
                # have another step that does logic and factorizes values
                # i'm not sure how the simplification step would work for the divisor
                dividends.append(sub_node.right)
            else:
                divisors.append(sub_node)
        else:
            divisors.append(sub_node)
    else:
        del sub_node

    def factorize_factors(
        factors: list[Atom],
        handle_zero: Callable[[Atom], Atom],
    ):
        i = 0
        end = len(factors)
        while i < end:
            factor = simplify(factors[i])

            if factor.token_type & TT_Numeric:
                if 1 == factor.value:
                    factors.pop(i)
                    end -= 1
                    continue
                elif 0 == factor.value:
                    print("simplify_division", node, "divisor is zero")
                    return handle_zero(node)

            if isinstance(factor, Int):

                int_factors = __get_factors(factor.value)

                if len(int_factors) <= 1:
                    i += 1
                    continue  # i do not know what this is doing
                factors[i] = int_factors[0]
                factors.extend(int_factors[1:])

            if isinstance(factor, Operation):
                if factor.token_type & TT_Exponent and isinstance(factor.right, Int):
                    assert factor.right.value > 0, "Negative Exponent not supported"
                    factors.extend(expand_exponentiation(factor))
                    factors.pop(i)
                    end -= 1
                else:
                    assert False, "Factorising operations not handled at the moment"
            i += 1

    def handle_zero_dividend(_):
        return Int((TT_Numeric | TT_Int, "0"))  # just skip the end result will be 0

    factorize_factors(dividends, handle_zero_dividend)  # factorize dividends

    def handle_zero_divisor(_):
        raise ValueError("divisor is zero")

    factorize_factors(divisors, handle_zero_divisor)  # factorize divisors

    # print(list(map(str, dividends)))
    # print(list(map(str, divisors)))

    # cancel out same values, don't know how to express what i'm actually doing
    i = 0
    while i < len(divisors):
        # NOTE: there might be a bug because some values could,
        # skate through without getting simplified

        divisor = divisors[i]
        j = 0
        flag = True
        while j < len(dividends) and flag:
            dividend = dividends[j]
            if compare_atoms(divisor, dividend):
                # atoms are equal
                dividends.pop(j)
                divisors.pop(i)
                # break NOTE: you cannot break out of just a inner loop in python

                # the few lines below are dumb but i works so no need to mess with it
                if i == 0:
                    i -= 1
                else:
                    i -= 2

                flag = False  # force the exit of inner loop

            j += 1

        i += 1

    # check special cases
    if 0 == len(divisors):
        if 0 == len(dividends):
            # this means that all factors got cancelled out
            return Int((TT_Int | TT_Numeric | TT_Numeric_Positive, str(1)))
        else:
            return __construct_node_from_factors(
                dividends, (RESERVED_IDENTITIES["*"], "*")
            )

    if 0 == len(dividends):
        dividend = Int((TT_Int | TT_Numeric | TT_Numeric_Positive, "1"))
    else:
        dividend = __construct_node_from_factors(
            dividends, (RESERVED_IDENTITIES["*"], "*")
        )

    divisor = __construct_node_from_factors(divisors, (RESERVED_IDENTITIES["*"], "*"))

    # TODO: break out identities from dividend
    # [a / 2] => [1 / 2 * a]
    # [(5 * a * b) / 4] => 5 / 4 * (a * b)

    return Operation((RESERVED_IDENTITIES["/"], "/"), dividend, divisor)


def print_simplification_status(node: Atom, expected: str, s=simplify):
    if __name__ != "__main__":
        return
    if not expected:
        expected = str(node)

    simplified = s(node)
    print(f"[{str(simplified) == expected}]", node, "=>", simplified)


node = build_tree(parse("-1 * (b + (a * 2)) * 2"))
print_simplification_status(
    node,
    "-4 * a + -2 * b",
    lambda x: simplify_multiplication_distribute(simplify_multiplication(x)),
)

node = build_tree(parse("-2 * (b + 2)"))
print_simplification_status(
    node,
    "-2 * b + -4",
    lambda x: simplify_addition(simplify_multiplication_distribute(simplify_multiplication(x))),
)

node = build_tree(parse("(a + 3) * (b + 2)"))
print_simplification_status(
    node,
    "(6 + 3 * b) + (2 * a + b * a)",
    lambda x: simplify_multiplication_distribute(simplify_multiplication(x)),
)

node = build_tree(parse("(a + 3) * (b + 2) * 2"))
print_simplification_status(
    node,
    "(12 + 6 * b) + (4 * a + (2 * b) * a)",
    lambda x: simplify_multiplication_distribute(simplify_multiplication(x)),
)

node = build_tree(parse("(6 + 2 * a) + (3 * b + a * b)"))
print_simplification_status(node, "(2 * a + (a + 3) * b) + 6", simplify_addition)

node = build_tree(parse("a*x + b*c - (d * x + e * f)"))
# node = build_tree(parse("a*x + b*c + (-1*d * x + -1*e * f)"))
print_simplification_status(node, "((-1 * e) * f + b * c) + (a + -1 * d) * x", simplify)

"""
    a = (-1 * 8b + 2) / 2b
    a = (-8b + 2) / 2b
    a = -8b / 2b + 2 / 2b
    a = 1 / b + -4  
"""
