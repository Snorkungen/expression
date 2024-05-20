from expression_parser import parse
from expression_tree_builder2 import *
from utils import *


def _collect_dividends_and_divisors(
    node: TokenValue,
) -> Tuple[List[TokenValue], List[TokenValue]]:
    left: List[TokenValue] = []  # a list containing all values that would be a dividend
    right: List[TokenValue] = []  # a list containinga all dividends

    if node.token_type & TT_Mult and isinstance(node, Operation):
        for value in node.values:
            l, r = _collect_dividends_and_divisors(value)
            left.extend(l)
            right.extend(r)
    elif node.token_type & TT_Div and isinstance(node, Operation):
        l, r = _collect_dividends_and_divisors(node.left)
        left.extend(l)
        right.extend(r)
        l, r = _collect_dividends_and_divisors(node.right)
        left.extend(r)
        right.extend(l)
    else:
        left.append(node)

    return left, right


def _calculate(node: TokenValue) -> TokenValue:
    """Add and Multiply all integers, where possible"""
    if is_node_atomic(node) or not isinstance(node, Operation):
        return node

    if node.token_type & TT_Add:
        # gå igenom värden
        integer_sum = 0
        non_integers = []

        for value in node.values:
            value = _calculate(value)

            if value.token_type & TT_Int:
                integer_sum += value.token_value
            else:
                non_integers.append(value)

        if integer_sum == 0:
            return _construct_token_value_with_values(node, *non_integers)
        return _construct_token_value_with_values(
            node, Integer.create(integer_sum), *non_integers
        )

    if node.token_type & TT_Mult:
        integer_product = 1
        non_integers = []

        for value in node.values:
            value = _calculate(value)

            if value.token_type & TT_Int:
                integer_product *= value.token_value
            else:
                non_integers.append(value)

        if integer_product == 0:
            return Integer.create(0)
        if integer_product == 1:
            return _construct_token_value_with_values(node, *non_integers)
        return _construct_token_value_with_values(
            node, Integer.create(integer_product), *non_integers
        )

    if node.token_type & TT_Exponent:
        base = _calculate(node.left)
        exponent = _calculate(node.right)

        if base.token_type & TT_Int and exponent.token_type & TT_Int:
            return Integer.create(base.token_value**exponent.token_value)

        if base.token_type & TT_Exponent and isinstance(base, Operation):
            return Operation.create(
                "^", base.left, _calculate(Operation.create("*", base.right, exponent))
            )

        if base.token_value == 1:
            return base

        if exponent.token_value == 0:
            return Integer.create(1)
        if exponent.token_value == 1:
            return base

        return Operation(node.token, base, exponent)

    if node.token_type & TT_Div:
        # TODO: could do some factoring if possible
        dividend = _calculate(node.left)
        divisor = _calculate(node.right)

        if dividend.token_value == 0:
            return Integer.create(0)

        if divisor.token_value == 1:
            return dividend

        return Operation(node.token, dividend, divisor)

    raise NotImplementedError


def _construct_token_value_with_values(
    source: TokenValue, *values: Tuple[TokenValue, ...]
) -> TokenValue:
    if isinstance(source, str):
        source = Operation.create(source)

    if len(values) == 0:
        if source.token_type & TT_Mult:
            return Integer.create(1)
        if source.token_type & TT_Add:
            return Integer.create(0)

        raise NotImplementedError

    if len(values) == 1:
        return values[0]

    return Operation(
        source.token,
        *values,
    )


def is_node_atomic(a: TokenValue):
    """Wheter the node can be split"""
    return (
        bool(a.token_type & TT_Numeric)
        or b_nand(a.token_type, TT_INFO_MASK) == TT_Ident
    )


def compare_variables(a: TokenValue, b: TokenValue) -> bool:
    if a == b:
        return True

    if isinstance(a, Variable) and isinstance(b, Variable):
        return a.token_value == b.token_value

    return False


def compare_numeric(a: TokenValue, b: TokenValue) -> bool:
    if a == b:
        return True

    return (a.token_type | b.token_type) & TT_Numeric and a.token_value == b.token_value


def _expand(node: TokenValue) -> TokenValue:
    """2 *(2 + 3) => 2 * 2 + 2 * 3"""

    if is_node_atomic(node) or not isinstance(node, Operation):
        return node

    if node.token_type & TT_Add:
        return Operation(node.token, *map(_expand, node.values))

    if node.token_type & TT_Div:
        return Operation(node.token, _expand(node.left), _expand(node.right))

    if node.token_type & TT_Exponent:
        exponent = _expand(node.right)
        base = _expand(node.left)

        if base.token_type & TT_Add:
            return Operation.create(
                "+", *map(lambda v: Operation(node.token, v, exponent), base.values)
            )

        if base.token_type & TT_Div:
            return Operation.create(
                "/", *map(lambda v: Operation(node.token, v, exponent), base.values)
            )

        return Operation.create("^", base, exponent)

    if node.token_type & TT_Mult:
        node = Operation.create("*", *map(_expand, node.values))

        for i, value in enumerate(node.values):
            if value.token_type & TT_Add == 0 or not isinstance(value, Operation):
                continue

            factors = tuple(
                map(
                    lambda v: v[1],
                    filter(lambda j: j[0] != i, enumerate(node.values)),
                )
            )

            return _expand(
                _construct_token_value_with_values(
                    value,
                    *map(lambda f: Operation.create("*", f, *factors), value.values),
                )
            )

        return node

    print(node)
    raise NotImplementedError


def factor_terms_n_stuff(node: Operation):
    """
    2 * a * b + 3 * a * b => (2 + 3) * a * b
    2 * a * b + 3 * a * c => (2 + 3 * c) * a * b
    2 * a * b / 2 - 4 * a * b => (2 / 2 - 4) * a * b
    2 * a * b / 2 - 4 * a * b => (2 / 2 - 4) * a * b
    """
    assert node.token_type & TT_Add

    term_info: List[Tuple[TokenValue, Iterable[TokenValue]]] = []

    for term in node.values:
        dividends, divisors = _collect_dividends_and_divisors(term)

        left_factors = []
        factors = []

        # NOTE: divisors exist and are problematic
        if len(divisors) > 0:
            # if divisors are only an integer then
            for divisor in divisors:
                divisor = _calculate(_expand(divisor))

                factor = Operation.create("/", Integer.create(1), divisor)
                if is_node_atomic(divisor) and divisor.token_type & TT_Numeric:
                    left_factors.append(factor)
                else:
                    # check that divisor does not contain special stuff
                    # NOTE Float exist but are going to be ignored and not function as expected
                    factors.append(factor)

        integer_product = 1
        for factor in dividends:
            if factor.token_type & TT_Int:
                integer_product *= factor.token_value
                continue
            factors.append(factor)

        left_factor: TokenValue
        if len(left_factors) <= 0:
            left_factor = Integer.create(integer_product)
        elif integer_product == 1:
            left_factor = _construct_token_value_with_values("*", *left_factors)
        else:
            left_factor = _construct_token_value_with_values(
                "*", Integer.create(integer_product), *left_factors
            )

        term_info.append((left_factor, factors))

    def value_in_factors(factor: TokenValue, factors: Iterable[TokenValue]):
        for f in factors:
            if compare_values(factor, f):
                return True
        return False

    factors = term_info[0][1]

    # get the shortest amount of factors
    # that all terms have in common
    for _, term_factors in term_info[1:]:
        factors = tuple(filter(lambda n: value_in_factors(n, term_factors), factors))

    if len(factors) <= 0:
        return Operation.create("*", Integer.create(1), node)  # there is nothing to do

    terms = []

    for left_factor, right_factors in term_info:
        additional_left_factors = filter(
            lambda f: not value_in_factors(f, factors), right_factors
        )
        left_factor = _construct_token_value_with_values(
            "*", left_factor, *additional_left_factors
        )

        terms.append(left_factor)

    term = _calculate(_construct_token_value_with_values("+", *terms))
    return _construct_token_value_with_values("*", term, *factors)


def is_node_negative(node: TokenValue) -> bool:
    """Returns if node can be perceived as being negative

    -1 => True

    2 - 10 => True

    -1 * a => True

    2 * a - 4 * a => True
    """
    if node.token_type & TT_Numeric and node.token_value < 0:
        return True
    if is_node_atomic(node):
        return False

    # NOTE recursive call to coerce into fraction
    coerced_fraction = _calculate(_coerce_into_fraction(node))
    dividends, divisors = _collect_dividends_and_divisors(coerced_fraction)

    # for terms factor

    flip_flop = False
    for factor in (*dividends, *divisors):
        # factor = _calculate(factor) # should have been calculated earlier
        if is_node_atomic(factor) and is_node_negative(factor):
            flip_flop = not flip_flop

        if factor.token_type & TT_Add and isinstance(node, Operation):
            # NOTE Recursive call to compare variables
            # NOTE the below function returns the left most as the interesting value
            left = factor_terms_n_stuff(factor).left

            # TODO: in future the below check is wrong Float can be known if theyre negative or positive
            if is_node_negative(left):
                flip_flop = not flip_flop

    # what does it mean for an expression to be negative
    # the following will be handled
    # -1 * a
    # -1 * -2 * (1 + -2)

    return flip_flop


def _coerce_into_fraction(node: TokenValue):
    """
    Make node a fraction
    1 => 1 / 1
    2 + 1 / 2 => (2 * 2 + 1) / 2
    (1 + 1 / 2) / (3 / 4) =>
    """
    ONE_INTEGER = Integer.create(1)
    fileter_ones = lambda n: n.token_type & TT_Numeric == 0 or not compare_numeric(
        n, ONE_INTEGER
    )

    if is_node_atomic(node) or not isinstance(node, Operation):
        return Operation.create("/", node, ONE_INTEGER)

    node = _expand(node)

    # recursively walk each node and _coerce_into_fraction

    if node.token_type & TT_Exponent:
        assert isinstance(node, Operation)

        if is_node_negative(node.right):
            # node is negative
            exponent = _calculate(Operation.create("*", node.right, Integer.create(-1)))

            return Operation.create(
                "/", ONE_INTEGER, Operation.create("^", node.left, exponent)
            )

        return Operation.create("/", node, ONE_INTEGER)

    if node.token_type & TT_Div:
        dividends, divisors = _collect_dividends_and_divisors(node)
    else:
        # collect dividends and divisors for Addition and multiplication
        i = 0
        dividends: List[TokenValue] = []
        divisors: List[TokenValue] = []

        while i < len(node.values):
            # hope fully if it ends up working correctly _coerce_into_fraction will flatten wher possible
            value = _coerce_into_fraction(node.values[i])

            # associative lists
            dividends.append(value.left)
            divisors.append(value.right)
            # what do you do with negative exponentss?

            i += 1

    if node.token_type & (TT_Mult | TT_Div):
        # multiply dividends and divisor and return early

        dividends = list(dividends)
        divisors = list(divisors)

        for i, value in enumerate(dividends):
            value = _coerce_into_fraction(value)
            dividends[i] = value.left

            if value.right.token_type & TT_Numeric == 0 or not compare_numeric(
                value.right, ONE_INTEGER
            ):
                divisors.append(value.right)

        for i, value in enumerate(divisors):
            value = _coerce_into_fraction(value)
            divisors[i] = value.left

            if value.right.token_type & TT_Numeric == 0 or not compare_numeric(
                value.right, ONE_INTEGER
            ):
                dividends.append(value.right)

        return Operation.create(
            "/",
            _construct_token_value_with_values(
                Operation.create("*"), *filter(fileter_ones, dividends)
            ),
            _construct_token_value_with_values(
                Operation.create("*"), *filter(fileter_ones, divisors)
            ),
        )

    # if the node is an addition operation first unify the divisor

    assert node.token_type & TT_Add

    for i in range(len(dividends)):
        dividends[i] = _construct_token_value_with_values(
            Operation.create("*"),
            *filter(
                fileter_ones,
                map(
                    lambda v: v[1],
                    filter(lambda j: j[0] != i, enumerate(divisors)),
                ),
            ),
            dividends[i],
        )

    return Operation.create(
        "/",
        _construct_token_value_with_values(Operation.create("+"), *dividends),
        _construct_token_value_with_values(
            Operation.create("*"), *filter(fileter_ones, divisors)
        ),
    )


def _gather_terms(node: TokenValue):
    if is_node_atomic(node) or not isinstance(node, Operation):
        return (node,)

    if node.token_type & TT_Add:
        return node.values

    # for all intents and purposes the data should have been expanded
    # and there should not be any fractions withing the data

    return (node,)


def _exponentiate_node(node: TokenValue) -> TokenValue:
    terms = list(_gather_terms(node))

    for term_idx, term in enumerate(terms):
        factors, _ = _collect_dividends_and_divisors(term)

        i = 0
        while i < len(factors):
            factor = factors[i]
            if isinstance(factor, Variable):
                # in future could support floats and other complex objects
                base = factor
                exponent = Integer.create(1)
            elif isinstance(factor, Operation) and factor.token_type & TT_Exponent:
                base = factor.left
                exponent = factor.right
            elif factor.token_type & TT_Numeric:
                i += 1
                continue
            else:
                raise NotImplementedError("I do not know, anymore", str(factor))

            exponents = [exponent]

            for j, other_factor in enumerate(factors[(i + 1) :]):
                if isinstance(other_factor, Variable):
                    # in future could support floats and other complex objects
                    other_base = other_factor
                    other_exponent = Integer.create(1)
                elif (
                    isinstance(other_factor, Operation)
                    and other_factor.token_type & TT_Exponent
                ):
                    other_base = other_factor.left
                    other_exponent = other_factor.right
                elif other_factor.token_type & TT_Numeric:
                    i += 1
                    continue
                else:
                    raise NotImplementedError(
                        "I do not know, anymore", str(other_factor)
                    )

                if not compare_values(base, other_base):
                    continue

                exponents.append(other_exponent)
                factors.pop(i + j)

            exponent = _calculate(_construct_token_value_with_values("+", *exponents))

            if compare_numeric(exponent, Integer.create(1)):
                factors[i] = base
            else:
                factors[i] = Operation.create("^", base, exponent)

            i += 1

        # set term
        terms[term_idx] = _construct_token_value_with_values("*", *factors)

    return _construct_token_value_with_values("+", *terms)


def _gather_inventory(node: TokenValue):
    terms = _gather_terms(_calculate(_exponentiate_node(node)))

    integer_sum = 0
    node_factors: Dict[str, TokenValue] = {}

    for i, value in enumerate(terms):
        if value.token_type & TT_Int:
            integer_sum += value.token_value
            continue
        elif is_node_atomic(value):
            dividends = [value, Integer.create(1)]
        elif isinstance(value, Operation):
            dividends, divisors = _collect_dividends_and_divisors(value)
            if len(divisors) > 0:
                raise ValueError(str(node), "Division is not expexted")

            assert len(dividends) > 0
        else:
            raise ValueError("functions, no thank you")

        # calculate should have ensured that the numbers are the same
        for j, factor in enumerate(dividends):
            # assume the only thing of interest are variables
            if factor.token_type & TT_Int:
                continue
            elif (
                not isinstance(factor, Variable) and not factor.token_type & TT_Exponent
            ):
                raise NotImplementedError(
                    "Exponents are a thing but not yet supported", str(value)
                )

            if str(factor) in node_factors:
                continue

            # now we can assume this a variable

            this_factor = _construct_token_value_with_values(
                Operation.create("*"), *everything_but_my_index(dividends, j)
            )

            this_factors_terms = [this_factor]

            for term in terms[i + 1 :]:
                term_factors, _ = _collect_dividends_and_divisors(term)

                for k, term_factor in enumerate(term_factors):
                    if compare_variables(factor, term_factor):
                        other_factor = _construct_token_value_with_values(
                            Operation.create("*"),
                            *everything_but_my_index(term_factors, k),
                        )

                        this_factors_terms.append(other_factor)

            node_factors[str(factor)] = _calculate(
                _construct_token_value_with_values(
                    Operation.create("+"), *this_factors_terms
                )
            )

    return integer_sum, node_factors


def compare_values(a: TokenValue, b: TokenValue):
    """Compare two token values"""

    if compare_variables(a, b) or compare_numeric(a, b):
        return True

    if is_node_atomic(a) and is_node_atomic(b):
        return False

    if (a.token_type | b.token_type) & TT_Equ:
        raise NotImplementedError("comparing equations is not implemented")

    # format values in a unified way
    # make both values a unified fraction
    afrac = _coerce_into_fraction(a)
    bfrac = _coerce_into_fraction(b)

    # multiply by eachothers divisors and ignore divisor
    # assume the values have been expanded
    # now the aim is to calculate i make it simpler for the later steps

    # calculate might be able to doit
    # recursive call to compare_values ? problem

    a = _calculate(_expand(Operation.create("*", afrac.left, bfrac.right)))
    b = _calculate(_expand(Operation.create("*", bfrac.left, afrac.right)))

    a_sum, a_inventory = _gather_inventory(a)
    b_sum, b_inventory = _gather_inventory(b)

    if a_sum != b_sum:
        return False

    if len(a_inventory) != len(b_inventory):
        return False

    for key in a_inventory:
        if key not in b_inventory:
            return False
        a_factors = a_inventory[key]
        b_factors = b_inventory[key]

        # print ("a", a_factors)
        # print ("b", b_factors)

        if not compare_values(a_factors, b_factors):
            return False

    return True


def pb(s: str):
    return build_tree2(parse(s))


if __name__ == "__main__":
    for a, expected in (
        ("2 * a * c - 4 * a * b", "(2 * c + -4 * b) * a"),
        ("2 * a * b - 4 * a * b * c", "(2 + -4 * c) * a * b"),
        ("2 * a * b - 4 * a * b", "-2 * a * b"),
        ("2 * a * b - 4 * a * b * c", "(2 + -4 * c) * a * b"),
        ("2 * a * 1 / (2 + 2) - 4 * a", "(-4 + 2 * (1 / 4)) * a"),
        ("2 * a * 1 / (b) - 4 * a", "(-4 + 2 * (1 / b)) * a"),
        # ("-4 + 2 * (1 / b)", ""), # How should that work? (-4 + 2) * (1 / b)
    ):
        ftns = factor_terms_n_stuff(pb(a))
        if str(ftns) != expected:
            print(ftns, "Expected", expected)

    for a, b in (
        ("-1", True),
        ("(-b)", True),
        ("1 / 2 - 1", True),
        ("2 * a * b - 4 * a * b", True),
        ("2 * a * b - 4 * a * b * c", False),
    ):
        res = is_node_negative(pb(a))
        if b != res:
            print(a, res)

    for a, b in (
        ("1", "1 / 1"),
        ("1 / 2 + 2", "(1 + 2 * 2) / 2"),
        ("(1 / 2) / 2", "1 / (2 * 2)"),
        ("1 * 2 / 4", "2 / 4"),
        ("a ^ 2", "a ^ 2 / 1"),
        ("a ^ -2", "1 / a ^ 2"),
        ("(1 / 2 + 2) / 2", "(1 + 2 * 2) / (2 * 2)"),
        ("a + 2 * a / 2", "(2 * a + 2 * a) / 2"),
        ("2 * (1 / 2 + 3)", "(2 + 2 * 3 * 2) / 2"),
        ("2 * (2+ 1) * (3 + 3)", "(3 * 2 * 2 + 3 * 2 * 2 + 3 * 2 + 3 * 2) / 1"),
        ("a ^ -1", "1 / a ^ 1"),
        ("a ^ (-b)", "1 / a ^ b"),
    ):
        fraction = _coerce_into_fraction(pb(a))
        if str(fraction) != b:
            print(a, "|", fraction)

    # for a in ("a * (b + a) * 2", "(b + a) * (b + c)", "(a + b) ^ c"):
    #     print(_expand(pb(a)))

    for a, b, expected in (
        ("1", "1", True),
        ("b", "b", True),
        ("1", "2", False),
        ("a", "b", False),
        ("a + a", "2 * a", True),
        ("5 * (-8 / 5 + 2)", "8", False),
        ("a + 4 * a / 2", "3 * a", True),
        ("2 + 2 ", "5 - 1", True),
        ("1 / 2", "2 / 4", True),
        ("5 * (8 / 5 - 2 / 7)", "8  - 5 * 2 / 7", True),
        ("5 * (8 / 5 + -1 * 2 + 2)", "(8 * (8 / 5 + -1 * 2)) / (8 / 5 + -1 * 2)", True),
        (
            "(7 * (b / (7 / 2 + 2))) / 2 + b / (7 / 2 + 2)",
            "b + -1 * (b / (7 / 2 + 2))",
            True,
        ),
        ("(a + 2) ^ 2", "a * a + 4", True),
        ("1 / a", "1 / (2 * a) * 2", True),
        ("a * b + 100", "(50 + (a * b) / 2) * 2", True),
        ("a * b + 100", "(50 + (a * b) / 2) * 2", True),
        ("a * b ^ (-1)", "a / b", True),
    ):
        res = compare_values(pb(a), pb(b))
        if res != expected:
            print(f"{a} =? {b} |", res)
