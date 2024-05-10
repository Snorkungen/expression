from typing import Callable
from expression_parser import *

TB_TOKEN_VALUE = TT_RESERVED_1


class TokenValue:
    token_type: int
    token_value: Any

    def __init__(self, token: Token) -> None:
        self.token_type = token[0]
        self.token_value = token[1]

    def __str__(self) -> str:
        return str(self.token_value)

    @property
    def token(self) -> Token:
        return (self.token_type, self.token_value)


class Integer(TokenValue):
    token_value: int

    def __init__(self, token: Token) -> None:
        try:
            value = int(token[1])
        except ValueError as e:
            print(
                Integer.__name__ + "Failed: to initialize value is not a valid integer",
                token,
            )
            raise e

        super().__init__((token[0], value))

    @staticmethod
    def create(value: int):
        return Integer(
            (
                TT_Numeric
                | TT_Int
                | (TT_Numeric_Negative if value < 0 else TT_Numeric_Positive),
                value,
            )
        )


class Float(TokenValue):
    token_value: float

    def __init__(self, token: Token) -> None:
        try:
            value = float(token[1])
        except ValueError as e:
            print(
                Integer.__name__ + "Failed: to initialize value is not a valid float",
                token,
            )
            raise e

        super().__init__((token[0], value))

    @staticmethod
    def create(value: int):
        return Float((TT_Numeric | TT_Float, value))


class Variable(TokenValue):
    token_value: str

    def __init__(self, token: Token) -> None:
        super().__init__((token[0], str(token[1])))

    @staticmethod
    def create(name: str):
        return Variable((TT_Ident, name))


class Operation(TokenValue):
    """
    Operations "Addition"(+), "Multiplication"(*) , "Division"(/), "exponentiation"
    """

    values: Iterable[TokenValue]
    token_value: str

    def __init__(self, token: Token, *values: TokenValue, self_flatten=True) -> None:
        super().__init__((token[0], str(token[1])))
        self.values = values

        if self_flatten:
            if self.token_type & TT_Add:
                flatten_terms(self, inplace=True)
            elif self.token_type & TT_Mult:
                flatten_factors(self, inplace=True)

    @property
    def left(self) -> TokenValue:
        return self.values[0]

    @property
    def right(self) -> TokenValue:
        return self.values[1]

    def __str__(self) -> str:
        # TODO: this should be a more extensive function with more thinking

        return f" {self.token_value} ".join(
            map(
                lambda node: (
                    # welcome nested terneraries
                    str(node)
                    if node.token_type & TT_Operation == 0
                    else (
                        str(node)
                        if (node.token_type & TT_OOO_MASK)
                        > (self.token_type & TT_OOO_MASK)
                        else f"({node})"
                    )
                ),
                self.values,
            )
        )

    @staticmethod
    def create(
        symbol: str, *values: TokenValue, RESERVED_IDENTITIES=RESERVED_IDENTITIES
    ):
        if symbol not in RESERVED_IDENTITIES:
            raise ValueError(f"{symbol} not int RESERVED_IDENTITIES")
        token_type = RESERVED_IDENTITIES[symbol]
        return Operation((token_type, symbol), *values)


class Function(TokenValue):
    values: list[TokenValue]
    token_value: str

    def __init__(self, token: Token) -> None:
        raise NotImplementedError
        super().__init__(token)


def collect_factors(node: Operation, factors: list[TokenValue] = []):
    assert node.token_type & TT_Mult
    assert isinstance(node, Operation)

    if factors == None:
        factors = []

    nodes = list(node.values)
    while len(nodes) > 0:
        sub_node = nodes.pop()

        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Mult:
            nodes.extend(sub_node.values)
            continue
        elif isinstance(sub_node, Operation) and sub_node.token_type & TT_Div:
            # TODO: collect division factors
            # this would require the modification of values
            # which is not what wanted
            pass

        factors.append(sub_node)

    return factors


def collect_factors_ordered(node: Operation, inplace=True):
    assert node.token_type & TT_Mult
    assert isinstance(node, Operation)

    # ensure that values is a list that can be modified
    values = list(node.values)

    i = 0
    while i < len(values):
        sub_node = values[i]

        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Mult:
            factors = collect_factors_ordered(
                sub_node, inplace=inplace
            )  # does not matter because the operations will get GCed soon after

            values[i] = factors[0]

            for j in range(1, len(factors)):
                values.insert(i + j, factors[j])
            else:
                i += len(factors) - 1

        elif isinstance(sub_node, Operation) and sub_node.token_type & TT_Div:
            # TODO: collect division factors
            # this would require the modification of values
            # which is not what wanted
            pass

        i += 1

    if inplace:
        node.values = values
    return values


def collect_terms(node: Operation, terms: list[TokenValue] = []):
    assert node.token_type & TT_Add
    assert isinstance(node, Operation)

    if terms == None:
        terms = []

    # operation: keep the order

    nodes = list(node.values)
    while len(nodes) > 0:
        # is there a better more efficient way of removing the first value
        sub_node = nodes.pop()
        if isinstance(sub_node, Operation) and sub_node.token_type & TT_Add:
            nodes.extend(sub_node.values)
            continue

        terms.append(sub_node)

    return terms


def collect_terms_ordered(node: Operation, inplace=True):
    """Collects terms onto node"""
    assert node.token_type & TT_Add
    assert isinstance(node, Operation)

    # ensure that values is a list that can be modified
    values = list(node.values)

    i = 0
    while i < len(values):
        sub_node = values[i]

        if sub_node.token_type & TT_Add and isinstance(sub_node, Operation):
            terms = collect_terms_ordered(
                sub_node, inplace=inplace
            )  # does not matter because the operations will get GCed soon after

            values[i] = terms[0]

            for j in range(1, len(terms)):
                values.insert(i + j, terms[j])
            else:
                i += len(terms) - 1

        i += 1
    if inplace:
        node.values = values
    return values


def flatten_factors(node: Operation, inplace=True):
    assert node.token_type & TT_Mult
    assert isinstance(node, Operation)

    values = collect_factors_ordered(node, inplace=inplace)

    if inplace:
        return node

    return Operation((node.token_type, node.token_value), *values)


def flatten_terms(node: Operation, inplace=True):
    assert node.token_type & TT_Add
    assert isinstance(node, Operation)

    values = collect_terms_ordered(node, inplace=inplace)

    if inplace:
        return node

    return Operation((node.token_type, node.token_value), *values)


def build_tree2(
    tokens: Iterable[Token], RESERVED_IDENTITIES=RESERVED_IDENTITIES
) -> TokenValue:
    tokens = list(tokens)

    def _create_value(token: Token) -> TokenValue:
        # zero all info mask bits
        token_type = token[0] ^ (token[0] & TT_INFO_MASK)

        if token_type == (TT_Numeric | TT_Int):
            return Integer(token)
        if token_type == (TT_Numeric | TT_Float):
            return Float(token)
        if token_type == TT_Ident:
            return Variable(token)
        if token_type == TT_Tokens:
            # TODO: add some kind of note to denote that this thing was surrounded by brackets
            return build_tree2(token[1], RESERVED_IDENTITIES=RESERVED_IDENTITIES)
        if token_type == TB_TOKEN_VALUE:
            return token[1]

        print(token)
        raise ValueError("token not recognised " + token)

    def _do_operation(
        test: int, OperValue: Callable[[Token, Tuple[TokenValue, ...]], TokenValue]
    ):
        i = 0
        while i < len(tokens) and len(tokens) > 0:
            token = tokens[i]
            if not (token[0] & TT_Operation):  # TODO: remember functions
                i += 1
                continue

            if token[0] & (test) == 0:
                i += 1
                continue

            if i == 0 or i + 1 >= len(tokens):
                raise ValueError("Cannot get left value or right value")

            left = _create_value(tokens[i - 1])
            right = _create_value(tokens[i + 1])

            operation = OperValue(token, left, right)

            tokens[i] = (TB_TOKEN_VALUE, operation)
            tokens.pop(i + 1)  # pop right value
            tokens.pop(i - 1)  # pop left value
            continue

    if len(tokens) == 1:
        return _create_value(tokens[0])

    _do_operation(TT_Func, Function)
    _do_operation(TT_Exponent, Operation)

    # _do_operation(
    #     TT_Mult | TT_Div,
    #     lambda token, left, right: (
    #         Operation(token, left, right)
    #         if token[0] & TT_Mult
    #         else (
    #             Operation(
    #                 (RESERVED_IDENTITIES["*"], "*"),
    #                 left,
    #                 Operation(
    #                     (RESERVED_IDENTITIES["/"], "/"), Integer.create(1), right
    #                 ),
    #             )
    #         )
    #     ),
    # )
    _do_operation(TT_Mult | TT_Div, Operation)

    _do_operation(
        TT_Add | TT_Sub,
        lambda token, left, right: (
            Operation(token, left, right)
            if token[0] & TT_Add
            else (
                Operation(
                    (RESERVED_IDENTITIES["+"], "+"),
                    left,
                    Operation(
                        (RESERVED_IDENTITIES["*"], "*"), Integer.create(-1), right
                    ),
                )
            )
        ),
    )
    _do_operation(TT_Equ, Operation)

    if len(tokens) != 1:
        """
        something has gone terribly wrong,
        the tree should be left with on node,
        input tokens were probably bad
        """
        raise ValueError

    root = tokens[0][1]

    if tokens[0][0] != TB_TOKEN_VALUE:
        """The last token is not an atom something failed"""
        return ValueError

    # attempt to flatten addition and multiplication operations
    if isinstance(root, Operation) or isinstance(root, Function):
        nodes = [root]

        idx = 0

        while idx < len(nodes):
            node = nodes[idx]
            idx += 1

            if node.token_type & TT_Add and isinstance(node, Operation):
                collect_terms_ordered(node, inplace=True)
            elif node.token_type & TT_Mult:
                collect_factors_ordered(node, inplace=True)
            elif isinstance(node, Operation) or isinstance(root, Function):
                nodes.extend(node.values)

    return root


parsed = parse("1 + 3 * 2 - 6 + 3 / 4")
node = build_tree2(parsed)
assert isinstance(node, Operation)
# parsed = parse("(1 + 2 + 3 + 4) * (5 * (6 + 7) + (8 + 9)) * 10")
node = build_tree2(parsed)
assert isinstance(node, Operation)

# TODO: write tests