from typing import Callable, List
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
    Operations "Addition"(+), "Multiplication"(*) , "Division"(/), "exponentiation"(^)
    """

    values: Tuple[TokenValue]
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
    def create(symbol: str, *values: TokenValue):
        if symbol not in RESERVED_IDENTITIES:
            raise ValueError(f"{symbol} not int RESERVED_IDENTITIES")
        token_type = RESERVED_IDENTITIES[symbol]
        return Operation((token_type, symbol), *values)


class Function(TokenValue):
    values: Tuple[TokenValue]
    token_value: str

    def __init__(self, token: Token, *values: TokenValue) -> None:
        super().__init__(token)
        self.values = values

    def __str__(self) -> str:
        parameters = ", ".join(map(str, self.values))
        return f"{self.token_value}({parameters})"

    @property
    def name(self) -> str:
        return self.token_value

    @staticmethod
    def create(name: str, *values: TokenValue):
        return Function((TT_Ident | TT_Func, name), *values)


def collect_factors_ordered(node: Operation, inplace=True):
    """collect all direct sub_nodes that are also multiplication

    Operation[*]
    ├──Operation[*]
    │  ├──Variable[a]
    │  └──Variable[b]
    └──Variable[c]

    => a, b, c
    """
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


def collect_terms_ordered(node: Operation, inplace=True):
    """collect all direct sub_nodes that are also addition

    Operation[+]
    ├──Operation[+]
    │  ├──Variable[a]
    │  └──Variable[b]
    └──Variable[c]

    => a, b, c
    """
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
    """take a multiplication node and return a new node with flattened factors"""
    assert node.token_type & TT_Mult
    assert isinstance(node, Operation)

    values = collect_factors_ordered(node, inplace=inplace)

    if inplace:
        return node

    return Operation((node.token_type, node.token_value), *values)


def flatten_terms(node: Operation, inplace=True):
    """take a addition node and return a new node with flattened terms"""
    assert node.token_type & TT_Add
    assert isinstance(node, Operation)

    values = collect_terms_ordered(node, inplace=inplace)

    if inplace:
        return node

    return Operation((node.token_type, node.token_value), *values)


def token_to_token_value(token: Token) -> TokenValue:
    # zero all info mask bits
    token_type = token[0] ^ (token[0] & TT_INFO_MASK)

    if b_nand(token_type, TT_INFO_MASK) == (TT_Numeric | TT_Int):
        return Integer(token)
    if b_nand(token_type, TT_INFO_MASK) == (TT_Numeric | TT_Float):
        return Float(token)
    if b_nand(token_type, TT_INFO_MASK) == TT_Ident:
        return Variable(token)
    if b_nand(token_type, TT_INFO_MASK) == TT_Tokens:
        # TODO: add some kind of note to denote that this thing was surrounded by brackets
        return build_tree2(token[1])
    if b_nand(token_type, TT_INFO_MASK) == TB_TOKEN_VALUE:
        return token[1]

    if b_nand(token_type, TT_INFO_MASK) == (TT_Func | TT_Ident):
        return Function(token)

    print(token)
    raise ValueError("token not recognised", str(token))


def build_tree2(tokens: Iterable[Token]) -> TokenValue:
    tokens = list(tokens)

    def _do_operation(
        test: int, OperValue: Callable[[Token, Tuple[TokenValue, ...]], TokenValue]
    ):
        """Go through tokens and replace tokens with `TokenValue`s"""
        i = 0
        while i < len(tokens) and len(tokens) > 0:
            token = tokens[i]

            if token[0] & (test) == 0:
                i += 1
                continue

            next_token = tokens[i + 1]

            if token[0] & TT_Func:
                # handle a function consume the tokens
                if (i + 1) >= len(tokens):
                    tokens[i] = (
                        TB_TOKEN_VALUE,
                        OperValue(token),
                    )  # A function is not guaranteed to have values
                    continue

                if next_token[0] & TT_Func and i + 2 < len(tokens):
                    # find the first good none function value
                    j = i + 2
                    while j < len(tokens) and tokens[j][0] & TT_Func:
                        j += 1

                    if len(tokens) == j:
                        j -= 1
                        tokens[j] = (TB_TOKEN_VALUE, OperValue(tokens[j]))

                    # recursively walk slice backwards
                    while j > i:
                        j -= 1

                        # should probably refactor so that this would not have to call build_tree2
                        tokens[j] = (
                            TB_TOKEN_VALUE,
                            build_tree2((tokens[j], tokens[j + 1])),
                        )
                        tokens.pop(j + 1)

                    continue  # IMPORTANT

                values: List[TokenValue] = []

                if next_token[0] & TT_Tokens:
                    begin = 0
                    j = 0
                    while j < len(next_token[1]):
                        t = next_token[1][j]
                        if t[0] & TT_Comma:
                            values.append(build_tree2(next_token[1][begin:j]))
                            begin = j + 1
                        j += 1
                    else:
                        if begin < len(next_token[1]):
                            values.append(build_tree2(next_token[1][begin:]))
                else:
                    values.append(token_to_token_value(next_token))

                tokens.pop(i + 1)  # remove next_token
                tokens[i] = (TB_TOKEN_VALUE, OperValue(token, *values))
                i += 1
            elif token[0] & TT_Operation:
                # handle an operation
                if i == 0 or i + 1 >= len(tokens):
                    raise ValueError("Cannot get left value or right value")

                left = token_to_token_value(tokens[i - 1])
                right = token_to_token_value(tokens[i + 1])

                operation = OperValue(token, left, right)

                tokens[i] = (TB_TOKEN_VALUE, operation)
                tokens.pop(i + 1)  # pop right value
                tokens.pop(i - 1)  # pop left value

    if len(tokens) == 1:
        return token_to_token_value(tokens[0])

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
        raise ValueError(tokens)

    root = tokens[0][1]

    if tokens[0][0] != TB_TOKEN_VALUE:
        """The last token is not an atom something failed"""
        return ValueError

    # attempt to flatten addition and multiplication operations
    # for ease of implementation the previous steps constructed a **binary** tree
    # and to rectify this later walk the tree and flatten `TokenValue`s
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
            elif isinstance(node, Operation) or isinstance(node, Function):
                nodes.extend(node.values)

    return root
