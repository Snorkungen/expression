from typing import Tuple, Any
from copy import deepcopy
from expression_parser import (
    TT_RESERVED_1,
    TT_Equ,
    TT_Exponent,
    parse,
    TT_Operation,
    TT_Mult,
    TT_Div,
    TT_Numeric,
    TT_Int,
    TT_Ident,
    TT_Float,
    TT_Add,
    TT_Sub,
)

TT_Atom = TT_RESERVED_1


class Atom:
    value: any
    token_type: int

    def __str__(self) -> str:
        return str(self.value)


class Variable(Atom):
    value: str

    def __init__(self, token: Tuple[int, str]) -> None:
        super().__init__()
        self.token_type = token[0]
        self.value = token[1]


class Int(Atom):
    value: int

    def __init__(self, token: Tuple[int, str]) -> None:
        super().__init__()
        self.token_type = token[0]
        self.value = int(token[1])


class Float(Atom):
    value: float

    def __init__(self, token: Tuple[int, str]) -> None:
        super().__init__()
        self.token_type = token[0]
        self.value = float(token[1])


class Operation(Atom):
    right: Atom
    left: Atom
    token_type: int
    value: str

    def __init__(self, token: Tuple[int, str], left: Atom, right: Atom) -> None:
        super().__init__()
        self.token_type = token[0]
        self.value = token[1]
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} {self.value} {self.right}"


class Equals(Operation):
    pass


def build_tree(tokens: list[Tuple[int, Any]]) -> Operation:
    # make a copy of the tokens
    tokens = list(deepcopy(tokens))

    def _create_atom(token: Tuple[int, Any]) -> Atom:
        m = {
            (TT_Numeric | TT_Int): Int,
            (TT_Numeric | TT_Float): Float,
            TT_Ident: Variable,
        }

        if not token[0] in m:
            if token[0] == TT_Atom:
                return token[1]
            # this is for now becaus i could not be bothered
            raise ValueError

        return m[token[0]](token)

    def _do_operation(test: int, OperAtom: Operation = Operation):
        i = 0

        while i < len(tokens) and len(tokens) > 0:
            token = tokens[i]
            if not (token[0] & TT_Operation):
                i += 1
                continue

            if token[0] & (test):
                if i == 0 or i + 1 == len(tokens):
                    print(tokens)
                    raise "Cannot get left or right tokens"

                left_idx = i - 1
                right_idx = i + 1
                left = _create_atom(tokens[left_idx])
                right = _create_atom(tokens[right_idx])
                operation = OperAtom(token, left, right)

                # replace the current token with an atom
                tokens[i] = (TT_Atom, operation)
                tokens.pop(left_idx)
                tokens.pop(right_idx - 1)  # this is hacky
                # i -= 1
                continue
            i += 1

    # TODO: add support for bracketed information should build a tree and attach the head as node
    # but can't be bothered right

    _do_operation((TT_Exponent))
    _do_operation((TT_Mult | TT_Div))
    _do_operation((TT_Add | TT_Sub))
    _do_operation(TT_Equ, Equals)

    if len(tokens) != 1:
        """
        something has gone terribly wrong,
        the tree should be left with on node,
        input tokens were probably bad
        """
        raise ValueError

    if tokens[0][0] != TT_Atom:
        """The last token is not an atom something failed"""
        return ValueError

    return tokens[0][1]


expression = "10 + a * 20 / 10 = b"

print(build_tree(parse(expression)))
