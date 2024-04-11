from typing import Tuple, Any, Iterable
from copy import deepcopy
from expression_parser import (
    RESERVED_IDENTITIES,
    TT_OOO_MASK,
    TT_RESERVED_1,
    TT_Equ,
    TT_Exponent,
    TT_Operation_Commutative,
    parsed_to_string,
    TT_Operation,
    TT_Mult,
    TT_Div,
    TT_Numeric,
    TT_Int,
    TT_Ident,
    TT_Float,
    TT_Add,
    TT_Sub,
    TT_Tokens,
    TT_INFO_MASK,
    TT_Func,
    TT_Comma,
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


class Function(Atom):
    value: str
    parameters = Iterable[Atom]

    def __init__(self, token: Tuple[int, str], parameters: Iterable[Atom]) -> None:
        super().__init__()
        self.token_type = token[0]
        self.value = token[1]
        self.parameters = parameters

    def __str__(self) -> str:
        return parsed_to_string(flatten_tree(self))


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
        return parsed_to_string(flatten_tree(self))


class Equals(Operation):
    pass


def build_tree(tokens: Iterable[Tuple[int, Any]]) -> Atom:
    # make a copy of the tokens
    tokens = list(deepcopy(tokens))

    def _create_atom(token: Tuple[int, Any]) -> Atom:
        m = {
            (TT_Numeric | TT_Int): Int,
            (TT_Numeric | TT_Float): Float,
            TT_Ident: Variable,
        }

        # zero all info mask bits
        token_type = token[0] ^ (token[0] & TT_INFO_MASK)

        if not token_type in m:
            if token_type == TT_Atom:
                return token[1]
            elif token_type == TT_Tokens:
                return build_tree(token[1])
            # this is for now because i could not be bothered
            raise ValueError

        return m[token_type](token)

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

    if len(tokens) == 1:
        # The below thing fails if there is a problem
        return _create_atom(tokens[0])

    i = 0
    while i < len(tokens) and len(tokens) > 0:
        token = tokens[i]

        if token[0] & TT_Func:
            # read the next parameters
            if i + 1 < len(tokens):
                parameters: Iterable[Atom]
                next_token = tokens[i + 1]
                if next_token[0] & TT_Tokens:
                    # (2, 1, 4, 5)
                    sub_tokens = next_token[1]
                    parameters = []

                    # sub tokens split for comma
                    j = 0
                    k = 0
                    while j < len(sub_tokens):
                        if sub_tokens[j][0] & TT_Comma:
                            if j == 0:
                                raise ValueError

                            parameters.append(build_tree(sub_tokens[k:(j)]))
                            k = j + 1
                        j += 1
                    else:
                        parameters.append(build_tree(sub_tokens[k:]))
                else:
                    parameters = [_create_atom(next_token)]
                    # remove the next token
                print(parameters)
                tokens.pop(i + 1)
                # print(parameters)
                tokens[i] = (TT_Atom, Function(token, parameters))
            else:
                raise ValueError

        i += 1

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


def flatten_tree(node: Atom) -> Iterable[Tuple[int, str]]:
    """Flatten a tree to a list of tokens"""
    # TODO: rewrite to not rely on recursion, it feels "Yucky", I do not understand how this works
    token = (node.token_type, str(node.value))

    # TODO: if int is negative wrap number in tokens

    if isinstance(node, Operation):
        left_tokens = flatten_tree(node.left)
        right_tokens = flatten_tree(node.right)

        if (
            node.left.token_type & TT_Operation
            and (node.token_type & TT_OOO_MASK)
            >= (  # if division wasn't real ">" would suffice # TODO: add commutative flag and use that aswell
                node.left.token_type & TT_OOO_MASK
            )
            and not (
                node.token_type & TT_Operation_Commutative
                and node.token_type & TT_OOO_MASK == node.left.token_type & TT_OOO_MASK
            )
        ):
            left_tokens = [(TT_Tokens, left_tokens)]
        if (
            node.right.token_type & TT_Operation
            and (node.token_type & TT_OOO_MASK) >= (node.right.token_type & TT_OOO_MASK)
            and not (
                node.token_type & TT_Operation_Commutative
                and node.token_type & TT_OOO_MASK == node.right.token_type & TT_OOO_MASK
            )
        ):
            right_tokens = [(TT_Tokens, right_tokens)]

        return [*left_tokens, token, *right_tokens]
    elif isinstance(node, Function):
        parameters = []
        for param in node.parameters:
            # dumb solution but can't be bothered to do actual thinking
            parameters.extend([*flatten_tree(param), (RESERVED_IDENTITIES[","], ",")])
        return [token, (TT_Tokens, parameters[:-1])]

    return [token]


def compare_varible(a: Variable, b: Variable) -> bool:
    return (
        a.value == b.value
    )  # in future the variable might have more information contained, such as metadata


def is_variable_in_tree(root: Atom, node: Atom) -> bool:
    assert type(node) == Variable

    if not isinstance(root, Operation):
        if root.token_type == TT_Ident:
            return compare_varible(root, node)
        return False

    atoms = [root.left, root.right]
    while len(atoms) > 0:
        atom = atoms.pop()

        if isinstance(atom, Operation):
            atoms.extend((atom.left, atom.right))
            continue

        if type(atom) == Variable and compare_varible(atom, node):
            return True

    return False


# TODO: write tests
