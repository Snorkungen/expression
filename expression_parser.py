from typing import Iterable, NewType, Tuple, Any, Final, List, Dict

from utils import b_nand

"""
    first four(4) bits are reserved for encoding information
    0b0000 - is reserved
"""

TT_INFO_MASK: Final = 0b1111
"""1-based counting"""

TT_Numeric: Final = 1 << 4  # Integers and floats
"""Numeric Numbers are special"""
TT_Ident: Final = 1 << 5  #  Identity Everything that is not a number
TT_Tokens: Final = 1 << 6  # A type containing the contents of a bracket

TT_Int: Final = 1 << 7
TT_Float: Final = 1 << 8


"""
    Encode information such as a Numeric token being signed
    1. Positive, Explicit, otherwise assumed
    2. Signed Negative
"""
TT_Numeric_Positive: Final = 1
TT_Numeric_Negative: Final = 2

TT_Operation: Final = 1 << 9
TT_Operation_Commutative: Final = 1 << 3  # 8 0b1000

"""
    Encode order of operations into the Operation type
    4. Parentheses
    3. Exponentiation
    2. Multiplication & Division
    1. Addition & subtraction
"""
TT_OOO_MASK: Final = 0b111
TT_OOO_ADD: Final = 1  # 0b001
TT_OOO_Mult: Final = 2  # 0b010
TT_OOO_Expo: Final = 3  # 0b011
TT_OOO_Paren: Final = 4  # 0b100 # this is superoflous, due to the TT_Tokens type

TT_Add: Final = 1 << 10
TT_Sub: Final = 1 << 11
TT_Mult: Final = 1 << 12
TT_Div: Final = 1 << 13
TT_Equ: Final = 1 << 14
TT_Exponent: Final = 1 << 15
TT_Func: Final = 1 << 16

TT_Comma: Final = 1 << 17  # this value is special

TT_RESERVED_0: Final = 0
"""this type is reserved, and will not be used by the parser, with the caveat that if there is an error then the token type might be 0"""
TT_RESERVED_1: Final = 1 << 18
"""this type is reserved, and will not be used by the parser"""
TT_RESERVED_2: Final = 1 << 19
"""this type is reserved, and will not be used by the parser"""
TT_RESERVED_3: Final = 1 << 20
"""this type is reserved, and will not be used by the parser"""

RESERVED_IDENTITIES = {
    "+": TT_Add | TT_Operation | TT_Operation_Commutative | TT_Ident | TT_OOO_ADD,
    "-": TT_Sub | TT_Operation | TT_Ident | TT_OOO_ADD,
    "*": TT_Mult | TT_Operation | TT_Operation_Commutative | TT_Ident | TT_OOO_Mult,
    "/": TT_Div | TT_Operation | TT_Ident | TT_OOO_Mult,
    "=": TT_Equ | TT_Operation | TT_Ident,
    "^": TT_Exponent | TT_Operation | TT_Ident | TT_OOO_Expo,
    "__testfunc": TT_Func | TT_Ident,
    ",": TT_Comma | TT_Operation | TT_Ident,
}

Token = NewType("Token", Tuple[int, Any])


def ismathematical_alphanumeric_symbol(char: str) -> bool:
    return ord(char[0]) >= 0x1D400 and ord(char[0]) <= 0x1D7FF


def issubscript(char: str) -> bool:
    n = ord(char[0])

    if n >= 0x2080 and n <= 0x209C and n != 0x208F:
        return True

    if n == 0x2C7C:
        return True

    if n >= 0x1D62 and n <= 0x1D6A:
        return True

    return False


def is_opening_bracket(char: str) -> bool:
    return char in "([{"


def find_closing_bracket_location(input_text: str, opening_bracket: str, i: int) -> int:
    bracket_pairs = ("()", "{}", [])
    # determind the bracket that is relevant

    for bracket_pair in bracket_pairs:
        if bracket_pair[0] == opening_bracket:
            brackets = bracket_pair
        break
    else:
        # should raise an error claiming that the given bracket is not a valid opening bracket
        pass

    brackets = bracket_pairs[0]
    depth = 0
    while i < len(input_text):
        char = input_text[i]
        i += 1
        if char == brackets[0]:
            depth += 1
        elif char == brackets[1]:
            depth -= 1
            if depth < 0:
                return i  # return early found the end

    return i + 1


def is_candidate_for_implicit_multiplication(token: Token):
    """Token is either tokens, numeric or a variable"""
    return b_nand(token[0], TT_INFO_MASK) == TT_Ident or token[  # exclusive TT_Ident
        0
    ] & (
        TT_Numeric | TT_Tokens
    )  # either numeric or a tokens token


def modify_tokens_to_make_more_sense(tokens: List[Token], tokens_positions: List[int]):
    """modify the tokens"""
    i = 0
    token: Tuple[int, Any]
    while i < len(tokens):
        if (i + 1) == len(tokens):
            break
        # check that next token has an operation
        token = tokens[i]
        next_token = tokens[i + 1]

        if token[0] & TT_Mult and next_token[0] & TT_Mult:
            # cast "**" to exponentiation
            tokens.pop(i)
            tokens_positions.pop(i)

            tokens[i] = (RESERVED_IDENTITIES["^"], "^")

        if is_candidate_for_implicit_multiplication(
            token
        ) and is_candidate_for_implicit_multiplication(next_token):
            i += 1
            tokens.insert(i, (RESERVED_IDENTITIES["*"], "*"))
            tokens_positions.insert(i, -1)
            continue

        if token[0] & TT_Operation and i == 0:
            if token[0] & (TT_Add | TT_Sub) == 0:
                # TODO: Emit a warning
                i += 1
                continue

            # NOTE i = 0
            tokens.pop(i)
            tokens_positions.pop(i)

            if token[0] & TT_Add:
                continue

            # NOTE token is now a subtraction token

            if next_token[0] & TT_Numeric:
                tokens[i] = (
                    (b_nand(next_token[0], TT_INFO_MASK)) | TT_Numeric_Negative,
                    token[1] + next_token[1],
                )
            elif next_token[0] & TT_Ident:
                tokens.insert(0, ((TT_Numeric | TT_Numeric_Negative | TT_Int), "-1"))
                tokens.insert(1, (RESERVED_IDENTITIES["*"], "*"))
                tokens_positions.insert(1, -1)
            else:
                # TODO: emit a
                pass

        if token[0] & TT_Operation and next_token[0] & TT_Operation:
            if next_token[0] & (TT_Add | TT_Sub) == 0:
                # TODO: emit an error or warning
                i += 1
                continue

            # see if there is a token after the next token
            if len(tokens) < i + 2:
                # TODO: this is most definitively an error
                break

            # handle *, -, a
            if tokens[i + 2][0] & TT_Numeric and next_token[0] & (TT_Sub | TT_Add):
                tokens[i + 2] = (
                    (
                        (b_nand(tokens[i + 2][0], TT_INFO_MASK)) | TT_Numeric_Negative
                        if next_token[0] & TT_Sub
                        else TT_Numeric_Positive
                    ),
                    next_token[1] + tokens[i + 2][1],
                )

                tokens.pop(i + 1)
                tokens_positions.pop(i + 1)
                continue
            elif b_nand(tokens[i + 2][0], TT_INFO_MASK) == TT_Ident and (
                next_token[0] & (TT_Sub | TT_Add)
            ):
                tokens.pop(i + 1)
                tokens_positions.pop(i + 1)

                if next_token[0] & TT_Add:
                    continue

                tokens[i + 1] = (
                    TT_Tokens,
                    [
                        ((TT_Numeric | TT_Numeric_Negative | TT_Int), "-1"),
                        (RESERVED_IDENTITIES["*"], "*"),
                        tokens[i + 1],
                    ],
                )
                continue

        if token[0] & TT_Operation and next_token[0] & TT_Operation:
            if token[0] & TT_Add and next_token[0] & TT_Add:
                # TODO: Emit a warning
                tokens.pop(i)
                tokens_positions.pop(i)
            elif token[0] & TT_Add and next_token[0] & TT_Sub:
                tokens.pop(i)
                tokens_positions.pop(i)

                tokens[i] = next_token
            elif token[0] & TT_Sub and next_token[0] & TT_Add:
                tokens.pop(i)
                tokens_positions.pop(i)

                tokens[i] = token
            elif token[0] & TT_Sub and next_token[0] & TT_Sub:
                tokens.pop(i)
                tokens_positions.pop(i)

                tokens[i] = RESERVED_IDENTITIES["+"]

        i += 1


def parse(
    input_text: str, additional_identities: Dict[str, int] = None
) -> Iterable[Tuple[int, Any]]:
    tokens: List[Tuple[int, Any]] = []
    tokens_positions = []
    token_type = 0
    buffer = ""
    i = 0

    if not additional_identities:
        identities = RESERVED_IDENTITIES
    else:
        identities = {**RESERVED_IDENTITIES, **additional_identities}

    while i < len(input_text):
        char = input_text[i]
        i += 1

        if (
            not char.isascii()
            and not issubscript(char)
            and not ismathematical_alphanumeric_symbol(char)
        ):
            raise ValueError(f"{char} : unsupported character")

        if char.isspace():
            if token_type == 0:
                continue
            tokens.append((token_type, buffer))
            tokens_positions.append(i)
            buffer = ""
            token_type = 0
            continue

        if token_type == 0:
            # there is nothing to initialize stuff
            if char.isnumeric() and not issubscript(char):
                # what we're dealing with is something numeric
                token_type = TT_Numeric | TT_Int
            else:
                token_type = TT_Ident

        if is_opening_bracket(char):
            if buffer:
                tokens.append((token_type, buffer))
                tokens_positions.append(i)
            # reset buffer and token type
            buffer = ""
            token_type = 0
            start = i
            end_pos = find_closing_bracket_location(input_text, char, start)
            tokens.append(
                (
                    TT_Tokens,
                    parse(input_text[start : end_pos - 1], additional_identities),
                )
            )
            tokens_positions.append(end_pos)
            i = end_pos
        if token_type & TT_Numeric:
            if char.isnumeric() and not issubscript(char):
                buffer += char
            elif char == ".":
                buffer += "."
                token_type = (
                    token_type | TT_Float
                ) ^ TT_Int  # set type to a numeric float only
            else:
                # add the current buffer and pressume it is an ident
                tokens.append((token_type, buffer))
                tokens_positions.append(i)
                buffer = ""
                token_type = TT_Ident

        if token_type & TT_Ident:
            # first check that the char is not reserved and special
            if char in identities or (char.isnumeric() and not issubscript(char)):
                if buffer:  # prevent adding an empty buffer to tokens
                    tokens.append((token_type, buffer))
                    tokens_positions.append(i)
                buffer = char
            else:
                buffer += char

            if char.isnumeric() and not issubscript(char):
                token_type = TT_Int

        # final thing in loop check buffer for a reserved identity
        if buffer in identities:
            # select the longest match
            identity_options = filter(lambda name: name.startswith(buffer), identities)
            selected_identity = buffer
            for name in identity_options:
                diff = len(name) - len(buffer)
                if diff == 0:
                    pass
                elif (
                    len(selected_identity) < len(name)
                    and len(input_text) > i + diff - 1
                    and input_text[i : i + diff] == name[-diff:]
                ):
                    selected_identity = name
            else:
                diff = len(selected_identity) - len(buffer)
                i += diff

            tokens.append((identities[selected_identity], selected_identity))
            tokens_positions.append(i)

            buffer = ""
            token_type = 0

    # final stuff
    if token_type != 0:
        tokens.append((token_type, buffer))
        tokens_positions.append(i)

    # Do some checking that the given tokens make sense
    # The following should not be a part of the tokeniser
    modify_tokens_to_make_more_sense(tokens, tokens_positions)

    # TODO: verify that the use of "," is correct

    return tokens


def parsed_to_string(parsed: Iterable[Tuple[int, Any]], space: str = " ") -> str:
    def token_to_string(token: Tuple[int, str]):
        return (
            f"({parsed_to_string(token[1])})"
            if token[0] == TT_Tokens
            else str(token[1])
        )

    return space.join(map(token_to_string, parsed))


assert parsed_to_string(parse("12.01 + 1"), space="") == "12.01+1"
assert len(parse("-1")) == 1
assert parsed_to_string(parse("-a"), space="") == "-1*a"
assert parsed_to_string(parse("a₀ + 1 - 𝜃 + 𝛑")) == "a₀ + 1 - 𝜃 + 𝛑"
