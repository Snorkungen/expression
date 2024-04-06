from typing import Iterable, Tuple, Any

"""
    first four(4) bits are reserved for encoding information
    0b0000 - is reserved
"""

TT_INFO_MASK = 0b1111
"""1-based counting"""

TT_Numeric = 1 << 4  # Integers and floats
"""Numeric Numbers are speci"""
TT_Ident = 1 << 5  #  Identity Everything that is not a number
TT_Tokens = 1 << 6  # A type containing the contents of a bracket

TT_Int = 1 << 7
TT_Float = 1 << 8

TT_Operation = 1 << 9

"""
    Encode order of operations into the Operation type
    4. Parentheses
    3. Exponentiation
    2. Multiplication & Division
    1. Addition & subtraction
"""
TT_OOO_ADD = 1  # 0b0001
TT_OOO_Mult = 2  # 0b0010
TT_OOO_Expo = 3  # 0b0011
TT_OOO_Paren = 4  # 0b0100 # this is superoflous, due to the TT_Tokens type

TT_Add = 1 << 10
TT_Sub = 1 << 11
TT_Mult = 1 << 12
TT_Div = 1 << 13
TT_Equ = 1 << 14
TT_Exponent = 1 << 15
TT_Func = 1 << 16

TT_Comma = 1 << 17  # this value is special

TT_RESERVED_0 = 0
"""this type is reserved, and will not be used by the parser, with the caveat that if there is an error then the token type might be 0"""
TT_RESERVED_1 = 1 << 18
"""this type is reserved, and will not be used by the parser"""
TT_RESERVED_2 = 1 << 19
"""this type is reserved, and will not be used by the parser"""
TT_RESERVED_3 = 1 << 20
"""this type is reserved, and will not be used by the parser"""

RESERVED_IDENTITIES = {
    "+": TT_Add | TT_Operation | TT_Ident | TT_OOO_ADD,
    "-": TT_Sub | TT_Operation | TT_Ident | TT_OOO_ADD,
    "*": TT_Mult | TT_Operation | TT_Ident | TT_OOO_Mult,
    "/": TT_Div | TT_Operation | TT_Ident | TT_OOO_Mult,
    "=": TT_Equ | TT_Operation | TT_Ident,
    "^": TT_Exponent | TT_Operation | TT_Ident | TT_OOO_Expo,
    "__testfunc": TT_Func | TT_Operation | TT_Ident,
    ",": TT_Comma | TT_Operation | TT_Ident,
}


def parse(input: str) -> Iterable[Tuple[int, Any]]:
    tokens: list[Tuple[int, Any]] = []
    tokens_positions = []
    token_type = 0
    buffer = ""
    i = 0

    while i < len(input):
        char = input[i]
        i += 1
        if not char.isascii():
            raise "this program does only support unicode characters"

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
            if char.isnumeric():
                # what we're dealing with is something numeric
                token_type = TT_Numeric | TT_Int
            else:
                token_type = TT_Ident

        if char == "(":
            if buffer:
                tokens.append((token_type, buffer))
                tokens_positions.append(i)
            buffer = ""
            token_type = 0
            start = i
            depth = 0
            brackets = "()"
            while i < len(input):
                char = input[i]
                i += 1
                if char == brackets[0]:
                    depth += 1
                elif char == brackets[1]:
                    depth -= 1
                    if depth < 0:
                        tokens.append((TT_Tokens, parse(input[start : i - 1])))
                        tokens_positions.append(i)
                        break
        if token_type & TT_Numeric:
            if char.isnumeric():
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
            if char in RESERVED_IDENTITIES or char.isnumeric():
                if buffer:  # prevent adding an empty buffer to tokens
                    tokens.append((token_type, buffer))
                    tokens_positions.append(i)
                buffer = char
            else:
                buffer += char

            if char.isnumeric():
                token_type = TT_Int

        # final thing in loop check buffer for a reserved identity
        if buffer in RESERVED_IDENTITIES:
            tokens.append((RESERVED_IDENTITIES[buffer], buffer))
            tokens_positions.append(i)

            buffer = ""
            token_type = 0

    # final stuff
    if token_type != 0:
        tokens.append((token_type, buffer))
        tokens_positions.append(i)

    # TODO: handle implicit multiplication
    # TODO: handle ** as to the power of

    i = 0
    token: Tuple[int, Any]
    while i < len(tokens):
        if (i + 1) == len(tokens):
            break
        # check that next token has an operation
        token = tokens[i]
        next_token = tokens[i + 1]

        def print_warning():
            print(
                f"[WARNING]: '{input[min(tokens_positions[i] -10, 0):max(tokens_positions[i]+10, (len(input) -1))]}'"
            )

        if token[0] & TT_Operation:
            if token[0] == next_token[0]:
                if token[0] & TT_Mult:
                    tokens.remove(token)
                    tokens[i] = (RESERVED_IDENTITIES["^"], "^")
                # TODO: check for stuff like "--" & "+-" & "(- 1)"
                else:
                    print_warning()
        elif not (next_token[0] & TT_Operation):
            if next_token[0] & TT_Numeric:
                # TODO: have the cabability to issue a warning if two TT_Numerics are being implicitly multiplied
                print_warning()
            i += 1
            tokens.insert(i, (TT_Mult, "*"))
            tokens_positions.insert(i, -1)
        i += 1

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
