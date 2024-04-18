from typing import Iterable, Tuple, Any, Final

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


def parse(
    input: str, RESERVED_IDENTITIES=RESERVED_IDENTITIES
) -> Iterable[Tuple[int, Any]]:
    tokens: list[Tuple[int, Any]] = []
    tokens_positions = []
    token_type = 0
    buffer = ""
    i = 0

    while i < len(input):
        char = input[i]
        i += 1

        if not char.isascii() and not issubscript(char) and not ismathematical_alphanumeric_symbol(char):
            raise ValueError(f"{char} : unsupported charachter")

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
            if char in RESERVED_IDENTITIES or (
                char.isnumeric() and not issubscript(char)
            ):
                if buffer:  # prevent adding an empty buffer to tokens
                    tokens.append((token_type, buffer))
                    tokens_positions.append(i)
                buffer = char
            else:
                buffer += char

            if char.isnumeric() and not issubscript(char):
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
                    tokens.pop(i)
                    tokens_positions.pop(i)

                    tokens[i] = (RESERVED_IDENTITIES["^"], "^")
                # TODO: check for stuff like "--" & "+-" & "(- 1)"
                else:
                    print_warning()

            # look at -
            if i <= 0 or tokens[i - 1][0] & TT_Operation:
                if token[0] & (TT_Sub | TT_Add):
                    if next_token[0] & TT_Numeric:
                        if (next_token[0] & TT_INFO_MASK) > 0:
                            # this means it has been touched
                            raise "Something went wrong"
                        # Set the new token and remove
                        tokens[i + 1] = (
                            next_token[0]
                            | (
                                TT_Numeric_Positive
                                if token[0] & TT_Add
                                else TT_Numeric_Negative
                            ),
                            token[1] + next_token[1],
                        )
                        tokens.pop(i)
                        tokens_positions.pop(i)
                        i -= 1
                    elif (
                        next_token[0] & TT_Ident and not next_token[0] & TT_Operation
                    ):  # check that next token is only an identity
                        # implicit multiplication
                        if token[0] & TT_Add:
                            tokens[i] = (
                                (TT_Numeric | TT_Int | TT_Numeric_Positive),
                                "1",
                            )
                        else:
                            tokens[i] = (
                                (TT_Numeric | TT_Int | TT_Numeric_Negative),
                                "-1",
                            )
                        i += 1
                        tokens.insert(i, (RESERVED_IDENTITIES["*"], "*"))
                        tokens_positions.insert(i, -1)

        elif not (next_token[0] & TT_Operation):
            if next_token[0] & TT_Numeric:
                # TODO: have the cabability to issue a warning if two TT_Numerics are being implicitly multiplied
                print_warning()
            i += 1
            tokens.insert(i, (RESERVED_IDENTITIES["*"], "*"))
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
assert len(parse("-1")) == 1
assert parsed_to_string(parse("-a"), space="") == "-1*a"
assert parsed_to_string(parse("a₀ + 1 - 𝜃 + 𝛑")) == "a₀ + 1 - 𝜃 + 𝛑"

