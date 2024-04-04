from typing import Union, Iterable, Final, Tuple

ADDITION: Final = 128 << 0
SUBTRACTION: Final = 128 << 1
MULTIPLICATION: Final = 128 << 2

OPERAND_TABLE = {ADDITION: "+", SUBTRACTION: "-", MULTIPLICATION: "*"}

class Symbol:
    pass


class Expression:
    values: Iterable[Union[float, int, Symbol]]
    pass


class Symbol(Symbol):
    name: str
    # values: Iterable[Tuple[int, Union[int, float]]]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = str(name)

    def __add__(self, value):
        if type(value) == type(self):
            print(value, "Symbol")
            pass  # value is a symbol
        elif type(value) == Expression:
            print(value, "Expression")
            pass
        else:
            print(value, "number")

        return Expression()

    def __eq__(self, __value: object) -> bool:
        print(self.name, __value)

        return True

    def __str__(self) -> str:
        return self.name


a = Symbol("a")
# b = Symbol("b")

print( 1+ a)

# a == b + 1 + 1 + b
