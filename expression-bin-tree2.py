from typing import Any, Union, Tuple, Callable


class Atom:
    """
    This idea is "inspired" by sympy source code
    """


class Value(Atom):
    value: Any


class IntValue(Value):
    value: int

    def __init__(self, value: int) -> None:
        super().__init__()

        self.value = value


class Operation(Value):
    left: Value
    right: Value

    def __str__(self) -> str:
        return str(self.left.value) + "+" + str(self.right.value)



class AddOperation(Operation):
    value = "+"

    def __init__(self, left: Value, right: Value) -> None:
        super().__init__()
        self.left = left
        self.right = right




def generate_value(input: str) -> Union[Tuple[Value, None], Tuple[None, str]]:
    if input.isnumeric():
        return IntValue(int(input)), None

    return None, "Failed to create a value"


def parse(input: str):
    end_idx = len(input)
    idx = end_idx - 1

    right: Value
    operation_constructor: Callable

    while idx >= 0:  # read from right to left
        char = input[idx]
        if char.isspace():  # ignore white-space
            idx -= 1
            continue

        # !TODO: be aware of brackets, i.e. scopes
        # !NOTE: I want recursion becaus it's super cool

        if char == "+":
            # print a slice of the earlier statement
            right_expression = input[(idx + 1) : (end_idx)].strip()
            value, err_msg = generate_value(right_expression)
            if err_msg:
                raise err_msg

            right = value
            operation_constructor = AddOperation
            end_idx = idx

        idx = idx - 1

    if idx + 1 != end_idx:
        # resolve by compting final left side
        left_expression = input[(idx + 1) : (end_idx)].strip()
        value, err_msg = generate_value(left_expression)
        if err_msg:
            raise err_msg
        
        op = operation_constructor(value, right)
        print (op)
        




expression_1 = "1 + 2"

parse(expression_1)
