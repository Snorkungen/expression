from typing import Optional
from expression_parser import parse, TT_Equ
from expression_tree_builder2 import *
from expression_compare import *


def b_nand(a: int, b: int) -> int:
    return a ^ (b & a)


def is_target_variable_in_tree(node: Operation, target: Variable) -> bool:
    if not isinstance(node, Operation):
        return compare_variables(node, target)

    nodes = list(node.values)
    while len(nodes) > 0:
        sub_node = nodes.pop()

        if compare_variables(target, sub_node):
            return True

        if isinstance(sub_node, Operation):
            nodes.extend(sub_node.values)

    return False


def _subtract(node: Operation, destin_idx: int, value: TokenValue):
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)

    print(_subtract.__name__, node, f"[{value}, {destin_idx}]")

    node.values[destin_idx] = _simplify_terms(
        Operation.create(
            "+",
            node.values[destin_idx],
            Operation.create("*", Integer.create(-1), value),
        )
    )


def _multiply(node: Operation, destin_idx: int, value: TokenValue):
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)
    print(_multiply.__name__, node, f"[{value}, {destin_idx}]")

    node.values[destin_idx] = _simplify_factors(
        Operation.create("*", node.values[destin_idx], value)
    )


def _divide(node: Operation, destin_idx: int, value: TokenValue):
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)
    print(_divide.__name__, node, f"[{value}, {destin_idx}]")

    node.values[destin_idx] = Operation.create("/", node.values[destin_idx], value)


def _construct_token_value_with_values(
    source: TokenValue, *values: Tuple[TokenValue, ...]
) -> Optional[TokenValue]:
    if len(values) == 0:
        if source.token_type & TT_Mult:
            return Integer.create(1)
        if source.token_type & TT_Add:
            return Integer.create(0)
        return None

    if len(values) == 1:
        return values[0]

    return Operation(
        source.token,
        *values,
    )


def _simplify_factors(node: Operation) -> Operation:
    """calculate the product of all integers"""
    assert node.token_type & TT_Mult, str(node)

    integer_product = 1
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Add:
            value = _simplify_terms(value)
            # this could cause infinite recursion

        if isinstance(value, Integer):
            integer_product *= value.token_value
            values.pop(i)
            continue

        i += 1

    if integer_product != 1:
        values.append(Integer.create(integer_product))
        values.reverse()

    new_node = _construct_token_value_with_values(node, *values)
    if new_node == None:
        raise ValueError

    # print(_simplify_factors.__name__, f"[{node} => {new_node}]")
    # print(node)
    return new_node


def _simplify_terms(node: Operation) -> Operation:
    """calculate the sum of all integers"""
    assert node.token_type & TT_Add

    integer_idx = -1
    integer_sum = 0
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Mult:
            values[i] = _simplify_factors(value)
            value = values[i]

        if isinstance(value, Integer):
            if integer_idx >= 0:
                values[integer_idx].token_value += value.token_value
                values.pop(i)
                continue
            else:
                integer_idx = i

        i += 1

    if integer_sum != 0:
        values.append(Integer.create(integer_sum))

    new_node = _construct_token_value_with_values(node, *values)

    if new_node == None:
        raise ValueError

    # print(_simplify_terms.__name__, f"[{node} => {new_node}]")

    return new_node


def _simplify_distribute_factor(node: Operation, idx: int):
    assert len(node.values) > idx and idx >= 0
    assert node.token_type & TT_Mult


    values = list(node.values)
    value = values.pop(idx)

    if not isinstance(value, Operation):
        # this is ignoring functions, but functions, i do not undestand
        return node  # there is nothing to do

    if value.token_type & TT_Add:
        distributed_value =  _simplify_terms(  # simplify_terms called due to the fact that this is a add operation
            _construct_token_value_with_values(
                value,
                *map(
                    lambda v: _construct_token_value_with_values(node, *values, v),
                    value.values,
                ),
            )
        )
        print(_simplify_distribute_factor.__name__, f"[{node} => {distributed_value}, {idx}]")

        return distributed_value
    raise NotImplementedError("simplifying is not implemented for this operation")

    # get the values which are not the thing of interest


def solve_for2(node: Operation, target: Variable) -> Operation:
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)

    assert is_target_variable_in_tree(node, target), "Target variable not in node!"

    node = Operation((node.token_type, node.token_value), *node.values)
    node.values = list(node.values)  # ensure that the values are a list

    # first things first handle the target variable being on both sides
    if is_target_variable_in_tree(node.left, target) and is_target_variable_in_tree(
        node.right, target
    ):
        # move variable from 1-side to the other side

        # default to move from right to left
        if isinstance(node.right, Operation):
            if node.right.token_type & TT_Add:
                # loop through values and subtract the target value from the node
                values = list(node.right.values)
                i = 0
                while i < len(values):
                    value = values[i]
                    if not is_target_variable_in_tree(value, target):
                        i += 1
                        continue

                    _subtract(node, 0, value)
                    values.pop(i)

                    # this is verbose but that is the whole point right
                    node.values[1] = _construct_token_value_with_values(
                        node.values[1], *values
                    )
            elif node.right.token_type & TT_Mult:
                values = list(node.right.values)
                i = 0
                while i < len(values):
                    value = values[i]
                    if not is_target_variable_in_tree(value, target):
                        i += 1
                        continue

                    _divide(node, 0, value)
                    values.pop(i)
                    node.values[1] = _construct_token_value_with_values(
                        node.values[1], *values
                    )
            elif node.right.token_type & TT_Div:
                if is_target_variable_in_tree(node.right.right, target):
                    if (
                        isinstance(node.right.right, Operation)
                        and node.right.right.token_type & TT_Mult
                    ):
                        values = list(node.right.right.values)
                        i = 0
                        while i < len(values):
                            value = values[i]
                            if not is_target_variable_in_tree(value, target):
                                i += 1
                                continue
                            _multiply(node, 0, value)
                            values.pop(i)

                            if len(values) == 0:
                                node.values[1] = node.right.left

                            node.values[1] = Operation(
                                node.right.token,
                                node.right.left,
                                _construct_token_value_with_values(
                                    node.right.right, *values
                                ),
                            )
                    else:
                        # the variable is in the divisor
                        _multiply(node, 0, node.right.right)
                        node.values[1] = node.right.left
                if node.right.token_type & TT_Div and is_target_variable_in_tree(
                    node.right.left, target
                ):
                    if (
                        isinstance(node.right.left, Operation)
                        and node.right.left.token_type & TT_Mult
                    ):
                        values = list(node.right.left.values)
                        i = 0
                        while i < len(values):
                            value = values[i]
                            if not is_target_variable_in_tree(value, target):
                                i += 1
                                continue
                            _multiply(node, 0, value)
                            values.pop(i)

                            if len(values) == 0:
                                node.values[1] = node.right.left

                            node.values[1] = Operation(
                                node.left.token,
                                _construct_token_value_with_values(
                                    node.right.left, *values
                                ),
                                node.right.right,
                            )
                    else:
                        _divide(node, 0, node.right.left)
                        node.values[1] = Operation(
                            node.right.token, Integer.create(1), node.right.right
                        )
            else:
                print(node.right)
                raise NotImplementedError
        else:
            # the only value on the right side is the target variable
            _subtract(node, 0, node.right)
            node.values[1] = Integer.create(0)

    # determine where the variable is

    target_idx = 0 if is_target_variable_in_tree(node.left, target) else 1
    destin_idx = 1 if is_target_variable_in_tree(node.left, target) else 0

    def multiple_targets_in_values(values: Iterable, target: Variable) -> bool:
        count = 0
        for value in values:
            if is_target_variable_in_tree(value, target):
                count += 1
            if count > 1:
                return True
        return False

    while not compare_variables(node.values[target_idx], target):
        if is_target_variable_in_tree(node.left, target) and is_target_variable_in_tree(
            node.right, target
        ):
            # NOTE: this statement will not run if a = a + b,
            # or any similar situation where the only value on the target side is the target variable

            # do the same logic as above
            if node.values[destin_idx].token_type & TT_Operation:
                destin = node.values[destin_idx]
                assert isinstance(destin, Operation)

                # this is a operation, do operation specific actions
                if destin.token_type & TT_Add:
                    values = list(destin.values)

                    i = 0
                    while i < len(values):
                        value = values[i]
                        if not is_target_variable_in_tree(value, target):
                            i += 1
                            continue

                        _subtract(node, target_idx, value)
                        values.pop(i)

                        # this is verbose but that is the whole point right
                        node.values[destin_idx] = _construct_token_value_with_values(
                            node.values[destin_idx], *values
                        )
                else:
                    raise NotImplemented("operation not handled")
            else:
                raise NotImplemented("Not an operation")

        root = node.values[target_idx]
        if not isinstance(root, Operation):
            raise NotImplementedError(
                "This could be the variable has switched sides or this is a funtction"
            )
        left = root.left
        right = root.right

        if multiple_targets_in_values(root.values, target):
            if root.token_type & TT_Div:
                if (
                    left.token_type & TT_Add
                    and isinstance(left, Operation)
                    and isinstance(right, Variable)
                ):
                    # solving a specific case, with a specific solution

                    print("_distribute_division", node, f"[{root}, {target_idx}]")
                    values = list(left.values)
                    for i, value in enumerate(values):
                        values[i] = Operation(root.token, value, right)

                    node.values[target_idx] = Operation(left.token, *values)
                else:
                    _multiply(node, destin_idx, root.right)
                    node.values[target_idx] = root.left
                    tmp = target_idx
                    target_idx = destin_idx
                    destin_idx = tmp
            elif root.token_type & TT_Add:
                # this is just a hyperspecific action because i'm trying to solve a specific equation

                # here have a pre step that distributes values if necessary
                values = list(root.values)
                i = 0
                while i < len(values):
                    value = values[i]
                    if value.token_type & TT_Operation == 0 or not isinstance(
                        value, Operation
                    ):
                        i += 1
                        continue  # ignore not interested right now and as mentioned functions do not exist
                    if value.token_type & TT_Mult:
                        # here check if there are any values that need to be distributed
                        distributed_value = None
                        for j in range(len(value.values)):
                            if (
                                not is_target_variable_in_tree(value.values[j], target)
                                or value.values[j].token_type & TT_Operation == 0
                            ):
                                continue

                            if b_nand(
                                value.values[j].token_type, TT_INFO_MASK
                            ) == b_nand(root.token_type, TT_INFO_MASK):
                                distributed_value = _simplify_distribute_factor(
                                    value, j
                                )
                                break

                        if distributed_value and b_nand(
                            distributed_value.token_type, TT_INFO_MASK
                        ) == b_nand(root.token_type, TT_INFO_MASK):
                            values.extend(values[i + 1 :])
                            for j in range(len(distributed_value.values)):
                                values[i + j] = distributed_value.values[j]

                            continue

                    i += 1

                action_taken = False

                factor_terms: list[TokenValue] = []
                factor_indices: list[int] = []

                i = 0
                while i < len(values):
                    value = values[i]

                    # now check for the specific case where a / a = 1
                    if value.token_type & TT_Div and isinstance(value, Operation):
                        if compare_values(value.left, value.right):
                            print(
                                "_division_combine",
                                node,
                                f"[{value}, {target_idx}]",
                            )
                            values[i] = Integer.create(1)
                            node.values[target_idx] = (
                                _construct_token_value_with_values(
                                    node.values[target_idx], *values
                                )
                            )
                            action_taken = True
                    elif compare_values(value, target):
                        factor_terms.append(Integer.create(1))
                        factor_indices.append(i)
                    elif (
                        is_target_variable_in_tree(value, target)
                        and isinstance(value, Operation)
                        and value.token_type & TT_Mult
                    ):
                        if multiple_targets_in_values(value.values, target):
                            raise NotImplementedError(
                                "This whole spaghetti needs to get thrown out!"
                            )

                        """HERE, it is no good to be"""
                        sub_values = tuple(
                            filter(
                                lambda x: not compare_variables(x, target), value.values
                            )
                        )
                        if len(sub_values) == len(value.values):
                            raise NotImplementedError(
                                "As in the line a bit above this is more difficult, than I appreciate"
                            )

                        factor_terms.append(
                            _construct_token_value_with_values(value, *sub_values)
                        )
                        factor_indices.append(i)

                    i += 1

                if len(factor_indices) > 0:
                    print(
                        "_addition_combine",
                        node,
                        f"[{_construct_token_value_with_values(root, *factor_terms)}, {target_idx}]",
                    )
                    for i in reversed(factor_indices[1:]):
                        values.pop(i)

                    values[factor_indices[0]] = _simplify_factors(
                        Operation.create(
                            "*",
                            _construct_token_value_with_values(root, *factor_terms),
                            Variable.create(target.token_value),
                        )
                    )
                    node.values[target_idx] = _construct_token_value_with_values(
                        root, *values
                    )
                    action_taken = True

                if not action_taken:
                    print("Action unknown, quittin loop")
                    break
            else:
                print("Action unknown, quittin loop")
                break
        elif root.token_type & TT_Add:
            values = list(root.values)
            i = 0
            while i < len(values):
                value = values[i]

                if not is_target_variable_in_tree(value, target):
                    _subtract(node, destin_idx, value)
                    values.pop(i)
                    node.values[target_idx] = _construct_token_value_with_values(
                        root, *values
                    )
                    continue

                i += 1
        elif root.token_type & TT_Mult:
            values = list(root.values)
            i = 0
            while i < len(values):
                value = values[i]

                if not is_target_variable_in_tree(value, target):
                    _divide(node, destin_idx, value)
                    values.pop(i)
                    node.values[target_idx] = _construct_token_value_with_values(
                        root, *values
                    )
                    continue

                i += 1
        elif root.token_type & TT_Div:
            # multiply the divisor, if the variable is in the divisor swap target and destin
            _multiply(node, destin_idx, right)
            node.values[target_idx] = left

            if is_target_variable_in_tree(right, target):
                tmp = target_idx
                target_idx = destin_idx
                destin_idx = tmp

        else:
            print("Action unknown, quittin loop")
            break

    if target_idx == 1:
        tmp = node.values[0]
        node.values[0] = node.values[1]
        node.values[1] = tmp

    return node


a = Variable.create("a")
b = Variable.create("b")

expr = build_tree2(parse("10 + a = b * a"))
solved = solve_for2(expr, a)

print("Source: ", expr)
print("Solved: ", solved)
solved = solve_for2(expr, b)

print("Source: ", expr)
print("Solved: ", solved)

expr = build_tree2(parse("10 + a = b - a"))
solved = solve_for2(expr, a)

print("Source: ", expr)
print("Solved: ", solved)

expr = build_tree2(parse("a * b + c * d = a * e + c * f"))
solved = solve_for2(expr, a)

print("Source: ", expr)
print("Solved: ", solved)

expr = build_tree2(parse("f = 1 / a"))
solved = solve_for2(expr, a)

print("Source: ", expr)
print("Solved: ", solved)

expr = build_tree2(parse("b = c * ((a + v) / a)"))
solved = solve_for2(expr, a)

print("Source: ", expr)
print("Solved: ", solved)

expr = build_tree2(parse("a = v / (b / c + -1)"))
solved = solve_for2(expr, b)

print("Source: ", expr)
print("Solved: ", solved)

expr = build_tree2(parse("b = (a + 1) / (a + 3)"))
solved = solve_for2(expr, a)

print("Source: ", expr)
print("Solved: ", solved)
