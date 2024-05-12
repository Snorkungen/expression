from typing import Literal, Optional, TypedDict
from expression_parser import parse, TT_Equ
from expression_tree_builder2 import *
from expression_compare import *


class SolveActionEntry(TypedDict):
    type: Literal["global", "simplify"]
    node_str: str
    method_name: Callable

    parameters: Iterable[Any]
    derrived_values: Iterable[Any]


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


def _subtract(
    node: Operation,
    destin_idx: int,
    value: TokenValue,
    solve_action_list: list[SolveActionEntry],
):
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)

    node_str = str(node)

    node.values[destin_idx] = _simplify_terms(
        Operation.create(
            "+",
            node.values[destin_idx],
            Operation.create("*", Integer.create(-1), value),
        ),
        solve_action_list=solve_action_list,
    )

    solve_action_list.append(
        {
            "type": "global",
            "node_str": node_str,
            "method_name": _subtract.__name__,
            "parameters": (destin_idx, value),
            "derrived_values": (str(node)),
        }
    )


def _multiply(
    node: Operation,
    destin_idx: int,
    value: TokenValue,
    solve_action_list: list[SolveActionEntry],
):
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)

    node_str = str(node)
    # print(_multiply.__name__, node, f"[{value}, {destin_idx}]")

    node.values[destin_idx] = _simplify_factors(
        Operation.create("*", node.values[destin_idx], value),
        solve_action_list=solve_action_list,
    )

    solve_action_list.append(
        {
            "type": "global",
            "node_str": node_str,
            "method_name": _multiply.__name__,
            "parameters": (destin_idx, value),
            "derrived_values": (str(node)),
        }
    )


def _divide(
    node: Operation,
    destin_idx: int,
    value: TokenValue,
    solve_action_list: list[SolveActionEntry],
):
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)

    node_str = str(node)

    node.values[destin_idx] = Operation.create("/", node.values[destin_idx], value)

    solve_action_list.append(
        {
            "type": "global",
            "node_str": node_str,
            "method_name": _divide.__name__,
            "parameters": (destin_idx, value),
            "derrived_values": (str(node)),
        }
    )


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


def _simplify_factors(
    node: Operation, solve_action_list: list[SolveActionEntry]
) -> Operation:
    """calculate the product of all integers"""
    assert node.token_type & TT_Mult, str(node)

    node_str = str(node)

    integer_product = 1
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Add:
            value = _simplify_terms(value, solve_action_list=solve_action_list)
            # this could cause infinite recursion

        if isinstance(value, Integer):
            integer_product *= value.token_value
            values.pop(i)
            continue

        i += 1

    if integer_product != 1:
        values.insert(0, Integer.create(integer_product))

    new_node = _construct_token_value_with_values(node, *values)
    if new_node == None:
        raise ValueError

    solve_action_list.append(
        {
            "type": "simplify",
            "method_name": _simplify_factors.__name__,
            "node_str": node_str,
            "parameters": (),
            "derrived_values": (str(new_node),),
        }
    )

    return new_node


def _simplify_terms(
    node: Operation, solve_action_list: list[SolveActionEntry]
) -> Operation:
    """calculate the sum of all integers"""
    assert node.token_type & TT_Add

    node_str = str(node)

    integer_idx = -1
    integer_sum = 0
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Mult:
            values[i] = _simplify_factors(value, solve_action_list=solve_action_list)
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

    solve_action_list.append(
        {
            "type": "simplify",
            "method_name": _simplify_terms.__name__,
            "node_str": node_str,
            "parameters": (),
            "derrived_values": (str(new_node),),
        }
    )

    return new_node


def _simplify_division(
    node: Operation, solve_action_list: list[SolveActionEntry]
) -> Operation:
    assert node.token_type & TT_Div

    # TODO: write a better version of this that actually does this thing properly
    # Can't even be bothered to log this action

    if compare_values(node.left, node.right):
        solve_action_list.append(
            {
                "type": "simplify",
                "method_name": _simplify_division.__name__,
                "node_str": str(node),
                "parameters": (),
                "derrived_values": (str(1),),
            }
        )
        return Integer.create(1)

    return node


def _simplify_distribute_factor(
    node: Operation, idx: int, solve_action_list: list[SolveActionEntry]
):
    assert len(node.values) > idx and idx >= 0
    assert node.token_type & TT_Mult

    node_str = str(node)

    values = list(node.values)
    value = values.pop(idx)

    if not isinstance(value, Operation):
        # this is ignoring functions, but functions, i do not undestand
        return node  # there is nothing to do

    if value.token_type & TT_Add:
        distributed_value = _simplify_terms(  # simplify_terms called due to the fact that this is a add operation
            _construct_token_value_with_values(
                value,
                *map(
                    lambda v: _construct_token_value_with_values(node, *values, v),
                    value.values,
                ),
            ),
            solve_action_list=solve_action_list,
        )

        solve_action_list.append(
            {
                "type": "simplify",
                "method_name": _simplify_distribute_factor.__name__,
                "node_str": node_str,
                "parameters": (idx,),
                "derrived_values": (str(distributed_value),),
            }
        )

        return distributed_value
    raise NotImplementedError("simplifying is not implemented for this operation")

    # get the values which are not the thing of interest


def _simplify_distribute_dividend(
    node: Operation, idx: int, solve_action_list: list[SolveActionEntry]
) -> Operation:
    assert node.token_type & TT_Div

    # TODO: should this simplify divison or is to agressive and will do weird unexpected things

    if not isinstance(node.left, Operation):
        return node

    if not node.left.token_type & TT_Add:
        raise NotImplementedError(
            "distribute division, only supports distributing terms"
        )

    node_str = str(node)
    values = list(node.left.values)
    terms: list[TokenValue] = list()

    if idx < 0:
        terms = values
        values = ()
    elif idx >= len(values):
        values[idx]
        pass  # this will eventually bubble up an IndexError
    else:
        terms = [values[idx]]

        values.pop(idx)

    # append the existing term to terms
    if values:
        terms.insert(0, _construct_token_value_with_values(node.left, *values))

    distributed_value = _construct_token_value_with_values(
        node.left,
        *map(
            lambda term: (_construct_token_value_with_values(node, (term), node.right)),
            terms,
        ),
    )

    solve_action_list.append(
        {
            "type": "simplify",
            "method_name": _simplify_distribute_dividend.__name__,
            "node_str": str(node_str),
            "parameters": (idx,),
            "derrived_values": (str(distributed_value),),
        }
    )

    return distributed_value


def _simplify_factor_target(
    node: Operation, target: TokenValue, solve_action_list: list[SolveActionEntry]
) -> Operation:
    assert node.token_type & TT_Add
    node_str = str(node)

    # here have a pre step that distributes values if necessary
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]
        if value.token_type & TT_Operation == 0 or not isinstance(value, Operation):
            i += 1
            continue  # ignore not interested right now and as mentioned functions do not exist
        if value.token_type & TT_Mult:
            # here check if there are any values that need to be distributed
            distributed_value = None
            for j in range(len(value.values)):
                if (
                    # TODO: to make this more general check if the specific value is in tree instead of just checking for variable
                    not is_target_variable_in_tree(value.values[j], target)
                    or value.values[j].token_type & TT_Operation == 0
                ):
                    continue

                if b_nand(value.values[j].token_type, TT_INFO_MASK) == b_nand(
                    node.token_type, TT_INFO_MASK
                ):
                    distributed_value = _simplify_distribute_factor(
                        value, j, solve_action_list=solve_action_list
                    )
                    break

            if distributed_value and b_nand(
                distributed_value.token_type, TT_INFO_MASK
            ) == b_nand(node.token_type, TT_INFO_MASK):
                values.extend(values[i + 1 :])
                for j in range(len(distributed_value.values)):
                    values[i + j] = distributed_value.values[j]

                continue

        i += 1

    factor_terms: list[TokenValue] = []
    factor_indices: list[int] = []

    i = 0
    while i < len(values):
        value = values[i]

        if compare_values(value, target):
            factor_terms.append(Integer.create(1))
            factor_indices.append(i)
        elif (
            is_target_variable_in_tree(value, target)
            and isinstance(value, Operation)
            and value.token_type & TT_Mult
        ):
            # only remove the first occurence of the target
            # TODO: i want this thing to be smart enough to factor something like the following  a*b*c + a*b*d = (c + d)*a*b
            removed_target = False
            sub_values = []
            for sub_value in value.values:
                if compare_values(sub_value, target):
                    if not removed_target:
                        removed_target = True
                        continue

                sub_values.append(sub_value)
            factor_terms.append(_construct_token_value_with_values(value, *sub_values))
            factor_indices.append(i)

        i += 1

    if len(factor_indices) <= 0:
        return node

    for i in reversed(factor_indices[1:]):
        values.pop(i)

        values[factor_indices[0]] = _simplify_factors(
            Operation.create(
                "*",
                _construct_token_value_with_values(node, *factor_terms),
                Variable.create(target.token_value),
            ),
            solve_action_list=solve_action_list,
        )

    factored_value = _construct_token_value_with_values(node, *values)

    solve_action_list.append(
        {
            "type": "simplify",
            "method_name": _simplify_factor_target.__name__,
            "node_str": str(node_str),
            "parameters": (str(target),),
            "derrived_values": (str(factored_value),),
        }
    )

    return factored_value


def solve_for2(
    node: Operation, target: Variable, solve_action_list: list[SolveActionEntry] = []
) -> Operation:
    assert node.token_type & TT_Equ, "root must be a equals operation"
    assert isinstance(node, Operation)

    assert is_target_variable_in_tree(node, target), "Target variable not in node!"

    node = Operation((node.token_type, node.token_value), *node.values)
    node.values = list(node.values)  # ensure that the values are a list

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

    while not compare_variables(
        node.values[target_idx], target
    ) or is_target_variable_in_tree(node.values[destin_idx], target):
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

                        _subtract(
                            node, target_idx, value, solve_action_list=solve_action_list
                        )
                        values.pop(i)

                        # this is verbose but that is the whole point right
                        node.values[destin_idx] = _construct_token_value_with_values(
                            node.values[destin_idx], *values
                        )
                elif destin.token_type & TT_Mult:
                    values = list(destin.values)
                    i = 0
                    while i < len(values):
                        value = values[i]
                        if not is_target_variable_in_tree(value, target):
                            i += 1
                            continue

                        _divide(
                            node, target_idx, value, solve_action_list=solve_action_list
                        )
                        values.pop(i)
                        node.values[destin_idx] = _construct_token_value_with_values(
                            node.values[destin_idx], *values
                        )
                elif destin.token_type & TT_Div:
                    values = list(destin.values)

                    _multiply(
                        node,
                        target_idx,
                        destin.right,
                        solve_action_list=solve_action_list,
                    )

                    node.values[destin_idx] = destin.left

                    if is_target_variable_in_tree(node.left, target):
                        tmp = target_idx
                        target_idx = destin_idx
                        destin_idx = tmp
                else:
                    raise NotImplementedError("operation not handled")
            else:
                raise NotImplementedError("Not an operation")

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
                    node.values[target_idx] = _simplify_distribute_dividend(
                        root, -1, solve_action_list=solve_action_list
                    )

                    # NOTE: the following lines simplifies each divison, but should be removed whe _simplify_distribute_dividend starts simplifying
                    node.values[target_idx].values = tuple(
                        map(
                            lambda v: _simplify_division(v, solve_action_list),
                            node.values[target_idx].values,
                        )
                    )

                else:
                    _multiply(
                        node,
                        destin_idx,
                        root.right,
                        solve_action_list=solve_action_list,
                    )
                    node.values[target_idx] = root.left
                    tmp = target_idx
                    target_idx = destin_idx
                    destin_idx = tmp
            elif root.token_type & TT_Add:
                # this is just a hyperspecific action because i'm trying to solve a specific equation

                factored_value = _simplify_factor_target(
                    root, target, solve_action_list
                )


                if multiple_targets_in_values(factored_value.values, target):
                    print(node)
                    raise NotImplementedError

                node.values[target_idx] = factored_value
            else:
                print("Action unknown, quittin loop")
                break
        elif root.token_type & TT_Add:
            values = list(root.values)
            i = 0
            while i < len(values):
                value = values[i]

                if not is_target_variable_in_tree(value, target):
                    _subtract(
                        node, destin_idx, value, solve_action_list=solve_action_list
                    )
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
                    _divide(
                        node, destin_idx, value, solve_action_list=solve_action_list
                    )
                    values.pop(i)
                    node.values[target_idx] = _construct_token_value_with_values(
                        root, *values
                    )
                    continue

                i += 1
        elif root.token_type & TT_Div:
            # multiply the divisor, if the variable is in the divisor swap target and destin
            _multiply(node, destin_idx, right, solve_action_list=solve_action_list)
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


def collect_all_variables(node: TokenValue) -> list[Variable]:
    if not isinstance(node, Operation):
        if isinstance(node, Variable):
            return [node]
        return []

    variables: list[Variable] = []

    nodes = list(node.values)
    while len(nodes):
        sub_node = nodes.pop()

        if isinstance(sub_node, Operation):
            nodes.extend(sub_node.values)
            continue

        if isinstance(sub_node, Variable):
            variables.append(sub_node)
            continue

    return variables


def replace_variables(node: Operation, *values: Tuple[Operation, ...]):
    node_values = list(node.values)
    for value in values:
        assert isinstance(value, Operation)
        assert value.token_type & TT_Equ

        # assume left is the variable
        variable = value.left

        for i in range(len(node_values)):
            if compare_variables(node_values[i], variable):
                node_values[i] = value.right
            elif is_target_variable_in_tree(node_values[i], variable):
                node_values[i] = replace_variables(node_values[i], value)

        node = _construct_token_value_with_values(node, *node_values)
    return node


def _print_solve_action_list(solve_action_list: list[SolveActionEntry]):
    for solve_action in solve_action_list:
        if solve_action["type"] == "global":
            print(
                (
                    solve_action["method_name"],
                    solve_action["node_str"],
                    *map(str, solve_action["parameters"]),
                )
            )
        elif solve_action["type"] == "simplify":

            if solve_action["node_str"] == solve_action["derrived_values"][0]:
                continue
            print(
                (
                    solve_action["method_name"],
                    solve_action["node_str"],
                    *map(str, solve_action["parameters"]),
                    solve_action["derrived_values"][0],
                )
            )


def _test_solve_for2(expr, target, show_solve_action_list: Literal[0, 1, 2] = 0):
    """
    show_solve_action_list:\n
        0-show list only if the solution does not pass the test \n
        1-always show solve actions \n
        2-never show actions \n
    """
    if __name__ != "__main__":
        return

    expr = build_tree2(parse(expr))
    solve_action_list: list[SolveActionEntry] = []

    try:
        solved = solve_for2(expr, target, solve_action_list=solve_action_list)
    except NotImplementedError as e:
        _print_solve_action_list(solve_action_list)
        print(solve_action_list[-1]["derrived_values"])
        raise e

    print("Source: ", expr)
    print("Solved: ", solved)

    # TODO: write tests that issues valuels for each value

    if show_solve_action_list == 1:
        _print_solve_action_list(solve_action_list)

    # TODO: implement a method that checks that the created value does indeed match
    if True and show_solve_action_list == 0:
        _print_solve_action_list(solve_action_list)

    print("-" * 20)


a = Variable.create("a")
b = Variable.create("b")


_test_solve_for2("10 + a = b * a", a)
_test_solve_for2("10 + a = b * a", b)

_test_solve_for2("10 + a = b - a", a)

_test_solve_for2("a * b + c * d = a * e + c * f", a)

_test_solve_for2("f = 1 / a", a, 0)

_test_solve_for2("b = c * ((a + v) / a)", a)

_test_solve_for2("a = v / (b / c + -1)", b)

_test_solve_for2("b = (a + 1) / (a + 3)", a)

_test_solve_for2("a = (a + b) /  3",a)
# _test_solve_for2("1 / a = (a + b) /  3",a)
