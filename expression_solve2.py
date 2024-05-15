from typing import Literal, Optional, TypedDict
from expression_parser import parse, TT_Equ
from expression_tree_builder2 import *
from expression_compare import *
from utils import *


class SolveActionEntry(TypedDict):
    type: Literal["global", "simplify", "simplify_modify"]
    node_str: str
    method_name: Callable

    parameters: Iterable[Any]
    derrived_values: Iterable[Any]


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


def multiple_targets_in_values(values: Iterable, target: Variable) -> bool:
    count = 0
    for value in values:
        if is_target_variable_in_tree(value, target):
            count += 1
        if count > 1:
            return True
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
            "derrived_values": (str(node),),
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
            "derrived_values": (str(node),),
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
            "derrived_values": (str(node),),
        }
    )


def _fix_last_entries_derrived_value(
    node: Operation,
    solve_action_list: list[SolveActionEntry],
):
    """Solve, slight inconvenience wher the global solve actions do not have access to the step"""
    assert isinstance(node.values, list)
    assert len(solve_action_list) > 0
    assert solve_action_list[-1]["type"] == "global"

    solve_action_list[-1]["derrived_values"] = (str(node), *solve_action_list[-1]["derrived_values"][1:])

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
    """
    calculate the product of all integers
    Yet be modify as little as possible
    """
    assert node.token_type & TT_Mult, str(node)

    node_str = str(node)

    integer_idx = -1
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Add:
            value = _simplify_terms(value, solve_action_list=solve_action_list)
            # this could cause infinite recursion

        if isinstance(value, Operation) and value.token_type & TT_Div:
            # What to do when multiplying with fractions
            # An approach where i convert every thing to a fraction and simplify
            # Then pull out the divisor => a / b => a * (1 / b)
            value = _simplify_division(value, solve_action_list)

        if value.token_type & TT_Numeric and value.token_value == 1:
            values.pop(i)
            continue

        if isinstance(value, Integer):
            if integer_idx >= 0:
                values[integer_idx].token_value *= value.token_value
                values.pop(i)
                continue
            else:
                integer_idx = i
                values[integer_idx] = Integer.create(value.token_value)

        i += 1

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
    values = list(node.values)
    i = 0
    while i < len(values):
        value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Mult:
            values[i] = _simplify_factors(value, solve_action_list=solve_action_list)
            value = values[i]

        if isinstance(value, Operation) and value.token_type & TT_Div:
            values[i] = _simplify_division(value, solve_action_list=solve_action_list)
            value = values[i]

        if value.token_type & TT_Numeric and value.token_value == 0:
            values.pop(i)
            continue

        if isinstance(value, Integer):

            if integer_idx >= 0:
                values[integer_idx].token_value += value.token_value
                values.pop(i)
                continue
            else:
                integer_idx = i
                values[integer_idx] = Integer.create(value.token_value)

        i += 1

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


def _simplify_division_flatten(
    node: Operation, solve_action_list: list[SolveActionEntry]
) -> Operation:
    assert node.token_type & TT_Div

    def collect_dividends_and_divisors(
        node: TokenValue,
    ) -> Tuple[list[TokenValue], list[TokenValue]]:
        left: list[TokenValue] = (
            []
        )  # a list containing all values that would be a dividend
        right: list[TokenValue] = []  # a list containinga all dividends

        if node.token_type & TT_Mult and isinstance(node, Operation):
            for value in node.values:
                l, r = collect_dividends_and_divisors(value)
                left.extend(l)
                right.extend(r)
        elif node.token_type & TT_Div and isinstance(node, Operation):
            l, r = collect_dividends_and_divisors(node.left)
            left.extend(l)
            right.extend(r)
            l, r = collect_dividends_and_divisors(node.right)
            left.extend(r)
            right.extend(l)
        else:
            left.append(node)

        return left, right

    dividends: list[TokenValue] = []
    divisors: list[TokenValue] = []

    node_str = str(node)

    l, r = collect_dividends_and_divisors(node.left)
    dividends.extend(l)
    divisors.extend(r)
    l, r = collect_dividends_and_divisors(node.right)
    dividends.extend(r)
    divisors.extend(l)

    mult_hack = Operation.create("*")
    node = _construct_token_value_with_values(
        node,
        _construct_token_value_with_values(mult_hack, *dividends),
        _construct_token_value_with_values(mult_hack, *divisors),
    )

    if str(node) != node_str:
        solve_action_list.append(
            {
                "type": "simplify",
                "method_name": _simplify_division_flatten.__name__,
                "node_str": node_str,
                "parameters": (),
                "derrived_values": (str(node),),
            }
        )

    return node


def _simplify_division(
    node: Operation, solve_action_list: list[SolveActionEntry]
) -> Operation:
    assert node.token_type & TT_Div

    node = _simplify_division_flatten(node, solve_action_list=solve_action_list)

    if node.left.token_type & TT_Mult and isinstance(node.left, Operation):
        dividends = list(node.left.values)
    else:
        dividends = [node.left]

    if node.right.token_type & TT_Mult and isinstance(node.right, Operation):
        divisors = list(node.right.values)
    else:
        divisors = [node.right]

    i = 0
    while i < len(dividends):
        j = 0
        while j < len(divisors):
            if compare_values(dividends[i], divisors[j]):
                dividends.pop(i)
                divisors.pop(j)
                break
            j += 1
        else:
            i += 1

    mult_hack = Operation.create("*")
    node_str = str(node)
    node = _construct_token_value_with_values(
        node,
        _construct_token_value_with_values(mult_hack, *dividends),
        _construct_token_value_with_values(mult_hack, *divisors),
    )

    if isinstance(node, Operation) and node.token_type & TT_Div:
        if node.left.token_type & TT_Numeric and node.left.token_value == 0:
            node = Integer.create(0)
        elif node.right.token_type & TT_Numeric and node.right.token_value == 1:
            node = node.left

    solve_action_list.append(
        {
            "type": "simplify",
            "method_name": _simplify_division.__name__,
            "node_str": node_str,
            "parameters": (),
            "derrived_values": (str(node),),
        }
    )

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
                "type": "simplify_modify",
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
            lambda term: _simplify_division(
                _construct_token_value_with_values(node, (term), node.right),
                solve_action_list,
            ),
            terms,
        ),
    )

    solve_action_list.append(
        {
            "type": "simplify_modify",
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
                # TODO: to make this more general check if the specific value is in tree instead of just checking for variable
                if (
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
        elif value.token_type & TT_Div:
            if is_target_variable_in_tree(value.right, target):
                # 1 / a + a => (1 + a * a) / a
                # this would require, that this is done significantly differently
                # because now the algorithm would have to muliply the divisor and subtract something
                # and now you have a quadratic equation
                # c + a * a = b * a
                # a * a - b * a + c = 0
                # a^2 - b * a + c = 0
                raise NotImplementedError(
                    "unsupported, when the target is in the divisor:", str(value)
                )
            elif is_target_variable_in_tree(value.left, target):
                # (a + 2) / 2 => 1 / 2 * (a + 2)
                # (a * 2) / 2 => 2 / 2 * a
                # ((a + 2) * 2) / 2 => 2 / 2 * (a + 2)
                # where do i check if there actually is the target within the operation
                # TODO: make this a seperate solver step
                value_str = str(value)
                if compare_values(value.left, target):
                    values[i] = Operation.create(
                        "*",
                        _construct_token_value_with_values(
                            value, Integer.create(1), value.right
                        ),
                        value.left,
                    )
                elif value.left.token_type & TT_Add:
                    """this might have to change if something like (a + a) / 2 => 2 / 2 * a"""
                    # TODO: support and be aware of more edge cases
                    values[i] = Operation.create(
                        "*",
                        _construct_token_value_with_values(
                            value, Integer.create(1), value.right
                        ),
                        value.left,
                    )
                elif value.left.token_type & TT_Mult and isinstance(
                    value.left, Operation
                ):
                    lvalues = list(value.left.values)
                    factors = []
                    j = 0
                    # grab every factor that contains the target
                    while j < len(lvalues):
                        lval = lvalues[j]
                        if is_target_variable_in_tree(lval, target):
                            factors.append(lval)
                            lvalues.pop(j)
                        j += 1

                    values[i] = Operation.create(
                        "*",
                        _construct_token_value_with_values(
                            value,
                            _construct_token_value_with_values(value.left, *lvalues),
                            value.right,
                        ),
                        *factors,
                    )
                else:
                    raise NotImplementedError

                solve_action_list.append(
                    {
                        "type": "simplify_modify",
                        "method_name": "_temp_name_ripout_target_from_fraction",
                        "node_str": value_str,
                        "derrived_values": (str(values[i]),),
                        "parameters": (),
                    }
                )

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
            "type": "simplify_modify",
            "method_name": _simplify_factor_target.__name__,
            "node_str": str(node_str),
            "parameters": (str(target),),
            "derrived_values": (str(factored_value),),
        }
    )

    return factored_value


def _simplify(
    node: TokenValue, solve_action_list: list[SolveActionEntry] = []
) -> TokenValue:
    if node.token_type & TT_Add:
        return _simplify_terms(node, solve_action_list=solve_action_list)
    if node.token_type & TT_Mult:
        return _simplify_factors(node, solve_action_list=solve_action_list)
    if node.token_type & TT_Div:
        return _simplify_division(node, solve_action_list=solve_action_list)
    return node


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

                        _fix_last_entries_derrived_value(node, solve_action_list)
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
                        _fix_last_entries_derrived_value(node, solve_action_list)
                elif destin.token_type & TT_Div:
                    values = list(destin.values)

                    _multiply(
                        node,
                        target_idx,
                        destin.right,
                        solve_action_list=solve_action_list,
                    )

                    node.values[destin_idx] = destin.left
                    _fix_last_entries_derrived_value(node, solve_action_list)
                    continue
                else:
                    raise NotImplementedError("operation not handled")
            else:
                _subtract(
                    node,
                    target_idx,
                    node.values[destin_idx],
                    solve_action_list=solve_action_list,
                )
                node.values[destin_idx] = Integer.create(0)
                _fix_last_entries_derrived_value(node, solve_action_list)

        root = node.values[target_idx]
        if not isinstance(root, Operation):
            raise NotImplementedError(
                "This could be the variable has switched sides or this is a funtction "
                + str(node)
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

                elif left.token_type & TT_Div or right.token_type & TT_Div:
                    raise NotImplementedError(
                        "here there should be some logic to simplify the fraction",
                        str(root),
                    )
                else:
                    print(root, left, right)
                    _multiply(
                        node,
                        destin_idx,
                        root.right,
                        solve_action_list=solve_action_list,
                    )
                    node.values[target_idx] = root.left
                    _fix_last_entries_derrived_value(node, solve_action_list)

                    tmp = target_idx
                    target_idx = destin_idx
                    destin_idx = tmp
            elif root.token_type & TT_Add:
                # this is just a hyperspecific action because i'm trying to solve a specific equation

                factored_value = _simplify_factor_target(
                    root, target, solve_action_list
                )

                if multiple_targets_in_values(factored_value.values, target):
                    print(node, factored_value)
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
                    _fix_last_entries_derrived_value(node, solve_action_list)
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
                    _fix_last_entries_derrived_value(node, solve_action_list)
                    continue

                i += 1
        elif root.token_type & TT_Div:
            # multiply the divisor, if the variable is in the divisor swap target and destin
            _multiply(node, destin_idx, right, solve_action_list=solve_action_list)
            node.values[target_idx] = left
            _fix_last_entries_derrived_value(node, solve_action_list)

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


def find_values_that_would_cause_an_undefined_result(
    node: Operation, bad_values: list[Tuple[Variable, TokenValue]]
) -> list[Tuple[Variable, TokenValue]]:
    """
    The aim with this function is determine values that would cause a undefined result.
    "a = 10 / b", b != 0
    "a = 10 / (b - 4)", b != 4
    """

    # first find nodes where divison occurs
    nodes = [node]
    while len(nodes):
        sub_node = nodes.pop()

        if sub_node.token_type & TT_Operation == 0 and not isinstance(
            sub_node, Operation
        ):
            continue  # ignore none operations in this case

        if len(collect_all_variables(node)) == 0:
            continue  # ignore if there aren't any variables in tied to the operation

        if sub_node.token_type & TT_Div == 0:
            nodes.extend(sub_node.values)
            continue

        # first check if sub_node is a division operation, and contains variables

        # now comes the fun part only bother about the divisor

        sub_node = _simplify_division_flatten(sub_node, [])

        if isinstance(sub_node.left, Operation):
            nodes.extend(sub_node.values)
        if isinstance(sub_node.right, Operation):
            nodes.extend(sub_node.values)

        variables = collect_all_variables(sub_node.right)
        if len(variables) == 0:
            continue

        for variable in variables:
            solved = solve_for2(
                Operation.create("=", sub_node.right, Integer.create(0)), variable
            )

            if len(collect_all_variables(solved.right)) > 0:
                # TODO: somehow return a function instead
                bad_values.append((variable, _simplify(solved.right)))
            else:
                bad_values.append((variable, _simplify(solved.right)))


def _print_solve_action_list(solve_action_list: list[SolveActionEntry]):
    for solve_action in solve_action_list:
        if solve_action["type"] == "global":
            print(
                (
                    solve_action["method_name"],
                    solve_action["node_str"],
                    # *map(str, solve_action["parameters"]),
                    str(solve_action["parameters"][1]),
                    "=>",
                    solve_action["derrived_values"][0],
                )
            )
        elif (solve_action["type"] == "simplify" and True) or solve_action[
            "type"
        ] == "simplify_modify":

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
        if len(solve_action_list):
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

    # try to check what values some thing can be to make an equation true
    bad_values = []
    find_values_that_would_cause_an_undefined_result(solved, bad_values)
    print([*map(lambda v: (str(v[0]), str(v[1])), bad_values)])

    print("-" * 20)


def pb(s: str):
    return build_tree2(parse(s))


a = Variable.create("a")
b = Variable.create("b")

_test_solve_for2("10 + a = b * a", a)
_test_solve_for2("10 + a = b * a", b)
_test_solve_for2("10 + c * a = b * a", a)

_test_solve_for2("10 + a = b - a", a)

_test_solve_for2("a * b + c * d = a * e + c * f", a)

_test_solve_for2("f = 1 / a", a, 0)

# _test_solve_for2("b = c * ((a + v) / a)", a)

# _test_solve_for2("a = v / (b / c + -1)", b)

# _test_solve_for2("b = (a + 1) / (a + 3)", a)

# _test_solve_for2("a = (a + b) /  3",a)

# TODO: add some form of logic that checks that the input fraction is valid
# _test_solve_for2("a / 2  = a", a)
# TODO: add some of logic when there are an infinite amount of answers
# _test_solve_for2("a / 2  = a * b", a)

# _test_solve_for2("a / 2  = a * b + 1", a)

# print(_simplify_factor_target(build_tree2(parse("a / 2 + -1 * a")), a, []))
# print(_simplify_factor_target(build_tree2(parse("a / (2 * a) + -1 * a")), a, []))


# steps = []
# v = _simplify_division(pb("(2 * c) / ((c / d) * 2)"), steps)
# _print_solve_action_list(steps)

# print("-" * 40)


# for expr in [
#     "2 * a * 2",  # 4 * a
#     "1 * a",  # a
#     "2 * a * b",  # 2 * a * bx
#     "2 * (a + b)",  # 2 * (a + b)
#     "2 * (a / b)",  # 2 * (a / b)
#     "2 * (a / (b * 2))",  # a * (1 / b)
#     "3 * (a / (b * 2))",  # 3 * a * (1 / (b * 2))
# ]:
#     steps = []
#     v = _simplify_factors(pb(expr), steps)
#     _print_solve_action_list(steps)
#     print(v)
#     print("-" * 40)

# print(_simplify_factor_target(build_tree2(parse("(a + 2) / 2 + -1 * a")), a, steps))

# print(_simplify_factor_target(build_tree2(parse("(a * 2 * b * a) / 2 + -1 * a")), a, steps))
# _print_solve_action_list(steps)

# print(_simplify_factor_target(build_tree2(parse(" a * a + -1 * a")), a, steps))
# print(_simplify_factor_target(build_tree2(parse("(a * 2 * b) / 2 + -1 * a")), a, steps))
# _print_solve_action_list(steps)
