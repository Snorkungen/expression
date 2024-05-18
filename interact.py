import readline
from typing import Union
from expression_parser import *
from expression_tree_builder2 import *
from expression_solve2 import *


class RegisteredFunctionDescription(TypedDict):
    short_description: str
    signatures: Iterable[str]


class State:
    loop_flag: bool
    history: list[str]

    prompt: str

    env: dict[str, Any]
    functions: dict[
        str,
        Union[
            Tuple[Callable[[Any, Function], None], RegisteredFunctionDescription], str
        ],
    ]

    def __init__(self) -> None:
        self.loop_flag = True
        self.env = {}
        self.functions = {}
        self.history = []
        self.prompt = "enter:"

        readline.parse_and_bind("tab: complete")
        readline.set_completer(self.tab_completer)

    @property
    def reserved_identities(self):
        return {k: TT_Ident | TT_Func for k in self.functions}

    def dispatch(self, text: str):
        if not text.strip():
            return

        tokens = parse(text, additional_identities=self.reserved_identities)

        if len(tokens) < 1 or tokens[0][0] & TT_Func == 0:
            print("unrecognized value:", text)
            return

        self.history.append(text)
        try:
            node = build_tree2(tokens)
        except:
            print("failed to read:", text)
            return

        if not (
            node.token_type & TT_Func
            and isinstance(node, Function)
            and node.name in self.functions
        ):
            print("unrecognized value:", text)
            return

        func = self.functions[node.name]

        if isinstance(func, str):
            func = self.functions[func]

        assert isinstance(func, Tuple)

        # call dispatch function
        try:
            func[0](self, node)
        except:
            print("funcion failed")

    def register_alias(self, alias_name: str, name: str):
        self.functions[alias_name] = name

    def register(
        self,
        name: str,
        dispatch: Callable[[Any, Function], None],
        description: RegisteredFunctionDescription,
    ):
        self.functions[name] = (dispatch, description)

    def tab_completer(self, text: str, state: int):
        return tuple(filter(lambda key: key.startswith(text), self.functions))[state]


ENV_SOLVE_ACTIONS = "solve_actions"
TESTING_EQUATION = "5 * (a + 2) = (8 / a) * a"


description_solve_for: RegisteredFunctionDescription = {
    "name": "solve_for",
    "short_description": "solve an equation",
    "signatures": ("solve_for(c = a + b, a)",),
}


def dispatch_solve_for(state: State, node: Function):
    assert len(node.values) == 2
    assert isinstance(node.values[0], Operation)
    assert isinstance(node.values[1], Variable)

    solve_action_list: list[SolveActionEntry] = []
    try:
        solved = solve_for2(node.values[0], node.values[1], solve_action_list)
    except NotImplemented:
        print("the current solver is not capable of solving the given equation")
        print("TODO: put last derived global value [here]")

    print("\nSource: ", node.values[0])
    print("Solved: ", solved)

    # do some crunching

    actions: list[Tuple[SolveActionEntry, Iterable[SolveActionEntry]]] = []
    action_buffer = []
    for solve_action in solve_action_list:
        if solve_action["type"] != "global":
            action_buffer.append(solve_action)
            continue

        actions.append((solve_action, tuple(action_buffer)))

        action_buffer.clear()

    if not ENV_SOLVE_ACTIONS in state.env:
        state.env[ENV_SOLVE_ACTIONS] = []

    state.env[ENV_SOLVE_ACTIONS].append(actions)

    for i, action in enumerate(actions):
        action = action[0]
        method_name, pre_str, value, after_str = (
            action["method_name"],
            action["node_str"],
            action["parameters"][1],
            action["derrived_values"][0],
        )

        print(
            f"""[{i}] {method_name} {pre_str} by {value}
 =>   {after_str}"""
        )


description_inspect_solve: RegisteredFunctionDescription = {
    "short_description": "Show the solve actions that were not shown",
    "signatures": ("inspect_solve (0)", "inspect_solve (1)"),
}


def dispatch_inspect_solve(state: State, f: Function):
    if not ENV_SOLVE_ACTIONS in state.env:
        return

    # always select the most recent
    actions = state.env[ENV_SOLVE_ACTIONS][-1]

    for value in f.values:
        if value.token_type & TT_Int == 0:
            continue

        idx = value.token_value
        if idx < 0 or idx >= len(actions):
            continue

        action, sub_actions = actions[idx]

        for sub_action in sub_actions:
            print(
                (
                    sub_action["method_name"],
                    sub_action["node_str"],
                    *map(str, sub_action["parameters"]),
                    sub_action["derrived_values"][0],
                )
            )

        method_name, pre_str, value, after_str = (
            action["method_name"],
            action["node_str"],
            action["parameters"][1],
            action["derrived_values"][0],
        )

        print(
            f"""[{idx}] {method_name} {pre_str} by {value}
 =>   {after_str}"""
        )


description_evaluate_solution: RegisteredFunctionDescription = {
    "name": "evaluate_solution",
    "short_description": "solve an equation",
    "signatures": ("solve_for(c = a + b, a)",),
}


def dispatch_evaluate_solution(env: Any, node: Function):
    assert len(node.values) == 2
    assert isinstance(node.values[0], Operation)
    assert isinstance(node.values[1], Operation)
    source, solved = node.values

    print(evaluate_solution(source, solved))


description_quit: RegisteredFunctionDescription = {
    "short_description": "exit the program",
    "signatures": ("exit",),
}


def dispatch_quit(state: State, _: Function):
    state.loop_flag = False
    print("Bye!")


description_help: RegisteredFunctionDescription = {
    "short_description": "show the available functions",
    "signatures": ("help", "help quit"),
}


def dispatch_help(state: State, f: Function):
    if len(f.values) > 0:
        value = f.values[0]
        if isinstance(value, Function) and value.token_value in state.functions:
            print(state.functions[value.token_value][1]["signatures"])
            return

    for name in state.functions:
        if isinstance(state.functions[name], str):
            continue
            name = state.functions[name]
        print(f"{name} - {state.functions[name][1]['short_description']}")


description_alias: RegisteredFunctionDescription = {
    "short_description": "create a alias for a function",
    "signatures": ("alias(alias, function_to_alias)"),
}


def dispatch_alias(state: State, f: Function):
    values = list(f.values)
    if len(values) < 2:
        state.dispatch(f"help({f.name})")
        print()

        for func in state.functions:
            if isinstance(state.functions[func], str):
                print(func, "-", state.functions[func])

        return

    if values[1].token_type & TT_Func == 0:
        print("bad value:", f.values[1])

    if not isinstance(values[0], Variable):
        print("bad value:", values[0].token_value)

    alias_name = values[0].token_value
    name = values[1].token_value

    state.register_alias(alias_name, name)

    print(f"{alias_name} registered as an alias of {name}.")


if __name__ == "__main__":
    state = State()

    state.register("quit", dispatch_quit, description_quit)
    state.register("help", dispatch_help, description_help)
    state.register("alias", dispatch_alias, description_alias)
    state.register("solve_for", dispatch_solve_for, description_solve_for)
    state.register("inspect_solve", dispatch_inspect_solve, description_inspect_solve)
    state.register(
        "evaluate_solution", dispatch_evaluate_solution, description_evaluate_solution
    )

    state.register_alias("exit", "quit")
    state.register_alias("evalsol", "evaluate_solution")

    state.dispatch(f"solve_for({TESTING_EQUATION}, a)")
    state.dispatch("inspect_solve 1")

    state.dispatch("help")

    while state.loop_flag:
        text = input(state.prompt)
        state.dispatch(text)
