from typing import Any, Callable, Iterable, Tuple, TypedDict, Union, List, Dict
from importlib import reload as importlib_reload
import expression_parser as parser
import expression_tree_builder2 as tree_builder2
import expression_solve2 as solve2


class RegisteredFunctionDescription(TypedDict):
    short_description: str
    signatures: Iterable[str]


class State:
    loop_flag: bool
    history: List[str]

    prompt: str

    env: Dict[str, Any]
    functions: Dict[
        str,
        Union[
            Tuple[
                Callable[[Any, tree_builder2.Function, Iterable[parser.Token]], None],
                RegisteredFunctionDescription,
            ],
            str,
        ],
    ]

    def __init__(self) -> None:
        self.loop_flag = True
        self.env = {}
        self.functions = {}
        self.history = []
        self.prompt = "enter:"

        try:
            import readline

            readline.parse_and_bind("tab: complete")
            readline.set_completer(self.tab_completer)
        except:
            pass

    @property
    def reserved_identities(self):
        return {k: parser.TT_Ident | parser.TT_Func for k in self.functions}

    def dispatch(self, text: str):
        if not text.strip():
            return

        tokens = parser.parse(text, additional_identities=self.reserved_identities)

        if len(tokens) < 1 or tokens[0][0] & parser.TT_Func == 0:
            print("unrecognized value:", text)
            return

        self.history.append(text)
        try:
            node = tree_builder2.build_tree2(tokens)
        except:
            print("failed to read:", text)
            return

        if not (
            node.token_type & parser.TT_Func
            and isinstance(node, tree_builder2.Function)
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
        except BaseException as e:
            print("function failed")
            print(e)

    def register_alias(self, alias_name: str, name: str):
        self.functions[alias_name] = name

    def register(
        self,
        name: str,
        dispatch: Callable[[Any, tree_builder2.Function], None],
        description: RegisteredFunctionDescription,
    ):
        self.functions[name] = (dispatch, description)

    def tab_completer(self, text: str, state: int):
        return tuple(filter(lambda key: key.startswith(text), self.functions))[state]

    def reload_dependencies(self):
        importlib_reload(parser)
        importlib_reload(tree_builder2)
        importlib_reload(solve2)


ENV_SOLVE_ACTIONS = "solve_actions"
TESTING_EQUATION = "5 * (a + 2) = (8 / a) * a"


description_solve_for: RegisteredFunctionDescription = {
    "name": "solve_for",
    "short_description": "solve an equation",
    "signatures": ("solve_for(c = a + b, a)",),
}


def dispatch_solve_for(state: State, node: tree_builder2.Function):
    assert len(node.values) == 2
    assert isinstance(node.values[0], tree_builder2.Operation)
    assert isinstance(node.values[1], tree_builder2.Variable)

    solve_action_list: List[solve2.SolveActionEntry] = []
    try:
        solved = solve2.solve_for2(node.values[0], node.values[1], solve_action_list)
    except NotImplemented:
        print("the current solver is not capable of solving the given equation")
        print("TODO: put last derived global value [here]")

    print("\nSource: ", node.values[0])
    print("Solved: ", solved)

    # do some crunching

    actions: List[Tuple[solve2.SolveActionEntry, Iterable[solve2.SolveActionEntry]]] = (
        []
    )
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
    "signatures": ("inspect_solve (step_number)", "inspect_solve (1)"),
}


def dispatch_inspect_solve(state: State, f: tree_builder2.Function):
    if not ENV_SOLVE_ACTIONS in state.env:
        return

    # always select the most recent
    actions = state.env[ENV_SOLVE_ACTIONS][-1]

    for value in f.values:
        if value.token_type & parser.TT_Int == 0:
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


def dispatch_evaluate_solution(_, node: tree_builder2.Function):
    assert len(node.values) == 2
    assert isinstance(node.values[0], tree_builder2.Operation)
    assert isinstance(node.values[1], tree_builder2.Operation)
    source, solved = node.values

    print(solve2.evaluate_solution(source, solved))


description_quit: RegisteredFunctionDescription = {
    "short_description": "exit the program",
    "signatures": ("quit",),
}


def dispatch_quit(state: State, _: tree_builder2.Function):
    state.loop_flag = False
    print("Bye!")


description_help: RegisteredFunctionDescription = {
    "short_description": "show the available functions",
    "signatures": ("help", "help quit"),
}


def dispatch_help(state: State, f: tree_builder2.Function):
    if len(f.values) > 0:
        for value in f.values:
            if (
                isinstance(value, tree_builder2.Function)
                and value.token_value in state.functions
            ):
                if isinstance(state.functions[value.token_value], str):
                    value.token_value = state.functions[value.token_value]

                print("Signatures for:", value.token_value)
                for signature in state.functions[value.token_value][1]["signatures"]:
                    print(signature)
        else:
            print("-" * len(value.token_value))
            return

    for name in state.functions:
        if isinstance(state.functions[name], str):
            continue
            name = state.functions[name]
        print(f"{name} - {state.functions[name][1]['short_description']}")


description_alias: RegisteredFunctionDescription = {
    "short_description": "create a alias for a function",
    "signatures": ("alias(alias, function_to_alias)", "alias(qq, quit)"),
}


def dispatch_alias(state: State, f: tree_builder2.Function):
    values = list(f.values)
    if len(values) < 2:
        state.dispatch(f"help({f.name})")
        print()

        for func in state.functions:
            if isinstance(state.functions[func], str):
                print(func, "-", state.functions[func])

        return

    if values[1].token_type & parser.TT_Func == 0:
        print("bad value:", f.values[1])

    if not isinstance(values[0], tree_builder2.Variable):
        print("bad value:", values[0].token_value)

    alias_name = values[0].token_value
    name = values[1].token_value

    state.register_alias(alias_name, name)

    print(f"{alias_name} registered as an alias of {name}.")


description_reflect_parser: RegisteredFunctionDescription = {
    "short_description": "print the output of the parser",
    "signatures": ("reflect_parser (*parameters)",),
}


def dispatch_reflect_parser(state: State, _):
    text = state.history[-1]
    tokens = parser.parse(text, state.reserved_identities)

    assert (
        tokens[0][0] & parser.TT_Func
    ), "How is this function even called if this assertion fails"

    if tokens[1][0] & parser.TT_Tokens == 0:
        print("ERROR: invalid input")
        state.dispatch("help reflect_parser")
        return

    tokens = tokens[1][1]
    print(tokens)


description_reflect_token_type: RegisteredFunctionDescription = {
    "short_description": "print information about the given token types, bit-fields",
    "signatures": (
        "reflect_token_type (*parameters)",
        "reflect_token_type (64, 4650, 146)",
        "reflect_token_type 65568",
    ),
}


def dispatch_reflect_token_type(_, func):
    assert isinstance(func, tree_builder2.Function)

    BIT_FIELD_LEN = 20

    for value in func.values:
        if not isinstance(value, tree_builder2.Integer):
            continue

        print(f"{value.token_value:b}".zfill(BIT_FIELD_LEN), "-", value.token_value)

        info: List[str] = []
        flags = value.token_value

        if flags & parser.TT_Numeric:
            info.append("Numeric")
            if flags & parser.TT_Int:
                info.append("Integer")
            elif flags & parser.TT_Float:
                info.append("Float")
            if flags & parser.TT_Numeric_Positive:
                info.append("Explicitly Positive")
            if flags & parser.TT_Numeric_Negative:
                info.append("Explicitly Negative")
        elif flags & parser.TT_Ident:
            info.append("Identitiy")
            if flags & parser.TT_Operation:
                info.append("Operation")
                if flags & parser.TT_Add:
                    info.append("Addition")
                if flags & parser.TT_Sub:
                    info.append("Subtraction")
                if flags & parser.TT_Mult:
                    info.append("Multiplication")
                if flags & parser.TT_Div:
                    info.append("Division")
                if flags & parser.TT_Exponent:
                    info.append("Exponentiation")
                if flags & parser.TT_Equ:
                    info.append("Equality")

                if flags & parser.TT_Operation_Commutative:
                    info.append("Commutative")

                if (flags & parser.TT_OOO_MASK) == 1:
                    info.append("TT_OOO_ADD=1")
                elif (flags & parser.TT_OOO_MASK) == 2:
                    info.append("TT_OOO_Mult=2")
                elif (flags & parser.TT_OOO_MASK) == 3:
                    info.append("TT_OOO_Expo=3")
            if flags & parser.TT_Func:
                info.append("Function")
            if flags & parser.TT_Comma:
                info.append("Comma")
        elif flags & parser.TT_Tokens:
            info.append("Tokens")
        elif flags & parser.TT_RESERVED_0:
            info.append("RESERVED 0")
        elif flags & parser.TT_RESERVED_1:
            info.append("RESERVED 1")
        elif flags & parser.TT_RESERVED_2:
            info.append("RESERVED 2")
        elif flags & parser.TT_RESERVED_3:
            info.append("RESERVED 3")
        print(info)


description_reflect_tree: RegisteredFunctionDescription = {
    "short_description": "print the shape of the constructed tree",
    "signatures": ("reflect_tree (*parameters)",),
}


def dispatch_reflect_tree(state: State, func: tree_builder2.Function):
    assert isinstance(func, tree_builder2.Function)

    def print_token_value(
        node: tree_builder2.TokenValue, level_markers: Iterable[bool] = []
    ):
        marker_char = (
            "├──" if len(level_markers) <= 0 or not level_markers[-1] else "└──"
        )
        pipe_char = "│"
        pad_char = " "

        left_text = ""
        for b in level_markers[:-1]:
            if not b:
                left_text += pipe_char + pad_char * (len(marker_char) - len(pipe_char))
            else:
                left_text += pad_char * len(marker_char)
        if len(level_markers) > 0:
            left_text += marker_char

        node_str = f"{type(node).__name__}[{node.token_value}]"
        # node_str = f"{type(node).__name__}[{node.token_value}] ({node})"

        print(left_text + node_str)

        if "values" not in vars(node):
            return

        for i, value in enumerate(node.values):
            is_last = len(node.values) - 1 == i
            print_token_value(
                value,
                level_markers=[*level_markers, is_last],
            )

    print_token_value(func)

    print()
    print(func)


description_reloaddeps: RegisteredFunctionDescription = {
    "short_description": "reload dependencies",
    "signatures": ("reloaddeps",),
}


def dispatch_reloaddeps(state: State, _):
    state.reload_dependencies()


if __name__ == "__main__":
    # initialize a state instance
    state = State()

    # Register base functionality
    state.register("quit", dispatch_quit, description_quit)
    state.register("help", dispatch_help, description_help)
    state.register("alias", dispatch_alias, description_alias)
    state.register("reloaddeps", dispatch_reloaddeps, description_reloaddeps)

    # Register reflection functions
    state.register(
        "reflect_parser", dispatch_reflect_parser, description_reflect_parser
    )
    state.register(
        "reflect_token_type",
        dispatch_reflect_token_type,
        description_reflect_token_type,
    )
    state.register("reflect_tree", dispatch_reflect_tree, description_reflect_tree)

    # Register solve2 functions
    state.register("solve_for", dispatch_solve_for, description_solve_for)
    state.register("inspect_solve", dispatch_inspect_solve, description_inspect_solve)
    state.register(
        "evaluate_solution", dispatch_evaluate_solution, description_evaluate_solution
    )

    # Register aliases
    state.register_alias("exit", "quit")
    state.register_alias("evalsol", "evaluate_solution")
    state.register_alias("solve", "solve_for")
    state.register_alias("reflect_tt", "reflect_token_type")

    # Call the help function
    state.dispatch("help")

    while state.loop_flag:
        text = input(state.prompt)
        state.dispatch(text)
