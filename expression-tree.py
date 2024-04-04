from typing import Tuple, Any


expr_1 = "a = 1 + b"

"""
 a = 2b
 a   2 * b
"""

CONSTANT = 0
VARIABLE = 1
EXPRESSION = 2

OPERANDS_OPPOSITE = {"+": "-", "-": "+"}  # , "*": "/", "/": "*"}


class ExpressionNode:
    value: Any
    type: int

    def __init__(self, val) -> None:
        if type(val) is float or type(val) is int:
            self.type = CONSTANT
            self.value = val
        else:
            raise ValueError

    def __str__(self) -> str:
        return str(self.value)

    def clone(self):
        return ExpressionNode(self.value)


class ExpressionVariable(ExpressionNode):
    name: str

    def __init__(self, name) -> None:
        self.type = VARIABLE
        self.name = name
        self.value = ""

    def __str__(self) -> str:
        return "[" + self.value + self.name + "]"

    def clone(self):
        return ExpressionVariable(self.name)


class Expression(ExpressionNode):
    left: ExpressionNode
    right: ExpressionNode

    def __init__(
        self, t: Tuple[ExpressionNode, ExpressionNode], kind="=", parent=None
    ) -> None:
        self.type = EXPRESSION
        self.value = kind

        lhs, rhs = t
        self.left = lhs
        self.right = rhs
        self.parent = parent

        self.__walk_nodes__(self.left)
        self.__walk_nodes__(self.right)

    def __walk_nodes__(self, expr_node):
        expr_node.parent = self
        if expr_node.type != EXPRESSION:
            return
        expr_node.__walk_nodes__(expr_node.left)
        expr_node.__walk_nodes__(expr_node.right)

    def __str__(self) -> str:
        if self.value == "=":
            return str(self.left) + super().__str__() + str(self.right)
        return "(" + str(self.left) + super().__str__() + str(self.right) + ")"

    def clone(self):
        return Expression(
            (self.left.clone(), self.right.clone()), self.value, self.parent
        )

    def simplify(self):
        # add constants together
        # this can walk the tree

        # lhs and rhs are constants
        if self.left.type == CONSTANT and self.right.type == CONSTANT:
            val = 0
            if self.value == "+":
                val = self.left.value + self.right.value
            elif self.value == "-":
                val = self.left.value - self.right.value

            return ExpressionNode(val)
        else:
            c = self.clone()
            if self.left.type == EXPRESSION:
                c.left = self.left.simplify()
            if self.right.type == EXPRESSION:
                c.right = self.right.simplify()
        return c

    def solve_for(self, node: ExpressionNode):
        if node.type != VARIABLE and type(node) is ExpressionVariable:
            raise ValueError
        if self.value != "=":
            raise ValueError

        # move all occurences of the variable to the left
        # walk through tree

        clone = self.clone()
        stack = list((clone.left, clone.right))
        while len(stack) > 0:
            e = stack.pop()
            p = e.parent

            if type(e) is ExpressionVariable and e.name == node.name:
                if p.left == e:
                    continue  # no need to swap
                p.right = p.left

                if p.value == "-":
                    e.value = "-"
                    p.value = "+"
                p.left = e
            if e.type != EXPRESSION:
                continue

            stack.append(e.left)
            stack.append(e.right)

        """
            The below block just moves the variable up the tree "NOTGOOG"
        """
        # stack = [clone.right]
        # while len(stack) > 0:
        #     e = stack.pop()
        #     p: Expression = e.parent

        #     if type(e) is ExpressionVariable and e.name == node.name:
        #         if p.parent == clone:
        #             continue

        #         # move to the top left
        #         e.parent.parent.right = p.right
        #         e.parent.parent.left = Expression(
        #             (e.parent.left, e), e.parent.parent.value, e.parent.parent
        #         )

        #     if e.type != EXPRESSION:
        #         continue

        #     stack.append(e.left)
        #     stack.append(e.right)

        """
            Move the right most to the left side of the "=" sign
        """

        stack = [clone.right]
        while len(stack) > 0:
            e = stack.pop()
            if e.type == EXPRESSION:
                stack.append(e.right)
                continue
            
            if type(e) is ExpressionVariable and e.name == node.name:
                break

            # move thing to the left
            e.parent.left.parent = e.parent.parent
            e.parent.parent.right = e.parent.left
            clone.left = Expression((clone.left, e), e.parent.value, clone)

            stack = [clone.right]
        return clone


a = ExpressionVariable("a")
b = ExpressionVariable("b")

expression: list[ExpressionNode] = []

e1 = Expression(
    (
        a,
        Expression(
            (
                Expression(
                    (
                        Expression(
                            (
                                Expression(
                                    (ExpressionNode(10), ExpressionNode(10)), "+"
                                ),
                                b,
                            ),
                            "+",
                        ),
                        b,
                    ),
                    "-",
                ),
                b,
            ),
            "-",
        ),
    ),
    "=",
)

print(e1.solve_for(b).simplify(), e1)
