import unittest

from expression_parser import parse
from expression_tree_builder import build_tree
from expression_simplify import *


def assert_simplified(self: unittest.TestCase, simplify_fn=simplify):
    def assert_equal(node_str: str, expected: str, message=None):
        node = build_tree(parse(node_str))
        simplified = simplify_fn(node)
        self.assertEqual(str(simplified), expected, message if message else node_str)
        return

    return assert_equal


class Addition(unittest.TestCase):
    def test_initial(self):
        test = assert_simplified(self, simplify_addition)

        test("2 + 2 + 2 + a", "a + 6")

        test("2 + (-5)", "-3")

        test("(-2) + 5", "3")

        test("a + a", "2 * a")

        test("a + a + a + a", "4 * a")

        test("1 + a + 1 + a + a + 1", "3 * a + 3")

        test("2 + 2 + 3 + 2", "9")

        test("2 + 2 + a", "a + 4")

        test("2 + a + 2", "a + 4")

        test("a + a + 2", "2 * a + 2")

        test("2 + a + a", "2 * a + 2")  # the reason why is arbitrary

        test("2 + a + a + 2", "2 * a + 4")

        test("2 + (2 + 2)", "6")

        test("a + 2 * a", "3 * a")

        test("2 + 2 +  a + a + (2 + a + (3 * a + 3))", "6 * a + 9")

        test("3 * a + a *2 + 1 + 1", "5 * a + 2")

        test("1 + 1/2 + 1 + a + a", "(2 * a + 1 / 2) + 2")

        test("-2 + (-2 + 2)", "-2")


class Division(unittest.TestCase):
    def test_inital(self):
        test = assert_simplified(self, simplify_division)

        test("2 / 2", "1", "[1]")
        test("2 / 4", "1 / 2", "[2]")
        test("2 / 4 / 2", "1 / 4", "[3]")
        test("(2 / 4) / (1 / 2)", "1", "[4]")
        test("1 / (1 / 1 / 2)", "2", "[5]")
        test("(2 * a) / a", "2", "[6]")
        test("(2 * a) / 2", "a", "[7]")


class Multiplication(unittest.TestCase):
    def test_initial(self):
        test = assert_simplified(self, simplify_multiplication)

        test("2 * 4 * -1", "-8")
        test("4 * a * -1", "-4 * a")
        test("a * 4 * -1", "-4 * a")
        test("4 * a * -1 * a", "-4 * a ^ 2")
        test(
            "4 * a * b * -1 * a",
            "(-4 * b) * a ^ 2",
            "this could fail due to sorting of factors not doing a satisfactory job",
        )
        test("a * a ^ 2 * b * b", "b ^ 2 * a ^ 3")

    def test_dependant_on_simplify_division(self):
        test = assert_simplified(self, simplify_multiplication)

        test("((a) * 1 / 3) * 6", "2 * a")
        test("(a / 3) * 3", "a")
        test("(a / 3) * 3 * 4", "4 * a")
        test("(a / 3) * (3 / a) * 4", "4")


class Subtraction(unittest.TestCase):
    def test_initial(self):
        test = assert_simplified(self, simplify_subtraction)

        test("2 - 2", "0")  #   0
        test("2 - 2 - 2" , "-2")  # -2 + 2 - 2 => -2 + -2 + 2
        test("-a - 2 - a - 4 * a", "-6 * a + -2")
        # TODO: think about rectifying [a + -b] => [a - b]


if __name__ == "__main__":
    unittest.main()
