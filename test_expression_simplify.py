import unittest

from expression_parser import parse
from expression_tree_builder import build_tree
from expression_simplify import *


class Addition(unittest.TestCase):
    def test_initial(self):
        node = build_tree(parse("2 + 2 + 2 + a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "a + 6")

        node = build_tree(parse("2 + (-5)"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "-3")

        node = build_tree(parse("(-2) + 5"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "3")

        node = build_tree(parse("a + a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "2 * a")

        node = build_tree(parse("a + a + a + a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "4 * a")

        node = build_tree(parse("1 + a + 1 + a + a + 1"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "3 * a + 3")

        node = build_tree(parse("2 + 2 + 3 + 2"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "9")

        node = build_tree(parse("2 + 2 + a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "a + 4")

        node = build_tree(parse("2 + a + 2"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "a + 4")

        node = build_tree(parse("a + a + 2"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "2 * a + 2")

        node = build_tree(parse("2 + a + a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "2 * a + 2")  # the reason why is arbitrary

        node = build_tree(parse("2 + a + a + 2"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "2 * a + 4")

        node = build_tree(parse("2 + (2 + 2)"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "6")

        node = build_tree(parse("a + 2 * a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "3 * a")

        node = build_tree(parse("2 + 2 +  a + a + (2 + a + (3 * a + 3))"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "6 * a + 9")

        node = build_tree(parse("3 * a + a *2 + 1 + 1"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "5 * a + 2")

        node = build_tree(parse("1 + 1/2 + 1 + a + a"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "2 * a + 1 / 2 + 2")

        node = build_tree(parse("-2 + (-2 + 2)"))
        simplified = simplify_addition(node)
        self.assertEqual(str(simplified), "-2")


class Multiplication (unittest.TestCase):
    def test_initial (self):
        node = build_tree(parse("2 * 4 * -1"))
        simplified = simplify_multiplication(node)
        self.assertEqual(str(simplified), "-8")

        node = build_tree(parse("4 * a * -1"))
        simplified = simplify_multiplication(node)
        self.assertEqual(str(simplified), "-4 * a")

class Subtraction(unittest.TestCase):
    def test_initial(self):
        node = build_tree(parse("2 - 2"))
        simplified = simplify_subtraction(node)
        self.assertEqual(str(simplified), "0")  #   0

        node = build_tree(parse("2 - 2 - 2"))  # -2 + 2 - 2 => -2 + -2 + 2
        simplified = simplify_subtraction(node)
        self.assertEqual(str(simplified), "-2")

        # TODO: do solve multiplication
        node = build_tree(parse("-a - 2 - a - 4 * a"))  # -2 + 2 - 2 => -2 + -2 + 2
        simplified = simplify_subtraction(node)
        self.assertEqual(str(simplified), "-6 * a + -2", "This can fail because simplifying multiplication is not implemented yet.")


if __name__ == "__main__":
    unittest.main()
