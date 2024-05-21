import unittest
from expression_parser import *
from expression_tree_builder2 import *


class BuildTreeTester(unittest.TestCase):
    def test_build_tree1(self):
        tokens = parse("1 + 2 * 2")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Operation))
        self.assertTrue(node.token_type & TT_Add)
        self.assertEqual(len(node.values), 2)

        self.assertTrue(isinstance(node.values[1], Operation))
        self.assertTrue(node.values[1].token_type & TT_Mult)
        self.assertEqual(len(node.values[1].values), 2)

    def test_build_tree2(self):
        tokens = parse("(1 + 2) * 2")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Operation))
        self.assertTrue(node.token_type & TT_Mult)
        self.assertEqual(len(node.values), 2)

        self.assertTrue(isinstance(node.values[0], Operation))
        self.assertTrue(node.values[0].token_type & TT_Add)
        self.assertEqual(len(node.values[0].values), 2)

    def test_build_tree3(self):
        tokens = parse("2 / a ^ 2")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Operation))
        self.assertTrue(node.token_type & TT_Div)
        self.assertEqual(len(node.values), 2)

        self.assertTrue(isinstance(node.values[1], Operation))
        self.assertTrue(node.values[1].token_type & TT_Exponent)
        self.assertEqual(len(node.values[1].values), 2)

    def test_build_tree4(self):
        tokens = parse("2 / a ^ 2 = (a + b) * a")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Operation))
        self.assertTrue(node.token_type & TT_Equ)
        self.assertEqual(len(node.values), 2)

        self.assertTrue(isinstance(node.values[0], Operation))
        self.assertTrue(node.values[0].token_type & TT_Div)

        self.assertTrue(isinstance(node.values[1], Operation))
        self.assertTrue(node.values[1].token_type & TT_Mult)

    def test_build_tree5(self):
        tokens = parse("a + (a + a + a)")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Operation))
        self.assertTrue(node.token_type & TT_Add)
        self.assertEqual(len(node.values), 4)

    def test_build_tree_no_subtraction(self):
        tokens = parse("a - b") # a + -1 * b
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Operation))
        self.assertTrue(node.token_type & TT_Add)
        self.assertEqual(len(node.values), 2)

        self.assertTrue(isinstance(node.values[1], Operation))
        self.assertEqual(len(node.values[1].values), 2)

    def test_build_tree_function1(self):
        tokens = parse("__testfunc 2")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Function))
        self.assertEqual(len(node.values), 1)

        self.assertTrue(isinstance(node.values[0], Integer))

    def test_build_tree_function2(self):
        tokens = parse("__testfunc (a, b ,c)")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Function))
        self.assertEqual(len(node.values), 3)

        self.assertTrue(isinstance(node.values[0], Variable))
        self.assertTrue(isinstance(node.values[1], Variable))
        self.assertTrue(isinstance(node.values[2], Variable))

    def test_build_tree_function3(self):
        tokens = parse("__testfunc __testfunc a")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Function))
        self.assertEqual(len(node.values), 1)

        self.assertTrue(isinstance(node.values[0], Function))
        self.assertEqual(len(node.values[0].values), 1)
        self.assertTrue(isinstance(node.values[0].values[0], Variable))

    def test_build_tree_function31(self):
        tokens = parse("__testfunc __testfunc __testfunc (a, b)")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Function))
        self.assertEqual(len(node.values), 1)

        self.assertTrue(isinstance(node.values[0], Function))
        self.assertEqual(len(node.values[0].values), 1)

        self.assertTrue(isinstance(node.values[0].values[0], Function))
        self.assertEqual(len(node.values[0].values[0].values), 2)

    def test_build_tree_function32(self):
        tokens = parse("__testfunc __testfunc __testfunc")
        node = build_tree2(tokens)

        self.assertTrue(isinstance(node, Function))
        self.assertEqual(len(node.values), 1)

        self.assertTrue(isinstance(node.values[0], Function))
        self.assertEqual(len(node.values[0].values), 1)

        self.assertTrue(isinstance(node.values[0].values[0], Function))
        self.assertEqual(len(node.values[0].values[0].values), 0)


if __name__ == "__main__":
    unittest.main()
