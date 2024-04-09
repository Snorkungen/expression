import unittest
from expression_tree_builder import *
from expression_parser import parse


def assert_operation(self: unittest.TestCase, node: Atom, oper_flag: int):
    self.assertEqual(type(node), Operation)
    self.assertTrue(node.token_type & oper_flag)
    assert type(node) == Operation


class TreeBuilder(unittest.TestCase):
    def test_build_tree_basic(self):
        """
            =
        b      +
          1      *
             1.0   ^
                 a    1
        """
        tokens = parse("b = 1 + 1.0 * a^1")
        root = build_tree(tokens)

        self.assertEqual(type(root), Equals)

        assert_operation(self, root.right, TT_Add)
        assert_operation(self, root.right.right, TT_Mult)
        assert_operation(self, root.right.right.right, TT_Exponent)

        self.assertEqual(type(root.right.left), Int)
        self.assertEqual(type(root.right.left), Int)
        self.assertEqual(type(root.right.right.left), Float)
        self.assertEqual(type(root.right.right.right.left), Variable)

    def test_build_tree_tt_tokens(self):
        """
            *
        a       -
            1       2
        """

        tokens = parse("a * (1 - 2)")
        root = build_tree(tokens)

        assert_operation(self, root, TT_Mult)
        assert_operation(self, root.right, TT_Sub)


def b_nand(a: int, b: int) -> int:
    """I think this is a functioning nand operation"""
    return a ^ (b & a)


def assert_flattened_variable(self: unittest.TestCase, token: Tuple[int, Any]):
    # convoluted check becouse this has to be just an ident
    self.assertEquals(b_nand(token[0], TT_INFO_MASK), TT_Ident)


class FlattenTree(unittest.TestCase):
    def test_flatten_tree(self):
        """
            *
        a       -
            b       1.1

        a * (b - 1.1)
        """
        tree = build_tree(parse("a * (b - 1.1)"))
        tokens = flatten_tree(tree)

        assert_flattened_variable(self, tokens[0])
        self.assertTrue(tokens[1][0] & (TT_Operation | TT_Mult))

        self.assertTrue(b_nand(tokens[2][0], TT_INFO_MASK) == TT_Tokens)

        sub_tokens = tokens[2][1]
        assert_flattened_variable(self, sub_tokens[0])

        self.assertTrue(sub_tokens[1][0] & (TT_Operation | TT_Sub))
        self.assertTrue(sub_tokens[2][0] & (TT_Float))


if __name__ == "__main__":
    unittest.main()
