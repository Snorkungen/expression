import unittest
from expression_parser import *

class ParserTester(unittest.TestCase):
    def test_read_numeric(self):
        self.assertTrue(parse("  12.012  ")[0][0] & TT_Numeric, "#1")
        self.assertTrue(parse("  1212  ")[0][0] & TT_Numeric, "#2")
        self.assertTrue(parse("  12.012  ")[0][0] & TT_Float, "#3")
        self.assertTrue(parse("  1212.")[0][0] & TT_Float, "#4")
        self.assertTrue(parse("  1212  ")[0][0] & TT_Int, "#5")
        # get the above all working firtst
        # self.assertTrue(parse("-1212  ")[0][0] & TT_Int, "#6")
        # self.assertTrue(parse("+1212")[0][0] & TT_Int, "#7")
        # self.assertTrue(parse("+1212.1")[0][0] & TT_Float, "#8")

    def test_read_ident(self):
        self.assertTrue(
            parse("a")[0][0] == TT_Ident,
        )
        self.assertTrue(parse("abba")[0][0] == TT_Ident)

    def test_read_operator(self):
        self.assertTrue(parse("1+2")[1][0] & TT_Operation)

    def test_cast_double_multiplication_to_exponent(self):
        self.assertTrue(parse(" 1 ** 2")[1][0] & TT_Exponent)

    def test_implicit_multiplication(self):
        tokens = parse("5a")
        self.assertTrue(tokens[0][0] & TT_Int)
        self.assertTrue(tokens[1][0] & TT_Mult)
        self.assertTrue(tokens[2][0] & TT_Ident)
        # there are different situation where this is doing things

    def test_brackets(self):
        tokens = parse("(a + 2 (2 * 2))")
        self.assertEqual(tokens[0][0], TT_Tokens)
        self.assertEqual(len(tokens), 1)
        tokens = parse(
            "[a + 2 {2 * 2+ (1)}]"
        )  # TODO: support different bracket types []{}
        # self.assertEqual(tokens[0][0], TT_Tokens)
        # self.assertEqual(len(tokens), 1)
        # self.assertEqual(tokens[0][1][0][0], TT_Tokens)
        # self.assertEqual(len(tokens[0][1]), 1)
        # self.assertEqual(len(tokens[0][1][0][1]), 1)


if __name__ == "__main__":
    unittest.main()
