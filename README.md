# Expression

Something about Computer Algebra

## Get started

```sh
# download code
git clone https://github.com/Snorkugen/expression.git

# change into directory containing code
cd expression

# run interact script
# NOTE the script has only been tested on python version 3.10
python ./interact.py
```

## interact.py

> The following contains contain a slightly modified output of the interact.py script, when running the following functions **reflect_parser**, **reflect_token_type** and **reflect_tree**.

```
python ./interact.py

...

enter:reflect_parser(5 * a - b = 100)

[(144, '5'), (4650, '*'), (32, 'a'), (2593, '-'), (32, 'b'), (16928, '='), (144, '100')]

enter:reflect_token_type(144, 4650, 2593, 32, 16928)

00000000000010010000 - 144
['Numeric', 'Integer']

00000001001000101010 - 4650
['Identitiy', 'Operation', 'Multiplication', 'Commutative', 'TT_OOO_Mult=2']

00000000101000100001 - 2593
['Identitiy', 'Operation', 'Subtraction', 'TT_OOO_ADD=1']

00000000000000100000 - 32
['Identitiy']

00000100001000100000 - 16928
['Identitiy', 'Operation', 'Equality']

enter:reflect_tree(5 * a - b = 100)

Function[reflect_tree]
└──Operation[=]
   ├──Operation[+]
   │  ├──Operation[*]
   │  │  ├──Integer[5]
   │  │  └──Variable[a]
   │  └──Operation[*]
   │     ├──Integer[-1]
   │     └──Variable[b]
   └──Integer[100]

reflect_tree(5 * a + -1 * b = 100)

enter:

```

### interact.py functions

- **quit** - exit the program
- **help** - display the available functions
- **help** `function name` - display information about the given function
-
- **reflect_parser** `(expression)` - display the output of the tokenizer
- **reflect_token_type** `number` - display what does the bit-field represent
- **reflect_tree** `(expression)` - display the relationship between nodes
-
- **solve_for** `(equation, variable)` - Attempt to solve the equation for the the given variable
- **inspect_solve** `number` - display information about a specific solution step in the previous **solve_for** call
- **evaluate_solutions** `(equation, variable = expression)` - see if replacing the solution with the variable retains the truth in the equation 