from expression_tree_builder2 import *

def compare_values (a: TokenValue, b: TokenValue):
    if compare_variables(a, b):
        return True
    
    # compare numeric values
    if a.token_type & TT_Numeric and b.token_type & TT_Numeric and a.token_value == b.token_value:
        return True
    
    return False

def compare_variables(a: TokenValue, b: TokenValue) -> bool:
    if a == b:
        return True

    if isinstance(a, Variable) and isinstance(b, Variable):
        return a.token_value == b.token_value

    return False
