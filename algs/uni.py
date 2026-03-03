import tokenize
from io import StringIO
from typing import List, Dict

""" Psuedocode
def unification( exp1, exp2):
    # base cases
    if exp1 and exp2 are constants or the empty list:
        if exp1 = exp2 then return {}
        else return FAIL

    if exp1 is a variable:
        if exp1 occurs in exp2 then return FAIL
        else return {exp1/exp2}

    if exp2 is a variable:
        if exp2 occurs in exp1 then return FAIL
        else return {exp2/exp1}

    # inductive step
    first1 = first element of exp1
    first2 = first element of exp2

    result1 = unification( first1, first2)
    if result1 = FAIL then return FAIL

    apply result1 to rest of exp1 and exp2

    result2 = unification( rest of exp1, rest of exp2)
    if result2 = FAIL then return FAIL
    return composition of result1 and result2
"""


def atom(next, token):
    if token[1] == "(":
        out = []
        token = next()
        while token[1] != ")":
            out.append(atom(next, token))
            token = next()
            if token[1] == " ":
                token = next()
        return out
    elif token[1] == "?":
        token = next()
        return "?" + token[1]
    else:
        return token[1]


def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())


def is_variable(exp):
    return isinstance(exp, str) and exp[0] == "?"


def is_constant(exp):
    return isinstance(exp, str) and not is_variable(exp)


def is_equal(exp1, exp2):
    # expressions must be literally the same (identical) except if one or the other (or both) has a variable in that "spot".
    if len(exp1) != len(exp2):
        return False

    for i, j in zip(exp1, exp2):
        if is_variable(i) or is_variable(j):
            continue
        if i != j:
            return False
    return True


def occurs(variable, expression):
    return variable in expression


def split_first(exp):
    return exp[0], exp[1:]


def apply(result, list_expression1, list_expression2):
    list_expression1 = [result[i] if is_constant(i) and i in result else i for i in list_expression1]
    list_expression2 = [result[i] if is_constant(i) and i in result else i for i in list_expression2]
    return list_expression1, list_expression2


def composition(result1, result2, debug=False):
    result = result1
    if result2:
        result.update(result2)
    if debug:
        print("result: ", result1, result2, result)
    return result


def deatomize(exp):
    return "(" + " ".join(exp) + ")"


def assign(variable, expression):
    if isinstance(expression, list):
        expression = deatomize(expression)
    return {variable: expression}


def unification(list_expression1, list_expression2, debug=False) -> None | Dict:
    if debug:
        print(list_expression1, list_expression2)

    if (is_constant(list_expression1) and is_constant(list_expression2)) or (len(list_expression1) == 0 or len(list_expression2) == 0):
        if is_equal(list_expression1, list_expression2):
            return {}
        return None

    if is_variable(list_expression1):
        if occurs(list_expression1, list_expression2):
            return None
        return assign(list_expression1, list_expression2)

    if is_variable(list_expression2):
        if occurs(list_expression2, list_expression1):
            return None
        return assign(list_expression2, list_expression1)

    first1, rest1 = split_first(list_expression1)
    first2, rest2 = split_first(list_expression2)

    result1 = unification(first1, first2)

    if result1 is None:
        return None

    list_expression1, list_expression2 = apply(result1, list_expression1, list_expression2)

    result2 = unification(rest1, rest2, debug)
    if result2 is None:
        return None

    if result1 != result2 and any(key in result1 for key in result2):  # Contradiction
        if debug:
            print(f"Contradiction: {result1} {result2}")
        return None

    return composition(result1, result2, debug)


def list_check(parsed_expression):
    if isinstance(parsed_expression, list):
        return parsed_expression
    return [parsed_expression]


def unify(s_expression1, s_expression2):
    list_expression1 = list_check(parse(s_expression1))
    list_expression2 = list_check(parse(s_expression2))
    return unification(list_expression1, list_expression2)


self_check_test_cases = [
    ["(son Barney Barney)", "(daughter Wilma Pebbles)", None],
    ["(knowns John ?x)", "(knowns John Jane)", {"?x": "Jane"}],
    ["(knowns John ?x)", "(knowns ?y Bill)", {"?x": "Bill", "?y": "John"}],
    ["(knowns John ?x)", "(knowns ?y (mother ?y))", {"?y": "John", "?x": "(mother ?y)"}],
    ["(knowns John ?x)", "(knowns Elizabeth ?x)", None],
    ["(knowns John ?x)", "(knowns ?x Elizabeth)", None],
    ["(son Barney Bam_Bam)", "(son ?y (son Barney))", None],
    ["(loves Fred Fred)", "(loves ?x ?x)", {"?x": "Fred"}],
    ["(loves George Fred)", "(loves ?y ?y)", None],
]

for case in self_check_test_cases:
    exp1, exp2, expected = case
    actual = unify(exp1, exp2)
    print(f"actual = {actual}")
    print(f"expected = {expected}")
    print("\n")
    assert expected == actual
