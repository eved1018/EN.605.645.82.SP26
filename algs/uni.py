import tokenize
from io import StringIO
from typing import List, Dict, Tuple

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


def occurs(variable: str, expression: List) -> bool:
    if variable == expression:
        return True

    if isinstance(expression, list):  # recurse
        for subexpressions in expression:
            if occurs(variable, subexpressions):
                return True
    return False


def list_check(parsed_expression):
    if isinstance(parsed_expression, list):
        return parsed_expression
    return [parsed_expression]


def apply(substitutions: Dict, expression: List) -> List:
    return [substitutions[i] if is_variable(i) and i in substitutions else i for i in expression]


def composition(substitutions1: Dict, substitutions2: Dict, trace: bool = False) -> Dict | None:
    if any(x in substitutions2 for x in substitutions1):
        return None

    result = {**substitutions1, **substitutions2}
    if trace:
        print(f"New substitution List: {substitutions1} + {substitutions2} -> {result}")
    return result


def deatomize(expression: str | List) -> str | None:
    if isinstance(expression, str):  # base-case
        return expression

    if isinstance(expression, list):  # recurse
        str_expression = []
        for subexpressions in expression:
            result = deatomize(subexpressions)
            if result is None:
                return None
            str_expression.append(result)
        return "(" + " ".join(str_expression) + ")"

    return None


def assign(variable: str, expression: List | str) -> Dict:
    if isinstance(expression, list):
        expression = deatomize(expression)  # type: ignore list_expression1 is list by isinstance test
    return {variable: expression}


def split_expression(list_expression: List) -> Tuple[str, List] | None:
    if len(list_expression) == 0:
        return None
    first1, rest1 = list_expression[0], list_expression[1:]
    return first1, rest1


def is_unmodifiable(list_expression1: List | str, list_expression2: List | str) -> bool:
    if is_constant(list_expression1) and is_constant(list_expression2):
        return True

    if (isinstance(list_expression1, list) and isinstance(list_expression2, list)) and (len(list_expression1) == 0 and len(list_expression2) == 0):
        return True
    return False


def unification(list_expression1: List | str, list_expression2: List | str, trace: bool = False) -> Dict | None:
    print(f"Trying to unify {list_expression1} with {list_expression2}") if trace else None
    if is_unmodifiable(list_expression1, list_expression2):
        return {} if list_expression1 == list_expression2 else None

    if is_variable(list_expression1):
        return None if occurs(list_expression1, list_expression2) else assign(list_expression1, list_expression2)  # type: ignore list_expression1 is str by test of is_variable

    if is_variable(list_expression2):
        return None if occurs(list_expression2, list_expression1) else assign(list_expression2, list_expression1)  # type: ignore list_expression2 is str by test of is_variable

    first1, rest1 = split_expression(list_expression1)  # type: ignore list_expression1 is str by test of is_variable
    first2, rest2 = split_expression(list_expression2)  # type: ignore list_expression2 is list by test of is_variable
    result1 = unification(first1, first2, trace)
    if result1 is None:
        return None

    rest1, rest2 = apply(result1, rest1), apply(result1, rest2)
    result2 = unification(rest1, rest2, trace)
    return None if result2 is None else composition(result1, result2, trace)


def unify(s_expression1, s_expression2):
    list_expression1 = list_check(parse(s_expression1))
    list_expression2 = list_check(parse(s_expression2))
    return unification(list_expression1, list_expression2)


def run_tests(test_cases):
    for i, case in enumerate(test_cases):
        exp1, exp2, expected, message = case
        actual = unify(exp1, exp2)
        print(f"[{i + 1}] Testing {exp1} & {exp2} - {message}...")
        pre = "PASS" if actual == expected else "FAIL"
        print(f"{pre} actual = {actual} | expected = {expected}")
        # print("\n")
        assert expected == actual


self_check_test_cases = [
    ["(son Barney Barney)", "(daughter Wilma Pebbles)", None, "non-equal constants"],
    ["(Fred)", "(Barney)", None, "self check case - invalid unification of non-equal constant"],
    ["(Pebbles)", "(Pebbles)", {}, "self check case - unification of the same constant returns an empty substitution list"],
    ["(quarry_worker Fred)", "(quarry_worker ?x)", {"?x": "Fred"}, "self check case - valid unification of single variable"],
    ["(son Barney ?x)", "(son ?y Bam_Bam)", {"?y": "Barney", "?x": "Bam_Bam"}, "self check case - valid unification of two variables"],
    ["(married ?x ?y)", "(married Barney Wilma)", {"?x": "Barney", "?y": "Wilma"}, "self check case - valid unification of two variable within the same expression"],
    ["(son Barney ?x)", "(son ?y (son Barney))", {"?y": "Barney", "?x": "(son Barney)"}, "self check case - valid assignment of a variable to an expression"],
    ["(son Barney ?x)", "(son ?y (son ?y))", {"?y": "Barney", "?x": "(son ?y)"}, "self check case - valid assignment of a variable to an expression with replacement"],
    ["(son Barney Bam_Bam)", "(son ?y (son Barney))", None, "self check case -  invalid unification of a constant function to a constant"],
    ["(loves Fred Fred)", "(loves ?x ?x)", {"?x": "Fred"}, "self check case - valid substitution of the same constant to the same variable"],
    ["(future George Fred)", "(future ?y ?y)", None, "self check case - invalid substitution of the different constants to the same variable"],
]
run_tests(self_check_test_cases)