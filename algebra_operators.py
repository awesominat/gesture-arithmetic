
def evaluate_expression(expression: str) -> float:
    """ Evaluates an expression using PEMDAS """

    if not expression:
        return 0.0

    parts = expression.split('+')
    summation = 0.0
    for part in parts:
        if 'x' in part:
            terms = part.split('x')
            product = 1.0
            for term in terms:
                product *= int(term)
            summation += product
        else:
            summation += int(part)
    return summation


def eval_if_allowed(previous, to_add):
    if isinstance(to_add, int):
        return True
    elif to_add == '+' or to_add == 'x':
        return len(previous) > 0 and previous[-1].isdigit()
    else:
        return False
