def none(x):
    return isinstance(x, None)


def some(x):
    return not none(x)
