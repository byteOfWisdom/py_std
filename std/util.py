def none(x):
    return isinstance(x, type(None))


def some(x):
    return not none(x)
