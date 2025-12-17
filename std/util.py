def none(x):
    return isinstance(x, type(None))


def some(x):
    return not none(x)


def readfile(fname, lines=True, binary=False):
    if binary:
        print('no!')
        return None
    with open(fname, "r") as handle:
        content = handle.readlines()
        return content if lines else "".join(content)
