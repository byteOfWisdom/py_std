def pairs(iter):
    return None


def mesh(a, b, c=None, d=None):
    if some(c) and some(d):
        for first in a:
            for second in b:
                for third in c:
                    for fourth in d:
                        yield first, second, third, fourth
    elif some(c):
        for first in a:
            for second in b:
                for third in c:
                    yield first, second, third
    else:
        for first in a:
            for second in b:
                yield first, second

