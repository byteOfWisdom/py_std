from propeller import GenericOp

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


def texify(value):
    if isinstance(value, GenericOp):
        return ("$ \\num{" + value.format() + "}$").replace("(inf)", "")
    if isinstance(value, float):
        return str(round(value, 5))
    return str(value)


def print_tex_table(data, file):
    keys = data.keys()
    rows = max(map(len, data.values()))
    eol = "\\\\\n"
    content = "&".join(keys) + eol
    content += "\\hline\n"
    for i in range(rows):
        content += "&".join([texify(data[k][i]) if i < len(data[k]) else "" for k in keys])
        content += eol
    content += "\\hline"

    with open(file, "w") as handle:
        handle.write(content)


def print_csv_table(data, file):
    keys = data.keys()
    rows = max(map(len, data.values()))
    eol = "\n"
    content = "&".join(keys) + eol
    for i in range(rows):
        content += "; ".join([texify(data[k][i]) if i < len(data[k]) else "" for k in keys])
        content += eol

    with open(file, "w") as handle:
        handle.write(content)
        
