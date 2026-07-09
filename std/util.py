from propeller import GenericOp
import propeller as p


def none(x):
    return isinstance(x, type(None))


def some(x):
    return not none(x)


def load_csv(filename, delimiter=" ", skiprows=0):
    import numpy as np
    with open(filename, "r") as file:
        lines = file.readlines()[skiprows:]
        lines = list(map(lambda x: x.strip(), lines))
        lines = list(filter(lambda x: x.strip() != "", lines))
        elements = [list(map(p.from_string, line.split(delimiter))) for line in lines]
        return np.transpose(np.array(elements))


def readfile(fname, lines=True, binary=False):
    if binary:
        print('no!')
        return None
    with open(fname, "r") as handle:
        content = handle.readlines()
        return content if lines else "".join(content)


def write_file(fname, content):
    with open(fname, "w") as handle:
        handle.write(content)


def si_string(varname, ev, unit):
    return "$" + varname + " = \\SI{" + ev.format() + "}{" + unit + "}$"


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


def csvable(x):
    if isinstance(x, p.ErrVal):
        return x.format()
    else:
        return str(round(x, 4))


def print_csv_table(data, file):
    keys = data.keys()
    rows = max(map(len, data.values()))
    eol = "\n"
    content = " ".join(keys) + eol
    for i in range(rows):
        content += " ".join([csvable(data[k][i]) if i < len(data[k]) else "" for k in keys])
        content += eol

    with open(file, "w") as handle:
        handle.write(content)
        


# def default_plot(x, y, label=None):
#     if isinstance(x[0])
