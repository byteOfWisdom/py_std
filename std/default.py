error_bar_def = {"fmt": " ", "elinewidth": 0.75, "capsize": 2}


def plt_pretty(xlabel, ylabel):
    from matplotlib import pyplot as plt
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plt_finish(xlabel, ylabel, save_to=False):
    from matplotlib import pyplot as plt
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()
