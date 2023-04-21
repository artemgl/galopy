

def print_progress_bar(best_fitness, length=10, percentage=0., reprint=False):
    filled = int(length * percentage)
    s = "|" + "█" * filled + " " * (length - filled) + f"| {100. * percentage:.2f}%" + f"  Best fitness: {best_fitness}"
    if reprint:
        s = "\r" + s

    print(s, end='')
