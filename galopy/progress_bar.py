

def print_progress_bar(best_fitness, length=10, percentage=0., reprint=False):
    filled = int(length * percentage)
    if reprint:
        print("\r[" + "=" * filled + " " * (length - filled) + f"] {100. * percentage:.2f}%" +
              f"  Best fitness: {best_fitness}",
              end='')
    else:
        print("[" + "=" * filled + " " * (length - filled) + f"] {100. * percentage:.2f}%" +
              f"  Best fitness: {best_fitness}",
              end='')
