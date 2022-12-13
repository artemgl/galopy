import time


def print_progress_bar(best_fitness, length=10, percentage=0.):
    filled = int(length * percentage)
    print("[" + "=" * filled + " " * (length - filled) + f"] {100. * percentage:.2f}%" +
          f"  Best fitness: {best_fitness}",
          end='')


def reprint_progress_bar(best_fitness, length=10, percentage=0.):
    filled = int(length * percentage)
    print("\r[" + "=" * filled + " " * (length - filled) + f"] {100. * percentage:.2f}%" +
          f"  Best fitness: {best_fitness}",
          end='')


if __name__ == "__main__":
    # print("A")
    # print("B")
    # print("C", end='')
    # time.sleep(1.)
    # print("\033[F", end='')
    # print("D")

    print_progress_bar(None, 40, 0.)
    n = 73
    for i in range(n):
        time.sleep(0.1)
        reprint_progress_bar(None, 40, (i + 1) / n)
