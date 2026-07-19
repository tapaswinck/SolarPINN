from time import sleep

from solarpinn.utils.progress import ProgressBar, Spinner


def test_bar():

    bar = ProgressBar(
        total=100,
        description="Training",
    )

    for _ in range(100):
        sleep(0.03)
        bar.update()

    bar.finish()


def test_spinner():

    spinner = Spinner(
        description="Loading data",
    )

    for _ in range(60):
        sleep(0.05)
        spinner.update()

    spinner.finish()


if __name__ == "__main__":

    test_bar()

    print()

    test_spinner()