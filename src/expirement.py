import argparse

def _parse_args() -> dict:
    parser = argparse.ArgumentParser()

    return vars(parser.parse_args())


if __name__ == "__main__":
    pass
