import argparse
import sys
import logging


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(prog="rsc-toolkit", description="Respiratory Sound Classification Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    args = parser.parse_args()

