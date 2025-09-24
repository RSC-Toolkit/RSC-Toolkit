import argparse
import sys
import logging


from rsc_toolkit import datasets


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(prog="rsc-toolkit", description="Respiratory Sound Classification Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset command
    datasets.build_parser(subparsers)

    args, remaining = parser.parse_known_args()

    # entrypoint handling here
    if args.command == "datasets":
        if args.dataset_command == "download":
            datasets.handle_command(args)
