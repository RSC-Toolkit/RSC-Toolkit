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

    # create build_parser function for each entrypoint and add here
    datasets.build_parser(subparsers)

    # dataset download
    args = parser.parse_args()

    if args.command == "datasets":
        if args.dataset_command == "download":
            _dataset = datasets.DatasetFactory.get_dataset_class(args.dataset)()
            _dataset_downloader = _dataset.get_downloader(outputs_directory=args.datasets_download_path)
            _dataset_downloader.download()

