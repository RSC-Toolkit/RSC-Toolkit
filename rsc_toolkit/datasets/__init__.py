import importlib
import pkgutil

from rsc_toolkit.datasets.factory import DatasetFactory


DatasetFactory.register_datasets()



def build_parser(subparsers):
    """Attach the 'dataset' subcommand and its children."""

    dataset_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    download_parser = dataset_subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    download_parser.add_argument("--datasets_download_path", default="downloaded_datasets")
    return dataset_parser



def download_dataset(dataset_name: str) -> None:
    pass
