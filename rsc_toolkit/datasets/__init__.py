from rsc_toolkit.datasets.factory import DatasetFactory


DatasetFactory.register_datasets()


def build_parser(subparsers):
    """Attach the 'dataset' subcommand and its children."""

    dataset_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    download_parser = dataset_subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    download_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    download_parser.add_argument("--force", action="store_true", help="Despite valid dataset, re-download.")

    return dataset_parser


def create_dataset(args):
    dataset_class = DatasetFactory.get_dataset_class(dataset_name=args.dataset)
    downloader = dataset_class.make_downloader(outputs_directory=args.datasets_directory, force=args.force)
    processor = dataset_class.make_processor(outputs_directory=args.datasets_directory, force=args.force)
    dataset = dataset_class(
        downloader=downloader,
        processor=processor,
    )
    return dataset

def handle_command(args) -> None:
    dataset = create_dataset(args)
    dataset.download()
