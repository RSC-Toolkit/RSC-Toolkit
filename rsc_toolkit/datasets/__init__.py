import logging

from rsc_toolkit.datasets.factory import DatasetFactory


DatasetFactory.register_datasets()
logger = logging.getLogger(__name__)


def build_parser(subparsers):
    """Attach the 'dataset' subcommand and its children."""

    dataset_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    download_parser = dataset_subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    download_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    download_parser.add_argument("--force", action="store_true", help="Despite valid dataset, re-download.")

    preprocess_parser = dataset_subparsers.add_parser(
        "preprocess", 
        help="Preprocess dataset features (e.g., CLAP embeddings)"
    )
    preprocess_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    preprocess_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    preprocess_parser.add_argument("--force", action="store_true", help="Re-preprocess even if cached.")
    preprocess_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    preprocess_parser.add_argument("--batch_size", type=int, default=None, help="Batch size for GPU processing")
    preprocess_parser.add_argument("--raw_data_directory", default=None)

    index_parser = dataset_subparsers.add_parser(
        "build-index",
        help="Build dataset sample index",
    )
    index_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    index_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    index_parser.add_argument("--force", action="store_true", help="Rebuild index even if present.")
    index_parser.add_argument("--sample_level", choices=["cycle", "clip"], default="cycle")
    index_parser.add_argument("--raw_data_directory", default=None)

    validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate downloaded and raw dataset files",
    )
    validate_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    validate_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    validate_parser.add_argument("--force", action="store_true", help="Dummy..")
    validate_parser.add_argument("--raw_data_directory", default=None)

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
    
    if args.dataset_command == "download":
        dataset.download()
        return
    # elif args.dataset_command == "preprocess":
    #     # For preprocessing, we need to configure the processor
    #     # This would typically come from config, but for CLI we'll need
    #     # to set it up. For now, just process the dataset which may include
    #     # preprocessing if configured.
    #     processor = dataset.processor

    #     if getattr(args, "raw_data_directory", None) and hasattr(processor, "raw_data_directory"):
    #         processor.raw_data_directory = args.raw_data_directory
        
    #     # Update processor settings from CLI args
    #     if hasattr(processor, 'num_preprocess_workers'):
    #         processor.num_preprocess_workers = args.workers
    #     if hasattr(processor, 'preprocess_batch_size') and args.batch_size:
    #         processor.preprocess_batch_size = args.batch_size
        
    #     # Process dataset (which may trigger preprocessing)
    #     dataset.process()
    elif args.dataset_command == "build-index":
        processor = dataset.processor

        if args.raw_data_directory and hasattr(processor, "raw_data_directory"):
            processor.raw_data_directory = args.raw_data_directory

        if hasattr(processor, "build_index"):
            processor.build_index(sample_level=args.sample_level)
        else:
            raise NotImplementedError("Processor does not support index building")
        return
    elif args.dataset_command == "validate":
        processor = dataset.processor

        if args.raw_data_directory and hasattr(processor, "raw_data_directory"):
            processor.raw_data_directory = args.raw_data_directory

        dataset.downloader.validate_downloaded_files()
        processor.validate_processed_files()

        logger.info("✅ Dataset validation passed")
        return

    raise NotImplementedError(args.dataset_command)