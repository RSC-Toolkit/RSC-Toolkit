import logging

from rsc_toolkit.datasets.factory import DatasetFactory
import pathlib
from typing import Optional


DatasetFactory.register_datasets()
logger = logging.getLogger(__name__)

def _find_project_root(start: pathlib.Path) -> Optional[pathlib.Path]:
    for p in (start.resolve(), *start.resolve().parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return None


def _default_project_root() -> pathlib.Path:
    # Prefer resolving from the current working directory (useful in a repo checkout),
    # otherwise fall back to the installed package location.
    cwd_root = _find_project_root(pathlib.Path.cwd())
    if cwd_root is not None:
        return cwd_root
    pkg_root = _find_project_root(pathlib.Path(__file__).resolve())
    if pkg_root is not None:
        return pkg_root
    return pathlib.Path.cwd().resolve()


def _resolve_under_root(project_root: pathlib.Path, path: str) -> str:
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())


def build_parser(subparsers):
    """Attach the 'dataset' subcommand and its children."""

    dataset_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    download_parser = dataset_subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    download_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    download_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    download_parser.add_argument("--force", action="store_true", help="Despite valid dataset, re-download.")

    preprocess_parser = dataset_subparsers.add_parser(
        "preprocess", 
        help="Preprocess dataset features (e.g., CLAP embeddings)"
    )
    preprocess_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    preprocess_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    preprocess_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    preprocess_parser.add_argument("--force", action="store_true", help="Re-preprocess even if cached.")
    preprocess_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    preprocess_parser.add_argument("--batch_size", type=int, default=None, help="Batch size for GPU processing")
    preprocess_parser.add_argument("--raw_data_directory", default=None)
    preprocess_parser.add_argument("--sample_level", choices=["cycle", "clip"], default="cycle")
    preprocess_parser.add_argument("--cache_dir", default=None, help="Override feature cache directory")

    index_parser = dataset_subparsers.add_parser(
        "build-index",
        help="Build dataset sample index",
    )
    index_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    index_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    index_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    index_parser.add_argument("--force", action="store_true", help="Rebuild index even if present.")
    index_parser.add_argument("--sample_level", choices=["cycle", "clip"], default="cycle")
    index_parser.add_argument("--raw_data_directory", default=None)

    validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate downloaded and raw dataset files",
    )
    validate_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    validate_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    validate_parser.add_argument("--datasets_directory", default="downloaded_datasets")
    validate_parser.add_argument("--force", action="store_true", help="Dummy..")
    validate_parser.add_argument("--raw_data_directory", default=None)

    return dataset_parser


def create_dataset(args):
    project_root_arg = getattr(args, "project_root", None)
    project_root = (
        pathlib.Path(project_root_arg).expanduser().resolve()
        if project_root_arg
        else _default_project_root()
    )
    datasets_directory = _resolve_under_root(project_root, args.datasets_directory)
    dataset_class = DatasetFactory.get_dataset_class(dataset_name=args.dataset)
    downloader = dataset_class.make_downloader(outputs_directory=datasets_directory, force=args.force)
    processor = dataset_class.make_processor(outputs_directory=datasets_directory, force=args.force)
    dataset = dataset_class(
        downloader=downloader,
        processor=processor,
    )
    return dataset

def handle_command(args) -> None:
    project_root_arg = getattr(args, "project_root", None)
    project_root = (
        pathlib.Path(project_root_arg).expanduser().resolve()
        if project_root_arg
        else _default_project_root()
    )
    dataset = create_dataset(args)
    
    if args.dataset_command == "download":
        dataset.download()
        return
    elif args.dataset_command == "preprocess":
        processor = dataset.processor

        if getattr(args, "raw_data_directory", None) and hasattr(processor, "raw_data_directory"):
            processor.raw_data_directory = _resolve_under_root(project_root, args.raw_data_directory)

        if not getattr(processor, "feature_processor", None):
            raise RuntimeError("preprocess requires a feature_processor to be configured")

        # Ensure index exists; build if missing
        index_path = processor.index_dir_path / f"index_{args.sample_level}.csv"
        if not index_path.exists() or args.force:
            processor.build_index(sample_level=args.sample_level)
        processor.preprocess_features_from_index(
            index_csv=index_path,
            sample_level=args.sample_level,
            batch_size=args.batch_size or 1,
            num_workers=args.workers,
            cache_dir=_resolve_under_root(project_root, args.cache_dir) if args.cache_dir else None,
        )
        return
    elif args.dataset_command == "build-index":
        processor = dataset.processor

        if args.raw_data_directory and hasattr(processor, "raw_data_directory"):
            processor.raw_data_directory = _resolve_under_root(project_root, args.raw_data_directory)

        if hasattr(processor, "build_index"):
            processor.build_index(sample_level=args.sample_level)
        else:
            raise NotImplementedError("Processor does not support index building")
        return
    elif args.dataset_command == "validate":
        processor = dataset.processor

        if args.raw_data_directory and hasattr(processor, "raw_data_directory"):
            processor.raw_data_directory = _resolve_under_root(project_root, args.raw_data_directory)

        dataset.downloader.validate_downloaded_files()
        processor.validate_processed_files()

        logger.info("✅ Dataset validation passed")
        return

    raise NotImplementedError(args.dataset_command)
