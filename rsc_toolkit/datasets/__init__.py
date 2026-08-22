import logging

from rsc_toolkit.datasets.factory import DatasetFactory
import pathlib
from typing import Optional

from rsc_toolkit.datasets.paths import resolve_dataset_dir


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

def _configure_processor_raw_data_directory(
    *,
    processor,
    dataset_name: str,
    dataset_dir: pathlib.Path,
) -> None:
    if not hasattr(processor, "raw_data_directory"):
        return

    processor.raw_data_directory = str(dataset_dir.resolve())


def build_parser(subparsers):
    """Attach the 'dataset' subcommand and its children."""

    dataset_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    download_parser = dataset_subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    download_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    download_parser.add_argument("--data_dir", default=None, help="Dataset directory (defaults to OS user data dir)")
    download_parser.add_argument("--force", action="store_true", help="Despite valid dataset, re-download.")

    preprocess_parser = dataset_subparsers.add_parser(
        "preprocess", 
        help="Preprocess dataset features (e.g., CLAP embeddings)"
    )
    preprocess_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    preprocess_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    preprocess_parser.add_argument("--force", action="store_true", help="Re-preprocess even if cached.")
    preprocess_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    preprocess_parser.add_argument("--batch_size", type=int, default=None, help="Batch size for GPU processing")
    preprocess_parser.add_argument(
        "--data_dir",
        default=None,
        help="Dataset directory (defaults to OS user data dir)",
    )
    preprocess_parser.add_argument("--sample_level", choices=["cycle", "clip"], default="cycle")
    preprocess_parser.add_argument("--cache_dir", default=None, help="Override feature cache directory")

    index_parser = dataset_subparsers.add_parser(
        "build-index",
        help="Build dataset sample index",
    )
    index_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    index_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    index_parser.add_argument("--data_dir", default=None, help="Dataset directory (defaults to OS user data dir)")
    index_parser.add_argument("--force", action="store_true", help="Rebuild index even if present.")
    index_parser.add_argument("--sample_level", choices=["cycle", "clip"], default="cycle")

    validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate downloaded and raw dataset files",
    )
    validate_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    validate_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    validate_parser.add_argument("--data_dir", default=None, help="Dataset directory (defaults to OS user data dir)")
    validate_parser.add_argument("--force", action="store_true", help="Dummy..")

    return dataset_parser


def create_dataset(args):
    project_root_arg = getattr(args, "project_root", None)
    project_root = (
        pathlib.Path(project_root_arg).expanduser().resolve()
        if project_root_arg
        else _default_project_root()
    )
    dataset_dir = resolve_dataset_dir(
        args.dataset,
        dataset_dir=getattr(args, "data_dir", None),
        project_root=project_root,
    )
    dataset_class = DatasetFactory.get_dataset_class(dataset_name=args.dataset)
    downloader = dataset_class.make_downloader(outputs_directory=str(dataset_dir), force=args.force)
    processor = dataset_class.make_processor(outputs_directory=str(dataset_dir), force=args.force)
    dataset = dataset_class(
        downloader=downloader,
        processor=processor,
    )
    return dataset

def handle_command(args) -> int:
    """서브커맨드를 실행하고 프로세스 종료 코드(0=성공)를 돌려준다."""
    project_root_arg = getattr(args, "project_root", None)
    project_root = (
        pathlib.Path(project_root_arg).expanduser().resolve()
        if project_root_arg
        else _default_project_root()
    )
    dataset = create_dataset(args)
    dataset_dir = resolve_dataset_dir(
        args.dataset,
        dataset_dir=getattr(args, "data_dir", None),
        project_root=project_root,
    )
    
    if args.dataset_command == "download":
        dataset.download()
        return 0
    elif args.dataset_command == "preprocess":
        processor = dataset.processor

        _configure_processor_raw_data_directory(
            processor=processor,
            dataset_name=args.dataset,
            dataset_dir=dataset_dir,
        )

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
        return 0
    elif args.dataset_command == "build-index":
        processor = dataset.processor

        _configure_processor_raw_data_directory(
            processor=processor,
            dataset_name=args.dataset,
            dataset_dir=dataset_dir,
        )

        if hasattr(processor, "build_index"):
            processor.build_index(sample_level=args.sample_level)
        else:
            raise NotImplementedError("Processor does not support index building")
        return 0
    elif args.dataset_command == "validate":
        processor = dataset.processor

        _configure_processor_raw_data_directory(
            processor=processor,
            dataset_name=args.dataset,
            dataset_dir=dataset_dir,
        )

        downloaded_ok = dataset.downloader.validate_downloaded_files()
        processed_ok = processor.validate_processed_files()

        if downloaded_ok:
            logger.info("✅ 다운로드 파일 검증 통과")
        else:
            logger.error("❌ 다운로드 파일 검증 실패 (%s)", dataset_dir)

        if processed_ok is None:
            logger.warning("⏭️  전처리 산출물 없음 — 아직 build-index 를 안 했거나 검증 미구현 (실패 아님)")
        elif processed_ok:
            logger.info("✅ 전처리 산출물 검증 통과")
        else:
            logger.error("❌ 전처리 산출물 검증 실패")

        failed = (not downloaded_ok) or (processed_ok is False)
        if failed:
            logger.error("❌ 데이터셋 '%s' 검증 실패", args.dataset)
            return 1

        logger.info("✅ 데이터셋 '%s' 검증 통과", args.dataset)
        return 0

    raise NotImplementedError(args.dataset_command)
