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
        help="학습 config(configs/data/*.yaml) 기준으로 피처 캐시를 미리 채운다 (예: CLAP 임베딩)",
    )
    preprocess_parser.add_argument("dataset", choices=DatasetFactory.datasets.keys())
    preprocess_parser.add_argument(
        "--data_config",
        default=None,
        help="사용할 데이터 config 이름 (rsc_toolkit/configs/data/<이름>.yaml, 기본: dataset 이름)",
    )
    preprocess_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    preprocess_parser.add_argument(
        "--data_dir",
        default=None,
        help="Dataset directory (defaults to OS user data dir)",
    )
    preprocess_parser.add_argument(
        "--sample_level",
        choices=["cycle", "clip"],
        default=None,
        help="config 의 sample_level 재정의 (기본: config 값)",
    )
    preprocess_parser.add_argument("--workers", type=int, default=None, help="피처 계산 DataLoader 워커 수 (기본: config 값)")
    preprocess_parser.add_argument("--batch_size", type=int, default=None, help="피처 계산 배치 크기 (기본: config 값)")
    preprocess_parser.add_argument("--cache_dir", default=None, help="피처 캐시 디렉토리 재정의 (기본: config 값)")
    preprocess_parser.add_argument("--force", action="store_true", help="이 설정의 기존 캐시를 지우고 전부 다시 계산")

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

def _available_data_configs() -> list:
    import rsc_toolkit.configs as configs_pkg

    data_dir = pathlib.Path(configs_pkg.__file__).parent / "data"
    return sorted(p.stem for p in data_dir.glob("*.yaml"))


def _compose_data_config(data_config_name: str, project_root: pathlib.Path) -> dict:
    """
    학습이 쓰는 것과 동일한 hydra 데이터 config 를 합성해 plain dict 로 돌려준다.

    캐시 디렉토리·키가 학습(IndexDataModule)과 어긋나면 미리 뽑은 피처를
    학습이 읽지 못하므로, config 를 별도로 해석하지 않고 학습과 같은 소스를 쓴다.
    """
    try:
        from hydra import compose, initialize_config_module
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "preprocess 는 trainer 의존성이 필요하다: pip install 'rsc-toolkit[trainer]'"
        ) from exc

    available = _available_data_configs()
    if data_config_name not in available:
        raise ValueError(
            f"데이터 config '{data_config_name}' 가 없다. 사용 가능: {', '.join(available)}"
        )

    with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
        cfg = compose(config_name="default", overrides=[f"data={data_config_name}"])

    # config 의 project_root 는 ${hydra:runtime.cwd} 인데, 이 리졸버는 hydra 런타임
    # 밖(compose API)에서는 동작하지 않는다. CLI 가 정한 project_root 로 덮어쓴다.
    cfg.data.params.project_root = str(project_root)
    return OmegaConf.to_container(cfg.data, resolve=True)


def _handle_preprocess(args, *, project_root: pathlib.Path, dataset_dir: pathlib.Path) -> None:
    """
    학습 config 기준으로 피처 캐시를 미리 채운다.

    핵심 원칙: 캐시 규약(디렉토리·키·sample_id)을 여기서 복제하지 않는다.
    build_datamodule -> IndexDataModule 라는 학습 경로 그대로를 태워서,
    미리 뽑은 캐시를 학습이 반드시 재사용하게 만든다.
    """
    import shutil

    data_config_name = getattr(args, "data_config", None) or args.dataset
    data_cfg = _compose_data_config(data_config_name, project_root)

    params = data_cfg.get("params") or {}
    cfg_dataset_name = str(params.get("dataset_name", ""))
    if cfg_dataset_name != args.dataset:
        raise ValueError(
            f"데이터 config '{data_config_name}' 는 dataset_name={cfg_dataset_name!r} 용이다. "
            f"'{args.dataset}' 에는 쓸 수 없다. --data_config 로 맞는 config 를 지정하라. "
            f"(사용 가능: {', '.join(_available_data_configs())})"
        )

    features = params.get("features") or {}
    if not features.get("enabled"):
        raise ValueError(
            f"데이터 config '{data_config_name}' 는 features 를 쓰지 않는다 "
            "(features.enabled 가 false). 미리 뽑을 피처가 정의된 config 를 지정하라."
        )

    # CLI 재정의 (주지 않은 항목은 config 값 유지)
    params["data_dir"] = str(dataset_dir)
    if getattr(args, "sample_level", None):
        params["sample_level"] = args.sample_level
    if getattr(args, "cache_dir", None):
        features["cache_dir"] = _resolve_under_root(project_root, args.cache_dir)
        params["features"] = features
    if getattr(args, "workers", None) is not None:
        params["feature_precompute_num_workers"] = int(args.workers)
    if getattr(args, "batch_size", None) is not None:
        params["feature_precompute_batch_size"] = int(args.batch_size)

    # 워밍업은 아래에서 직접 돌린다 (--force 로 캐시를 비울 틈을 만들기 위해).
    # setup() 중 자동 워밍업이 먼저 돌면 안 되므로 꺼 둔다.
    params["feature_precompute"] = False

    # lightning 의존이라 여기서 import (datasets CLI 전체를 무겁게 만들지 않기 위해)
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule

    dm = build_datamodule(data_cfg)
    try:
        dm.setup(None)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n인덱스가 없다. 먼저 인덱스를 만들어라: "
            f"rsctk datasets build-index {args.dataset} --data_dir {dataset_dir} "
            f"--sample_level {params.get('sample_level', 'cycle')}"
        ) from exc

    targets = [("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)]

    if getattr(args, "force", False):
        cleared = set()
        for _, ds in targets:
            cache = getattr(getattr(ds, "fp", None), "feature_cache", None)
            features_dir = getattr(cache, "features_dir", None)
            if features_dir is None or str(features_dir) in cleared:
                continue
            cleared.add(str(features_dir))
            if pathlib.Path(features_dir).exists():
                logger.info("--force: 기존 피처 캐시 삭제 %s", features_dir)
                shutil.rmtree(features_dir)
                pathlib.Path(features_dir).mkdir(parents=True, exist_ok=True)

    # 학습(setup)이 쓰는 것과 동일한 워밍업 루틴을 split 별로 돌린다.
    for split_name, ds in targets:
        dm._warmup_feature_cache(ds, split_name=split_name)

    cache = getattr(getattr(dm.train_ds, "fp", None), "feature_cache", None)
    if cache is not None:
        stats = cache.get_cache_stats()
        logger.info(
            "✅ 피처 캐시 준비 완료: %s (.pt %s개)",
            stats.get("cache_dir"),
            stats.get("pt_files"),
        )


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
        _handle_preprocess(args, project_root=project_root, dataset_dir=dataset_dir)
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
