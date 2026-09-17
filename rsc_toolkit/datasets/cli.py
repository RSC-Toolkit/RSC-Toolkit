"""`rsctk datasets ...` subcommands (download, preprocess, build-index, validate).

Kept separate from `rsc_toolkit/datasets/__init__.py` so that imports that only need DatasetFactory
do not pull in argparse/hydra code. `__init__` re-exports these names for backward compatibility.

The data config for preprocess defaults to the package-internal `rsc_toolkit/configs/data/*.yaml`; `--config_dir`
adds an external directory (hydra searchpath) -- symmetric with `--config-dir` on the training path.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Optional

from rsc_toolkit.datasets.factory import DatasetFactory
from rsc_toolkit.datasets.paths import resolve_dataset_dir

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


def _dataset_choices():
    # Mapping keys -- datasets registered via entry points/plugins show up automatically.
    return sorted(DatasetFactory.datasets.keys())


def build_parser(subparsers):
    """Attach the 'dataset' subcommand and its children."""

    dataset_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    download_parser = dataset_subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("dataset", choices=_dataset_choices())
    download_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    download_parser.add_argument("--data_dir", default=None, help="Dataset directory (defaults to OS user data dir)")
    download_parser.add_argument("--force", action="store_true", help="Despite valid dataset, re-download.")

    preprocess_parser = dataset_subparsers.add_parser(
        "preprocess",
        help="Pre-fill the feature cache from a training config (configs/data/*.yaml) (e.g. CLAP embeddings)",
    )
    preprocess_parser.add_argument("dataset", choices=_dataset_choices())
    preprocess_parser.add_argument(
        "--data_config",
        default=None,
        help="Data config name to use (rsc_toolkit/configs/data/<name>.yaml or data/<name>.yaml under --config_dir; default: dataset name)",
    )
    preprocess_parser.add_argument(
        "--config_dir",
        default=None,
        help="Extra hydra config search directory -- for data configs from an external package (symmetric with --config-dir in training)",
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
        help="Override the config's sample_level (default: config value)",
    )
    preprocess_parser.add_argument("--workers", type=int, default=None, help="DataLoader workers for feature computation (default: config value)")
    preprocess_parser.add_argument("--batch_size", type=int, default=None, help="Batch size for feature computation (default: config value)")
    preprocess_parser.add_argument("--cache_dir", default=None, help="Override the feature cache directory (default: config value)")
    preprocess_parser.add_argument("--force", action="store_true", help="Delete the existing cache for this setting and recompute everything")

    index_parser = dataset_subparsers.add_parser(
        "build-index",
        help="Build dataset sample index",
    )
    index_parser.add_argument("dataset", choices=_dataset_choices())
    index_parser.add_argument("--project_root", default=None, help="Base directory for relative paths (defaults to repo root)")
    index_parser.add_argument("--data_dir", default=None, help="Dataset directory (defaults to OS user data dir)")
    index_parser.add_argument("--force", action="store_true", help="Rebuild index even if present.")
    index_parser.add_argument("--sample_level", choices=["cycle", "clip"], default="cycle")

    validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate downloaded and raw dataset files",
    )
    validate_parser.add_argument("dataset", choices=_dataset_choices())
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


def _compose_data_config(
    data_config_name: str, project_root: pathlib.Path, config_dir: Optional[str] = None
) -> dict:
    """
    Compose the same hydra data config that training uses and return it as a plain dict.

    If the cache directory or key diverged from training (IndexDataModule), training could not read
    the precomputed features, so the config is not interpreted separately; the same source as training is used.

    With `config_dir`, that directory is added to the hydra searchpath so external `data/<name>.yaml` files are accepted
    (the package-internal list check is skipped and composition errors are reported as-is).
    """
    try:
        from hydra import compose, initialize_config_module
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "preprocess requires the trainer dependencies: pip install 'rsc-toolkit[trainer]'"
        ) from exc

    overrides = [f"data={data_config_name}"]
    if config_dir:
        extra = pathlib.Path(config_dir).expanduser().resolve()
        if not extra.is_dir():
            raise FileNotFoundError(f"--config_dir directory does not exist: {extra}")
        overrides.append(f"hydra.searchpath=[file://{extra}]")
    else:
        available = _available_data_configs()
        if data_config_name not in available:
            raise ValueError(
                f"Data config '{data_config_name}' not found. Available: {', '.join(available)} "
                "(for external configs, add a search path with --config_dir)"
            )

    with initialize_config_module(config_module="rsc_toolkit.configs", version_base=None):
        cfg = compose(config_name="default", overrides=overrides)

    # The config's project_root is ${hydra:runtime.cwd}, but that resolver does not work
    # outside the hydra runtime (compose API). Overwrite it with the project_root chosen by the CLI.
    cfg.data.params.project_root = str(project_root)
    return OmegaConf.to_container(cfg.data, resolve=True)


def _handle_preprocess(args, *, project_root: pathlib.Path, dataset_dir: pathlib.Path) -> None:
    """
    Pre-fill the feature cache based on a training config.

    Key principle: the cache conventions (directory, key, sample_id) are not duplicated here.
    The exact training path build_datamodule -> IndexDataModule is used,
    so training is guaranteed to reuse the precomputed cache.
    """
    import shutil

    data_config_name = getattr(args, "data_config", None) or args.dataset
    config_dir = getattr(args, "config_dir", None)
    # Without config_dir, keep the 2-argument call (tests and external code substitute this signature)
    data_cfg = (
        _compose_data_config(data_config_name, project_root, config_dir=config_dir)
        if config_dir
        else _compose_data_config(data_config_name, project_root)
    )

    params = data_cfg.get("params") or {}
    cfg_dataset_name = str(params.get("dataset_name", ""))
    if cfg_dataset_name != args.dataset:
        raise ValueError(
            f"Data config '{data_config_name}' is for dataset_name={cfg_dataset_name!r}. "
            f"It cannot be used for '{args.dataset}'. Specify a matching config with --data_config. "
            f"(available: {', '.join(_available_data_configs())})"
        )

    features = params.get("features") or {}
    if not features.get("enabled"):
        raise ValueError(
            f"Data config '{data_config_name}' does not use features "
            "(features.enabled is false). Specify a config that defines features to precompute."
        )

    # CLI overrides (unspecified items keep the config value)
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

    # Warm-up runs explicitly below (to leave room for clearing the cache with --force).
    # Automatic warm-up during setup() must not run first, so it is disabled.
    params["feature_precompute"] = False

    # Imported here because it depends on lightning (keeps the datasets CLI lightweight)
    from rsc_toolkit.trainer.factories.build_datamodule import build_datamodule

    dm = build_datamodule(data_cfg)
    try:
        dm.setup(None)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nIndex not found. Build the index first: "
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
                logger.info("--force: deleting existing feature cache %s", features_dir)
                shutil.rmtree(features_dir)
                pathlib.Path(features_dir).mkdir(parents=True, exist_ok=True)

    # Run the same warm-up routine as training (setup) for each split.
    for split_name, ds in targets:
        dm._warmup_feature_cache(ds, split_name=split_name)

    cache = getattr(getattr(dm.train_ds, "fp", None), "feature_cache", None)
    if cache is not None:
        stats = cache.get_cache_stats()
        logger.info(
            "✅ Feature cache ready: %s (%s .pt files)",
            stats.get("cache_dir"),
            stats.get("pt_files"),
        )


def handle_command(args) -> int:
    """Run the subcommand and return the process exit code (0 = success)."""
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
            logger.info("✅ Downloaded files validated")
        else:
            logger.error("❌ Downloaded file validation failed (%s)", dataset_dir)

        if processed_ok is None:
            logger.warning("⏭️  No preprocessing artifacts -- build-index has not run yet or validation is not implemented (not a failure)")
        elif processed_ok:
            logger.info("✅ Preprocessing artifacts validated")
        else:
            logger.error("❌ Preprocessing artifact validation failed")

        failed = (not downloaded_ok) or (processed_ok is False)
        if failed:
            logger.error("❌ Dataset '%s' validation failed", args.dataset)
            return 1

        logger.info("✅ Dataset '%s' validated", args.dataset)
        return 0

    raise NotImplementedError(args.dataset_command)


__all__ = [
    "build_parser", "handle_command", "create_dataset",
    "_find_project_root", "_default_project_root", "_resolve_under_root",
    "_available_data_configs", "_compose_data_config", "_handle_preprocess",
    "_configure_processor_raw_data_directory",
]
