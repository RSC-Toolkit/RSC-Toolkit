# Portions of this file come from the original RSC-Toolkit code base by Miika Toikkanen; that code is under the MIT license.
# See THIRD_PARTY_NOTICES.md. The rest of the file is under the PolyForm Noncommercial License 1.0.0.
import argparse
import sys
import logging


from rsc_toolkit.datasets import cli as datasets_cli

# Registries shown by `rsc-toolkit list` — names match the hub variables in rsc_toolkit.registries (lowercase).
REGISTRY_KINDS = ("models", "datasets", "datamodules", "losses", "metrics", "optimizers",
                  "adapters", "processors", "augmentations", "meta_labels")


def list_registries(kinds, out) -> int:
    """Write registered names per kind to `out`. Unknown kinds print a message and return 2. (Imports happen only here, to keep CLI startup light.)"""
    import rsc_toolkit.registries as registries

    wanted = list(kinds) or list(REGISTRY_KINDS)
    unknown = [k for k in wanted if k not in REGISTRY_KINDS]
    if unknown:
        print(f"unknown kind {unknown} — choices: {', '.join(REGISTRY_KINDS)}", file=sys.stderr)
        return 2
    for kind in wanted:
        reg = getattr(registries, kind.upper())
        names = sorted(reg)
        out.write(f"{kind} ({len(names)}):\n")
        for n in names:
            out.write(f"  {n}\n")
        failed = getattr(reg, "failed_modules", {})
        for mod, why in sorted(failed.items()):
            out.write(f"  ! failed to import {mod}: {why}\n")
    return 0


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(prog="rsc-toolkit", description="Respiratory Sound Classification Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset command
    datasets_cli.build_parser(subparsers)
    # list command — registry listing
    p_list = subparsers.add_parser("list", help="show registered component names by kind")
    p_list.add_argument("kinds", nargs="*", metavar="KIND",
                        help=f"kinds to show (default: all). Choices: {', '.join(REGISTRY_KINDS)}")

    args = parser.parse_args()          # unknown options are an error (parse_known_args would silently drop typos)

    # entrypoint handling here
    if args.command == "datasets":
        # Propagate validation failures as the exit code so scripts/CI can detect them.
        sys.exit(datasets_cli.handle_command(args) or 0)
    if args.command == "list":
        sys.exit(list_registries(args.kinds, sys.stdout))


if __name__ == "__main__":      # `python -m rsc_toolkit.cli ...`
    main()
