import argparse
import sys
import logging


from rsc_toolkit.datasets import cli as datasets_cli


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(prog="rsc-toolkit", description="Respiratory Sound Classification Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset command
    datasets_cli.build_parser(subparsers)

    args, remaining = parser.parse_known_args()

    # entrypoint handling here
    if args.command == "datasets":
        # 검증 실패를 스크립트/CI 가 알 수 있도록 종료 코드로 전달한다.
        sys.exit(datasets_cli.handle_command(args) or 0)
