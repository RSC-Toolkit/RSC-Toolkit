import argparse
import sys
import logging


from rsc_toolkit.datasets import cli as datasets_cli

# `rsc-toolkit list` 가 보여주는 등록소 — 이름은 rsc_toolkit.registries 의 허브 변수와 같다 (소문자).
REGISTRY_KINDS = ("models", "datasets", "datamodules", "losses", "metrics", "optimizers",
                  "adapters", "processors", "augmentations", "meta_labels")


def list_registries(kinds, out) -> int:
    """종류별 등록 이름을 `out` 에 쓴다. 모르는 종류는 메시지와 함께 2. (import 는 여기서만 — CLI 시작을 무겁게 하지 않는다.)"""
    import rsc_toolkit.registries as registries

    wanted = list(kinds) or list(REGISTRY_KINDS)
    unknown = [k for k in wanted if k not in REGISTRY_KINDS]
    if unknown:
        print(f"모르는 종류 {unknown} — 선택: {', '.join(REGISTRY_KINDS)}", file=sys.stderr)
        return 2
    for kind in wanted:
        reg = getattr(registries, kind.upper())
        names = sorted(reg)
        out.write(f"{kind} ({len(names)}):\n")
        for n in names:
            out.write(f"  {n}\n")
        failed = getattr(reg, "failed_modules", {})
        for mod, why in sorted(failed.items()):
            out.write(f"  ! {mod} 을 불러오지 못했다: {why}\n")
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
    # list command — 등록소 목록 (README·train.sh 가 안내하던 `rsc-toolkit list` 가 실제로는 없었다: REVIEW 2026-09-10 §8 P3)
    p_list = subparsers.add_parser("list", help="등록된 컴포넌트 이름을 종류별로 보여준다")
    p_list.add_argument("kinds", nargs="*", metavar="KIND",
                        help=f"보여줄 종류 (기본: 전부). 선택: {', '.join(REGISTRY_KINDS)}")

    args = parser.parse_args()          # 모르는 옵션은 오류 (이전 parse_known_args 는 오타를 조용히 버렸다)

    # entrypoint handling here
    if args.command == "datasets":
        # 검증 실패를 스크립트/CI 가 알 수 있도록 종료 코드로 전달한다.
        sys.exit(datasets_cli.handle_command(args) or 0)
    if args.command == "list":
        sys.exit(list_registries(args.kinds, sys.stdout))


if __name__ == "__main__":      # `python -m rsc_toolkit.cli ...` — 가드가 없어 무동작이던 것을 고침 (2026-09-14)
    main()
