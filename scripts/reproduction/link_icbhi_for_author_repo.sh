#!/bin/bash
# 저자 저장소가 기대하는 data/icbhi_dataset/audio_test_data/ 를 **심볼릭 링크만으로** 만든다.
#   .datasets/icbhi 에는 녹음 920쌍 외에 filename_differences.txt·zip·index/ 등이 섞여 있는데,
#   저자 코드는 디렉터리를 통째로 glob 해 *.txt 를 전부 녹음으로 취급하므로 그대로 링크하면 KeyError 가 난다.
#   원본 데이터 불변 원칙에 따라 .datasets 는 읽기만 하고, 링크 디렉터리만 새로 만든다.
# 사용: bash scripts/reproduction/link_icbhi_for_author_repo.sh outputs/reference/<repo>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$ROOT/.datasets/icbhi
if [ $# -ne 1 ] || [ -z "$1" ] || [ ! -d "$ROOT/$1" ]; then
  echo "사용: $0 outputs/reference/<repo>  (저장소 루트 기준 상대경로, 존재하는 디렉터리)" >&2; exit 1
fi
case "$1" in /*|*..*) echo "절대경로·상위 디렉터리 참조는 받지 않는다: $1" >&2; exit 1;; esac
DST=$ROOT/$1/data/icbhi_dataset/audio_test_data
# 링크 디렉터리만 지운다 — 실제 파일이 들어 있으면(링크가 아닌 항목) 멈춘다
if [ -d "$DST" ] && find "$DST" -mindepth 1 -not -type l | grep -q .; then
  echo "$DST 에 심볼릭 링크가 아닌 항목이 있어 지우지 않는다" >&2; exit 1
fi
rm -rf "$DST"; mkdir -p "$DST"
n=0
for f in "$SRC"/[0-9]*_*.wav "$SRC"/[0-9]*_*.txt; do
  [ -e "$f" ] || continue
  ln -s "$f" "$DST/$(basename "$f")"; n=$((n+1))
done
echo "링크 $n 개 → $DST  (wav $(ls "$DST"/*.wav | wc -l) / txt $(ls "$DST"/*.txt | wc -l))"
