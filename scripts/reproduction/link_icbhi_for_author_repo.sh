#!/bin/bash
# Builds the data/icbhi_dataset/audio_test_data/ expected by the authors' repositories **using symlinks only**.
#   .datasets/icbhi contains filename_differences.txt, zip files, index/ etc. besides the 920 recording pairs,
#   and the authors' code globs the whole directory and treats every *.txt as a recording, so linking it as-is raises KeyError.
#   The original data stays immutable: .datasets is only read, and only the link directory is created.
# Usage: bash scripts/reproduction/link_icbhi_for_author_repo.sh outputs/reference/<repo>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$ROOT/.datasets/icbhi
if [ $# -ne 1 ] || [ -z "$1" ] || [ ! -d "$ROOT/$1" ]; then
  echo "usage: $0 outputs/reference/<repo>  (path relative to the repository root, existing directory)" >&2; exit 1
fi
case "$1" in /*|*..*) echo "absolute paths and parent-directory references are not accepted: $1" >&2; exit 1;; esac
DST=$ROOT/$1/data/icbhi_dataset/audio_test_data
# delete only the link directory — stop if it contains real files (entries that are not links)
if [ -d "$DST" ] && find "$DST" -mindepth 1 -not -type l | grep -q .; then
  echo "$DST contains entries that are not symlinks; not deleting it" >&2; exit 1
fi
rm -rf "$DST"; mkdir -p "$DST"
n=0
for f in "$SRC"/[0-9]*_*.wav "$SRC"/[0-9]*_*.txt; do
  [ -e "$f" ] || continue
  ln -s "$f" "$DST/$(basename "$f")"; n=$((n+1))
done
echo "linked $n files → $DST  (wav $(ls "$DST"/*.wav | wc -l) / txt $(ls "$DST"/*.txt | wc -l))"
