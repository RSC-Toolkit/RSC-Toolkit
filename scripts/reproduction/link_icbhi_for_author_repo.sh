#!/bin/bash
# 저자 저장소가 기대하는 data/icbhi_dataset/audio_test_data/ 를 **심볼릭 링크만으로** 만든다.
#   .datasets/icbhi 에는 녹음 920쌍 외에 filename_differences.txt·zip·index/ 등이 섞여 있는데,
#   저자 코드는 디렉터리를 통째로 glob 해 *.txt 를 전부 녹음으로 취급하므로 그대로 링크하면 KeyError 가 난다.
#   원본 데이터 불변 원칙에 따라 .datasets 는 읽기만 하고, 링크 디렉터리만 새로 만든다.
# 사용: bash scripts/reproduction/link_icbhi_for_author_repo.sh outputs/reference/<repo>
set -e
ROOT=/raid/doyoung/projects/RSC-Toolkit
SRC=$ROOT/.datasets/icbhi
DST=$ROOT/$1/data/icbhi_dataset/audio_test_data
rm -rf "$DST"; mkdir -p "$DST"
n=0
for f in "$SRC"/[0-9]*_*.wav "$SRC"/[0-9]*_*.txt; do
  [ -e "$f" ] || continue
  ln -s "$f" "$DST/$(basename "$f")"; n=$((n+1))
done
echo "링크 $n 개 → $DST  (wav $(ls "$DST"/*.wav | wc -l) / txt $(ls "$DST"/*.txt | wc -l))"
