"""
BEATs (Microsoft, MIT) — wa976/PAFA 저장소의 BEATs/ 디렉토리를 **그대로** 옮긴 것 (import 경로만 상대화).

원류: https://github.com/microsoft/unilm/tree/master/beats (BEATs.py, backbone.py, modules.py).
PAFA 논문 수치는 이 수정본으로 얻어졌으므로 unilm 원본이 아니라 이쪽을 기준으로 한다.
unilm 원본과의 차이 (2026-08-25 diff 확인):
- BEATs.preprocess 에서 `waveform * 2**15` 스케일링이 제거됨 (주석 처리) — fbank 정규화 상수
  (mean 15.41663 / std 6.55582)는 스케일된 입력 기준이라 분포가 어긋나지만 원본 동작 그대로 둔다.
- BEATs.extract_features 에 spec_transform(SpecAugment) 훅 추가 (PAFA 스크립트는 --nospec 이라 미사용).
- backbone.TransformerEncoder 의 `x = x.clone() + x_conv` (in-place 방지), 그 외는 black 포매팅.
quantizer.py / Tokenizers.py 는 미세조정에 안 쓰여 옮기지 않았다.
"""
from .BEATs import BEATs, BEATsConfig

__all__ = ["BEATs", "BEATsConfig"]
