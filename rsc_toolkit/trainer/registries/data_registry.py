"""하위호환 창구. 등록의 원천은 rsc_toolkit.registries.DATAMODULES 다.

예전에는 여기서 논문별 datamodule 10개를 모두 최상단 import 해 수동 dict 에 적었다 — 한 모듈의
선택 의존성(librosa 등)이 빠지면 이 파일 import 자체가 죽어 전체 학습이 막혔다. 이제 내장 항목은
허브에 lazy 로 선언되고 첫 조회 때만 import 된다. 새 datamodule 은 허브 선언이나
`@DATAMODULES.register("name")` 데코레이터로 등록한다.
"""
from rsc_toolkit.registries import DATAMODULES

__all__ = ["DATAMODULES"]
