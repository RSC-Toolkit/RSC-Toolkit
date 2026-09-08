"""하위호환 창구. 등록의 원천은 rsc_toolkit.registries.METRICS (내장은 lazy 선언, 외부는 데코레이터) 다."""
from rsc_toolkit.registries import METRICS

__all__ = ["METRICS"]
