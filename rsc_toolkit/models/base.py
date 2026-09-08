"""
툴킷이 모델에 요구하는 계약(contract).

서드파티가 자기 모델을 붙일 때 지켜야 하는 것은 이 파일 하나다.
`BaseRSCModel` 을 `nn.Module` 과 함께 상속하고 `training_forward` 를 구현한 뒤
`ModelFactory.register_model_class(...)` 로 등록하면 학습·검증 루프가 그대로 돌아간다.

    from torch import nn
    from rsc_toolkit.models.base import BaseRSCModel
    from rsc_toolkit.models.factory import ModelFactory

    @ModelFactory.register_model_class("my_model")
    class MyModel(BaseRSCModel, nn.Module):
        def __init__(self, n_cls: int = 4, **params):
            super().__init__()
            self.net = ...          # n_cls 는 configs 의 model.params.n_cls 로 주입된다

        def training_forward(self, audio_batch, labels, loss_func, **kwargs):
            logits = self.net(audio_batch)
            return logits, loss_func(logits, labels)

(등록 경로는 `factory.py` 를 보라 — 패키지 내부 모듈이면 자동 스캔되고,
외부 패키지면 `rsc_toolkit.models` 엔트리포인트나 `RSC_TOOLKIT_PLUGINS` 로 붙인다.)
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Tuple


logger = logging.getLogger(__name__)


class BaseRSCModel(ABC):
    """
    호흡음 분류 모델 인터페이스.

    학습 루프(`rsc_toolkit.trainer.adapters.pl_adapter.PLAdapter`)는 매 스텝에서
    **오직 `training_forward` 하나만** 호출한다. `forward` 는 툴킷이 부르지 않으므로
    필요에 따라 자유롭게 정의해도 된다.
    """

    @abstractmethod
    def training_forward(
        self,
        audio_batch: Any,
        labels: Any,
        loss_func: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        한 배치에 대해 logits 와 loss 를 함께 돌려준다.

        학습 스텝과 검증 스텝이 **같은 메서드**를 부른다. 둘을 다르게 처리해야 하면
        `self.training`(`nn.Module` 이 관리한다) 으로 분기하라 — 대조학습·마스킹처럼
        학습에서만 켜지는 항이 있는 모델이 이 방식을 쓴다.

        인자
        ----
        audio_batch
            입력 배치. `PLAdapter` 가 `batch[0].squeeze(1)` 을 넘기므로 채널 축이 없는
            상태로 도착한다. 구체적 모양(파형이냐 스펙트로그램이냐, 프레임 수)은
            **데이터 config 가 정하고 모델이 그에 맞춘다** — 툴킷은 강제하지 않는다.
        labels
            정수 클래스 라벨 `(B,)`. ICBHI 는 4-class, SPRSound 는 7-class 다.
        loss_func
            `configs/loss/*.yaml` 로 만들어진 손실 객체. 가중 CE 처럼 데이터셋 분포에
            의존하는 손실이 여기로 주입되므로, **모델 안에서 손실을 새로 만들지 말고
            받은 것을 쓰라.** 자체 항(대조손실 등)은 여기에 더하면 된다.
        **kwargs
            선택 인자. 현재 툴킷이 넘길 수 있는 것은 `meta_labels` 하나이며,
            datamodule 의 `meta_label_mode` 가 켜진 경우에만 배치 3번째 원소로 들어온다
            (SG-SCL·DAT 는 학습에서, BTS 계열은 검증에서도 받는다).
            **쓰지 않는 모델도 `**kwargs` 를 열어 두어야 한다** — 그래야 메타를 붙이는
            데이터 config 와 붙이지 않는 config 양쪽에서 같은 모델이 돌아간다.
            `meta_labels` 의 의미는 datamodule 마다 다르다(청진기 id·환자 번호·AQS float·
            ssl_label·meta7 벡터…). 받는 모델은 클래스 속성 `accepts_meta = ("dev", ...)` 로
            어떤 datamodule 의 `meta_kind` 를 이해하는지 선언하는 것을 권장한다 — 어댑터가
            학습 시작 전에 조합을 대조해 조용한 오학습을 막는다(미선언이면 검사하지 않는다).

        반환
        ----
        `(logits, loss)`
            `logits` 는 `(B, n_cls)` — 지표 계산이 이 모양을 전제한다.
            `loss` 는 역전파 가능한 0-차원 텐서다.
        """
        raise NotImplementedError
