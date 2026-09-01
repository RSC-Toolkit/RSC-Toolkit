# RSC-Toolkit

Respiratory Sound Classification Toolkit — **ICBHI 2017** 과 **SPRSound** 위에서
호흡음 분류 논문을 같은 파이프라인·같은 프로토콜로 재현하고 비교하기 위한 툴킷이다.

핵심은 **공식 스플릿 위의 공정 비교**다. 논문마다 다른 스플릿·지표·모델 선택 규칙을
하나의 데이터·특징·평가 경로에 올려, "방법이 실제로 얼마나 기여하는가" 를 잴 수 있게 한다.

현재 **모델 33종**이 등록돼 있고(SFT 기준선부터 대조학습·어댑터·파운데이션 모델까지),
논문별 골든 텐서 회귀 테스트로 이식 충실도를 고정한다.

## 설치

```bash
pip install -e ".[trainer,models]"
```

| extra | 무엇이 들어오나 |
|---|---|
| `trainer` | Hydra · Lightning · TensorBoard — 학습을 돌리려면 필요하다 |
| `models` | timm · transformers 등 모델 백본 의존성 |
| `ssl-weights` | ResNet50 SimCLR 가중치를 쓸 때만 |
| `dev` | pytest · black · flake8 · isort |

## 데이터 준비

```bash
rsc-toolkit datasets download    --dataset icbhi
rsc-toolkit datasets build-index --dataset icbhi
rsc-toolkit datasets validate    --dataset icbhi
```

원본 데이터는 **불변**으로 다룬다 — 스플릿·인덱스는 전부 파생 파일로만 만든다.

## 학습

```bash
rsc-toolkit-train --config-name patchmix_icbhi seed=1 run.name=my-first-run
```

설정은 Hydra 구성이라 CLI 에서 그대로 덮어쓸 수 있다:

```bash
rsc-toolkit-train --config-name patchmix_icbhi \
  model.params.method=ce trainer.max_epochs=10 seed=3
```

결과는 `outputs/<계열>/<run.name>/` 에 쌓이고, `best.json` 이 최종 지표를 담는다
(`{"epoch", "S_p", "S_e", "icbhi_score"}` — `icbhi_score = (S_e + S_p) / 2`).

## 자기 모델 붙이기

**툴킷을 포크할 필요가 없다.** `BaseRSCModel` 을 상속하고 `training_forward` 하나만
구현한 뒤 등록하면 학습·검증 루프가 그대로 돈다.

```python
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
        return logits, loss_func(logits, labels)   # (logits, loss) 를 함께 돌려준다
```

계약 전문은 [`rsc_toolkit/models/base.py`](rsc_toolkit/models/base.py) 에 있다.
학습 스텝과 검증 스텝이 같은 메서드를 부르므로, 둘을 나눠야 하면 `self.training` 으로 분기한다.

등록 경로는 셋이다:

1. **패키지 내부** — `rsc_toolkit/models/` 에 모듈을 두면 자동 스캔된다.
2. **엔트리포인트** — 외부 패키지는 설치만으로 등록된다:

   ```toml
   [project.entry-points."rsc_toolkit.models"]
   my_models = "my_pkg.models"
   ```

3. **환경변수** — 아직 패키징하지 않은 로컬 모듈용:

   ```bash
   RSC_TOOLKIT_PLUGINS=my_models rsc-toolkit-train --config-name ... model.name=my_model
   ```

등록한 이름을 `model.name` 으로 넘기면 된다.

## 학습·검증 루프 바꾸기

콜백은 Hydra `_target_` 으로 완전히 열려 있어 외부 클래스도 그대로 붙는다
(`rsc_toolkit/configs/callbacks/*.yaml` 참고). 체크포인트 선택 규칙, best 지표 기준,
학습 재개 정책이 전부 콜백이다.

## 테스트

```bash
pytest -m "not slow"      # 빠른 회귀
pytest                    # 골든 텐서 포함 전체
```

## 문서

- [`DEV.md`](DEV.md) — 개발 환경
- [`AGENTS.md`](AGENTS.md) — 저장소 작업 규약
- [`docs/RESULTS_BY_DATASET.md`](docs/RESULTS_BY_DATASET.md) — 데이터셋별 재현 결과와 이슈
- [`docs/REPRODUCTION_TARGETS.md`](docs/REPRODUCTION_TARGETS.md) — 재현 대상 논문 목록과 정찰 결과

## 알려진 제약

- 배포용 휠은 아직 검증되지 않았다 — 현재는 **editable 설치**(`pip install -e`)를 전제로 하고,
  일부 config 가 리포 작업 디렉터리 기준 경로를 쓴다.
- 평가 전용 엔트리포인트가 없다. 학습은 `rsc-toolkit-train` 하나로 되지만,
  체크포인트 재평가·집계는 아직 `scripts/reproduction/` 의 스크립트가 담당한다.
- CI 가 없다.

## 라이선스

MIT — [`LICENSE`](LICENSE)
