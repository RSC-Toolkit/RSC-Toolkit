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
셸에서는 `scripts/train.sh <config> [overrides]` 가 같은 일을 한다(`GPU=` 로 카드 선택, 스레드 캡 기본).

## 평가 (학습된 weight 재평가)

```bash
rsc-toolkit-eval outputs/patchmix/icbhi-patchmix-ce-s1                 # 스냅샷 자동 탐색 → strict 로드 → test → best.json 대조
rsc-toolkit-eval outputs/qlung/icbhi-qlung-s1 --override data.params.batch_size=16 --out eval.json
scripts/eval.sh outputs/bts/icbhi-bts-s1                               # 셸 래퍼 (GPU=, 샌드박스)
```

run 디렉터리의 hydra 스냅샷으로 학습 때와 같은 팩토리에서 모델·데이터를 다시 세운다. 학습 산출물은 읽기만 하고
(쓰기는 샌드박스), `best.json` 이 있으면 PASS/MISMATCH 를 판정한다. 여러 run 전수 검증은 `scripts/verify_paper_weights.sh`.

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
배치 3번째 원소(meta)를 받는 모델은 `accepts_meta = ("dev", ...)` 를 선언하면 어댑터가 datamodule 의
`meta_kind` 와 학습 시작 전에 대조해 준다(미선언이면 검사하지 않는다).

### 모델만이 아니다 — 등록소 10종

모델·손실·지표·옵티마이저·datamodule·raw 데이터셋·어댑터·피처 프로세서·피처 증강·meta 라벨이
전부 같은 방식의 등록소([`rsc_toolkit/registries.py`](rsc_toolkit/registries.py))다.
`ModelFactory.register_model_class` 는 `MODELS.register` 의 별칭이다.

```python
from torch import nn
from rsc_toolkit.registries import LOSSES, METRICS, PROCESSORS, META_LABELS

@LOSSES.register("focal")                 # configs 의 loss.name=focal
class FocalLoss(nn.Module): ...

@METRICS.register("hflung_jaccard")       # metrics 의 val: [{name: hflung_jaccard}]
class HFLungJaccard(Metric): ...

@PROCESSORS.register("hs_embed")          # data.params.features.kind=hs_embed
class HeartSoundSSLProcessor(BaseAudioProcessor):
    is_spectrogram = False                # True 면 학습 배치에 SpecAugment 를 걸 수 있다
    @classmethod
    def from_options(cls, opts, *, sampling_rate, feature_cache):   # features 블록(+ features.params)
        return cls(ckpt=opts["params"]["ckpt"], sampling_rate=sampling_rate, feature_cache=feature_cache)
    def get_processor_config(self):       # = 피처 캐시 키. 계산에 영향 주는 설정을 전부 넣는다
        return {"kind": "hs_embed", "ckpt": self.ckpt, "sr": self.sampling_rate}

@META_LABELS.register("hospital")         # data.params.meta_label_mode=hospital (팩토리 → item -> int)
def build_hospital():
    return lambda item: int(item["meta"]["hospital"])
```

커스텀 인덱스 데이터셋은 `IndexDataset` 을 상속하고 config 로 주입한다 — datamodule 을 서브클래싱할 필요가 없다:

```yaml
data:
  params:
    datamodule_name: index
    index_dataset_cls: my_pkg.data.HeartSoundIndexDataset
    index_dataset_params: { bandpass_hz: [25, 400], fade: false }   # 생성자 인자가 그대로 전달된다
    collate_fn: my_pkg.data.my_collate                               # (선택) dotted path
    features: { enabled: true, kind: hs_embed, params: { ckpt: /path/to/ssl.pt } }
```

등록 경로는 셋이다:

1. **패키지 내부** — `rsc_toolkit/models/` 에 모듈을 두면 자동 스캔된다 (모델). 다른 종류는 허브에 lazy 로 선언한다.
2. **엔트리포인트** — 외부 패키지는 설치만으로 등록된다. 그룹 하나로 모든 종류를 등록한다:

   ```toml
   [project.entry-points."rsc_toolkit.plugins"]
   my_plugin = "my_pkg"          # import 되면 데코레이터가 전부 실행되는 모듈
   ```

   (예전 그룹 `rsc_toolkit.models` 도 계속 읽는다.)
3. **환경변수** — 아직 패키징하지 않은 로컬 모듈용:

   ```bash
   RSC_TOOLKIT_PLUGINS=my_pkg rsc-toolkit-train --config-name ... model.name=my_model
   ```

자기 config 는 `--config-dir` 로 얹는다(학습), `rsctk datasets preprocess … --config_dir` 로 얹는다(피처 사전계산).
등록한 이름을 `model.name` / `loss.name` / `features.kind` 등으로 넘기면 된다. 미지 이름은 가용 목록과 함께 즉시 실패한다.

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

- 배포용 휠은 `scripts/packaging/smoke_installed_wheel.sh` 로 검증한다(빈 venv 설치 → 저장소 밖 cwd 에서
  import·등록소·번들 데이터·config 합성·콘솔 스크립트). config 의 상대 경로(`.datasets/…`, `outputs/…`,
  `pretrained_models/…`)는 **실행한 작업 디렉터리** 기준이다. 사전학습 가중치는 `RSCTK_PRETRAINED_ROOT=/path/to/weights` 로
  한 디렉터리에 모아 둘 수 있고(후보 경로의 `pretrained_models/` 접두사를 뗀 상대 경로 또는 파일명으로 찾는다), 피처 캐시는
  `RSCTK_FEATURE_CACHE_ROOT`, 저장소 루트 추정은 `RSCTK_PROJECT_ROOT` 로 바꾼다.
- 논문별 **집계**(seed 평균·표) 는 아직 `scripts/reproduction/aggregate_*.py` 와 `scripts/paper/` 가 담당한다
  (재평가 자체는 `rsc-toolkit-eval`).
- CI 는 `.github/workflows/ci.yml`(휠 스모크 + CPU 단위 테스트) — GPU 가 필요한 골든 테스트는 로컬에서 돈다.

## 라이선스

MIT — [`LICENSE`](LICENSE)
