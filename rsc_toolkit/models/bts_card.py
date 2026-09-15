"""
BTS-CARD (ICASSP 2026) — 반사실·적대 편향 제거 다중모달 RSC, 공식 코드 충실 이식 (P0).

출처:
- 논문: Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial
  Debiasing for OOD Robustness (arXiv 2510.22263 v2)
- 공식 코드: https://github.com/RSC-Toolkit/BTS-CARD (a9982a0, Apache-2.0)
  main_causal_adversarial_debiasing_logging.py set_model()/train()/validate(),
  models/{clap,tinybert,discriminators}.py, util/misc.py(fusion_function, kl_loss_function,
  ReverseLayerF), util/icbhi_util.py(randomly_mask_metadata :137-144, :259-261).

구조 (원본 그대로):
- TE 분기 = BTSModel(CLAP 텍스트 ⊕ 오디오 concat → Linear(1024, 4)) → o_multi        (main:268-273)
- NDE 분기 = nlpie/tiny-clinicalbert CLS → Linear(h, 256) → ReLU → Dropout(0.1) → Linear(256, 4)
  → o_text                                                                        (models/tinybert.py)
- RUBI 융합  o1 = o_multi · σ(o_text),  반사실 o2 = c · σ(o_text),  c = LearnableConstant(1.0)
  (main:276-280; util/misc.py fusion_function mode="rubi")
- GRL(λ=1.0) → Location(7)/Device(4) 판별기 LayerNorm(4)→Linear(4,96)→GELU→Dropout(0.1)→Linear(96,K),
  입력 = o_text **4-d 로짓** (main:294-299; models/discriminators.py)
- 손실 = CE(o1) + CE(o2) + KL + CE(o_text) + λ_loc·CE(loc) + λ_dev·CE(dev)          (main:282-301)
- 추론 TIE  output = o1 − α·o2                                                     (main:416)

원본 quirk (기본값으로 유지, 플래그로만 논문형 전환):
- KL 항 무효: kl_div(log_softmax(c·σ(o_text.detach())), log_softmax(o1).detach()) — 학습 파라미터로의
  그래디언트 경로가 없다 (c 는 optimizer 미등록). `kl_dead=True`(기본). 논문 Eq.8 형태는 `kl_dead=False`.
- LearnableConstant c: nn.Parameter 이지만 optimizer 6 그룹에 없음 → 고정 1.0 (main:195-212). 툴킷은
  `requires_grad=False` 로 동일 효과(PLAdapter 가 model.parameters() 전부를 optimizer 에 넣으므로).
  `const_trainable=True` 면 학습 대상. MA 대상에서도 제외(원본 main:325-331).
- 판별기 입력 = 4-d 로짓 (특징 벡터가 아님).
- 메타 마스킹(counterfactual metadata augmentation) = 학습 시 tiny-bert 문장만 속성별 p=0.25 로 값을
  None 으로 바꾸고 문장에서 **생략**(icbhi_util.py:137-144 → meta_description None 분기). 논문 §2.3 의
  'neutral placeholder(unknown)' 치환은 코드에 없다(test_drop_key 평가 ablation 전용) → `mask_mode="unknown"`
  옵션. CLAP 문장은 마스킹 없음.
- AMP 경계: 원본 train() 은 CLAP+classifier 만 autocast, TinyBERT·융합·손실은 fp32 (main:267-301);
  validate() 는 전 과정 autocast. 여기서는 training 중일 때만 후반부 autocast 를 끈다.
- MA(β 0.5, 매 step) 대상 = model/classifier/projector/meta_classifier 만 (판별기·상수 제외)
  → trainer/callbacks/btscard.py BTSCARDWeightAveraging.

난수: 원본 마스킹은 python `random` 전역 상태(num_workers=0 전제, seed 111)로 dataset __getitem__ 에서
뽑는다. 툴킷은 모델 안에서 torch.Generator(seed = torch.initial_seed(), 즉 L.seed_everything 종속)로
뽑는다 — 시퀀스 자체는 원본과 다르며(재현은 seed 평균으로 비교), 같은 seed 면 툴킷 내부에서는 재현된다.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bts import BTSModel, _as_4d
from .factory import ModelFactory
from .grl import ReverseLayerF
from rsc_toolkit.paths import resolve_pretrained

logger = logging.getLogger(__name__)

META_KEYS = ("age", "sex", "loc", "dev")  # 원본 meta_dict 삽입 순서 = 마스킹 난수 소비 순서
DEFAULT_NDE_LOCAL = "pretrained_models/upstream/hf/tiny-clinicalbert"
NDE_HUB_NAME = "nlpie/tiny-clinicalbert"
MA_PREFIXES = ("audio_features.", "classifier.", "meta_classifier.")  # 원본 MA 대상(projector=Identity 는 무상태)


# ----------------------------------------------------------------------------- 원본 함수
def fusion_function(o1: torch.Tensor, o2: torch.Tensor, mode: str = "rubi") -> torch.Tensor:
    """util/misc.py:57-67."""
    eps = 1e-12
    if mode == "sum":
        return torch.log(torch.sigmoid(o1 + o2) + eps)
    if mode == "hm":
        o = torch.sigmoid(o1) * torch.sigmoid(o2)
        return torch.log(o + eps) - torch.log1p(o)
    if mode == "rubi":
        return o1 * torch.sigmoid(o2)
    raise ValueError("Invalid fusion mode")


def kl_loss_function(o1_fusion: torch.Tensor, o2_fusion: torch.Tensor) -> torch.Tensor:
    """util/misc.py:139-143 — target = log_softmax(o1).detach(), input = log_softmax(o2)."""
    log_p_te = F.log_softmax(o1_fusion, dim=-1).detach()
    log_p_nde = F.log_softmax(o2_fusion, dim=-1)
    return F.kl_div(log_p_nde, log_p_te, reduction="batchmean", log_target=True)


def randomly_mask_metadata(meta_dict: Dict[str, Optional[str]], drop_prob: float = 0.25,
                           generator: Optional[torch.Generator] = None,
                           mask_value: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    util/icbhi_util.py:137-144. 키 순서대로 난수 하나씩 뽑아 `< drop_prob` 면 마스킹.
    원본은 random.random() 이고 마스킹 값은 None(문장에서 생략). mask_value='unknown' 이면 논문형 치환.
    """
    masked = {}
    for key, value in meta_dict.items():
        u = torch.rand(1, generator=generator).item()
        masked[key] = mask_value if u < drop_prob else value
    return masked


def apply_mask(meta_dict: Dict[str, Optional[str]], mask: Sequence[bool],
               mask_value: Optional[str] = None) -> Dict[str, Optional[str]]:
    """골든 재생용: 속성별 bool 마스크를 그대로 적용."""
    return {k: (mask_value if m else v) for (k, v), m in zip(meta_dict.items(), mask)}


class TinyBertTextModel(nn.Module):
    """models/tinybert.py — 속성 이름 bert/classifier 를 저자 ckpt 'meta_classifier' 키와 맞춘다."""

    def __init__(self, pretrained_name: str = NDE_HUB_NAME, output_size: int = 4):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size),
        )

    def cls_hidden(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        return self.classifier(self.cls_hidden(input_ids, attention_mask))


class MetaDiscriminator(nn.Module):
    """models/discriminators.py Location(7)/Device(4) — in_dim=4 (o_text 로짓), hidden 96."""

    def __init__(self, in_dim: int = 4, hidden: int = 96, n_classes: int = 7, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class LearnableConstant(nn.Module):
    """main:195-198. 원본은 optimizer 미등록 → 사실상 고정."""

    def __init__(self, initial_value: float = 1.0, trainable: bool = False):
        super().__init__()
        self.dummy_vector = nn.Parameter(torch.tensor(float(initial_value)), requires_grad=bool(trainable))


NDE_HUB_FALLBACK_ENV = "RSCTK_NDE_HUB_FALLBACK"   # "1" 이면 로컬 사본이 없을 때 HF 허브 nlpie/tiny-clinicalbert 로 대체


def _resolve_nde_name(name: str) -> str:
    """NDE(TinyBERT) 가중치 자리: 로컬 경로(cwd → RSCTK_PRETRAINED_ROOT 아래)를 찾고, 없으면 그대로(허브 id) 돌려준다.

    기본값 DEFAULT_NDE_LOCAL 이 없을 때 예전엔 경고 한 줄만 남기고 HF 허브로 **조용히 대체**해 네트워크 없는 환경에서
    학습 시작 뒤에 죽거나, 다른 스냅샷의 토크나이저를 쓰게 됐다 (REVIEW 2026-09-10 §6 P2). 이제 대체는 옵트인이다.
    """
    found = resolve_pretrained(explicit=name)          # 절대/존재하는 상대 경로 → ROOT 아래 → 그대로
    if found is not None and found.exists():
        return str(found)
    if name == DEFAULT_NDE_LOCAL:
        if os.environ.get(NDE_HUB_FALLBACK_ENV, "").strip() == "1":
            logger.warning("%s 가 없어 HF 허브 %s 를 쓴다 (%s=1)", name, NDE_HUB_NAME, NDE_HUB_FALLBACK_ENV)
            return NDE_HUB_NAME
        raise FileNotFoundError(
            f"BTS-CARD NDE 가중치 {name} 가 없다 (cwd 와 RSCTK_PRETRAINED_ROOT 아래 모두). "
            f"`{NDE_HUB_NAME}` 스냅샷을 그 자리에 두거나, model.params.nde_pretrained_name={NDE_HUB_NAME} 로 허브 id 를 "
            f"명시하거나, {NDE_HUB_FALLBACK_ENV}=1 로 자동 대체를 허용하라."
        )
    return name


# ----------------------------------------------------------------------------- 모델
@ModelFactory.register_model_class("bts_card")
class BTSCARDModel(BTSModel):
    def __init__(
        self,
        n_cls: int = 4,
        pretrained_name: str = "laion/clap-htsat-unfused",
        final_feat_dim: int = 512,
        clap_final: str = "concat",
        te_alpha: float = 0.5,
        meta_mode: str = "all",
        text_max_length: int = 64,
        meta_dialect: str = "icbhi",    # "sprsound" = SPRSound 로 직접 학습할 때 (bts.py 참고)
        nde_pretrained_name: str = DEFAULT_NDE_LOCAL,
        alpha: float = 0.0,             # 추론 TIE 계수 (논문 §3.2.4 권장 0)
        lambda_loc: float = 0.01,
        lambda_dev: float = 0.1,
        grl_lambda: float = 1.0,
        p_mask: float = 0.25,
        kl_dead: bool = True,           # True = 원본(무효 항), False = 논문 Eq.8 형태
        const_trainable: bool = False,  # 원본: optimizer 미등록(고정 1.0)
        mask_mode: str = "omit",        # "omit" = 원본(None 생략) | "unknown" = 논문 서술(placeholder)
        n_loc: int = 7,
        n_dev: int = 4,
        use_counterfactual: bool = True,  # ablation (a) 플래그 — 정의는 저자 확인 전 잠정
        use_adversarial: bool = True,     # ablation (b)
        use_meta_aug: bool = True,        # ablation (c)
        fusion_mode: str = "rubi",
    ):
        super().__init__(n_cls=n_cls, pretrained_name=pretrained_name, final_feat_dim=final_feat_dim,
                         clap_final=clap_final, te_alpha=te_alpha, meta_mode=meta_mode,
                         text_max_length=text_max_length, meta_dialect=meta_dialect)
        from transformers import AutoTokenizer

        if mask_mode not in {"omit", "unknown"}:
            raise ValueError(f"mask_mode must be omit|unknown, got {mask_mode!r}")
        nde_name = _resolve_nde_name(nde_pretrained_name)
        self.meta_classifier = TinyBertTextModel(nde_name, output_size=n_cls)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(nde_name)
        self.loc_discriminator = MetaDiscriminator(n_cls, 96, n_loc, 0.1)
        self.dev_discriminator = MetaDiscriminator(n_cls, 96, n_dev, 0.1)
        self.constant = LearnableConstant(1.0, trainable=const_trainable)

        self.n_cls = n_cls
        self.alpha = float(alpha)
        self.lambda_loc = float(lambda_loc)
        self.lambda_dev = float(lambda_dev)
        self.grl_lambda = float(grl_lambda)
        self.p_mask = float(p_mask)
        self.kl_dead = bool(kl_dead)
        self.mask_mode = mask_mode
        self.use_counterfactual = bool(use_counterfactual)
        self.use_adversarial = bool(use_adversarial)
        self.use_meta_aug = bool(use_meta_aug)
        self.fusion_mode = fusion_mode
        self.loss_terms: Dict[str, float] = {}
        self._mask_gen: Optional[torch.Generator] = None

    # ------------------------------------------------------------------ 텍스트
    @property
    def mask_generator(self) -> torch.Generator:
        if self._mask_gen is None:
            g = torch.Generator()
            g.manual_seed(int(torch.initial_seed()) % (2 ** 63))
            self._mask_gen = g
        return self._mask_gen

    def meta_dicts(self, meta7_batch: torch.Tensor) -> List[Dict[str, Optional[str]]]:
        return self.meta_field_dicts(meta7_batch)

    def describe_bert(self, meta7_batch: torch.Tensor, masking: bool) -> List[str]:
        """tiny-bert 문장: 학습(masking=True)이면 속성별 p_mask 마스킹 (icbhi_util.py:259-261)."""
        out = []
        # SPRSound 방언은 마스킹 값이 반드시 "unknown" 이어야 한다 — 저자의 SPRSound 문장 생성기가
        # "unknown" 을 명시적으로 처리하는 반면(util/sprsound_util.py:1-24), None 은 처리 분기가 없어
        # "a None patient" 같은 문장이 나온다. ICBHI 는 원본대로 mask_mode 를 따른다.
        if self.meta_dialect == "sprsound":
            mask_value = "unknown"
        else:
            mask_value = "unknown" if self.mask_mode == "unknown" else None
        for d in self.meta_dicts(meta7_batch):
            if masking and self.p_mask > 0:
                d = randomly_mask_metadata(d, self.p_mask, self.mask_generator, mask_value)
            out.append(self.render_meta_sentence(d))
        return out

    def encode_bert(self, texts: Sequence[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        """icbhi_dataset.py:42 — max_length 64, padding max_length, truncation."""
        enc = self.bert_tokenizer(list(texts), max_length=self.text_max_length, padding="max_length",
                                  truncation=True, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def encode_clap_texts(self, texts: Sequence[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(text=list(texts), return_tensors="pt", padding="max_length",
                             max_length=self.text_max_length)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ------------------------------------------------------------------ 분기
    def forward_te(self, x: torch.Tensor, clap_texts: Sequence[str]) -> torch.Tensor:
        ids, mask = self.encode_clap_texts(clap_texts, x.device)
        return self.classifier(self.forward_features(x, ids, mask))

    def forward_nde(self, bert_texts: Sequence[str], device) -> torch.Tensor:
        ids, mask = self.encode_bert(bert_texts, device)
        return self.meta_classifier(ids, mask)

    def counterfactual_pair(self, o_multi: torch.Tensor, o_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """main:276-280 — o1 = fusion(o_multi, o_text), o2 = fusion(c·1, o_text)."""
        o_multi_c = self.constant.dummy_vector * torch.ones_like(o_multi)
        o1 = fusion_function(o_multi, o_text, self.fusion_mode)
        o2 = fusion_function(o_multi_c, o_text, self.fusion_mode)
        return o1, o2

    def tie(self, o1: torch.Tensor, o2: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        a = self.alpha if alpha is None else float(alpha)
        return o1 - a * o2

    def forward_all(self, x: torch.Tensor, clap_texts: Sequence[str], bert_texts: Sequence[str],
                    fp32_tail: bool = False) -> Dict[str, torch.Tensor]:
        """모든 중간 출력. fp32_tail=True 면 CLAP 이후를 autocast 밖(fp32)에서 계산 (원본 train())."""
        o_multi = self.forward_te(x, clap_texts)
        ctx = torch.autocast(device_type=x.device.type, enabled=False) if fp32_tail else contextlib.nullcontext()
        with ctx:
            if fp32_tail:
                o_multi = o_multi.float()
            o_text = self.forward_nde(bert_texts, x.device)
            o1, o2 = self.counterfactual_pair(o_multi, o_text)
            rev = ReverseLayerF.apply(o_text, self.grl_lambda)
            loc_logits = self.loc_discriminator(rev)
            dev_logits = self.dev_discriminator(rev)
        return {"o_multi": o_multi, "o_text": o_text, "o1": o1, "o2": o2,
                "loc_logits": loc_logits, "dev_logits": dev_logits}

    def compute_losses(self, out: Dict[str, torch.Tensor], labels: torch.Tensor, loc: torch.Tensor,
                       dev: torch.Tensor, criterion: nn.Module) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """main:282-301 6항 손실. 반환: (총합, 항별 dict)."""
        o1, o2, o_text = out["o1"], out["o2"], out["o_text"]
        terms: Dict[str, torch.Tensor] = {}
        terms["o1"] = criterion(o1, labels)
        if self.use_counterfactual:
            terms["o2"] = criterion(o2, labels)
            if self.kl_dead:
                # 원본 main:285 — 입력 쪽 o_text.detach(), c 는 고정 → 그래디언트 없음
                o2_detached = fusion_function(self.constant.dummy_vector * torch.ones_like(o1),
                                              o_text.detach(), self.fusion_mode)
                terms["kl"] = kl_loss_function(o1, o2_detached)
            else:
                # 논문형(그래디언트 흐름). 실제 계산 방향: kl_loss_function(o1, o2) 는
                # F.kl_div(input=log_softmax(o2), target=log_softmax(o1).detach(), log_target=True)
                # = KL(softmax(o1) ‖ softmax(o2)) 다 (원본 util/misc.py:139-143 과 동일).
                # 논문 Eq.8 표기 KL(Y_{t,m*} ‖ Y_{t,m}) 와 방향이 다를 수 있음(저자 확인 대기).
                # ※ 이전 주석은 방향을 반대로 적었었다 — Eq.8 비교 판정 시 이 정정 기준으로 볼 것.
                terms["kl"] = kl_loss_function(o1, o2)
        terms["nde"] = criterion(o_text, labels)
        if self.use_adversarial:
            terms["loc"] = criterion(out["loc_logits"], loc)
            terms["dev"] = criterion(out["dev_logits"], dev)
        total = terms["o1"] + terms["nde"]
        if self.use_counterfactual:
            total = total + terms["o2"] + terms["kl"]
        if self.use_adversarial:
            total = total + self.lambda_loc * terms["loc"] + self.lambda_dev * terms["dev"]
        return total, terms

    # ------------------------------------------------------------------ 인터페이스
    def forward(self, x: torch.Tensor, meta7: torch.Tensor, alpha: Optional[float] = None,
                bert_texts: Optional[Sequence[str]] = None, clap_texts: Optional[Sequence[str]] = None) -> torch.Tensor:
        """TIE 로짓 o1 − α·o2 (평가: 마스킹 없음)."""
        clap_texts = list(clap_texts) if clap_texts is not None else self.describe(meta7)
        bert_texts = list(bert_texts) if bert_texts is not None else self.describe_bert(meta7, masking=False)
        out = self.forward_all(x, clap_texts, bert_texts)
        return self.tie(out["o1"], out["o2"], alpha)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if meta_labels is None:
            raise ValueError("BTS-CARD 는 배치에 meta(7필드) 가 필요하다 — data=bts_precomputed 를 쓸 것")
        meta7 = meta_labels
        clap_texts = self.describe(meta7)
        training = self.training
        bert_texts = self.describe_bert(meta7, masking=training and self.use_meta_aug)
        out = self.forward_all(audio_batch, clap_texts, bert_texts, fp32_tail=training)
        # 학습 분기의 반환 로짓은 tie(o1,o2) 융합값이다. 원본 train() 의 학습 정확도는
        # 융합 전 o_multimodel 로 계산되므로 train top-1 로그의 의미가 원본과 다르다
        # (손실·그래디언트·검증 지표는 동일). 원본 학습 곡선과 직접 대조할 때 주의.
        output = self.tie(out["o1"], out["o2"])
        if training:
            loc = meta7[:, 5].long()
            dev = meta7[:, 6].long()
            loss, terms = self.compute_losses(out, labels, loc, dev, loss_func)
            self.loss_terms = {k: float(v.detach()) for k, v in terms.items()}
            return output, loss
        # 원본 validate(): loss = CE(o1 − α·o2, y)
        return output, loss_func(output, labels)

    # ------------------------------------------------------------------ 저자 체크포인트
    def load_reference_checkpoint(self, path, strict_backbone: bool = True) -> Dict[str, List[str]]:
        """
        best.pth {'alpha','epoch','best_acc','model','classifier','meta_classifier'} 로드.
        model/classifier/meta_classifier 는 키 변환 없이(원본 속성명 유지) 로드; 판별기·상수는 ckpt 에 없다
        (원본 validate 저장 대상 밖) → 그 키만 missing 으로 허용하고 나머지는 strict.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model"])
        state.update({f"classifier.{k}": v for k, v in ckpt["classifier"].items()})
        state.update({f"meta_classifier.{k}": v for k, v in ckpt["meta_classifier"].items()})
        res = self.load_state_dict(state, strict=False)
        allowed = ("loc_discriminator.", "dev_discriminator.", "constant.")
        bad_missing = [k for k in res.missing_keys if not k.startswith(allowed)]
        if strict_backbone and (bad_missing or res.unexpected_keys):
            raise RuntimeError(f"BTS-CARD ckpt 키 불일치: missing={bad_missing} unexpected={res.unexpected_keys}")
        logger.info("BTS-CARD 저자 ckpt 로드: %s (alpha=%s epoch=%s best_acc=%s; 판별기·상수 제외 missing=%d)",
                    path, ckpt.get("alpha"), ckpt.get("epoch"), ckpt.get("best_acc"), len(res.missing_keys))
        return {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys),
                "alpha": ckpt.get("alpha"), "epoch": ckpt.get("epoch"), "best_acc": ckpt.get("best_acc")}

    def load_toolkit_checkpoint(self, path) -> None:
        """ICBHIBestScoreCheckpoint 산출물 {'model': state_dict, ...} 을 strict 로드."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["model"], strict=True)

    @staticmethod
    def ma_state_keys(state_dict_keys) -> List[str]:
        """원본 MA 대상(model/classifier/projector/meta_classifier)에 해당하는 state_dict 키."""
        return [k for k in state_dict_keys if k.startswith(MA_PREFIXES)]
