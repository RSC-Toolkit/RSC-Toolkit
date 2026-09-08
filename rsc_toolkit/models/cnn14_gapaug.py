"""
GaP-Aug(Chang et al., ICASSP 2024) 분류기 — PANNs 원형 Cnn14 + 논문 Algorithm 1 학습 절차. **논문 명세 재구현**(공식·비공식 코드 없음).

백본(공식 PANNs 코드 qiuqiangkong/audioset_tagging_cnn pytorch/models.py Cnn14 와 동일 구조 — 골든 대조 P0):
  파형 → 파워 STFT(window, center, reflect) → mel(librosa slaney) → 10·log10(clamp amin 1e-10, ref 1, top_db None)
  → bn0(BatchNorm2d(64), mel 축) → conv_block1–6(각 2×(3×3 conv+BN+ReLU), avg pool 2×2 / 마지막 1×1, dropout 0.2)
  → 주파수(mel) 축 평균(mean dim=3) → 시간 축 max+mean [= pooled 2048] → dropout 0.5 → fc1(2048→2048)+ReLU [= feat] → dropout 0.5 [= emb, PANNs 'embedding']
  → 분류 fc(2048→n_cls) — 입력은 head_input(emb|feat) 플래그.
  ConvBlock 은 툴킷 models/backbone/cnn.py 의 것(공식과 동일: 3×3 ×2, bias 없음, xavier/BN 1·0 초기화) 을 재사용.
  사전학습: AudioSet Cnn14 16 kHz 판(Cnn14_16k_mAP=0.438.pth, zenodo 3987831; 자매 논문 JMIR AI 2025 가 16 kHz 판 사용) —
  spectrogram/logmel/fc_audioset 을 제외한 전 키를 strict 로 로드. 32 kHz 판(zenodo 3960586) 은 pretrained="32k" 플래그
  (32 kHz·fmax 14 kHz 학습분이라 sample_rate/fmax/데이터 config 를 함께 바꿔야 한다 — 경고 출력).

특징(논문 §3.1 '64-dim log Mel filterbank, 25 ms / 6.25 ms'): 16 kHz, 10 s, 64 mel → n_fft 400 / hop 100 (feature_preset="paper").
  공식 16k ckpt 는 512/160 으로 학습됐다(feature_preset="panns16k"; 골든 대조는 이 설정).
  **특징 정의 = PANNs 식(가정, 질문 #5)**: 파워 스펙트럼 → slaney mel(fmin 50 / fmax 8000) → 10·log10 dB(ref 1, amin 1e-10, top_db None).
  논문 문구는 kaldi fbank(HTK mel·자연로그) 로도 읽힐 수 있으나 백본이 PANNs 이므로 PANNs 프론트엔드 정의를 택한다.
  창 함수(window): 공식 ckpt 는 hann 으로 학습 → 기본 hann; 자매 논문(jmir_full.txt:667-669) 은 같은 연구실 CNN14 파이프라인에
  'Hamming window 512 / hop 160' 을 명시 → window="hamming" 플래그(가정).

가정(저자 확인 전 — AUTHOR_QUESTIONS GaP-Aug #1–#8; 전부 config 플래그):
  norm_mode        : log-mel 비음수화 = 샘플별 min-max [0,1] (모델 입력도 그 스케일; 학습·평가 동일 적용). 대안 batch_minmax / db_offset / none /
                     sample_minmax_restore(γ 만 [0,1] 에서 적용하고 모델 입력은 원 dB 스케일로 복원 — 질문 #1 후반의 갈림길).
                     부작용: 사전학습 bn0 running stats 는 PANNs dB 입력(mel 별 mean −46..−18, std ~20) 기준이라 [0,1] 입력에선
                     초기 eval 출력이 거의 상수가 되고, train 중 bn0 running stats 가 재적응(momentum 0.1, ~수십 step) 한다.
  mixup_alpha      : 1.0 (근거: 공식 PANNs 파이프라인 main.py:174 Mixup(mixup_alpha=1.) + do_mixup 짝/홀 쌍 = Algo. 1 line 7-12 구조;
                     단 PANNs 는 bn0 **뒤** 에서 mixup, 여기서는 bn0 앞 스펙트로그램) — 여전히 가정 #2,  mixup_loss: soft (대안 pair)
  triplet          : **w_trip 1.0** = 논문 Algorithm 1 line 14 'L ← L_NLL + L_triplet' 비가중 합(문면). 0.01 은 자매 논문(JMIR AI 2025,
                     CE 기반·다른 세팅, λ=0.01) 근거의 **의도적 이탈 플래그**(w_trip=0.01) — 수정 회차 1 까지의 기본값이었음.
                     margin 1.0, X̃_a 샘플(triplet_source), 무작위 삼중항,
                     **triplet_layer** = emb(PANNs 'embedding' = fc1 ReLU 후 dropout 0.5; 학습 모드에선 dropout 마스크가 걸린다)
                     | feat(fc1 ReLU, dropout 전) | pooled(conv 풀링 2048, fc1 전) — 질문 #3 의 '어느 층' 미기재 항목
  head_input       : **emb** (공식 PANNs finetune_template.py Transfer_Cnn14 = post-dropout embedding 에 fc_transfer → log_softmax → NLL;
                     논문 §2.3 'NLL loss' 문구와 부합) | feat (dropout 없이 fc1 ReLU 출력에 직접) — 질문 #4 관련, 미기재
  joint_forward    : **False(기본, 문면)** = Algo. 1 line 13 대로 M(X̃_mix)·M(X̃_a) 두 번 forward(BN 배치통계 분리).
                     True 면 16개를 한 배치로 묶어 한 번 forward(BN 통계 공유; 속도 최적화, 비문면)
  head_init        : 분류 fc xavier_uniform(bias 0) (PANNs init_layer 관례)
  window           : hann (ckpt 학습 설정) | hamming (자매 논문 관례)
  데이터 측 가정   : 10 s 초과 사이클은 앞 절단(pad_type repeat; 논문 'divided into 10-second segments' 는 다중 세그먼트 분할로도
                     읽힘 — 질문 #6), 녹음 단위 fade 없음(configs/data/icbhi_gapaug.yaml `fade: false`; 논문 언급 없음)
  specaug/patchmask: augmentations/gap_aug.py 참고

method → (증강, mixup): gapaug(gamma, on) | gapaug_nomixup(gamma, off) | naive(없음, off) | specaug(spec, on) |
  patchmask(mask, on) | mixup_only(없음, on). `mixup` 인자로 개별 override 가능(specaug/patchmask w/o Mixup).
손실은 모든 method 에서 동일(NLL + w_trip·triplet, 기본 w_trip 1; Table 2 는 '증강 비교' 이므로 손실은 고정 — 가정). 평가: loss_func(CE).
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from rsc_toolkit.augmentations.gap_aug import GammaPatchAug, SpecAugPANNs, normalize_nonneg, sample_minmax_restore, sample_minmax_split
from rsc_toolkit.methods.gapaug import mixup_pairs, pair_nll, soft_target_nll, split_odd_even, triplet_loss
from rsc_toolkit.models.backbone.cnn import ConvBlock, _init_bn, _init_layer

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)

METHODS: Dict[str, tuple] = {
    "gapaug": ("gamma", True),
    "gapaug_nomixup": ("gamma", False),
    "naive": ("none", False),
    "specaug": ("spec", True),
    "patchmask": ("mask", True),
    "mixup_only": ("none", True),
}

PANNS_CNN14 = {
    "16k": {
        "url": "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
        "path": "pretrained_models/upstream/Cnn14_16k_mAP=0.438.pth",
        "sha256": "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6",  # 2026-08-26 확인 (358,668,570 B)
        "size": 358_668_570,
        "sample_rate": 16000, "fmax": 8000.0,
    },
    "32k": {
        "url": "https://zenodo.org/record/3960586/files/Cnn14_mAP%3D0.431.pth?download=1",
        "path": "pretrained_models/upstream/Cnn14_mAP=0.431.pth",
        "sha256": None,  # 미다운로드 — 첫 다운로드 시 provenance.json 에 기록
        "size": None,
        "sample_rate": 32000, "fmax": 14000.0,
    },
}
PANNS_LICENSE = "MIT (PANNs, qiuqiangkong/audioset_tagging_cnn) — 패키지 동봉·재배포 금지, 개별 다운로드"
FEATURE_PRESETS = {
    "paper": dict(n_fft=400, hop=100),      # 25 ms / 6.25 ms @16 kHz (논문 §3.1)
    "panns16k": dict(n_fft=512, hop=160),   # 공식 16k ckpt 학습 설정 (README) — 골든 대조
}
WINDOWS = {"hann": torch.hann_window, "hamming": torch.hamming_window}
EXCLUDE_PREFIXES = ("spectrogram_extractor.", "logmel_extractor.", "fc_audioset.")
HEAD_INPUTS = ("emb", "feat")
TRIPLET_LAYERS = ("emb", "feat", "pooled")
NORM_MODES = ("sample_minmax", "sample_minmax_restore", "batch_minmax", "db_offset", "none")


class PANNsLogMel(nn.Module):
    """torchlibrosa Spectrogram(power) + LogmelFilterBank(librosa slaney mel, log10, ref 1, amin 1e-10, top_db None) 의 torchaudio 재현."""

    def __init__(self, sample_rate=16000, n_fft=512, hop=160, n_mels=64, fmin=50.0, fmax=8000.0, ref=1.0, amin=1e-10, window="hann"):
        super().__init__()
        if window not in WINDOWS:
            raise ValueError(f"window must be one of {tuple(WINDOWS)}, got {window!r}")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop, f_min=fmin, f_max=fmax, n_mels=n_mels,
            window_fn=WINDOWS[window], power=2.0, center=True, pad_mode="reflect", norm="slaney", mel_scale="slaney",
        )
        self.window = window
        self.ref, self.amin = float(ref), float(amin)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, 1, T, n_mels) — PANNs 레이아웃."""
        m = self.mel(wav)                                   # (B, n_mels, T)
        db = 10.0 * torch.log10(torch.clamp(m, min=self.amin)) - 10.0 * torch.log10(torch.tensor(max(self.amin, self.ref)))
        return db.transpose(1, 2).unsqueeze(1).contiguous()


class Cnn14Backbone(nn.Module):
    """PANNs Cnn14 의 bn0 이후 부분 (입력 (B,1,T,F) log-mel).
    반환 dict(pooled=conv 풀링 2048(fc1 전, dropout 전), feat=fc1 ReLU 출력(dropout 전), emb=dropout 후 = PANNs 'embedding')."""

    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        _init_bn(self.bn0)
        _init_layer(self.fc1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for i, blk in enumerate((self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6)):
            x = blk(x, pool_size=(1, 1) if i == 5 else (2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        pooled = x1 + x2
        x = F.dropout(pooled, p=0.5, training=self.training)
        feat = F.relu_(self.fc1(x))
        emb = F.dropout(feat, p=0.5, training=self.training)
        return {"pooled": pooled, "feat": feat, "emb": emb}


@ModelFactory.register_model_class("cnn14_gapaug")
class Cnn14GaPAug(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        method: str = "gapaug",
        mixup: Optional[bool] = None,          # None = method 기본; True/False 로 override (SpecAug/PatchMask w/o Mixup)
        pretrained: Optional[str] = "16k",     # 16k | 32k | null
        pretrained_path: Optional[str] = None, # 직접 경로 지정 시
        # ---- 특징 ----
        sample_rate: int = 16000,
        feature_preset: str = "paper",         # paper(400/100) | panns16k(512/160)
        n_fft: Optional[int] = None,
        hop: Optional[int] = None,
        n_mels: int = 64,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        window: str = "hann",                  # hann(ckpt 학습 설정) | hamming(자매 논문 관례)  (가정)
        norm_mode: str = "sample_minmax",      # sample_minmax | sample_minmax_restore | batch_minmax | db_offset | none  (가정 #1)
        db_offset: float = 100.0,
        # ---- GaP-Aug / PatchMask ----
        n_patches: int = 32,
        max_freq: int = 16,
        max_time: int = 16,
        min_freq: int = 1,
        min_time: int = 1,
        gamma_min: float = 1.7,
        gamma_max: float = 2.0,
        gain: float = 1.0,
        gamma_per: str = "patch",
        aug_p: float = 1.0,
        # ---- SpecAug 비교군 (PANNs 기본값 가정) ----
        specaug_time_width: int = 64,
        specaug_time_stripes: int = 2,
        specaug_freq_width: int = 8,
        specaug_freq_stripes: int = 2,
        # ---- mixup / 손실 ----
        mixup_alpha: float = 1.0,              # 가정 #2
        mixup_loss: str = "soft",              # soft | pair
        w_trip: float = 1.0,                   # Algo. 1 line 14 비가중 합(문면). 0.01 = 자매 논문 근거 이탈 플래그 (#3)
        triplet_margin: float = 1.0,
        triplet_p: float = 2.0,
        triplet_source: str = "xa",            # xa (Algo. 1) | all (비혼합 전체 배치; 대안)
        triplet_layer: str = "emb",            # emb | feat | pooled  (가정 #3: 임베딩 층 미기재)
        head_input: str = "emb",               # emb (공식 Transfer_Cnn14 관행) | feat  (가정 #4: 미기재)
        joint_forward: bool = False,           # False = Algo. 1 line 13 문면(2회 forward) | True = 결합 1회(비문면)
        head_init: str = "xavier",             # xavier | default
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {tuple(METHODS)}, got {method!r}")
        if feature_preset not in FEATURE_PRESETS:
            raise ValueError(f"feature_preset must be one of {tuple(FEATURE_PRESETS)}")
        if mixup_loss not in ("soft", "pair"):
            raise ValueError("mixup_loss must be soft|pair")
        if triplet_source not in ("xa", "all"):
            raise ValueError("triplet_source must be xa|all")
        if triplet_layer not in TRIPLET_LAYERS:
            raise ValueError(f"triplet_layer must be one of {TRIPLET_LAYERS}, got {triplet_layer!r}")
        if head_input not in HEAD_INPUTS:
            raise ValueError(f"head_input must be one of {HEAD_INPUTS}, got {head_input!r}")
        self.method = method
        self.aug_kind, default_mix = METHODS[method]
        self.use_mixup = default_mix if mixup is None else bool(mixup)
        self.n_cls = int(n_cls)
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")
        self.norm_mode, self.db_offset = norm_mode, float(db_offset)
        self.mixup_alpha, self.mixup_loss = float(mixup_alpha), mixup_loss
        self.w_trip, self.triplet_margin, self.triplet_p, self.triplet_source = float(w_trip), float(triplet_margin), float(triplet_p), triplet_source
        self.triplet_layer, self.head_input = triplet_layer, head_input
        self.joint_forward = bool(joint_forward)
        self.sample_rate, self.fmax, self.window = int(sample_rate), float(fmax), window

        preset = FEATURE_PRESETS[feature_preset]
        self.n_fft, self.hop = int(n_fft or preset["n_fft"]), int(hop or preset["hop"])
        self.frontend = PANNsLogMel(sample_rate, self.n_fft, self.hop, n_mels, fmin, fmax, window=window)
        self.backbone = Cnn14Backbone(n_mels)
        self.fc = nn.Linear(2048, self.n_cls)
        if head_init == "xavier":
            _init_layer(self.fc)

        self.gap = GammaPatchAug(n_patches, max_freq, max_time, min_freq, min_time, (gamma_min, gamma_max), gain, gamma_per, aug_p, mode="gamma")
        self.patchmask = GammaPatchAug(n_patches, max_freq, max_time, min_freq, min_time, (gamma_min, gamma_max), gain, gamma_per, aug_p, mode="mask")
        self.specaug = SpecAugPANNs(specaug_time_width, specaug_time_stripes, specaug_freq_width, specaug_freq_stripes, aug_p)

        self.pretrained_info = None
        if pretrained_path is not None or pretrained:
            self.load_panns_weights(pretrained_path, pretrained or "16k")
        logger.info("Cnn14GaPAug(method=%s, aug=%s, mixup=%s, feat=%d/%d %s, norm=%s, w_trip=%.3g@%s, alpha=%.2f, head=%s)",
                    method, self.aug_kind, self.use_mixup, self.n_fft, self.hop, window, norm_mode, self.w_trip, triplet_layer,
                    self.mixup_alpha, head_input)

    # ------------------------------------------------------------------ 사전학습
    def load_panns_weights(self, path: Optional[str], variant: str = "16k") -> Dict[str, list]:
        if path is None:
            spec = PANNS_CNN14[variant]
            if spec["sample_rate"] != self.sample_rate or spec["fmax"] != self.fmax:
                logger.warning("PANNs Cnn14 %s ckpt 는 %d Hz / fmax %.0f 로 학습됐는데 모델은 %d Hz / fmax %.0f — "
                               "데이터 config(sampling_rate)·fmax 를 함께 맞춰야 한다",
                               variant, spec["sample_rate"], spec["fmax"], self.sample_rate, self.fmax)
            from rsc_toolkit.paths import download_target, resolve_pretrained

            ckpt = resolve_pretrained([spec["path"]]) or download_target(spec["path"])  # RSCTK_PRETRAINED_ROOT → cwd 순
            if not ckpt.exists():
                from rsc_toolkit.datasets.download_utils import download_file, record_download_provenance

                ckpt.parent.mkdir(parents=True, exist_ok=True)
                logger.info("PANNs Cnn14 %s 다운로드: %s", variant, spec["url"])
                download_file(spec["url"], ckpt)
                record_download_provenance(ckpt, url=spec["url"], known_size=spec["size"], known_sha256=spec["sha256"])
                self._append_provenance_extras(ckpt, variant)
        else:
            ckpt = pathlib.Path(path)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        kept = {k: v for k, v in state.items() if not k.startswith(EXCLUDE_PREFIXES)}
        self.backbone.load_state_dict(kept, strict=True)
        self.pretrained_info = {"path": str(ckpt), "variant": variant, "n_loaded": len(kept),
                                "excluded": sorted(k for k in state if k.startswith(EXCLUDE_PREFIXES))}
        logger.info("PANNs Cnn14 로드: %s (%d 키, 제외 %d)", ckpt, len(kept), len(self.pretrained_info["excluded"]))
        return self.pretrained_info

    @staticmethod
    def _append_provenance_extras(ckpt: pathlib.Path, variant: str) -> None:
        """record_download_provenance 가 쓰지 않는 라이선스·용도 필드를 provenance.json 에 병합(재다운로드 시 소실 방지)."""
        p = ckpt.with_name(ckpt.name + ".provenance.json")
        try:
            rec = json.loads(p.read_text()) if p.exists() else {}
            rec.update({"license": PANNS_LICENSE, "used_by": f"rsc_toolkit/models/cnn14_gapaug.py (GaP-Aug, {variant} 판)"})
            p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        except OSError as e:  # provenance 기록 실패는 학습을 막지 않는다
            logger.warning("provenance 추가 기록 실패 %s: %s", p, e)

    # ------------------------------------------------------------------ 특징
    def _as_wave(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"expected waveform (B, L), got {tuple(x.shape)}")
        return x

    def compute_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """파형 (B,L) → PANNs log-mel (B,1,T,F) (정규화 전; 골든 대조용)."""
        return self.frontend(self._as_wave(wav))

    def make_input(self, wav: torch.Tensor) -> torch.Tensor:
        """log-mel → 비음수화(norm_mode) — 학습·평가 공통 입력 스케일 (sample_minmax_restore 는 모델 입력이 원 dB 스케일)."""
        with torch.no_grad():
            return normalize_nonneg(self.compute_logmel(wav), self.norm_mode, self.db_offset)

    def make_augmented_input(self, wav: torch.Tensor) -> torch.Tensor:
        """학습 경로: 비음수화 → 증강(Algo. 1 line 5-6). sample_minmax_restore 면 [0,1] 에서 γ 를 적용한 뒤 dB 스케일로 복원."""
        with torch.no_grad():
            lm = self.compute_logmel(wav)
            if self.norm_mode == "sample_minmax_restore":
                x01, mn, mx = sample_minmax_split(lm)
                return sample_minmax_restore(self.augment(x01), mn, mx)
            return self.augment(normalize_nonneg(lm, self.norm_mode, self.db_offset))

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.aug_kind == "gamma":
            return self.gap(x)
        if self.aug_kind == "mask":
            return self.patchmask(x)
        if self.aug_kind == "spec":
            return self.specaug(x)
        return x

    # ------------------------------------------------------------------ forward
    def forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(x)
        out["logits"] = self.fc(out[self.head_input])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_backbone(self.make_input(x))["logits"]

    def _trip(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return out[self.triplet_layer]

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if not self.training:
            logits = self.forward(audio_batch)
            return logits, loss_func(logits, labels)

        x = self.make_augmented_input(audio_batch)  # (B,1,T,F): 비음수화 + Algo. 1 line 5-6 샘플별 증강
        B = x.shape[0]

        if not self.use_mixup or B < 2:
            out = self.forward_backbone(x)
            loss = F.cross_entropy(out["logits"], labels)  # NLL(one-hot) == CE
            if self.w_trip > 0:
                loss = loss + self.w_trip * triplet_loss(self._trip(out), labels, self.triplet_margin, self.triplet_p)
            return out["logits"], loss

        idx_a, idx_b = split_odd_even(B, device=x.device)
        xa, xb, ya, yb = x[idx_a], x[idx_b], labels[idx_a], labels[idx_b]
        x_mix, y_mix, lam = mixup_pairs(xa, xb, ya, yb, self.n_cls, self.mixup_alpha)

        if self.joint_forward:
            out = self.forward_backbone(torch.cat([x_mix, xa], dim=0))
            n = x_mix.shape[0]
            logits_mix, logits_a, emb_a = out["logits"][:n], out["logits"][n:], self._trip(out)[n:]
        else:
            om = self.forward_backbone(x_mix)
            oa = self.forward_backbone(xa)
            logits_mix, logits_a, emb_a = om["logits"], oa["logits"], self._trip(oa)

        if self.mixup_loss == "soft":
            loss = soft_target_nll(logits_mix, y_mix)
        else:
            loss = pair_nll(logits_mix, ya, yb, lam)
        if self.w_trip > 0:
            if self.triplet_source == "xa":
                loss = loss + self.w_trip * triplet_loss(emb_a, ya, self.triplet_margin, self.triplet_p)
            else:
                ob = self.forward_backbone(xb)
                loss = loss + self.w_trip * triplet_loss(torch.cat([emb_a, self._trip(ob)]), torch.cat([ya, yb]), self.triplet_margin, self.triplet_p)

        # 학습 지표용 로짓 (B, n_cls): a 위치 = M(X̃_a)(하드 라벨 대비 유효), b 위치 = M(X_mix)(혼합 입력이라 b 라벨과 무관).
        # → 학습 train/topk_accuracy 는 참고 불가(절반이 혼합 예측). 손실만 의미 있다. 평가 지표는 eval 경로(forward) 로 계산된다.
        full = x.new_zeros((B, self.n_cls))
        full[idx_a] = logits_a.detach().to(full.dtype)
        full[idx_b] = logits_mix.detach().to(full.dtype)
        if B % 2 == 1:
            full[B - 1] = logits_a.detach().mean(0)
        return full, loss

    # ------------------------------------------------------------------ 진단
    def describe(self) -> dict:
        n_all = sum(p.numel() for p in self.parameters())
        return {"params_total": n_all, "params_backbone": sum(p.numel() for p in self.backbone.parameters()),
                "method": self.method, "aug": self.aug_kind, "mixup": self.use_mixup, "n_fft": self.n_fft, "hop": self.hop,
                "window": self.window, "norm_mode": self.norm_mode, "head_input": self.head_input,
                "triplet_layer": self.triplet_layer, "pretrained": self.pretrained_info}
