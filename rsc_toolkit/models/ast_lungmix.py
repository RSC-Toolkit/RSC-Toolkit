"""
Lungmix(ICASSP 2025, Ge et al.) 의 AST 분류기 — **논문 명세 재구현** (공식 저장소에 모델·학습 코드 없음).

논문 §III-B: ImageNet+AudioSet 사전학습 AST, '첫 두 토큰(cls/dist) 대신 전 hidden state 평균' 풀링, 분류기 앞 BatchNorm,
입력 128-mel × 1024 frames(9 s 파형), AudioSet mean/std 정규화. 학습 시 배치 안에서 파형 Lungmix(온라인) 후 fbank.

구현: rsc_toolkit.models.ast_patchmix.PatchMixAST(AudioSet 10/10 ckpt 로딩·forward 골격·patch_mix) 를 상속하되,
입력 레이아웃은 **원 AST(input_fdim=128, input_tdim=1024)** 로 되돌린다 → AudioSet pos_embed(12×101) 를 보간 없이 그대로 쓴다
(patch-mix 의 뒤집힌 (798,128) quirk 는 물려받지 않는다 — 논문에 patch-mix 코드베이스 사용 근거 없음).

가정(저자 확인 전, 전부 플래그 — docs/AUTHOR_QUESTIONS_2026-08-26.md LungMix 절):
- #8 stride 미기재 → fstride=tstride=10 (AudioSet 10/10 ckpt). 전 토큰 평균에 cls/dist 포함 여부 미기재 → pool='mean_all'(포함);
     'mean_patches' 로 제외 가능. '전 hidden state' 는 최종 LayerNorm(v.norm) **뒤**의 토큰으로 가정(HF last_hidden_state 정의와 동일;
     norm 앞 평균은 미구현). BatchNorm = BatchNorm1d(768) 하나.
     head 미기재: 원 AST 분류기는 LayerNorm(768)+Linear 인데 논문은 '분류기 앞에 BatchNorm 추가' 만 적음 → 가정 head='bn_linear'
     (BN 이 LN 을 **대체**); 'bn_ln_linear' 는 BN → LayerNorm → Linear(원 mlp_head 유지 + BN 추가).
     SpecAugment 미언급 → specaugment.enabled=False (켜면 fbank 뒤 툴킷 SpecAugment, 정책 icbhi_ast_sup — 데이터 config 의
     specaugment 블록은 kind: waveform 에서 무효라 여기서 처리).
- #5 1024 frames: 9 s@10 ms ≈ 898 frame → 나머지 0-패딩. 패딩·정규화 순서 미기재 → fbank_pad_before_norm=True
     (원 AST 데이터로더 순서: 패딩 후 (x−mean)/(2·std)).
- #4 짝: 배치 내 randperm 짝·믹싱 샘플이 원본을 **대체** — **플래그 없음(고정)**; 믹싱 확률만 mix_prob=1.0 플래그.
- #5 믹싱 후 길이(max_len+shift2) → repeat 후 앞 9 s 절단(fit_mode='cut_first'; cut_last|random_crop 대안). raw 단계에서는 출력이
     ≤ 2×원 구간이라 대개 9 s 미만 → repeat 패딩이 지배하고 fit_mode 차이는 작다.
- #5 믹싱 위치: 논문 식 3 의 M ∈ {0,λ,1}^{1×maxlen(x_i,x_j)} 표기(길이가 다른 두 입력) 와 공식 함수의 길이 맞춤 구조(등길이 입력엔 항등)는
     '원 구간을 섞은 뒤 9 s' 를 시사 → mix_stage='raw'(**기본·주 런**): 배치 meta_labels 로 실린 원 구간 길이
     (data.params.lungmix.emit_valid_len=true 필요) 만 섞은 뒤 9 s 로 맞춘다. mix_stage='padded'(대안): 데이터셋이 9 s repeat 패딩한
     파형을 그대로 섞는다(공식 함수 구조상 shift2 상수 패딩 때문에 출력의 기대 25% 가 상수 mean(wav2)). 저자 확인 전.
- #1 apply_loudness_mask=False 가 공식 코드 출력(random mask 만) — 기본(P0). True 가 논문 식 3 재구현. 식 3 경로의 마스크 겹침값은
     overlap='lam'(공식 반환 mask, random∧loudness = λ) 기본 / 'one'(식 8 OR 해석 = 1) 대안.
- 'Lungmix + Patch-mix': 파형 Lungmix → 패치 임베딩 patch_mix(β=1, 원본 patch-mix) → 라벨은 patchmix_label_mode 가정:
     'mixup' = PatchMixLoss(y_or, y_or[index], λ) 선형 보간 / 'preserve' = CE(y_or) (§III-C 의 label-preservation Patch-mix 와 결합).
     논문은 결합판이 어느 쪽인지 적지 않았다(비교군, P1 큐 밖).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from rsc_toolkit.augmentations.lungmix import OVERLAP_MODES, ClassLayout, lungmix_batch, vanilla_mixup_batch
from rsc_toolkit.methods.lungmix import VARIANTS, LungmixLoss
from rsc_toolkit.processors.mel import generate_fbank

from .ast_patchmix import PatchMixAST
from .factory import ModelFactory

logger = logging.getLogger(__name__)

AUDIOSET_MEAN, AUDIOSET_STD = -4.2677393, 4.5689974


@ModelFactory.register_model_class("ast_lungmix")
class ASTLungmix(PatchMixAST):
    accepts_meta = ("lungmix_valid_len",)  # 학습 batch[2] = 원 구간 길이 (LungmixDataModule emit_valid_len)

    def __init__(
        self,
        n_cls: int = 4,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
        pretrained_path: Optional[str] = None,
        verbose: bool = False,
        # ---- 풀링·헤드 (논문 §III-B) ----
        pool: str = "mean_all",            # mean_all(cls/dist 포함 전 토큰 평균) | mean_patches | cls_dist(원 AST)
        use_bn: bool = True,
        head: str = "bn_linear",           # bn_linear(가정: BN 이 원 LayerNorm 대체) | bn_ln_linear(BN → LN → Linear)
        specaugment: Optional[dict] = None,  # {enabled: false, policy: icbhi_ast_sup, mask: mean} — 논문 미언급 → 기본 off
        # ---- 온라인 fbank ----
        sample_rate: int = 16000,
        n_mels: int = 128,
        target_frames: int = 1024,
        fbank_pad_before_norm: bool = True,
        fbank_mean: float = AUDIOSET_MEAN,
        fbank_std: float = AUDIOSET_STD,
        # ---- 믹싱·손실 ----
        variant: str = "non_linear",
        apply_loudness_mask: bool = False,
        overlap: str = "lam",              # 식 3 경로 마스크 겹침: lam(공식 반환 mask) | one(식 8 OR 해석)
        smooth: bool = False,
        mix_prob: float = 1.0,
        out_len_sec: float = 9.0,
        fit_mode: str = "cut_first",
        mix_stage: str = "raw",            # raw(기본·주 런; meta_labels=원 구간 길이 필요) | padded(대안)
        vanilla_alpha: float = 0.5,
        lambda1: float = 1.0,
        lambda2_mode: str = "ratio",
        lambda2: float = 1.0,
        mix_beta: float = 1.0,             # lungmix_patchmix 의 patch_mix Beta(β,β)
        patchmix_label_mode: str = "mixup",  # lungmix_patchmix: mixup(PatchMixLoss 선형 보간) | preserve(CE(y_or))
        time_domain: bool = False,
        class_layout: Optional[dict] = None,
        mix_num_threads: int = 1,
    ):
        super().__init__(
            n_cls=n_cls, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain, model_size=model_size,
            mix_beta=mix_beta, pretrained_path=pretrained_path, verbose=verbose, method="ce",
            time_domain=time_domain, input_layout="time_freq",
        )
        if variant not in VARIANTS:
            raise ValueError(f"variant 는 {VARIANTS} 중 하나: {variant!r}")
        if pool not in ("mean_all", "mean_patches", "cls_dist"):
            raise ValueError("pool 은 mean_all|mean_patches|cls_dist")
        if head not in ("bn_linear", "bn_ln_linear"):
            raise ValueError("head 는 bn_linear|bn_ln_linear")
        if mix_stage not in ("padded", "raw"):
            raise ValueError("mix_stage 는 padded|raw")
        if overlap not in OVERLAP_MODES:
            raise ValueError(f"overlap 은 {OVERLAP_MODES} 중 하나")
        if patchmix_label_mode not in ("mixup", "preserve"):
            raise ValueError("patchmix_label_mode 는 mixup|preserve")
        # 부모의 mlp_head/classifier(LayerNorm+Linear) 는 쓰지 않는다 → 파라미터 수에서 제외
        self.mlp_head = nn.Identity()
        self.classifier = nn.Identity()
        self.pool = pool
        self.head_mode = head
        self.bn = nn.BatchNorm1d(self.original_embedding_dim) if use_bn else nn.Identity()
        self.ln = nn.LayerNorm(self.original_embedding_dim) if head == "bn_ln_linear" else nn.Identity()
        self.head = nn.Linear(self.original_embedding_dim, n_cls)
        self.fstride, self.tstride = int(fstride), int(tstride)
        sa = dict(specaugment or {})
        self.specaugment = None
        if sa.get("enabled", False):
            from rsc_toolkit.augmentations.specaugment import SpecAugment

            self.specaugment = SpecAugment(policy=sa.get("policy", "icbhi_ast_sup"), mask=sa.get("mask", "mean"))

        self.sample_rate, self.n_mels, self.target_frames = int(sample_rate), int(n_mels), int(target_frames)
        self.fbank_pad_before_norm = bool(fbank_pad_before_norm)
        self.fbank_mean, self.fbank_std = float(fbank_mean), float(fbank_std)

        self.variant = variant
        self.apply_loudness_mask, self.smooth, self.overlap = bool(apply_loudness_mask), bool(smooth), overlap
        self.mix_prob, self.fit_mode, self.vanilla_alpha = float(mix_prob), fit_mode, float(vanilla_alpha)
        self.mix_stage, self.patchmix_label_mode = mix_stage, patchmix_label_mode
        self.out_len = int(round(out_len_sec * sample_rate))
        self.mix_num_threads = int(mix_num_threads)
        self.layout = ClassLayout(**(class_layout or {}))
        self.lungmix_loss = LungmixLoss(variant=variant, lambda1=lambda1, lambda2_mode=lambda2_mode, lambda2=lambda2)
        self.f_dim, self.t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)

    # ------------------------------------------------------------------ 특징
    def compute_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) 파형 → (B, target_frames, n_mels) kaldi fbank(AudioSet 상수 정규화). 입력은 9 s(out_len) 로 맞춘 파형."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        outs, n_valid = [], []
        for i in range(wav.shape[0]):
            fb = generate_fbank(wav[i : i + 1].float(), sampling_rate=self.sample_rate, n_mels=self.n_mels, normalize=False)
            n = fb.shape[0]
            if n < self.target_frames:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, self.target_frames - n))
            elif n > self.target_frames:
                fb = fb[: self.target_frames]
            outs.append(fb)
            n_valid.append(min(n, self.target_frames))
        fb = torch.stack(outs, 0)
        fb = (fb - self.fbank_mean) / (self.fbank_std * 2)
        if not self.fbank_pad_before_norm:
            # 정규화 후 0-패딩 (툴킷 MelBankAudioProcessor 순서) — 패딩 프레임이 0 이 된다
            for i, n in enumerate(n_valid):
                fb[i, n:] = 0.0
        return fb

    def forward_tokens(self, x: torch.Tensor, y=None, patch_mix: bool = False):
        """부모 forward_features 와 동일한 경로, 풀링만 뺀 토큰 출력 (B, 2+N, 768)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # 부모는 stride 10 하드코딩 — 여기서는 fstride/tstride 플래그를 따른다 (patch_embed 의 stride 와 일치)
        h_patch = int((x.size()[2] - 16) / self.fstride) + 1
        w_patch = int((x.size()[3] - 16) / self.tstride) + 1
        B = x.shape[0]
        x = self.v.patch_embed(x)
        mix = None
        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=self.time_domain, hw_num_patch=[h_patch, w_patch])
            mix = (y_a, y_b, lam, index)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return (x, mix) if patch_mix else x

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean_all":
            return tokens.mean(dim=1)
        if self.pool == "mean_patches":
            return tokens[:, 2:].mean(dim=1)
        return (tokens[:, 0] + tokens[:, 1]) / 2

    def forward_features(self, x, y=None, patch_mix=False, time_domain=False):  # noqa: D401 — 부모 시그니처 유지
        out = self.forward_tokens(x, y=y, patch_mix=patch_mix)
        if patch_mix:
            tokens, (y_a, y_b, lam, index) = out
            return self.pool_tokens(tokens), y_a, y_b, lam, index
        return self.pool_tokens(out)

    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln(self.bn(feats)))

    def apply_specaugment(self, fb: torch.Tensor) -> torch.Tensor:
        """(B, T, F) fbank 에 샘플별 SpecAugment (학습 시, enabled 일 때만)."""
        if self.specaugment is None or not self.training:
            return fb
        outs = [self.specaugment(fb[i].transpose(0, 1)).transpose(0, 1) for i in range(fb.shape[0])]
        return torch.stack(outs, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) 파형 → logits. (B, 1024, 128) fbank 를 직접 주면 그대로 쓴다."""
        if x.dim() == 2 and x.shape[-1] != self.n_mels:
            x = self.compute_fbank(x)
        return self.classify(self.forward_features(x))

    # ------------------------------------------------------------------ 학습
    def _mix_waveforms(self, audio_batch: torch.Tensor, labels: torch.Tensor, valid_len=None):
        """배치 내 짝 믹싱(CPU, 전역 RNG — 공식 함수와 같은 난수 열). None 이면 믹싱 안 함."""
        if self.variant == "none":
            return None
        if self.mix_prob < 1.0 and float(torch.rand(())) >= self.mix_prob:
            return None
        wav_cpu, y_cpu = audio_batch.detach().float().cpu(), labels.detach().cpu()
        vl_cpu = None
        if self.mix_stage == "raw":
            if valid_len is None:
                raise ValueError("mix_stage='raw' 는 배치 meta_labels 로 원 구간 길이가 필요하다 "
                                 "(data.params.lungmix.emit_valid_len=true)")
            vl_cpu = valid_len.detach().cpu()
        # CPU 믹싱은 요소별 연산이라 단일 스레드가 가장 빠르다 — 기본 스레드(64)로는 서버 부하(load 270+)에서 배치당 5–7 s,
        # 1 스레드로는 30 ms (OMP 경합). 수치 정의에는 무관 (mix_num_threads 로 조절, 0 이면 건드리지 않음).
        prev_threads = torch.get_num_threads()
        if self.mix_num_threads > 0:
            torch.set_num_threads(self.mix_num_threads)
        try:
            if self.variant == "vanilla_mixup":
                mb = vanilla_mixup_batch(wav_cpu, y_cpu, alpha=self.vanilla_alpha)
            else:
                mb = lungmix_batch(wav_cpu, y_cpu, out_len=self.out_len, smooth=self.smooth,
                                   apply_loudness_mask=self.apply_loudness_mask, layout=self.layout, fit_mode=self.fit_mode,
                                   valid_len=vl_cpu, overlap=self.overlap)
        finally:
            if self.mix_num_threads > 0:
                torch.set_num_threads(prev_threads)
        dev = audio_batch.device
        mb.wav, mb.y_or, mb.y_a, mb.y_b, mb.lam, mb.index = (
            mb.wav.to(dev), mb.y_or.to(dev), mb.y_a.to(dev), mb.y_b.to(dev), mb.lam.to(dev), mb.index.to(dev))
        return mb

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(1)
        mb = self._mix_waveforms(audio_batch, labels, valid_len=meta_labels) if self.training else None
        if mb is None:
            logits = self.classify(self.forward_features(self.apply_specaugment(self.compute_fbank(audio_batch))))
            return logits, loss_func(logits, labels)

        fb = self.apply_specaugment(self.compute_fbank(mb.wav))
        if self.variant == "lungmix_patchmix":
            from rsc_toolkit.methods.patchmix import PatchMixLoss

            feats, y_a, y_b, lam, index = self.forward_features(fb, y=mb.y_or, patch_mix=True)
            logits = self.classify(feats)
            if self.patchmix_label_mode == "preserve":
                return logits, loss_func(logits, mb.y_or)
            return logits, PatchMixLoss(criterion=loss_func)(logits, y_a, y_b, lam)

        logits = self.classify(self.forward_features(fb))
        loss = self.lungmix_loss(logits, mb.y_or, mb.y_a, mb.y_b, mb.lam, criterion=loss_func)
        return logits, loss
