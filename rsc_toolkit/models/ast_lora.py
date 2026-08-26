"""
LoRA-RSC(ICASSP 2025, deegy666/LoRA-RSC d1f0653) — HF AST(AudioSet 10/10) 의 K·V 투영에 LoRA 를 더한 ICBHI 분류기, 공식 코드 충실 이식 (P0).

원본 src/AST_LoRA_ICBHI.py AST_LoRA / src/AST.py AST (Cappellazzo PETL_AST 의 변형):
  ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=500, ignore_mismatched_sizes=True)
    → 12×49=588 패치 + cls/dist, position_embeddings(1,590,768) 는 ckpt(1214) 와 모양이 달라 미로드 → **0 그대로, 동결** (보간 없음)
  ASTSelfAttention_LoRA: key/value 에 lora_up(lora_down(h))·scaling 가산, lora_down: 768→round(768/96)=8, lora_up: 8→768 (bias 없음),
    scaling = alpha/rank_arg = 8/96 (rank 인자 96 은 사실 reduction rate; 진짜 rank 8)
  transformers 4.28.1 from_pretrained fast-init quirk: ckpt 에 없는 새 nn.Linear(lora_down/up) 에 AST._init_weights 가 다시 적용되어
    __init__ 의 zeros_/kaiming_uniform_ 가 **trunc_normal(std 0.02)** 로 덮어써진다 → 초기 LoRA 델타 ≠ 0 (골든으로 확인). 기본값으로 유지.
  embeddings·encoder 동결, LoRA + 최종 LayerNorm + Linear(768→4) 학습 (299,524 파라미터). train() 오버라이드로 encoder/embeddings 는 항상 eval.
  풀링: CLS 토큰(hidden[:,0]) 만 (HF pooler 의 (cls+dist)/2 아님).
  full-FT(src/AST.py): 같은 구성(pos_embed 0, 학습 가능) 전체 학습, lr 5e-5.  linear: 백본 동결 + 최종 LN·헤드만.

백본은 .venv 의 transformers 5.15.1 ASTModel(구성만; from_pretrained 미사용) 에 HF ckpt 를 4.28.1 키 → 5.15.1 키로 매핑해 직접 로드한다
(encoder.layer.i.attention.attention.{query,key,value} → layers.i.attention.{q,k,v}_proj, attention.output.dense → o_proj,
 intermediate.dense → mlp.fc1, output.dense → mlp.fc2). LoRA 는 k_proj/v_proj nn.Linear 를 LoRALinear 로 감싸 재현 (peft 미사용).
입력: 툴킷 fbank (B,128,T) — kaldi 128 mel·AudioSet 상수 정규화 = HF ASTFeatureExtractor 와 동일(골든), T>500 이면 앞 500 프레임만
(원본 max_length=500 절단 = 8 s 순환 패딩 파형의 앞 5 s).
골든(tests/golden/lorarsc, 4.28.1 참조 venv): fast-init quirk(LoRA down std 0.020)·pos_embed 0·파라미터 299,524/85,711,108·로짓/손실 일치 확인.
파형 잡음(미상): 공개 코드의 (1,N) 배열 + randn(len)=스칼라 DC 오프셋은 fbank 에서 제거돼 무효이지만, 그 호출 자체가 4.28.1 추출기에서
AssertionError 로 실행 불가(골든 orig_2d_call) → 저자 실행의 잡음 형태(스칼라 vs 1D 샘플별 N(0,0.3²)) 는 미상(저자 질문 #6). 기본 config 는
잡음 없음, 1D 해석은 data=icbhi_lorarsc_noise (datasets/lorarsc_noise.py).
"""
from __future__ import annotations

import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRSCModel
from .factory import ModelFactory

logger = logging.getLogger(__name__)

HF_AST_CANDIDATES = (
    "pretrained_models/upstream/hf/ast-finetuned-audioset-10-10-0.4593",   # cwd(리포 루트) 기준; 다른 위치는 pretrained_dir 인자
)
HF_PREFIX = "audio_spectrogram_transformer."


class LoRALinear(nn.Module):
    """nn.Linear + lora_up(lora_down(x))·scaling — 원본 ASTSelfAttention_LoRA 의 key/value 경로 (bias 없는 down/up)."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float):
        super().__init__()
        self.base = base
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = float(scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


def map_hf_ast_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """HF ckpt(4.28.1 시절 키, ASTForAudioClassification) → transformers 5.x ASTModel 키. classifier.* 는 버린다."""
    out = {}
    for k, v in state.items():
        if not k.startswith(HF_PREFIX):
            continue  # classifier.*
        k = k[len(HF_PREFIX):]
        if k.startswith("encoder.layer."):
            parts = k.split(".")
            i, rest = parts[2], ".".join(parts[3:])
            rest = (rest.replace("attention.attention.query.", "attention.q_proj.")
                        .replace("attention.attention.key.", "attention.k_proj.")
                        .replace("attention.attention.value.", "attention.v_proj.")
                        .replace("attention.output.dense.", "attention.o_proj.")
                        .replace("intermediate.dense.", "mlp.fc1.")
                        .replace("output.dense.", "mlp.fc2."))
            k = f"layers.{i}.{rest}"
        out[k] = v
    return out


def _load_hf_state(hf_dir: pathlib.Path) -> Dict[str, torch.Tensor]:
    st = hf_dir / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    return torch.load(hf_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)


@ModelFactory.register_model_class("ast_lora_rsc")
class ASTLoRARSC(BaseRSCModel, nn.Module):
    def __init__(
        self,
        n_cls: int = 4,
        method: str = "lora",                     # lora | full_ft | linear (원본 --method LoRA / full-FT / linear)
        max_length: int = 500,                    # train.yaml max_len_AST_ICBHI (특징 프레임 수; 8 s 파형의 앞 5 s)
        num_mel_bins: int = 128,
        pretrained_dir: Optional[str] = None,     # HF 스냅샷 디렉토리 (config.json + model.safetensors)
        pretrained: bool = True,
        lora_reduction: int = 96,                 # --reduction_rate_lora (원본이 'rank' 로 넘기는 값; 실제 rank = round(768/96) = 8)
        lora_alpha: float = 8.0,                  # --alpha_lora; scaling = alpha / reduction (= 8/96)
        lora_targets: Sequence[str] = ("k", "v"), # 원본 K,V (PETL_AST 원형은 Q,V)
        lora_init: str = "hf_fast_init",          # hf_fast_init: down/up 모두 trunc_normal(0.02) (4.28.1 quirk) | paper: down 0 / up kaiming_uniform(√5)
        initializer_range: float = 0.02,
        pos_embed: str = "zeros",                 # zeros: 원본 quirk(미로드·0) | interpolate: AudioSet pos_embed 를 12×49 로 bilinear 보간
        final_output: str = "CLS",                # CLS | ALL (train.yaml final_output)
        attn_implementation: str = "eager",       # 4.28.1 과 같은 eager 어텐션 (sdpa 도 수학적으로 동일)
        input_layout: str = "freq_time",          # 툴킷 fbank (B,128,T)
        freeze_backbone_bn_eval: bool = True,     # 원본 train() 오버라이드: encoder/embeddings 는 항상 eval (dropout 0 이라 무영향)
    ):
        super().__init__()
        from transformers import ASTConfig, ASTModel

        assert method in ("lora", "full_ft", "linear"), method
        assert final_output in ("CLS", "ALL"), final_output
        self.method, self.max_length, self.final_output, self.input_layout = method, int(max_length), final_output, input_layout
        self.freeze_backbone_bn_eval = bool(freeze_backbone_bn_eval)
        self.final_feat_dim = 768

        hf_dir = self._resolve_dir(pretrained_dir)
        config = ASTConfig.from_pretrained(str(hf_dir), max_length=self.max_length, num_mel_bins=num_mel_bins)
        config._attn_implementation = attn_implementation
        self.ast = ASTModel(config)               # position_embeddings (1, 12*49+2, 768) = 0 (5.x _init_weights zeros)
        self.hidden = config.hidden_size
        self.load_info = None
        if pretrained:
            self.load_info = self._load_pretrained(hf_dir, pos_embed)
        self.pos_embed_mode = pos_embed

        # LoRA (K,V) — from_pretrained 뒤에 만들어진 새 Linear 이므로 원본 순서(로드 → 초기화 quirk) 와 같다
        self.lora_rank = round(self.hidden / lora_reduction)
        self.lora_scaling = float(lora_alpha) / float(lora_reduction)
        self.lora_targets = tuple(lora_targets)
        if method == "lora":
            for layer in self.ast.layers:
                for t in self.lora_targets:
                    name = f"{t}_proj"
                    setattr(layer.attention, name, LoRALinear(getattr(layer.attention, name), self.lora_rank, self.lora_scaling))
            self._init_lora(lora_init, initializer_range)

        self.classification_head = nn.Linear(self.hidden, n_cls)  # nn.Linear 기본 초기화 (원본 그대로)
        self._set_trainable()

    # ----------------------------------------------------------------- 구성
    @staticmethod
    def _resolve_dir(pretrained_dir: Optional[str]) -> pathlib.Path:
        cands = [pretrained_dir] if pretrained_dir else list(HF_AST_CANDIDATES)
        for c in cands:
            if c and (pathlib.Path(c) / "config.json").exists():
                return pathlib.Path(c)
        raise FileNotFoundError(f"HF AST 스냅샷이 없다: {cands}")

    def _load_pretrained(self, hf_dir: pathlib.Path, pos_embed: str) -> dict:
        state = map_hf_ast_state_dict(_load_hf_state(hf_dir))
        pe_key = "embeddings.position_embeddings"
        ckpt_pe = state.pop(pe_key)
        missing, unexpected = self.ast.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) == {pe_key}, (missing, unexpected)
        if pos_embed == "interpolate":
            # 툴킷 확장(원본 아님): AudioSet 12×101 격자 → 12×49 bilinear (YuanGong ast_models.py 방식)
            fo, to = self.ast.embeddings.get_shape(self.ast.config)
            pe = ckpt_pe[:, 2:, :].transpose(1, 2).reshape(1, self.hidden, 12, 101)
            pe = F.interpolate(pe, size=(fo, to), mode="bilinear").reshape(1, self.hidden, fo * to).transpose(1, 2)
            with torch.no_grad():
                self.ast.embeddings.position_embeddings.copy_(torch.cat([ckpt_pe[:, :2, :], pe], dim=1))
        elif pos_embed != "zeros":
            raise ValueError(pos_embed)
        logger.info("LoRA-RSC HF AST 로드: %d 키, pos_embed=%s (ckpt %s 미로드)", len(state), pos_embed, tuple(ckpt_pe.shape))
        return {"loaded": len(state), "pos_embed": pos_embed, "ckpt_pos_embed_shape": list(ckpt_pe.shape)}

    def lora_modules(self):
        for layer in self.ast.layers:
            for t in self.lora_targets:
                m = getattr(layer.attention, f"{t}_proj")
                if isinstance(m, LoRALinear):
                    yield m

    def _init_lora(self, lora_init: str, std: float):
        for m in self.lora_modules():
            if lora_init == "hf_fast_init":
                for w in (m.lora_down.weight, m.lora_up.weight):
                    with torch.no_grad():
                        nn.init.trunc_normal_(w, mean=0.0, std=std)
            elif lora_init == "paper":
                nn.init.zeros_(m.lora_down.weight)
                nn.init.kaiming_uniform_(m.lora_up.weight, a=5 ** 0.5)
            else:
                raise ValueError(lora_init)

    def _set_trainable(self):
        if self.method == "full_ft":
            self.requires_grad_(True)
            return
        self.ast.requires_grad_(False)
        self.ast.layernorm.requires_grad_(True)
        self.classification_head.requires_grad_(True)
        if self.method == "lora":
            for m in self.lora_modules():
                m.lora_down.requires_grad_(True)
                m.lora_up.requires_grad_(True)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.method != "full_ft" and self.freeze_backbone_bn_eval:
            self.ast.embeddings.eval()
            self.ast.layers.eval()
        return self

    # ------------------------------------------------------------------ 실행
    def to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B,128,T) 또는 (B,1,128,T) → (B, max_length, 128); T>max_length 면 앞 프레임만 (HF max_length 절단), 짧으면 0 패딩.

        주의(미도달 경로): T<max_length 패딩은 정규화 **후** 0 이지만 HF ASTFeatureExtractor 는 정규화 전에 ZeroPad2d 하므로 패딩 프레임이
        (0-mean)/(2std)=+0.467 이다. 툴킷 데이터 config 는 8 s 순환 패딩(798 프레임 ≥ 500) 이라 이 분기는 실제로 발생하지 않는다."""
        if x.dim() == 4:
            x = x.squeeze(1)
        if self.input_layout == "freq_time":
            x = x.transpose(-2, -1)
        T = x.shape[1]
        if T > self.max_length:
            x = x[:, : self.max_length]
        elif T < self.max_length:
            x = F.pad(x, (0, 0, 0, self.max_length - T))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_model_input(x)
        hidden = self.ast(input_values=x).last_hidden_state  # 최종 layernorm 포함
        feat = hidden[:, 0] if self.final_output == "CLS" else hidden.mean(dim=1)
        return self.classification_head(feat)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        logits = self.forward(audio_batch)
        return logits, loss_func(logits, labels)
