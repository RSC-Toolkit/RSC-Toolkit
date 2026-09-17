"""
M2D 미세조정 묶음의 P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_m2d_goldens.py — 공식 nttcslab/m2d RuntimeM2D(+scl_icbhi2017 변환)를 우리 venv 에서 실행.
인코더는 툴킷 재구현(models/m2d/encoder.py)이며 공식 ckpt 를 같은 키로 로드해 임베딩을 대조한다.
"""
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.augmentations.specaugment_torchaudio import SpecAugmentTA
from rsc_toolkit.models.m2d import M2DEncoder, get_2d_sincos_pos_embed, load_m2d_checkpoint
from rsc_toolkit.processors.sclmeta_mel import SCLMetaMelProcessor, batch_minmax, standardize


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "m2d"
CKPT = REPO / "pretrained_models/upstream/m2d/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth"
needs_golden = pytest.mark.skipif(not (GOLDEN / "features.pt").exists(), reason="골든 없음")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="M2D ckpt 없음")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def test_encoder_layout_and_pos_embed():
    enc = M2DEncoder()
    assert len(enc.state_dict()) == 151 and enc.grid_size() == (5, 50)
    assert tuple(enc.pos_embed.shape) == (1, 251, 768) and not enc.pos_embed.requires_grad
    pe = get_2d_sincos_pos_embed(768, (5, 50), cls_token=True)
    assert pe.shape == (251, 768) and (pe[0] == 0).all()
    x = torch.randn(2, 1, 80, 801) * 0.1
    with torch.no_grad():
        feats = enc.encode_lms(x)
    assert feats.shape == (2, 201, 3840)  # 801 → pad 804 → 4x50 + 1


@needs_golden
@pytest.mark.exact_golden  # 비트 단위 비교 — 개발 머신 전용 (CI 러너마다 CPU 가 달라 마지막 자리에서 어긋난다)
def test_feature_stages_match_reference():
    g = torch.load(GOLDEN / "features.pt")
    proc = SCLMetaMelProcessor(n_fft=400, win_length=400, hop_length=160, n_mels=80, f_min=50, f_max=8000)
    # 참조는 배치(6,1,128000) 를 한 번에 STFT 했고 우리는 샘플별(캐시)이라 FFT 경로 차이로 ~1e-5 dB 오차
    dbmel = torch.stack([proc.make_features(w) for w in g["wav"]])
    assert torch.allclose(dbmel, g["dbmel"], atol=1e-3)
    assert torch.allclose(standardize(batch_minmax(dbmel), 0.3671, 0.2391), g["val"], atol=1e-3)
    sa = SpecAugmentTA(20, 40, 2, 2, 1.0)
    torch.manual_seed(g["seed"])
    v1 = standardize(sa(batch_minmax(dbmel)), 0.3671, 0.2391)
    v2 = standardize(sa(batch_minmax(dbmel)), 0.3671, 0.2391)
    assert torch.allclose(v1, g["view1"], atol=1e-3) and torch.allclose(v2, g["view2"], atol=1e-3)


@needs_golden
@needs_ckpt
def test_checkpoint_load_and_embeddings_match_reference():
    g = torch.load(GOLDEN / "encoder.pt"); f = torch.load(GOLDEN / "features.pt")
    enc = M2DEncoder()
    info = load_m2d_checkpoint(enc, CKPT)
    ours = {k: list(v.shape) for k, v in enc.state_dict().items()}
    assert ours == g["state_dict_shapes"]
    assert sorted(info["dropped"]) == g["dropped"]
    assert torch.equal(enc.norm_stats, g["norm_stats"]) and torch.equal(enc.pos_embed, g["pos_embed"])
    enc.eval()
    with torch.no_grad():
        emb = enc.encode_lms(f["val"])
    assert emb.shape == g["emb"].shape
    assert torch.allclose(emb, g["emb"], atol=1e-4), float((emb - g["emb"]).abs().max())


@needs_golden
@needs_ckpt
def test_head_and_loss_match_reference():
    from rsc_toolkit.models.m2d_finetune import M2DFinetune
    from rsc_toolkit.trainer.losses import WeightedCrossEntropyFromCounts

    g = torch.load(GOLDEN / "head.pt"); f = torch.load(GOLDEN / "features.pt")
    m = M2DFinetune(n_cls=4, pretrained_path=str(CKPT)).eval()
    for _, p in sorted(m.linear.named_parameters()):
        _det_fill(p)
    assert all(not p.requires_grad for p in m.encoder.patch_embed.parameters())
    with torch.no_grad():
        logits = m(f["val"])
    assert torch.allclose(logits, g["logits"], atol=1e-3)
    loss = WeightedCrossEntropyFromCounts([2063, 1215, 501, 363])(logits, g["labels"])
    assert torch.allclose(loss, g["loss"], atol=1e-4)


def test_split_iter_adapter_matches_manual_accumulation():
    """K 부분배치 backward 누적 + step 1회 = 원본 ce.py train_epoch; 나머지 샘플은 버린다."""
    from rsc_toolkit.trainer.adapters.split_iter_adapter import SplitIterAdapter

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(6, 3)

        def train_transform(self, x):
            return x

        def val_transform(self, x):
            return x

        def forward(self, x):
            return self.lin(x.mean(dim=1))

    torch.manual_seed(0)
    x = torch.randn(10, 2, 6); y = torch.randint(0, 3, (10,))
    ref = Tiny(); ad = Tiny(); ad.load_state_dict(ref.state_dict())
    ce = nn.CrossEntropyLoss()
    # 참조: K=4, D=2 → 8개만 사용, 손실 합의 gradient 로 SGD 1 step
    opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
    total = sum(ce(ref(x[i * 2:(i + 1) * 2]), y[i * 2:(i + 1) * 2]) for i in range(4))
    total.backward(); opt_ref.step()

    adapter = SplitIterAdapter(rsc_model=ad, loss_func=ce, optimizer_cfg={"name": "sgd", "params": {"lr": 0.1}}, split_iter=4)
    opt = adapter.configure_optimizers()
    adapter.optimizers = lambda: opt  # Trainer 없이 수동 호출
    adapter.manual_backward = lambda loss: loss.backward()
    adapter.log = lambda *a, **k: None
    adapter.training_step((x, y), 0)
    for p_ref, p_ad in zip(ref.parameters(), ad.parameters()):
        assert torch.allclose(p_ref, p_ad, atol=1e-6)


def test_split_iter_adapter_validation_covers_remainder():
    """검증은 D=B//K 서브배치 K 개 + 나머지 1 개 → 배치의 모든 샘플이 지표에 들어간다(원본 ce.py 는 나머지를 버렸다; 2026-09-16 수정).
    앞 K·D 건은 원본과 같은 서브배치 구성이라 로짓이 동일해야 한다(기존 1,428 건 예측 불변)."""
    from rsc_toolkit.trainer.adapters.split_iter_adapter import SplitIterAdapter

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(6, 3)

        def train_transform(self, x):
            return x

        def val_transform(self, x):
            return x

        def forward(self, x):
            return self.lin(x.mean(dim=1))

    seen = {}

    class Rec(nn.Module):
        def update(self, logits, target):
            seen["logits"], seen["y"] = logits, target

    torch.manual_seed(0)
    x = torch.randn(21, 2, 6); y = torch.randint(0, 3, (21,))       # SPRSound 마지막 배치와 같은 21 건
    model = Tiny(); ce = nn.CrossEntropyLoss()
    adapter = SplitIterAdapter(rsc_model=model, loss_func=ce, optimizer_cfg={"name": "sgd", "params": {"lr": 0.1}}, split_iter=4)
    adapter.log = lambda *a, **k: None
    adapter.val_metrics = nn.ModuleList([Rec()])
    assert adapter.val_chunks(21) == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 21)]
    assert adapter.val_chunks(64) == [(0, 16), (16, 32), (32, 48), (48, 64)]   # 나머지 없음 → 원본과 동일
    assert adapter.val_chunks(3) == [(0, 3)]                                    # B < K
    adapter.validation_step((x, y), 0)
    assert seen["y"].shape[0] == 21 and torch.equal(seen["y"], y)
    with torch.no_grad():
        ref = torch.cat([model(x[i * 5:(i + 1) * 5]) for i in range(4)])     # 원본 방식의 앞 20 건
    assert torch.allclose(seen["logits"][:20], ref)
