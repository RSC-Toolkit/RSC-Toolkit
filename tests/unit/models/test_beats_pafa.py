"""
PAFA 묶음(BEATs SFT · BEATs+PAFA)의 P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_pafa_goldens.py — 공식 wa976/PAFA 코드를 우리 venv 에서
실행해 생성 (BEATs 코드는 그 저장소의 수정본을 그대로 옮겼으므로 특징은 비트 단위로 같아야 한다).
"""
import pathlib

import pytest
import torch
import torch.nn as nn

from rsc_toolkit.methods.pafa import PAFALoss, ProjectionHead


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "pafa"
CKPT = REPO / "pretrained_models" / "upstream" / "BEATs_iter3_plus_AS2M.pt"
ICBHI = REPO / ".datasets" / "icbhi" / "index" / "index_cycle.csv"

needs_golden = pytest.mark.skipif(not (GOLDEN / "pafa_head_loss.pt").exists(), reason="골든 없음")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="BEATs iter3+ AS2M ckpt 없음")
needs_icbhi = pytest.mark.skipif(not ICBHI.exists(), reason="ICBHI 인덱스 없음")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


# ------------------------------------------------------------ 손실/헤드
@needs_golden
@pytest.mark.exact_golden  # 비트 단위 비교 — 개발 머신 전용 (CPU/BLAS/스레드 수가 다르면 마지막 자리에서 어긋난다)
def test_projection_head_and_pafa_loss_match_reference():
    g = torch.load(GOLDEN / "pafa_head_loss.pt")
    head = ProjectionHead(768, None, 768, attention=True, norm_type="ln", proj_type="end2end")
    for _, p in sorted(head.named_parameters()):
        _det_fill(p)
    head.eval()
    with torch.no_grad():
        proj = head(g["features"])
    assert torch.equal(proj, g["proj"])
    loss = PAFALoss()(proj, g["patient_ids"], lambda_pcsl=50.0, lambda_gpal=0.0005)
    assert torch.equal(loss.detach(), g["loss"])
    single = PAFALoss()(proj, torch.full_like(g["patient_ids"], 5), lambda_pcsl=50.0, lambda_gpal=0.0005)
    assert torch.equal(single.detach(), g["loss_single"]) and float(single) == 0.0


def test_pafa_loss_components():
    """PCSL = 환자 내 분산 / centroid 간 거리², GPAL = centroid-전역 centroid 거리² 평균 (원본 수식)."""
    f = torch.tensor([[0., 0.], [2., 0.], [10., 0.], [10., 2.]])
    pid = torch.tensor([1, 1, 2, 2])
    c1, c2 = torch.tensor([1., 0.]), torch.tensor([10., 1.])
    within = 1 + 1 + 1 + 1
    between = float(((c1 - c2) ** 2).sum())
    g = (c1 + c2) / 2
    gpal = float((((c1 - g) ** 2).sum() + ((c2 - g) ** 2).sum()) / 2)
    expected = 50.0 * within / (between + 1e-6) + 0.0005 * gpal
    assert torch.isclose(PAFALoss()(f, pid, 50.0, 0.0005), torch.tensor(expected))


# ------------------------------------------------------------------ 모델
@needs_golden
@needs_ckpt
def test_beats_features_match_reference():
    from rsc_toolkit.models.beats_pafa import BEATsClassifier

    g = torch.load(GOLDEN / "beats_features.pt")
    m = BEATsClassifier(n_cls=4, method="ce").eval()
    with torch.no_grad():
        feats = m.forward_features(g["wav"])
    assert feats.shape == g["features"].shape
    assert torch.allclose(feats, g["features"], atol=1e-5), float((feats - g["features"]).abs().max())


@needs_ckpt
def test_beats_classifier_ce_and_pafa_paths():
    from rsc_toolkit.models.beats_pafa import BEATsClassifier

    wav = torch.randn(4, 80000) * 0.05
    labels = torch.tensor([0, 1, 2, 3])
    pids = torch.tensor([101, 101, 226, 226])

    ce = BEATsClassifier(n_cls=4, method="ce")
    ce.train()
    logits, loss = ce.training_forward(wav, labels, nn.CrossEntropyLoss(), meta_labels=pids)
    assert logits.shape == (4, 4) and torch.isfinite(loss)
    assert isinstance(ce.projector, nn.Identity)

    pafa = BEATsClassifier(n_cls=4, method="pafa")
    pafa.train()
    logits2, loss2 = pafa.training_forward(wav, labels, nn.CrossEntropyLoss(), meta_labels=pids)
    assert torch.isfinite(loss2) and loss2.requires_grad
    assert pafa.projector.projection[0].out_features == 768
    with pytest.raises(ValueError, match="patient"):
        pafa.training_forward(wav, labels, nn.CrossEntropyLoss())
    pafa.eval()
    with torch.no_grad():
        _, loss_eval = pafa.training_forward(wav, labels, nn.CrossEntropyLoss())  # 평가는 CE 만
    assert torch.isfinite(loss_eval)


# ------------------------------------------------------------ 데이터
@needs_golden
@needs_icbhi
def test_waveform_pipeline_matches_reference():
    """IndexDataset(16 kHz, fade) + repeat_fade 5초 == 원본 slice + cut_pad_sample_torchaudio."""
    from rsc_toolkit.datasets.index_dataset import IndexDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor

    golden = torch.load(GOLDEN / "pipeline_golden.pt")
    pre = AudioPreprocessor(16000, AudioPreprocessConfig(desired_length_sec=5.0, pad_type="repeat_fade"))
    ds = IndexDataset(index_csv=ICBHI, data_root=REPO / ".datasets" / "icbhi", sampling_rate=16000,
                      sample_level="cycle", audio_preprocessor=pre, fade=True)
    by_id = {str(s): i for i, s in enumerate(ds.df["sample_id"])}
    for g in golden:
        item = ds[by_id[g["sample_id"]]]
        wav = item["audio"].squeeze(0)
        assert wav.shape == (80000,)
        assert torch.equal(wav, g["wav"]), g["sample_id"]


def test_patient_meta_label_mode():
    from rsc_toolkit.datasets.meta_labels import build_meta_label_fn

    assert build_meta_label_fn("patient")({"meta": {"patient_id": 226}}) == 226


def test_waveform_processor_passthrough():
    from rsc_toolkit.processors.waveform import WaveformProcessor

    p = WaveformProcessor(sampling_rate=16000)
    x = torch.randn(1, 80000)
    out = p.make_features_cached(x, sample_id="a")
    assert out.shape == (80000,) and torch.equal(out, x.squeeze(0))
    assert p.get_processor_config()["kind"] == "waveform"
