"""
AMS-D(CcnNnn/AMS-D) P0 골든·구성 테스트.

골든 출처: scripts/reproduction/dump_amsd_goldens.py — 공식 models/ams_d.py·custom_loss.py·utils/metric.py 를 timm 0.4.5 참조 venv(CPU) 에서
실행 (난수 파라미터는 det_fill 로 결정화, 실제 16/16 ckpt 로드). augment.pt 는 .venv 에서 공식 loader/mydataset.py ICBHIDataset 을 실제 실행한 값(합성 wav, loop/short 두 패딩 경로).
골든/ckpt 가 없으면 skip.
"""
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rsc_toolkit.methods.amsd import AMSDLoss
from rsc_toolkit.metrics.icbhi.binary_score import PaperBinaryScore
from rsc_toolkit.metrics.icbhi.score import ICBHIScore


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "amsd" / "model.pt"
AUG = REPO / "tests" / "golden" / "amsd" / "augment.pt"
CKPT = REPO / "pretrained_models/upstream/audioset_16_16_0.4422.pth"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="골든 없음")
needs_aug = pytest.mark.skipif(not AUG.exists(), reason="증강 골든 없음")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="AudioSet 16/16 ckpt 없음")
BUFFERS = {"all_predictions", "ids_masked", "current_epoch"}


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


def _same_mask(ours, ref):
    """마스크 집합 동일(정렬 후 equal) + 순서는 95% 이상 일치 — 동률에 가까운 점수(1e-6 차) 의 argsort 순서만 timm 버전 간에 흔들린다."""
    return torch.equal(ours.sort(dim=1).values, ref.sort(dim=1).values) and (ours == ref).float().mean() > 0.95


def _build(**kw):
    from rsc_toolkit.models.ast_amsd import AMSDAST

    m = AMSDAST(n_cls=4, pretrained_path=str(CKPT), num_train_samples=8, **kw)
    for mod in (m.mlp_head, m.gpredlayer):
        for p in mod.parameters():
            _det_fill(p)
    for p in m.token_weight.parameters():  # 골든과 동일: 비퇴화 sin 채움 (linspace 는 토큰 점수가 전부 동률)
        with torch.no_grad():
            p.copy_((torch.sin(torch.arange(p.numel(), dtype=torch.float32) * 0.7) * 0.05).reshape(p.shape))
    _det_fill(m.mask_embed)
    return m.eval()


# ------------------------------------------------------------------ 손실 / 지표 (골든 불필요)
def test_loss_double_softmax_teacher_epoch0():
    crit = AMSDLoss()
    s = torch.zeros(2, 4)
    t = torch.eye(4)[[0, 1]]
    total, ce, w, kl = crit(s, t, None, t, 0)
    teacher = F.softmax(t, 1)  # 이중 softmax: [0.475,0.175,0.175,0.175]
    assert torch.allclose(teacher[0], torch.tensor([0.4754, 0.1749, 0.1749, 0.1749]), atol=1e-3)
    ref_kl = F.kl_div(F.log_softmax(s, 1), teacher, reduction="batchmean")
    assert torch.allclose(kl, ref_kl) and torch.allclose(ce, F.cross_entropy(s, torch.tensor([0, 1]))) and float(w) == 0
    assert torch.allclose(total, ce + kl)
    # 가정 플래그: teacher 를 확률로 그대로
    total2, _, _, kl2 = AMSDLoss(teacher_is_prob=True)(s, t, None, t, 0)
    assert torch.allclose(kl2, F.kl_div(F.log_softmax(s, 1), t, reduction="batchmean"))


def test_binary_metric_vs_standard():
    logits = F.one_hot(torch.tensor([0, 1, 2, 2, 0, 1, 0]), 4).float()
    target = torch.tensor([0, 0, 1, 2, 3, 1, 0])
    pb, st = PaperBinaryScore(), ICBHIScore(num_classes=4)
    pb.update(logits, target); st.update(logits, target)
    p, s = pb.compute(), st.compute()
    # 이진: 비정상 4개 중 ŷ≠0 은 3개 → 75; 정상 3개 중 ŷ=0 은 2개 → 66.67
    assert abs(float(p["paper_S_e"]) - 75.0) < 1e-4 and abs(float(p["paper_S_p"]) - 200 / 3) < 1e-4
    # 표준 4-class: 비정상 정확 클래스는 (1→2 ✗, 2→2 ✓, 3→0 ✗, 1→1 ✓) 2/4 = 50
    assert abs(float(s["S_e"]) - 50.0) < 1e-4 and abs(float(s["S_p"]) - float(p["paper_S_p"])) < 1e-4
    assert float(p["paper_score"]) > float(s["icbhi_score"])


@needs_golden
def test_binary_metric_matches_reference_cases():
    g = torch.load(GOLDEN, weights_only=False)
    for case in g["metric_cases"]:
        m = PaperBinaryScore()
        m.update(F.one_hot(torch.tensor(case["pred"]), 4).float(), torch.tensor(case["target"]))
        r = m.compute()
        assert abs(float(r["paper_S_e"]) / 100 - case["tpr"]) < 1e-6 and abs(float(r["paper_S_p"]) / 100 - case["tnr"]) < 1e-6
        assert abs(float(r["paper_acc"]) / 100 - case["acc"]) < 1e-6


# ------------------------------------------------------------------ 데이터 증강
@needs_aug
@pytest.mark.parametrize("case", ["loop", "short"])
def test_augment_pipeline_matches_official_dataset(case):
    """툴킷 파이프라인(AudioPreprocessor → MelBankAudioProcessor(raw, 798) → AMSDFeatureDataset) = 공식 ICBHIDataset.__getitem__ 실행값.
    loop = 논문 8 s 반복 패딩(기본 가정, 질문 #10), short = 코드 fallback(fbank 798 raw 0-패딩; desired_length_sec=null pad_type=zero)."""
    import torchaudio

    from rsc_toolkit.datasets.amsd_data import AMSDFeatureDataset
    from rsc_toolkit.processors.audio import AudioPreprocessConfig, AudioPreprocessor
    from rsc_toolkit.processors.mel import MelBankAudioProcessor

    g = torch.load(AUG, weights_only=False)
    c = g["cases"][case]
    wav = g["wav_short"]  # 툴킷 입력은 항상 2.7 s 클립 — loop 케이스는 AudioPreprocessor 가 8 s 로 반복 패딩
    cfg = AudioPreprocessConfig(desired_length_sec=8.0, pad_type="repeat") if case == "loop" else AudioPreprocessConfig(None, pad_type="zero")
    audio = AudioPreprocessor(g["sr"], cfg)(wav)
    if case == "loop":
        assert torch.equal(audio, g["wav_loop"]) and audio.shape[1] == 8 * g["sr"]
    else:
        assert torch.equal(audio, wav)
    proc = MelBankAudioProcessor(n_mels=128, sampling_rate=g["sr"], target_frames=g["target_length"], frame_shift=10,
                                 window_type="hanning", dither=0.0, normalize=False, feature_cache=None)
    raw_ft = proc.make_features(audio)                        # (F, T) raw, 0-패딩/절단 완료
    # 공식 _wav2fbank 는 waveform 전역 평균을 먼저 뺀다 — kaldi remove_dc_offset(프레임별) 이 수학적으로 흡수하지만 float32 에서 최대 2.4e-4
    # (값 크기 ~10, 상대 2e-5) 의 수치 차가 남는다(전역 평균을 빼면 0.0). 공용 MelBankAudioProcessor 는 건드리지 않고 허용오차로 기록.
    assert torch.allclose(raw_ft.transpose(0, 1), c["raw"], atol=5e-4), (raw_ft.transpose(0, 1) - c["raw"]).abs().max()
    if case == "short":
        assert c["n_frames_before_pad"] < 798 and torch.count_nonzero(c["raw"][c["n_frames_before_pad"]:]) == 0  # fallback 0-패딩 분기
    else:
        assert c["n_frames_before_pad"] == 798                                                                   # 8 s → 패딩 분기 미실행

    ds = AMSDFeatureDataset.__new__(AMSDFeatureDataset)
    ds.training, ds.freqm, ds.timem, ds.mean, ds.std = True, g["freqm"], g["timem"], g["mean"], g["std"]
    ds.freq_mask = torchaudio.transforms.FrequencyMasking(g["freqm"] // 2)
    ds.time_mask = torchaudio.transforms.TimeMasking(g["timem"] // 2)
    torch.manual_seed(g["mask_seed"])
    train_tf = ds.transform(raw_ft).transpose(0, 1)           # (T, F): 마스킹 → 정규화
    assert torch.allclose(train_tf, c["train_nonoise"], atol=1e-4), (train_tf - c["train_nonoise"]).abs().max()  # 위 fbank 차 /(2·std)
    ds.training = False
    assert torch.allclose(ds.transform(raw_ft).transpose(0, 1), c["eval"], atol=1e-4)
    if case == "short":  # 패딩 영역 값 = (0-mean)/(2std) (raw 0-패딩 후 정규화 — 툴킷 기본 '정규화 후 0-패딩' 과 다름)
        assert torch.allclose(c["eval"][-1], torch.full((128,), (0 - g["mean"]) / (2 * g["std"])))
    # 잡음+roll: torch 부분은 모델 _noise_roll 과 같은 시드(2345+epoch+index); 진폭·shift 는 공식 전역 numpy RNG 값을 골든에서 재생
    gen = torch.Generator().manual_seed(2345 + g["epoch"] + g["index"])
    noised = c["train_nonoise"] + torch.rand(798, 128, generator=gen) * g["amp"]
    assert torch.allclose(torch.roll(noised, g["shift"], 0), c["train"], atol=1e-6)


# ------------------------------------------------------------------ 모델 골든
@needs_golden
@needs_ckpt
def test_model_matches_reference():
    g = torch.load(GOLDEN, weights_only=False)
    m = _build()
    d = g["dims"]
    assert (m.f_dim, m.t_dim, m.num_patches, m.num_masked) == (d["f_dim"], d["t_dim"], d["num_patches"], d["num_masked"]) == (8, 49, 392, 153)
    ours = {k: list(v.shape) for k, v in m.state_dict().items() if k not in BUFFERS and not k.startswith("criterion")}
    assert ours == g["state_keys"]
    assert torch.allclose(m.v.pos_embed, g["pos_embed"], atol=1e-6)
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["module.v.pos_embed"]
    cut = raw[:, 2:].reshape(1, 512, 768).transpose(1, 2).reshape(1, 768, 8, 64)[:, :, :, 8:57].reshape(1, 768, 392).transpose(1, 2)
    assert torch.allclose(m.v.pos_embed[:, 2:], cut)
    assert g["unfold_vs_conv_maxdiff"] < 1e-4  # (f) Unfold 순서 = conv 패치 순서

    x = g["input"]  # (2, 798, 128) time-freq
    with torch.no_grad():
        logits = m(x.transpose(-2, -1))  # 툴킷 배치 (B,128,798) 경로
        assert torch.allclose(logits, g["eval_logits"], rtol=1e-4, atol=1e-3), (logits - g["eval_logits"]).abs().max()
        w0, mse0, ids0, s0 = m.amsd_forward(x, 0, None)
        assert float(mse0) == 0 and _same_mask(ids0, g["ep0"]["ids_masked"])
        assert torch.allclose(s0, g["ep0"]["s_out"], rtol=1e-4, atol=1e-3) and torch.allclose(w0, g["ep0"]["weighted_x"], rtol=1e-4, atol=1e-4)
        w1, mse1, ids1, s1 = m.amsd_forward(x, 1, g["ep1"]["ids_in"])
        assert torch.allclose(s1, g["ep1"]["s_out"], rtol=1e-4, atol=1e-3), (s1 - g["ep1"]["s_out"]).abs().max()
        assert torch.allclose(w1, g["ep1"]["weighted_x"], rtol=1e-4, atol=1e-4)
        assert torch.allclose(mse1, g["ep1"]["mse"], rtol=1e-4), (float(mse1), float(g["ep1"]["mse"]))
        assert _same_mask(ids1, g["ep1"]["ids_masked"])
        tokens = m.v.patch_embed(x.unsqueeze(1).transpose(2, 3))
        xm, _ = m.attentive_mask(tokens, g["ep1"]["ids_in"])
        assert torch.allclose(xm, g["ep1"]["x_masked"], atol=1e-5)
        # (g) 다른 마스크 모드
        torch.manual_seed(g["masks"]["random_seed"])
        _, ids_rand = m.random_mask_patch(tokens.clone(), m.num_masked)
        assert torch.equal(ids_rand, g["masks"]["random"])
        assert torch.equal(m.time_interval_masking(tokens, m.t_dim, m.f_dim, m.num_masked)[1], g["masks"]["time_inter"])
        assert torch.equal(m.freq_interval_masking(tokens, m.t_dim, m.f_dim, m.num_masked)[1], g["masks"]["freq_inter"])
    # (d) Custom_Loss
    onehot = torch.eye(4)[g["labels"]]
    for key, teacher in (("onehot", onehot), ("prob", g["loss"]["t_prob"])):
        ours = m.criterion(g["ep1"]["s_out"], teacher, g["ep1"]["weighted_x"], onehot, g["ep1"]["mse"])
        for a, b in zip(ours, g["loss"][key]):
            assert torch.allclose(torch.as_tensor(a), b, rtol=1e-5), (key, float(a), float(b))
    ours0 = m.criterion(g["ep0"]["s_out"], onehot, g["ep0"]["weighted_x"], onehot, 0)
    for a, b in zip(ours0, g["loss"]["ep0"]):
        assert torch.allclose(torch.as_tensor(a), b, rtol=1e-5)


@needs_golden
@needs_ckpt
def test_training_forward_epoch_transition_and_buffers():
    """epoch 0: one-hot teacher 저장·마스킹 없음 → epoch 1: 저장된 ids 로 마스킹·확률 teacher (main.py:321-350)."""
    g = torch.load(GOLDEN, weights_only=False)
    m = _build().train()
    x = g["input"].transpose(-2, -1)  # (2,128,798)
    labels, idx = g["labels"], torch.tensor([3, 5])
    m.current_epoch.fill_(0)
    logits, loss = m.training_forward(x, labels, F.cross_entropy, meta_labels=idx)
    assert torch.allclose(logits, g["ep0"]["s_out"], rtol=1e-4, atol=1e-3)
    assert torch.allclose(m.all_predictions[idx], F.softmax(g["ep0"]["s_out"], 1), atol=1e-4)
    assert _same_mask(m.ids_masked[idx], g["ep0"]["ids_masked"]) and torch.count_nonzero(m.ids_masked[[0, 1, 2, 4, 6, 7]]) == 0
    assert torch.allclose(loss, g["loss"]["ep0"][0], rtol=1e-4)
    m.current_epoch.fill_(1)
    logits1, loss1 = m.training_forward(x, labels, F.cross_entropy, meta_labels=idx)
    assert torch.allclose(logits1, g["ep1"]["s_out"], rtol=1e-4, atol=1e-3)
    assert torch.allclose(loss1, g["loss"]["prob"][0], rtol=1e-4), (float(loss1), float(g["loss"]["prob"][0]))
    # 검증 경로(meta 없음) = AST 로짓 + CE, 버퍼 불변
    m.eval()
    snap = m.all_predictions.clone()
    vl, vloss = m.training_forward(x, labels, F.cross_entropy)
    assert torch.allclose(vl, g["eval_logits"], rtol=1e-4, atol=1e-3) and torch.equal(snap, m.all_predictions)
    # 버퍼가 state_dict 에 포함 (last.ckpt 저장·복원)
    assert BUFFERS <= set(m.state_dict())


@needs_ckpt
def test_method_variants_and_noise():
    torch.manual_seed(0)
    x = torch.randn(2, 128, 798)
    labels, idx = torch.tensor([0, 2]), torch.tensor([0, 1])
    for method in ("baseline", "sd_only", "sd_mask"):
        m = _build(method=method).train()
        m.current_epoch.fill_(1)
        logits, loss = m.training_forward(x, labels, F.cross_entropy, meta_labels=idx)
        assert logits.shape == (2, 4) and torch.isfinite(loss)
    m = _build(mask_mode="random").train()  # 비-attentive: epoch 0 부터 마스킹·MSE, 보조 없음
    with torch.no_grad():
        w, mse, ids, s = m.amsd_forward(x.transpose(-2, -1), 0, None)
    assert w is None and float(mse) > 0 and ids.shape == (2, 153)
    # 잡음·roll: 같은 (epoch, index) 면 결정적, index 가 다르면 다름
    m.current_epoch.fill_(10)
    a = m._noise_roll(x.transpose(-2, -1), 10, idx)
    b = m._noise_roll(x.transpose(-2, -1), 10, idx)
    c = m._noise_roll(x.transpose(-2, -1), 10, idx + 1)
    assert torch.equal(a, b) and not torch.equal(a, c) and a.shape == (2, 798, 128)
