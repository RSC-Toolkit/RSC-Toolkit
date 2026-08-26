"""
DDE-MAE 묶음 테스트.

P0 골든(백본 Audio-MAE): scripts/reproduction/dump_ddemae_goldens.py — 공식 facebookresearch/AudioMAE models_mae.py/models_vit.py 를
별도 venv(timm 0.4.5, CPU) 에서 실행. 인코더 latent·디코더 pred·손실·fc_norm 특징을 1e-4 로 대조.
DDE 부분(2 인코더·concat·Siamese·vCLUB·time-shuffle)은 논문 명세 재구현 — 골든 없음, 구성·불변식 테스트만.
"""
import pathlib

import pytest
import torch

from rsc_toolkit.methods.dde_mae import VCLUB, permute_token_noise, siamese_hinge, time_shuffle, time_unshuffle
from rsc_toolkit.models.audiomae_dde import (
    AUDIOMAE_CKPT, AudioMAEClassifier, AudioMAEEncoder, DDEMAE, MAEDecoder, load_audiomae_checkpoint, mae_recon_loss,
    patchify, random_masking, unpatchify,
)


def _repo_root() -> pathlib.Path:
    for c in pathlib.Path(__file__).resolve().parents:
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("리포 루트를 찾지 못했다")


REPO = _repo_root()
GOLDEN = REPO / "tests" / "golden" / "ddemae"
CKPT = REPO / AUDIOMAE_CKPT
needs_golden = pytest.mark.skipif(not (GOLDEN / "mae.pt").exists(), reason="골든 없음")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="Audio-MAE ckpt 없음")


def _det_fill(t):
    with torch.no_grad():
        t.copy_(torch.linspace(-0.5, 0.5, t.numel()).reshape(t.shape))


# ----------------------------------------------------------------------------------------- 구조 (골든·ckpt 불필요)
def test_encoder_layout_and_pos_embed():
    enc = AudioMAEEncoder()
    assert enc.patch_embed.grid_size == (64, 8) and enc.patch_hw == (8, 64) and enc.patch_embed.num_patches == 512
    assert tuple(enc.pos_embed.shape) == (1, 513, 768) and not enc.pos_embed.requires_grad
    assert (enc.pos_embed[0, 0] == 0).all()
    x = torch.randn(1, 1, 1024, 128) * 0.1
    with torch.no_grad():
        lat, mask, ids = enc.forward_tokens(x, 0.8)
        assert lat.shape == (1, 1 + 102, 768) and mask.sum() == 512 - 102 and ids.shape == (1, 512)
        assert enc.forward_features(x).shape == (1, 768)
        enc.global_pool = False
        ident = torch.arange(512, dtype=torch.float32).unsqueeze(0)
        assert torch.allclose(enc.forward_features(x), enc.forward_tokens(x, 0.0, ident)[0][:, 0], atol=1e-5)


def test_decoder_layout_and_patchify():
    dec = MAEDecoder(in_dim=1536, decoder_mode=1)
    assert dec.decoder_embed.in_features == 1536 and dec.decoder_embed.out_features == 512 and len(dec.decoder_blocks) == 16
    assert dec.decoder_blocks[1].shift_size == (2, 0) and dec.decoder_blocks[0].shift_size == (0, 0)
    assert dec.decoder_blocks[1].attn_mask is not None and dec.decoder_blocks[0].attn_mask is None
    dec0 = MAEDecoder(in_dim=768, decoder_mode=0)
    assert len(dec0.decoder_blocks) == 8
    x = torch.randn(2, 1, 1024, 128)
    assert torch.equal(unpatchify(patchify(x)), x)
    lat = torch.randn(2, 1 + 103, 1536)
    _, _, ids = random_masking(torch.randn(2, 512, 4), 0.8)
    with torch.no_grad():
        assert dec(lat, ids).shape == (2, 512, 256)
        assert dec0(torch.randn(2, 104, 768), ids).shape == (2, 512, 256)


def test_param_counts():
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    enc, dec = AudioMAEEncoder(), MAEDecoder(in_dim=768, decoder_mode=1)
    assert abs(n(enc) - 85.6497) < 0.01  # ViT-B 인코더 (85.65M, fc_norm 포함)
    assert abs(n(dec) - 51.3446) < 0.01  # Swin x16 디코더 (51.34M)
    dde = DDEMAE(variant="s3", pretrained_path=None)
    assert abs(n(dde) - 224.6126) < 0.01  # E_R + E_I + 디코더(1536 입력) + q_θ
    s2 = DDEMAE(variant="s2", pretrained_path=None)
    assert abs(n(s2) - 136.9943) < 0.01


def test_time_shuffle_invertible_and_mask_sync():
    torch.manual_seed(0)
    x = torch.randn(3, 1, 1024, 128)
    xs, perm = time_shuffle(x, "patch", 16)
    assert xs.shape == x.shape and torch.equal(time_unshuffle(xs, perm, "patch", 16), x)
    assert not torch.equal(xs, x) and torch.allclose(xs.sum(dim=2), x.sum(dim=2), atol=1e-4, rtol=1e-4)  # 스펙트럼 합 보존 (fp32 합산 순서 차 ~2e-5)
    xf, pf = time_shuffle(x, "frame")
    assert torch.equal(time_unshuffle(xf, pf, "frame"), x)
    # 물리 동기화: 순열된 noise 의 argsort 가 X̃ 에서 같은 물리 패치를 같은 순위로 남긴다
    noise = torch.rand(3, 512)
    ns = permute_token_noise(noise, perm, 64, 8)
    ks = torch.argsort(ns, dim=1)[:, :103]
    k = torch.argsort(noise, dim=1)[:, :103]
    t_s, f_s = ks // 8, ks % 8
    phys = torch.gather(perm, 1, t_s) * 8 + f_s
    assert torch.equal(phys, k)


def test_siamese_hinge_values():
    z = torch.randn(4, 768)
    assert float(siamese_hinge(z, z, 1.0)) == 0.0
    z2 = z + torch.tensor([3.0]) * torch.nn.functional.normalize(torch.randn(4, 768), dim=-1)
    assert abs(float(siamese_hinge(z, z2, 1.0)) - 2.0) < 1e-4
    assert float(siamese_hinge(z, z2, 5.0)) == 0.0


def test_vclub_bound_and_aux_step_isolation():
    torch.manual_seed(0)
    club = VCLUB(16, 16, 8)
    zr, zi = torch.randn(6, 16, requires_grad=True), torch.randn(6, 16, requires_grad=True)
    mi = club.mi_upper_bound(zr, zi)
    assert torch.isfinite(mi)
    mi_d = club.mi_upper_bound_detached_q(zr, zi)
    assert torch.allclose(mi, mi_d)
    mi_d.backward()
    assert all(p.grad is None for p in club.parameters()) and zr.grad is not None and zi.grad is not None
    # 보조 스텝: q_θ 만 바뀌고 입력(주 파라미터 대리)은 그대로
    m = DDEMAE(variant="s3", pretrained_path=None, club_hidden=8)
    main_before = {k: v.clone() for k, v in m.encoder_r.state_dict().items()}
    q_before = {k: v.clone() for k, v in m.club.state_dict().items()}
    m.train()
    m._club_step(torch.randn(4, 768), torch.randn(4, 768))
    assert all(torch.equal(v, m.encoder_r.state_dict()[k]) for k, v in main_before.items())
    assert any(not torch.equal(v, m.club.state_dict()[k]) for k, v in q_before.items())


def test_dde_variants_and_grad_routing():
    torch.manual_seed(0)
    x = torch.randn(2, 128, 1024) * 0.3  # 툴킷 특징 배치 (B, F, T)
    s3 = DDEMAE(variant="s3", pretrained_path=None, club_hidden=8, siamese_margin=0.0)
    assert s3.decoder.decoder_embed.in_features == 1536 and s3.encoder_i is not None
    s3.train()
    logits, loss = s3.training_forward(x, torch.tensor([0, 1]), torch.nn.CrossEntropyLoss())
    assert logits.shape == (2, 4) and torch.isfinite(loss)
    assert set(s3.last_losses) == {"recon", "siamese", "mi"} and s3.last_losses["siamese"] > 0
    loss.backward()
    assert all(p.grad is None for p in s3.club.parameters())
    assert s3.encoder_r.blocks[0].attn.qkv.weight.grad is not None and s3.encoder_i.blocks[0].attn.qkv.weight.grad is not None
    assert s3.decoder.decoder_embed.weight.grad is not None
    s2 = DDEMAE(variant="s2", pretrained_path=None)
    assert s2.decoder.decoder_embed.in_features == 768 and s2.encoder_i is None and s2.club is None
    s2.train()
    _, l2 = s2.training_forward(x, torch.tensor([0, 1]), torch.nn.CrossEntropyLoss())
    assert set(s2.last_losses) == {"recon"}
    abl = DDEMAE(variant="s3", pretrained_path=None, use_mi=False, use_siamese=False)
    abl.train()
    abl.training_forward(x, torch.tensor([0, 1]), torch.nn.CrossEntropyLoss())
    assert set(abl.last_losses) == {"recon"} and abl.club is None
    with pytest.raises(ValueError):
        DDEMAE(variant="s3", pretrained_path=None, shuffle_unit="frame", shuffle_mask="same_physical")


def test_stage2_loads_only_encoder_r(tmp_path):
    s3 = DDEMAE(variant="s3", pretrained_path=None, club_hidden=8)
    with torch.no_grad():
        for p in s3.encoder_i.parameters():
            p.add_(1.0)  # E_I 를 오염시켜 로드되면 티 나게
    torch.save({"model": s3.state_dict(), "epoch": 0}, tmp_path / "final_state.pth")
    cls = AudioMAEClassifier(n_cls=4, pretrained_path=None, encoder_ckpt=str(tmp_path / "final_state.pth"))
    for k, v in s3.encoder_r.state_dict().items():
        if not k.startswith("fc_norm."):
            assert torch.equal(cls.encoder.state_dict()[k], v), k
    assert cls.load_info["discarded_prefixes"] == ["club", "decoder", "encoder_i"]
    with pytest.raises(RuntimeError):
        AudioMAEClassifier(n_cls=4, pretrained_path=None, encoder_ckpt=str(tmp_path / "final_state.pth"), encoder_key="encoder_x.")


# ----------------------------------------------------------------------------------------- P0 골든 (공식 Audio-MAE)
@needs_golden
@needs_ckpt
def test_checkpoint_coverage_and_pos_embed():
    g = torch.load(GOLDEN / "mae.pt")
    enc, dec = AudioMAEEncoder(), MAEDecoder(in_dim=768, decoder_mode=g["decoder_mode"])
    info = load_audiomae_checkpoint(enc, CKPT, dec)
    assert info["dropped"] == [] and len(info["encoder_used"]) + len(info["decoder_used"]) == len(g["ckpt_shapes"]) == 430
    assert torch.equal(enc.pos_embed, g["pos_embed"]) and torch.equal(dec.decoder_pos_embed, g["decoder_pos_embed"])
    ours = {**{k: list(v.shape) for k, v in enc.state_dict().items() if not k.startswith("fc_norm.")},
            **{k: list(v.shape) for k, v in dec.state_dict().items()}}
    assert ours == g["ckpt_shapes"]
    # DDE concat 디코더: decoder_embed 만 건너뛰고 나머지 디코더 키 전부 로드
    dec2 = MAEDecoder(in_dim=1536, decoder_mode=1)
    info2 = load_audiomae_checkpoint(None, CKPT, dec2, skip_decoder_embed=True)
    assert set(info2["dropped"]) >= {"decoder_embed.weight", "decoder_embed.bias"} and "decoder_blocks.15.mlp.fc2.weight" in info2["decoder_used"]


@needs_golden
@needs_ckpt
def test_encoder_decoder_loss_match_reference():
    g, i = torch.load(GOLDEN / "mae.pt"), torch.load(GOLDEN / "input.pt")
    enc, dec = AudioMAEEncoder(global_pool=False), MAEDecoder(in_dim=768, decoder_mode=g["decoder_mode"])
    load_audiomae_checkpoint(enc, CKPT, dec)
    enc.eval(); dec.eval()
    with torch.no_grad():
        lat, mask, ids = enc.forward_tokens(i["x"], 0.8, i["noise"])
        assert torch.equal(mask, g["mask"]) and torch.equal(ids, g["ids_restore"])
        assert torch.allclose(lat, g["latent"], atol=1e-4), float((lat - g["latent"]).abs().max())
        pred = dec(lat, ids)
        assert torch.allclose(pred, g["pred"], atol=1e-4), float((pred - g["pred"]).abs().max())
        assert torch.allclose(mae_recon_loss(i["x"], pred, mask, False), g["loss"], atol=1e-4)
        assert torch.allclose(mae_recon_loss(i["x"], pred, mask, True), g["loss_norm_pix"], atol=1e-4)
        # recon_scope='all' (가정 플래그, 식 5 전체 MSE) = 마스크 무관 전 패치 평균
        all_ = mae_recon_loss(i["x"], pred, mask, False, scope="all")
        assert torch.isfinite(all_) and torch.allclose(all_, ((pred - patchify(i["x"])) ** 2).mean(), atol=1e-6)
        assert torch.equal(patchify(i["x"]), g["target_patches"])
        ident = torch.arange(512, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
        full, _, _ = enc.forward_tokens(i["x"], 0.0, ident)
        assert torch.allclose(full, g["latent_full"], atol=1e-4)


@needs_golden
@needs_ckpt
def test_finetune_features_and_head_match_reference():
    g, i = torch.load(GOLDEN / "vit.pt"), torch.load(GOLDEN / "input.pt")
    m = AudioMAEClassifier(n_cls=4, pretrained_path=str(CKPT), global_pool=True).eval()
    for _, p in sorted(m.head.named_parameters()):
        _det_fill(p)
    with torch.no_grad():
        feat = m.encoder.forward_features(i["x"])
        assert torch.allclose(feat, g["feat_gap"], atol=1e-4), float((feat - g["feat_gap"]).abs().max())
        logits = m(i["x"].squeeze(1).transpose(1, 2))  # 툴킷 입력 (B, F, T) 경로
        assert torch.allclose(logits, g["logits"], atol=1e-4)
        assert torch.allclose(torch.nn.functional.cross_entropy(logits, g["labels"]), g["loss"], atol=1e-4)
