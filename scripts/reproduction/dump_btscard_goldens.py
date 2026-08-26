"""
BTS-CARD P0 골든 — 공식 코드(RSC-Toolkit/BTS-CARD a9982a0)를 README 계보 환경(transformers 4.40.0, CPU)에서 실행.

  .venv-ref-btscard/bin/python scripts/reproduction/dump_btscard_goldens.py \
      --repo outputs/reference/btscard --out tests/golden/btscard [--ckpt pretrained_models/upstream/btscard/best.pth]

원본은 main 안에 .cuda() 가 하드코딩돼 있어(set_model/train) 여기서는 main 의 손실 조립(main:276-301, 416)을
공식 함수(models/*, util/misc.py fusion_function·kl_loss_function·ReverseLayerF, util/icbhi_util.py
get_meta_text_descriptions·randomly_mask_metadata)로 CPU 에서 그대로 재구성한다. HF 허브 접근 없이 로컬 사본을 쓴다
(nlpie/tiny-clinicalbert → pretrained_models/upstream/hf/tiny-clinicalbert; CLAP 은 HF 캐시).

골든 (tests/golden/btscard/):
- forward_golden.pt : 입력(training.pt 앞 N 배치 x/y/meta7), 랜덤 초기화 헤드 가중치(classifier, tinybert.classifier,
                      판별기), CLAP/BERT 토큰 id, bert CLS, o_multi/o_text/o1/o2/loc/dev 로짓, 6항 손실,
                      TIE(α 0/0.5/1) — fp32, eval 모드(dropout off)
- text_golden.json  : CLAP 문장(마스킹 없음), 마스킹 dict·문장(random.seed 고정), 4.40.0 토큰 id
- ckpt_forward_golden.pt (옵션): 저자 best.pth 로 같은 입력의 출력
- record.json       : 버전·설정
"""
import argparse
import json
import pathlib
import random
import sys
import types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="outputs/reference/btscard")
    ap.add_argument("--out", default="tests/golden/btscard")
    ap.add_argument("--train_pt", default="pretrained_models/upstream/bts/training.pt")
    ap.add_argument("--bert_local", default="pretrained_models/upstream/hf/tiny-clinicalbert")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--mask_seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import transformers

    repo = pathlib.Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    # 허브 이름 → 로컬 사본 (원본 tinybert.py 는 이름을 하드코딩)
    _orig_from_pretrained = transformers.AutoModel.from_pretrained.__func__
    bert_local = str(pathlib.Path(args.bert_local).resolve())

    def _redirect(cls, name, *a, **k):
        if name == "nlpie/tiny-clinicalbert":
            name = bert_local
        return _orig_from_pretrained(cls, name, *a, **k)

    transformers.AutoModel.from_pretrained = classmethod(_redirect)

    from models.clap import PretrainedCLAP  # noqa: E402
    from models.tinybert import TinyBertTextModel  # noqa: E402
    from models.discriminators import LocationDiscriminator, DeviceDiscriminator  # noqa: E402
    from util.misc import fusion_function, kl_loss_function, ReverseLayerF  # noqa: E402
    from util.icbhi_util import get_meta_text_descriptions, randomly_mask_metadata  # noqa: E402
    from transformers import AutoTokenizer, ClapProcessor  # noqa: E402

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    a = types.SimpleNamespace(clap_final="concat", te_alpha=0.5, meta_mode="all", test_wrong_label=False,
                              test_drop_key=False, test_drop_key_prob=0.1, test_unknown_all=False, test_bmi=False)

    # ---- 입력 ----
    items = torch.load(args.train_pt, map_location="cpu")[: args.n]
    x = torch.stack([it[0] for it in items])                     # (N, 1, 1001, 64) — 원본 images.squeeze(1) 은 (1,1001,64) 에 무효(dim1=1001) → 4D 그대로
    y = torch.tensor([int(it[1]) for it in items])
    meta = torch.stack([it[2] for it in items]).float()
    loc, dev = meta[:, 5].long(), meta[:, 6].long()

    # ---- 문장 (icbhi_dataset.py:36-45) ----
    clap_texts = [get_meta_text_descriptions(m, a, training=True, erasing=False) for m in meta]
    random.seed(args.mask_seed)
    bert_texts = [get_meta_text_descriptions(m, a, training=True, erasing=True) for m in meta]
    # 같은 난수열을 다시 돌려 마스크를 기록 (erasing 경로는 샘플당 random() 4회만 소비)
    random.seed(args.mask_seed)
    masks = []
    for m in meta:
        full = {"age": None, "sex": None, "loc": None, "dev": None}
        md = randomly_mask_metadata({k: "v" for k in full}, drop_prob=0.25)
        masks.append([md[k] is None for k in ("age", "sex", "loc", "dev")])
    unmasked_bert = [get_meta_text_descriptions(m, a, training=True, erasing=False) for m in meta]
    assert unmasked_bert == clap_texts
    # 마스크 재적용 문장이 random 경로 문장과 같은지 (기록 무결성)
    from util.meta_description import generate_meta_description  # noqa: E402
    from util.icbhi_util import get_meta_text_descriptions as _g  # noqa: F401
    replay = []
    for m, mk in zip(meta, masks):
        age = "adult" if int(m[0]) >= 19 else "pediatric"
        sex = "male" if int(m[1]) == 0 else "female"
        locs = ["trachea", "left anterior chest", "right anterior chest", "left posterior chest",
                "right posterior chest", "left lateral chest", "right lateral chest"]
        devs = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L"]
        d = {"age": age, "sex": sex, "loc": locs[int(m[5])], "dev": devs[int(m[6])]}
        d = {k: (None if mk[i] else v) for i, (k, v) in enumerate(d.items())}
        replay.append(generate_meta_description(**d))
    assert replay == bert_texts, (replay, bert_texts)

    Processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", sampling_rate=48000)
    bert_tok = AutoTokenizer.from_pretrained(bert_local)
    enc_c = Processor.tokenizer(text=clap_texts, return_tensors="pt", padding="max_length", max_length=64)
    enc_b = bert_tok(bert_texts, max_length=64, padding="max_length", truncation=True, return_tensors="pt")

    # ---- 모델 (set_model) ----
    torch.manual_seed(0)
    model = PretrainedCLAP("laion/clap-htsat-unfused", 512)
    classifier = nn.Linear(1024, 4)
    meta_classifier = TinyBertTextModel()
    loc_disc, dev_disc = LocationDiscriminator(), DeviceDiscriminator()
    constant = nn.Parameter(torch.tensor(1.0))
    heads = {"classifier": classifier.state_dict(), "meta_head": meta_classifier.classifier.state_dict(),
             "loc_discriminator": loc_disc.state_dict(), "dev_discriminator": dev_disc.state_dict()}
    criterion = nn.CrossEntropyLoss(reduction="mean")

    def run(alphas=(0.0, 0.5, 1.0)):
        for mod in (model, classifier, meta_classifier, loc_disc, dev_disc):
            mod.eval()
        with torch.no_grad():
            feats = model((enc_c["input_ids"], enc_c["attention_mask"], x), args=a, training=False)
            o_multi = classifier(feats)
            cls = meta_classifier.bert(input_ids=enc_b["input_ids"], attention_mask=enc_b["attention_mask"]).last_hidden_state[:, 0, :]
            o_text = meta_classifier(enc_b["input_ids"], enc_b["attention_mask"])
            o_multi_c = constant * torch.ones_like(o_multi)
            o1 = fusion_function(o_multi, o_text)
            o2 = fusion_function(o_multi_c, o_text)
            losses = {
                "o1": criterion(o1, y), "o2": criterion(o2, y),
                "kl": kl_loss_function(o1, fusion_function(o_multi_c, o_text.detach())),
                "nde": criterion(o_text, y),
            }
            rev = ReverseLayerF.apply(o_text, 1.0)
            loc_logits, dev_logits = loc_disc(rev), dev_disc(rev)
            losses["loc"] = criterion(loc_logits, loc)
            losses["dev"] = criterion(dev_logits, dev)
            losses["total"] = losses["o1"] + losses["o2"] + losses["kl"] + 0.01 * losses["loc"] + 0.1 * losses["dev"] + losses["nde"]
            tie = {f"{al:.1f}": (o1 - al * o2) for al in alphas}
        return {"feats": feats, "o_multi": o_multi, "bert_cls": cls, "o_text": o_text, "o1": o1, "o2": o2,
                "loc_logits": loc_logits, "dev_logits": dev_logits, "losses": {k: v.detach() for k, v in losses.items()},
                "tie": tie}

    g = run()
    golden = {"x": x, "y": y, "meta7": meta, "loc": loc, "dev": dev, "clap_input_ids": enc_c["input_ids"],
              "clap_attention_mask": enc_c["attention_mask"], "bert_input_ids": enc_b["input_ids"],
              "bert_attention_mask": enc_b["attention_mask"], "heads": heads, **g}
    torch.save(golden, out / "forward_golden.pt")
    text = {"clap_texts": clap_texts, "bert_texts_masked": bert_texts, "masks": masks, "mask_seed": args.mask_seed,
            "drop_prob": 0.25, "clap_input_ids": enc_c["input_ids"].tolist(), "bert_input_ids": enc_b["input_ids"].tolist(),
            "bert_attention_mask": enc_b["attention_mask"].tolist(), "meta7": meta.tolist()}
    json.dump(text, open(out / "text_golden.json", "w"), indent=1, ensure_ascii=False)
    print("losses", {k: round(float(v), 6) for k, v in g["losses"].items()})
    print("bert_texts", bert_texts)

    rec = {"transformers": transformers.__version__, "torch": torch.__version__, "repo": str(repo), "n": args.n,
           "device": "cpu", "note": "fp32, eval 모드(dropout off). 원본 train() 은 CLAP 만 autocast 지만 골든은 fp32 로 통일"}
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        classifier.load_state_dict(ck["classifier"], strict=True)
        meta_classifier.load_state_dict(ck["meta_classifier"], strict=True)
        g2 = run()
        torch.save({k: g2[k] for k in ("o_multi", "o_text", "o1", "o2", "tie", "losses", "bert_cls")}, out / "ckpt_forward_golden.pt")
        rec["ckpt"] = {"path": args.ckpt, "alpha": ck.get("alpha"), "epoch": ck.get("epoch"), "best_acc": ck.get("best_acc")}
        print("ckpt losses", {k: round(float(v), 6) for k, v in g2["losses"].items()})
    json.dump(rec, open(out / "record.json", "w"), indent=1, ensure_ascii=False)
    print("EXIT=0")


if __name__ == "__main__":
    main()
