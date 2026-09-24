"""
BTS-CARD official-code parity goldens — runs the official code (RSC-Toolkit/BTS-CARD a9982a0) in the README-lineage environment (transformers 4.40.0, CPU).

  .venv-ref-btscard/bin/python scripts/reproduction/dump_btscard_goldens.py \
      --repo outputs/reference/btscard --out tests/golden/btscard [--ckpt pretrained_models/upstream/btscard/best.pth]

The original hard-codes .cuda() inside main (set_model/train), so here the loss assembly of main (main:276-301, 416) is reconstructed
as-is on CPU from the official functions (models/*, util/misc.py fusion_function·kl_loss_function·ReverseLayerF, util/icbhi_util.py
get_meta_text_descriptions·randomly_mask_metadata). Local copies are used instead of HF Hub access
(nlpie/tiny-clinicalbert → pretrained_models/upstream/hf/tiny-clinicalbert; CLAP from the HF cache).

Goldens (tests/golden/btscard/):
- forward_golden.pt : inputs (first N batches of training.pt x/y/meta7), randomly initialized head weights (classifier, tinybert.classifier,
                      discriminators), CLAP/BERT token ids, bert CLS, o_multi/o_text/o1/o2/loc/dev logits, 6 loss terms,
                      TIE (α 0/0.5/1) — fp32, eval mode (dropout off)
- text_golden.json  : CLAP sentences (no masking), masking dicts and sentences (fixed random.seed), 4.40.0 token ids
- ckpt_forward_golden.pt (optional): outputs for the same inputs with the authors' best.pth
- record.json       : versions and settings
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
    # hub name → local copy (the original tinybert.py hard-codes the name)
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

    # ---- inputs ----
    items = torch.load(args.train_pt, map_location="cpu")[: args.n]
    x = torch.stack([it[0] for it in items])                     # (N, 1, 1001, 64) — the original images.squeeze(1) is a no-op on (1,1001,64) (dim1=1001) → kept 4D
    y = torch.tensor([int(it[1]) for it in items])
    meta = torch.stack([it[2] for it in items]).float()
    loc, dev = meta[:, 5].long(), meta[:, 6].long()

    # ---- sentences (icbhi_dataset.py:36-45) ----
    clap_texts = [get_meta_text_descriptions(m, a, training=True, erasing=False) for m in meta]
    random.seed(args.mask_seed)
    bert_texts = [get_meta_text_descriptions(m, a, training=True, erasing=True) for m in meta]
    # Replay the same random sequence to record the masks (the erasing path consumes only 4 random() calls per sample)
    random.seed(args.mask_seed)
    masks = []
    for m in meta:
        full = {"age": None, "sex": None, "loc": None, "dev": None}
        md = randomly_mask_metadata({k: "v" for k in full}, drop_prob=0.25)
        masks.append([md[k] is None for k in ("age", "sex", "loc", "dev")])
    unmasked_bert = [get_meta_text_descriptions(m, a, training=True, erasing=False) for m in meta]
    assert unmasked_bert == clap_texts
    # check that re-applying the masks gives the same sentences as the random path (record integrity)
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

    # ---- model (set_model) ----
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
           "device": "cpu", "note": "fp32, eval mode (dropout off). The original train() autocasts only CLAP, but the goldens use fp32 throughout"}
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
