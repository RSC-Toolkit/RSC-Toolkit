"""
BTS-CARD (ICASSP 2026) — multimodal RSC with counterfactual adversarial debiasing, exact port of the official code.

Sources:
- Paper: Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial
  Debiasing for OOD Robustness (arXiv 2510.22263 v2)
- Official code: https://github.com/RSC-Toolkit/BTS-CARD (a9982a0, Apache-2.0)
  main_causal_adversarial_debiasing_logging.py set_model()/train()/validate(),
  models/{clap,tinybert,discriminators}.py, util/misc.py(fusion_function, kl_loss_function,
  ReverseLayerF), util/icbhi_util.py(randomly_mask_metadata :137-144, :259-261).

Architecture (same as the original code):
- TE branch = BTSModel (CLAP text ⊕ audio concat → Linear(1024, 4)) → o_multi        (main:268-273)
- NDE branch = nlpie/tiny-clinicalbert CLS → Linear(h, 256) → ReLU → Dropout(0.1) → Linear(256, 4)
  → o_text                                                                        (models/tinybert.py)
- RUBI fusion  o1 = o_multi · σ(o_text),  counterfactual o2 = c · σ(o_text),  c = LearnableConstant(1.0)
  (main:276-280; util/misc.py fusion_function mode="rubi")
- GRL(λ=1.0) → Location(7)/Device(4) discriminators LayerNorm(4)→Linear(4,96)→GELU→Dropout(0.1)→Linear(96,K),
  input = o_text **4-d logits** (main:294-299; models/discriminators.py)
- Loss = CE(o1) + CE(o2) + KL + CE(o_text) + λ_loc·CE(loc) + λ_dev·CE(dev)          (main:282-301)
- Inference TIE  output = o1 − α·o2                                                 (main:416)

Quirks of the original code (kept as defaults; the paper form is available only via flags):
- Dead KL term: kl_div(log_softmax(c·σ(o_text.detach())), log_softmax(o1).detach()) — there is no gradient
  path to trainable parameters (c is not registered in the optimizer). `kl_dead=True` (default). The paper Eq.8 form is `kl_dead=False`.
- LearnableConstant c: an nn.Parameter, but not in any of the 6 optimizer groups → fixed at 1.0 (main:195-212). The toolkit
  gets the same effect with `requires_grad=False` (PLAdapter puts all of model.parameters() into the optimizer).
  Trainable with `const_trainable=True`. Also excluded from MA (original main:325-331).
- Discriminator input = 4-d logits (not the feature vector).
- Metadata masking (counterfactual metadata augmentation) = during training, only the tiny-bert sentence has each attribute
  set to None with p=0.25 and **omitted** from the sentence (icbhi_util.py:137-144 → meta_description None branch). The
  'neutral placeholder (unknown)' substitution of paper §2.3 is not in the code (only in the test_drop_key evaluation ablation) → `mask_mode="unknown"`
  option. The CLAP sentence is never masked.
- AMP boundary: the original train() autocasts only CLAP+classifier; TinyBERT, fusion and losses run in fp32 (main:267-301);
  validate() autocasts everything. Here the tail autocast is disabled only while training.
- MA (β 0.5, every step) covers only model/classifier/projector/meta_classifier (discriminators and constant excluded)
  → trainer/callbacks/btscard.py BTSCARDWeightAveraging.

Randomness: the original draws masks in dataset __getitem__ from the global python `random` state (assumes num_workers=0, seed 111).
The toolkit draws them inside the model with a torch.Generator (seed = torch.initial_seed(), i.e. tied to L.seed_everything).
The sequence differs from the original (reproduction is compared on seed means), but the same seed is reproducible within the toolkit.
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

META_KEYS = ("age", "sex", "loc", "dev")  # original meta_dict insertion order = order in which mask random numbers are consumed
DEFAULT_NDE_LOCAL = "pretrained_models/upstream/hf/tiny-clinicalbert"
NDE_HUB_NAME = "nlpie/tiny-clinicalbert"
MA_PREFIXES = ("audio_features.", "classifier.", "meta_classifier.")  # original MA targets (projector=Identity is stateless)


# ----------------------------------------------------------------------------- original functions
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
    util/icbhi_util.py:137-144. Draws one random number per key in order and masks if `< drop_prob`.
    The original uses random.random() and the mask value None (omitted from the sentence). mask_value='unknown' gives the paper-style substitution.
    """
    masked = {}
    for key, value in meta_dict.items():
        u = torch.rand(1, generator=generator).item()
        masked[key] = mask_value if u < drop_prob else value
    return masked


def apply_mask(meta_dict: Dict[str, Optional[str]], mask: Sequence[bool],
               mask_value: Optional[str] = None) -> Dict[str, Optional[str]]:
    """For golden replay: applies a per-attribute bool mask as given."""
    return {k: (mask_value if m else v) for (k, v), m in zip(meta_dict.items(), mask)}


class TinyBertTextModel(nn.Module):
    """models/tinybert.py — attribute names bert/classifier match the 'meta_classifier' keys of the authors' ckpt."""

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
    """models/discriminators.py Location(7)/Device(4) — in_dim=4 (o_text logits), hidden 96."""

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
    """main:195-198. Not registered in the original optimizer → effectively fixed."""

    def __init__(self, initial_value: float = 1.0, trainable: bool = False):
        super().__init__()
        self.dummy_vector = nn.Parameter(torch.tensor(float(initial_value)), requires_grad=bool(trainable))


NDE_HUB_FALLBACK_ENV = "RSCTK_NDE_HUB_FALLBACK"   # "1" = fall back to the HF hub nlpie/tiny-clinicalbert when there is no local copy


def _resolve_nde_name(name: str) -> str:
    """Location of the NDE (TinyBERT) weights: looks for a local path (cwd, then under RSCTK_PRETRAINED_ROOT); otherwise returns the name as is (hub id).

    When the default DEFAULT_NDE_LOCAL is missing, silently falling back to the HF hub would crash after training starts on
    machines without network access, or use a tokenizer from a different snapshot. The fallback is therefore opt-in.
    """
    found = resolve_pretrained(explicit=name)          # absolute/existing relative path → under ROOT → as is
    if found is not None and found.exists():
        return str(found)
    if name == DEFAULT_NDE_LOCAL:
        if os.environ.get(NDE_HUB_FALLBACK_ENV, "").strip() == "1":
            logger.warning("%s not found; using HF hub %s (%s=1)", name, NDE_HUB_NAME, NDE_HUB_FALLBACK_ENV)
            return NDE_HUB_NAME
        raise FileNotFoundError(
            f"BTS-CARD NDE weights {name} not found (neither under cwd nor under RSCTK_PRETRAINED_ROOT). "
            f"Place a `{NDE_HUB_NAME}` snapshot there, set the hub id explicitly with model.params.nde_pretrained_name={NDE_HUB_NAME}, "
            f"or allow the automatic fallback with {NDE_HUB_FALLBACK_ENV}=1."
        )
    return name


# ----------------------------------------------------------------------------- model
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
        meta_dialect: str = "icbhi",    # "sprsound" = when training directly on SPRSound (see bts.py)
        nde_pretrained_name: str = DEFAULT_NDE_LOCAL,
        alpha: float = 0.0,             # inference TIE coefficient (paper §3.2.4 recommends 0)
        lambda_loc: float = 0.01,
        lambda_dev: float = 0.1,
        grl_lambda: float = 1.0,
        p_mask: float = 0.25,
        kl_dead: bool = True,           # True = original (dead term), False = paper Eq.8 form
        const_trainable: bool = False,  # original: not registered in the optimizer (fixed 1.0)
        mask_mode: str = "omit",        # "omit" = original (None omitted) | "unknown" = paper description (placeholder)
        n_loc: int = 7,
        n_dev: int = 4,
        use_counterfactual: bool = True,  # ablation (a) flag — definition is provisional (not confirmed by the authors)
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

    # ------------------------------------------------------------------ text
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
        """tiny-bert sentences: per-attribute p_mask masking during training (masking=True) (icbhi_util.py:259-261)."""
        out = []
        # The SPRSound dialect must use "unknown" as the mask value — the authors' SPRSound sentence generator
        # handles "unknown" explicitly (util/sprsound_util.py:1-24) but has no branch for None, which
        # yields sentences like "a None patient". ICBHI follows mask_mode as in the original.
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

    # ------------------------------------------------------------------ branches
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
        """All intermediate outputs. With fp32_tail=True, everything after CLAP runs outside autocast (fp32) (original train())."""
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
        """main:282-301 six-term loss. Returns: (total, per-term dict)."""
        o1, o2, o_text = out["o1"], out["o2"], out["o_text"]
        terms: Dict[str, torch.Tensor] = {}
        terms["o1"] = criterion(o1, labels)
        if self.use_counterfactual:
            terms["o2"] = criterion(o2, labels)
            if self.kl_dead:
                # original main:285 — input side o_text.detach(), c is fixed → no gradient
                o2_detached = fusion_function(self.constant.dummy_vector * torch.ones_like(o1),
                                              o_text.detach(), self.fusion_mode)
                terms["kl"] = kl_loss_function(o1, o2_detached)
            else:
                # Paper form (gradient flows). Actual direction: kl_loss_function(o1, o2) is
                # F.kl_div(input=log_softmax(o2), target=log_softmax(o1).detach(), log_target=True)
                # = KL(softmax(o1) ‖ softmax(o2)) (same as original util/misc.py:139-143).
                # The direction may differ from the paper's Eq.8 notation KL(Y_{t,m*} ‖ Y_{t,m}) (unconfirmed by the authors).
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

    # ------------------------------------------------------------------ interface
    def forward(self, x: torch.Tensor, meta7: torch.Tensor, alpha: Optional[float] = None,
                bert_texts: Optional[Sequence[str]] = None, clap_texts: Optional[Sequence[str]] = None) -> torch.Tensor:
        """TIE logits o1 − α·o2 (evaluation: no masking)."""
        clap_texts = list(clap_texts) if clap_texts is not None else self.describe(meta7)
        bert_texts = list(bert_texts) if bert_texts is not None else self.describe_bert(meta7, masking=False)
        out = self.forward_all(x, clap_texts, bert_texts)
        return self.tie(out["o1"], out["o2"], alpha)

    def training_forward(self, audio_batch, labels, loss_func, meta_labels=None):
        if meta_labels is None:
            raise ValueError("BTS-CARD requires meta (7 fields) in the batch — use data=bts_precomputed")
        meta7 = meta_labels
        clap_texts = self.describe(meta7)
        training = self.training
        bert_texts = self.describe_bert(meta7, masking=training and self.use_meta_aug)
        out = self.forward_all(audio_batch, clap_texts, bert_texts, fp32_tail=training)
        # In the training branch the returned logits are the fused tie(o1,o2). The original train() computes
        # training accuracy from the pre-fusion o_multimodel, so the train top-1 log means something different
        # (losses, gradients and validation metrics are identical). Keep this in mind when comparing training curves with the original.
        output = self.tie(out["o1"], out["o2"])
        if training:
            loc = meta7[:, 5].long()
            dev = meta7[:, 6].long()
            loss, terms = self.compute_losses(out, labels, loc, dev, loss_func)
            self.loss_terms = {k: float(v.detach()) for k, v in terms.items()}
            return output, loss
        # original validate(): loss = CE(o1 − α·o2, y)
        return output, loss_func(output, labels)

    # ------------------------------------------------------------------ authors' checkpoint
    def load_reference_checkpoint(self, path, strict_backbone: bool = True) -> Dict[str, List[str]]:
        """
        Loads best.pth {'alpha','epoch','best_acc','model','classifier','meta_classifier'}.
        model/classifier/meta_classifier load without key remapping (original attribute names kept); the discriminators and
        constant are not in the ckpt (not saved by the original validate) → only those keys may be missing, the rest is strict.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model"])
        state.update({f"classifier.{k}": v for k, v in ckpt["classifier"].items()})
        state.update({f"meta_classifier.{k}": v for k, v in ckpt["meta_classifier"].items()})
        res = self.load_state_dict(state, strict=False)
        allowed = ("loc_discriminator.", "dev_discriminator.", "constant.")
        bad_missing = [k for k in res.missing_keys if not k.startswith(allowed)]
        if strict_backbone and (bad_missing or res.unexpected_keys):
            raise RuntimeError(f"BTS-CARD ckpt key mismatch: missing={bad_missing} unexpected={res.unexpected_keys}")
        logger.info("Loaded BTS-CARD authors' ckpt: %s (alpha=%s epoch=%s best_acc=%s; missing=%d, expected for discriminators/constant)",
                    path, ckpt.get("alpha"), ckpt.get("epoch"), ckpt.get("best_acc"), len(res.missing_keys))
        return {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys),
                "alpha": ckpt.get("alpha"), "epoch": ckpt.get("epoch"), "best_acc": ckpt.get("best_acc")}

    def load_toolkit_checkpoint(self, path) -> None:
        """Strictly loads the ICBHIBestScoreCheckpoint output {'model': state_dict, ...}."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["model"], strict=True)

    @staticmethod
    def ma_state_keys(state_dict_keys) -> List[str]:
        """state_dict keys covered by the original MA (model/classifier/projector/meta_classifier)."""
        return [k for k in state_dict_keys if k.startswith(MA_PREFIXES)]
