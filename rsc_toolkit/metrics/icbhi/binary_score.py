"""
AMS-D 논문/공식 코드의 **이진(비표준)** Se/Sp/Score — utils/metric.py:18-34 calculate_stats 의 이식.

  Se = #(y≠0 ∧ ŷ≠0) / #비정상     (비정상을 "아무 비정상 클래스"로 맞추면 정답)
  Sp = #(y=0 ∧ ŷ=0) / #정상
  Score = (Se + Sp) / 2            (main.py:416)
  acc = 4-class accuracy           (참고용)

논문 §3.1 "Sen./Spe. are computed in a binary setting by grouping all abnormal classes as positive" 이 이 산식이다.
표준 ICBHI 4-class Score(rsc_toolkit/metrics/icbhi/score.py, 정확한 비정상 클래스만 정답) 와 **비교 불가** — 키 이름을
"paper_S_e"/"paper_S_p"/"paper_score" 로 구분하고 표에는 "논문 지표(비표준)" 으로 표기한다. 값은 % 단위.
"""
from typing import Dict

import torch
from torchmetrics import Metric


class PaperBinaryScore(Metric):
    full_state_update = False

    def __init__(self, normal_index: int = 0, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.normal_index = int(normal_index)
        self.add_state("tp", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("num_pos", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("num_neg", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, C], got {logits.shape}")
        target = target.view(-1)
        pred = logits.argmax(dim=1)
        is_pos = target != self.normal_index
        self.num_pos += is_pos.sum()
        self.num_neg += (~is_pos).sum()
        self.tp += (is_pos & (pred != self.normal_index)).sum()
        self.tn += ((~is_pos) & (pred == self.normal_index)).sum()
        self.correct += (pred == target).sum()
        self.total += target.numel()

    def compute(self) -> Dict[str, torch.Tensor]:
        # 원본: num_pos/num_neg 가 0 이면 0 (utils/metric.py:30-31)
        se = self.tp.float() / self.num_pos.float() * 100.0 if int(self.num_pos) else self.tp.float() * 0.0
        sp = self.tn.float() / self.num_neg.float() * 100.0 if int(self.num_neg) else self.tn.float() * 0.0
        acc = self.correct.float() / self.total.float().clamp_min(1) * 100.0
        return {"paper_S_p": sp, "paper_S_e": se, "paper_score": (se + sp) / 2.0, "paper_acc": acc}
