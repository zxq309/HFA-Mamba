"""Scale-Balanced Loss (SBL).

Faithful implementation of Section 3.4 (Eqs. 15-20) of:
    "High-Frequency-Aware Mamba for Tiny Person Detection in Aerial Images".

SBL reweights the *classification* and *regression* terms of a decoupled
detection head as two independent, smooth functions of the image-normalized
ground-truth object area ``A_i = w_i * h_i / (H * W)``:

    lambda_reg(A_i) = exp( -alpha_R / (A_i + eps) )          (Eq. 15)
    lambda_cls(A_i) = 1 + beta_C * exp( -gamma_C * A_i )     (Eq. 16)

The total loss (Eq. 17), averaged over positive assignments ``P_pos``:

    L_SBL = mean_i [ lambda_cls(A_i) * L_BCE(S_i, S*_i)
                     + lambda_reg(A_i) * ( L_CIoU(B_i, B*_i)
                                           + gamma_dfl * L_DFL(D_i, D*_i) ) ]

Design intent (asymmetric):
    * classification is *amplified* for small instances (reduce missed detections);
    * regression + DFL are *down-weighted* for extremely tiny / noisy boxes
      (alleviate unstable gradients from annotation uncertainty).

This file provides:
    1. ``lambda_reg`` / ``lambda_cls``  - the two weighting functions.
    2. ``ScaleBalancedLoss``            - a thin reweighting wrapper that combines
       per-instance cls / CIoU / DFL losses. It is framework-independent; the
       integration helper for an Ultralytics/YOLOv8-style decoupled head is in
       ``docs/integration.md``.

Dependencies: PyTorch only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["lambda_reg", "lambda_cls", "ScaleBalancedLoss"]


def lambda_reg(area: torch.Tensor, alpha_r: float = 5.0e-4, eps: float = 1e-7) -> torch.Tensor:
    """Regression weight (Eq. 15): negative-exponential decay that suppresses
    unstable regression/DFL gradients for tiny boxes.

    As ``A_i -> 0``, ``lambda_reg -> 0`` (tiny boxes contribute little regression
    signal); as ``A_i`` grows, ``lambda_reg -> 1``.
    """
    return torch.exp(-alpha_r / (area + eps))


def lambda_cls(area: torch.Tensor, beta_c: float = 1.0, gamma_c: float = 800.0) -> torch.Tensor:
    """Classification weight (Eq. 16): boosting factor that focuses supervision
    on small instances.

    As ``A_i -> 0``, ``lambda_cls -> 1 + beta_C`` (maximum boost); as ``A_i``
    grows, ``lambda_cls -> 1`` (no change for large objects).
    """
    return 1.0 + beta_c * torch.exp(-gamma_c * area)


class ScaleBalancedLoss(nn.Module):
    """Scale-Balanced Loss reweighting wrapper.

    Args:
        alpha_r: regression suppression decay rate ``alpha_R`` (Eq. 15).
        beta_c: maximum classification enhancement ``beta_C`` (Eq. 16).
        gamma_c: classification enhancement decay rate ``gamma_C`` (Eq. 16).
        gamma_dfl: fixed DFL weight ``gamma_dfl`` (Eq. 17).
        eps: numerical-stability constant.

    Default values reproduce Table 1 (``alpha_R=5e-4, beta_C=1.0, gamma_C=800``),
    shared across HERIDAL / AFO / TinyPerson.
    """

    def __init__(
        self,
        alpha_r: float = 5.0e-4,
        beta_c: float = 1.0,
        gamma_c: float = 800.0,
        gamma_dfl: float = 1.5,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha_r = float(alpha_r)
        self.beta_c = float(beta_c)
        self.gamma_c = float(gamma_c)
        self.gamma_dfl = float(gamma_dfl)
        self.eps = float(eps)

    def normalized_area(self, gt_wh: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """A_i = w_i * h_i / (H * W). ``gt_wh`` holds per-instance (w, h) in pixels."""
        return (gt_wh[..., 0] * gt_wh[..., 1]) / (img_h * img_w + self.eps)

    def forward(
        self,
        cls_loss_i: torch.Tensor,    # (N,) per-instance BCE classification loss
        ciou_loss_i: torch.Tensor,   # (N,) per-instance CIoU regression loss
        dfl_loss_i: torch.Tensor,    # (N,) per-instance distribution focal loss
        gt_wh: torch.Tensor,         # (N, 2) ground-truth (w, h) in pixels
        img_hw: tuple[int, int],     # (H, W) image height/width in pixels
    ) -> torch.Tensor:
        """Return the scalar SBL averaged over positive assignments (Eq. 17)."""
        img_h, img_w = img_hw
        area = self.normalized_area(gt_wh, img_h, img_w)

        w_cls = lambda_cls(area, self.beta_c, self.gamma_c)
        w_reg = lambda_reg(area, self.alpha_r, self.eps)

        per_instance = (
            w_cls * cls_loss_i
            + w_reg * (ciou_loss_i + self.gamma_dfl * dfl_loss_i)
        )
        n_pos = max(per_instance.numel(), 1)
        return per_instance.sum() / n_pos

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"alpha_r={self.alpha_r}, beta_c={self.beta_c}, "
            f"gamma_c={self.gamma_c}, gamma_dfl={self.gamma_dfl}"
        )


# =========================================================================== #
# Drop-in detection loss that applies SBL inside the Ultralytics / Mamba-YOLO
# training loop. Injected at runtime by mbyolo_train.py (no upstream edits):
#     model.model.init_criterion = lambda: SBLDetectionLoss(model.model, **kw)
#
# Targets the Ultralytics 8.x ``v8DetectionLoss`` API. If your vendored fork's
# loss API differs, see docs/integration.md for the manual hook points.
# =========================================================================== #
try:
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors
    _HAS_ULTRA = True
except Exception:  # pragma: no cover - lets this file import without ultralytics
    _HAS_ULTRA = False

    class v8DetectionLoss:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("ultralytics is required for SBLDetectionLoss")


class SBLDetectionLoss(v8DetectionLoss):
    """``v8DetectionLoss`` with Scale-Balanced reweighting (Eqs. 15-17).

    Reuses the parent's assigner, anchor generation, box decoding, BCE and
    ``bbox_loss``. The only change is two per-positive weights keyed on the
    image-normalized target-box area:
      * classification BCE is scaled per anchor by ``lambda_cls`` (boost small);
      * the regression weight fed to ``bbox_loss`` (which drives BOTH CIoU and
        DFL) is scaled per positive anchor by ``lambda_reg`` (suppress tiny).
    """

    def __init__(self, model, alpha_r: float = 5.0e-4, beta_c: float = 1.0,
                 gamma_c: float = 800.0, tal_topk: int = 10):
        super().__init__(model, tal_topk=tal_topk)
        self.alpha_r = float(alpha_r)
        self.beta_c = float(beta_c)
        self.gamma_c = float(gamma_c)

    def __call__(self, preds, batch):
        import torch as _t

        loss = _t.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = _t.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = _t.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = _t.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # --- SBL per-positive area weights (target_bboxes are pixel xyxy here) ---
        img_h, img_w = imgsz[0], imgsz[1]
        wh = (target_bboxes[..., 2:4] - target_bboxes[..., 0:2]).clamp(min=1e-6)
        area = (wh[..., 0] * wh[..., 1]) / (img_h * img_w + 1e-7)      # (B, A)
        w_cls_map = _t.ones_like(area)
        w_reg_map = _t.ones_like(area)
        if fg_mask.any():
            w_cls_map[fg_mask] = lambda_cls(area[fg_mask], self.beta_c, self.gamma_c)
            w_reg_map[fg_mask] = lambda_reg(area[fg_mask], self.alpha_r)

        # --- classification: BCE scaled per anchor by lambda_cls (Eq. 16) ---
        bce = self.bce(pred_scores, target_scores.to(dtype))          # (B, A, nc), reduction='none'
        loss[1] = (bce * w_cls_map.unsqueeze(-1)).sum() / target_scores_sum

        # --- regression + DFL: scale the weight by lambda_reg (Eq. 15/17) ---
        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor
            target_scores_reg = target_scores.clone()
            target_scores_reg = target_scores_reg * w_reg_map.unsqueeze(-1)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores_reg, target_scores_sum, fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl   # hyp.dfl plays the role of gamma_dfl in Eq. 17
        return loss.sum() * batch_size, loss.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    N, (H, W) = 8, (2160, 2160)
    sbl = ScaleBalancedLoss()

    # Fake per-instance losses and a spread of box sizes (tiny -> large).
    cls = torch.rand(N).abs()
    ciou = torch.rand(N).abs()
    dfl = torch.rand(N).abs()
    wh = torch.tensor([[6, 8], [10, 12], [16, 20], [24, 30],
                       [40, 50], [64, 80], [120, 140], [200, 260]], dtype=torch.float)

    loss = sbl(cls, ciou, dfl, wh, (H, W))
    areas = sbl.normalized_area(wh, H, W)
    print("normalized areas:", areas.tolist())
    print("lambda_cls      :", [round(v, 4) for v in lambda_cls(areas).tolist()])
    print("lambda_reg      :", [round(v, 6) for v in lambda_reg(areas).tolist()])
    print("SBL scalar      :", float(loss))
