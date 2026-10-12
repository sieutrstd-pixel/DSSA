"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from ..util import DictTool
from ..util_learn import intersection_over_union, hungarian_matching
from scipy.optimize import linear_sum_assignment

class MetricWrap(nn.Module):
    """"""

    def __init__(self, detach=False, **metrics):
        super().__init__()
        self.detach = detach
        self.metrics = metrics

    def forward(self, **pack: dict) -> dict:
        if self.detach:
            with pt.inference_mode():
                return self._forward(**pack)
        else:
            return self._forward(**pack)

    def _forward(self, **pack: dict) -> dict:
        metrics = {}
        for key, value in self.metrics.items():
            kwds = {t: DictTool.getattr(pack, s) for t, s in value["map"].items()}
            # if self.detach:  # inference_mode has done this
            #     kwds = {k: v.detach() for k, v in kwds.items()}
            if "transform" in value:
                kwds = value["transform"](**kwds)
            assert isinstance(value["metric"], Metric)
            metric = value["metric"](**kwds)  # (loss/acc, valid)
            if "weight" in value:
                metric = (metric[0] * value["weight"], metric[1])
            metrics[key] = metric
        return metrics


class Metric(ABC, nn.Module):
    """
    mean all or specified dimensions; always keep batch dimension
    """

    def __init__(self, mean=()):
        """
        - mean: only support ``mean``; other operations like ``sum`` are seldomly used.
            - None: do nothing
            - len(mean) == 0: mean all non-batch/first dimensions
            - len(mean) > 0: mean the specified dimensions
        """
        super().__init__()
        assert mean is None or isinstance(mean, (list, tuple))
        if isinstance(mean, (list, tuple)) and len(mean) > 0:
            assert 0 not in mean  # batch/first dimension should not be included
        self.mean = mean

    def forward(self, *args, **kwds) -> tuple:
        if len(args) > 0:
            first_input_tensor = args[0]
        elif len(kwds) > 0:
            first_input_tensor = list(kwds.values())[0]
        else:
            raise ValueError
        assert isinstance(first_input_tensor, pt.Tensor)

        num = first_input_tensor.size(0)
        dtype = first_input_tensor.dtype
        device = first_input_tensor.device

        if num > 0:
            metric, valid = self.compute(*args, **kwds)
            if valid is None:
                valid = pt.ones(metric.size(0), dtype=pt.bool, device=device)
        else:
            metric = pt.tensor([], dtype=dtype, device=device)
            valid = pt.tensor([], dtype=pt.bool, device=device)

        """mean ``metric`` along dimensions ``self.mean``; flag ``valid`` samples
        - metric: shape=(b,..); dtype=float
        - valid: shape=(b,); dtype=bool
        """
        if self.mean is None:
            metric2 = metric  # (b,..)
        elif len(self.mean) == 0:
            if metric.ndim > 1:
                metric2 = metric.flatten(1).mean(1)  # (b,)
            else:
                metric2 = metric
        else:
            metric2 = metric.mean(self.mean)  # (b,..)

        return metric2, valid  # loss/acc (b,..), valid (b,)

    def compute(self, *args, **kwds) -> tuple:
        raise NotImplemented  # metric, valid


####


class CrossEntropyLoss(Metric):
    """``nn.CrossEntropyLoss``."""

    def compute(self, input, target):
        """
        - input: shape=(b,c,..), dtype=float
        - target: shape=(b,..), dtype=int64;
            or shape=(b,c,..), dtype=float
        """
        # loss = ptnf.cross_entropy(input, target, reduction="none")  # (b,..)
        loss = ptnf.cross_entropy(input, target)[None]  # (b=1,)
        return loss, None  # (b,) (b,)


class L1Loss(Metric):
    """``nn.L1Loss``."""

    def compute(self, input, target=None):
        if target is None:
            target = pt.zeros_like(input)
        assert input.ndim == target.ndim >= 1
        # loss = ptnf.l1_loss(input, target, reduction="none")  # (b,..)
        loss = ptnf.l1_loss(input, target)[None]  # (b=1,)
        return loss, None  # (b,) (b,)


class MSELoss(Metric):
    """``nn.MSELoss``."""

    def compute(self, input, target):
        assert input.ndim == target.ndim >= 1
        # TODO XXX why outside-mean is no better than builtin-mean ??? TODO XXX
        # loss = ptnf.mse_loss(input, target, reduction="none")  # (b,..)
        loss = ptnf.mse_loss(input, target)[None]  # (b=1,)
        return loss, None  # (b,) (b,)


####


class ClassAccuracy(Metric):
    """Classification accuracy."""

    def __init__(self, topk: int, dim=-1, mean=()):
        super().__init__(mean)
        self.topk = topk
        self.dim = dim  # the dim index of classes

    def compute(self, input, target):
        """
        input: shape=(b,..,c), dtype=int64
        target: shape=(b,..), dtype=int64
        """
        if self.topk == 1:
            predict = input.argmax(self.dim)
            assert predict.shape == target.shape
            correct = (predict == target).float()  # (b,..)
        else:
            predict = input.topk(self.topk, self.dim)[1]
            correct = (predict == target.unsqueeze(self.dim)).any(self.dim).float()
        return correct, None


class TensorSize(Metric):
    """For counting number of samples."""

    def __init__(self, dim, mean=()):
        super().__init__(mean)
        self.dim = dim

    def compute(self, input):
        size = input.shape[self.dim]
        size = pt.ones(1, device=input.device) * size
        return size, None


class BoxIoU(Metric):
    """
    No Hungarian matching, so not using ``mIoU`` as name.
    """

    def compute(self, input, target):
        """
        - input: shape=(b,c=4), dtype=float
        - target: shape=(b,c=4), dtype=float
        """
        assert input.shape == target.shape and input.ndim == 2 and input.size(1) == 4
        iou, valid = __class__.bbox_iou(input, target)  # (b,) (b,)
        return iou, valid

    @staticmethod
    def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        # Determine intersection coordinates
        x1 = pt.maximum(box1[:, 0], box2[:, 0])
        y1 = pt.maximum(box1[:, 1], box2[:, 1])
        x2 = pt.minimum(box1[:, 2], box2[:, 2])
        y2 = pt.minimum(box1[:, 3], box2[:, 3])

        # Calculate intersection area
        zero = pt.zeros_like(x1)
        intersection_area = pt.maximum(zero, x2 - x1) * pt.maximum(zero, y2 - y1)

        # Calculate areas of each box
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        # Calculate union area
        union_area = area1 + area2 - intersection_area

        # Avoid division by zero
        box_iou = intersection_area / union_area
        valid = union_area > 0
        return box_iou, valid


class ARI(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def compute(self, input, target):
        """
        - input: shape=(b,n,c), onehot segment
        - target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = __class__.skip_segment(target, self.skip)
        ari = __class__.adjusted_rand_index(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return ari, valid

    @pt.inference_mode()
    @staticmethod
    def adjusted_rand_index(oh_pd, oh_gt):
        """
        https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        # the following boolean op is even slower than floating point op
        # N = (oh_gt[:, :, :, None] & oh_pd[:, :, None, :]).sum(1)  # (b,d,c)  # oom
        # long, will be auto-cast to float  # int overflow
        N = pt.einsum("bnc,bnd->bcd", oh_pd.double(), oh_gt.double()).long()  # (b,c,d)
        # the following fixed point op shows negligible speedup vesus floating point op
        A = N.sum(1)  # (b,d)
        B = N.sum(2)  # (b,c)
        num = A.sum(1)  # (b,)

        ridx = (N * (N - 1)).sum([1, 2])  # (b,)
        aidx = (A * (A - 1)).sum(1)  # (b,)
        bidx = (B * (B - 1)).sum(1)  # (b,)

        expect_ridx = (aidx * bidx) / (num * (num - 1)).clip(1)
        max_ridx = (aidx + bidx) / 2
        denominat = max_ridx - expect_ridx
        ari = (ridx - expect_ridx) / denominat  # (b,)

        # two cases ``denominat == 0``
        # - both pd and gt assign all pixels to a single cluster
        #    max_ridx == expect_ridx == ridx == num * (num - 1)
        # - both pd and gt assign max 1 point to each cluster
        #    max_ridx == expect_ridx == ridx == 0
        # we want the ARI score to be 1.0
        ari[denominat == 0] = 1
        return ari

    @pt.inference_mode()
    @staticmethod
    def skip_segment(oh_gt, skip_idx):
        """
        - oh_gt: shape=(b,n,d), onehot segment
        - skip_idx: shape=(d,), dtype=long
        """
        b, n, d = oh_gt.shape
        arange = pt.arange(d, dtype=skip_idx.dtype, device=oh_gt.device)
        mask = ~pt.isin(arange, skip_idx)
        return oh_gt[:, :, mask]

    @pt.inference_mode()
    @staticmethod
    def find_valid(oh_gt):
        assert oh_gt.ndim == 3
        valid = oh_gt.flatten(1).any(1)
        return valid


class mBO(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def compute(self, input, target):
        """
        - input: shape=(b,n,c), onehot segment
        - target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = ARI.skip_segment(target, self.skip)
        mbo = __class__.mean_best_overlap(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return mbo, valid

    @pt.inference_mode()
    @staticmethod
    def mean_best_overlap(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = iou_all.max(1)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)


class mIoU(Metric):
    """"""

    def __init__(self, skip=[], mean=()):
        super().__init__(mean)
        self.skip = pt.from_numpy(np.array(skip, "int64"))

    def compute(self, input, target):
        """
        input: shape=(b,n,c), onehot segment
        target: shape=(b,n,d), onehot segment
        """
        assert input.ndim == target.ndim == 3
        if self.skip.numel():
            self.skip = self.skip.cuda()
            target = ARI.skip_segment(target, self.skip)
        miou = __class__.mean_intersection_over_union(input, target)  # (b,)
        valid = ARI.find_valid(target)  # (b,)
        return miou, valid

    @pt.inference_mode()
    @staticmethod
    def mean_intersection_over_union(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = hungarian_matching(iou_all, maximize=True)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)



class _TrackingLikeMetric(Metric):
    """Adapter to evaluate tracking-like metrics on slot masks."""

    def __init__(self, threshold=0.5, mean=()):
        super().__init__(mean)
        self.threshold = threshold

    @staticmethod
    def _compact_masks(oh_pd, oh_gt):
        """Keep only active masks and return IoU matrix."""
        pd_active = oh_pd.any(0)
        gt_active = oh_gt.any(0)
        pd_masks = oh_pd[:, pd_active].T.float()  # (np,n)
        gt_masks = oh_gt[:, gt_active].T.float()  # (ng,n)
        if pd_masks.numel() == 0 or gt_masks.numel() == 0:
            return pd_masks, gt_masks, pt.zeros(
                (gt_masks.size(0), pd_masks.size(0)),
                dtype=oh_pd.dtype,
                device=oh_pd.device,
            )
        inter = pt.einsum("gn,pn->gp", gt_masks, pd_masks)
        union = (
            gt_masks.sum(1, keepdim=True)
            + pd_masks.sum(1, keepdim=True).T
            - inter
        ).clamp_min(1e-12)
        return pd_masks, gt_masks, inter / union

    def _match_count(self, sim: pt.Tensor, threshold: float) -> tuple:
        ng, npd = sim.shape
        if ng == 0 or npd == 0:
            return 0, 0.0
        cost = -sim.detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        if rows.size == 0:
            return 0, 0.0
        vals = sim[pt.from_numpy(rows).to(sim.device), pt.from_numpy(cols).to(sim.device)]
        keep = vals >= threshold
        tp = int(keep.sum().item())
        motp = float(vals[keep].mean().item()) if tp > 0 else 0.0
        return tp, motp

    def _compute_idf1_sequence(self, input_seq, target_seq):
        """
        TrackEval-style global IDF1 for one sequence.

        input_seq:  (T, HW, S) — predicted slot masks
        target_seq: (T, HW, G) — gt instance masks
        Returns: (idtp, idfn, idfp) as floats
        """
        T = input_seq.size(0)
        S = input_seq.size(2)
        G = target_seq.size(2)

        potential_matches = np.zeros((G, S))
        gt_count      = np.zeros(G)
        tracker_count = np.zeros(S)

        for t in range(T):
            pred_t = input_seq[t].float()   # (HW, S)
            gt_t   = target_seq[t].float()  # (HW, G)

            gt_present   = (gt_t.sum(0) > 0).cpu().numpy()   # (G,)
            pred_present = (pred_t.sum(0) > 0).cpu().numpy() # (S,)
            gt_count      += gt_present.astype(float)
            tracker_count += pred_present.astype(float)

            if not gt_present.any() or not pred_present.any():
                continue

            inter = (gt_t.T @ pred_t).cpu().numpy()          # (G, S)
            gt_area   = gt_t.sum(0).cpu().numpy()[:, None]   # (G, 1)
            pred_area = pred_t.sum(0).cpu().numpy()[None, :]  # (1, S)
            union = gt_area + pred_area - inter
            iou = inter / np.maximum(union, 1e-12)

            matches = (iou >= self.threshold)
            matches &= gt_present[:, None] & pred_present[None, :]
            potential_matches += matches.astype(float)

        # Cost matrix exactly like TrackEval Identity
        INF = 1e10
        fn_mat = np.zeros((G + S, G + S))
        fp_mat = np.zeros((G + S, G + S))
        fp_mat[G:, :S] = INF
        fn_mat[:G, S:] = INF
        for g in range(G):
            fn_mat[g, :S]    = gt_count[g]
            fn_mat[g, S + g] = gt_count[g]
        for s in range(S):
            fp_mat[:G, s]    = tracker_count[s]
            fp_mat[G + s, s] = tracker_count[s]
        fn_mat[:G, :S] -= potential_matches
        fp_mat[:G, :S] -= potential_matches

        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)
        idfn = float(fn_mat[match_rows, match_cols].sum())
        idfp = float(fp_mat[match_rows, match_cols].sum())
        idtp = float(gt_count.sum()) - idfn
        return idtp, idfn, idfp


class IDF1(_TrackingLikeMetric):
    """Sequence-level IDF1 following TrackEval Identity metric."""

    SUPPORTED_FIELDS = {"IDTP", "IDFN", "IDFP", "IDR", "IDP", "IDF1"}

    def __init__(self, threshold=0.5, field="IDF1", mean=()):
        super().__init__(threshold=threshold, mean=mean)
        if field not in self.SUPPORTED_FIELDS:
            raise ValueError(f"Unsupported ID field: {field}")
        self.field = field

    def compute(self, input, target):
        # input:  (B, T, HW, S)
        # target: (B, T, HW, G)
        assert input.ndim == target.ndim == 4
        score = []
        # flatten T*HW so that find_valid works correctly
        valid = ARI.find_valid(target.flatten(1, 2))  # (B, T*HW, G) → valid (B,)
        for b in range(input.size(0)):
            idtp, idfn, idfp = self._compute_idf1_sequence(input[b], target[b])
            if self.field == "IDTP":
                score.append(float(idtp))
            elif self.field == "IDFN":
                score.append(float(idfn))
            elif self.field == "IDFP":
                score.append(float(idfp))
            elif self.field == "IDR":
                score.append(idtp / max(1.0, idtp + idfn))
            elif self.field == "IDP":
                score.append(idtp / max(1.0, idtp + idfp))
            else:
                score.append(idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn))
        return pt.tensor(score, dtype=pt.float32, device=input.device), valid


class MOTA(_TrackingLikeMetric):
    """CLEAR-style tracking metrics with temporal ID switch counting."""

    SUPPORTED_FIELDS = {
        "CLR_TP",
        "CLR_FN",
        "CLR_FP",
        "IDSW",
        "MT",
        "PT",
        "ML",
        "Frag",
        "CLR_Frames",
        "MOTA",
        "MOTP",
        "MODA",
        "CLR_Re",
        "CLR_Pr",
        "MTR",
        "PTR",
        "MLR",
        "sMOTA",
        "CLR_F1",
        "FP_per_frame",
        "MOTAL",
    }

    def __init__(self, threshold=0.5, field="MOTA", mean=()):
        super().__init__(threshold=threshold, mean=mean)
        if field not in self.SUPPORTED_FIELDS:
            raise ValueError(f"Unsupported CLEAR field: {field}")
        self.field = field

    def compute(self, input, target):
        assert input.ndim == target.ndim
        assert input.ndim in (3, 4)
        score = []
        valid = ARI.find_valid(target.flatten(1, -2) if target.ndim == 4 else target)
        for b in range(input.size(0)):
            inp_b = input[b]
            tgt_b = target[b]
            if inp_b.ndim == 2:
                inp_b = inp_b.unsqueeze(0)
                tgt_b = tgt_b.unsqueeze(0)

            # Temporal CLEAR accumulation.
            num_gt_ids = tgt_b.size(-1)
            gt_id_count = np.zeros(num_gt_ids, dtype=np.float64)
            gt_matched_count = np.zeros(num_gt_ids, dtype=np.float64)
            gt_frag_count = np.zeros(num_gt_ids, dtype=np.float64)
            prev_tracker_id = np.nan * np.zeros(num_gt_ids, dtype=np.float64)
            prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids, dtype=np.float64)

            tp_total = 0.0
            fn_total = 0.0
            fp_total = 0.0
            idsw = 0.0
            motp_sum = 0.0
            frames = float(inp_b.size(0))

            for t in range(inp_b.size(0)):
                oh_pd_t = inp_b[t]
                oh_gt_t = tgt_b[t]
                pd_active = oh_pd_t.any(0)
                gt_active = oh_gt_t.any(0)
                pd_ids = pd_active.nonzero(as_tuple=True)[0]
                gt_ids = gt_active.nonzero(as_tuple=True)[0]

                if gt_ids.numel() == 0:
                    fp_total += float(pd_ids.numel())
                    continue
                if pd_ids.numel() == 0:
                    fn_total += float(gt_ids.numel())
                    gt_id_count[gt_ids.detach().cpu().numpy()] += 1.0
                    continue

                pd_masks = oh_pd_t[:, pd_ids].T.float()
                gt_masks = oh_gt_t[:, gt_ids].T.float()
                inter = pt.einsum("gn,pn->gp", gt_masks, pd_masks)
                union = (
                    gt_masks.sum(1, keepdim=True)
                    + pd_masks.sum(1, keepdim=True).T
                    - inter
                ).clamp_min(1e-12)
                similarity = inter / union

                gt_ids_np = gt_ids.detach().cpu().numpy()
                pd_ids_np = pd_ids.detach().cpu().numpy()

                # Prefer previous assignment to reduce IDSW, then maximize IoU.
                prev_match = (
                    pd_ids_np[np.newaxis, :]
                    == prev_timestep_tracker_id[gt_ids_np][:, np.newaxis]
                )
                score_mat = 1000.0 * prev_match.astype(np.float64)
                score_mat += similarity.detach().cpu().numpy()
                score_mat[similarity.detach().cpu().numpy() < self.threshold] = 0.0

                rows, cols = linear_sum_assignment(-score_mat)
                if rows.size > 0:
                    keep = score_mat[rows, cols] > np.finfo(float).eps
                    rows = rows[keep]
                    cols = cols[keep]

                matched_gt_ids = gt_ids_np[rows] if rows.size > 0 else np.array([], dtype=np.int64)
                matched_tracker_ids = pd_ids_np[cols] if cols.size > 0 else np.array([], dtype=np.int64)

                if matched_gt_ids.size > 0:
                    prev_matched = prev_tracker_id[matched_gt_ids]
                    is_idsw = (~np.isnan(prev_matched)) & (matched_tracker_ids != prev_matched)
                    idsw += float(np.sum(is_idsw))

                gt_id_count[gt_ids_np] += 1.0
                if matched_gt_ids.size > 0:
                    gt_matched_count[matched_gt_ids] += 1.0

                not_previously_tracked = np.isnan(prev_timestep_tracker_id)
                if matched_gt_ids.size > 0:
                    prev_tracker_id[matched_gt_ids] = matched_tracker_ids
                prev_timestep_tracker_id[:] = np.nan
                if matched_gt_ids.size > 0:
                    prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
                currently_tracked = ~np.isnan(prev_timestep_tracker_id)
                gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

                num_matches = float(matched_gt_ids.size)
                tp_total += num_matches
                fn_total += float(gt_ids_np.size) - num_matches
                fp_total += float(pd_ids_np.size) - num_matches
                if num_matches > 0:
                    motp_sum += float(similarity[rows, cols].sum().item())

            tracked_ratio = np.zeros_like(gt_matched_count)
            has_obs = gt_id_count > 0
            tracked_ratio[has_obs] = gt_matched_count[has_obs] / gt_id_count[has_obs]
            mt = float(np.sum(tracked_ratio > 0.8))
            pt_count = float(np.sum(tracked_ratio >= 0.2) - mt)
            ml = float(num_gt_ids - mt - pt_count)
            frag = float(np.sum(gt_frag_count[gt_frag_count > 0] - 1.0))
            tp = tp_total
            fn = fn_total
            fp = fp_total
            motp = motp_sum / max(1.0, tp)

            if self.field == "CLR_TP":
                score.append(float(tp))
            elif self.field == "CLR_FN":
                score.append(float(fn))
            elif self.field == "CLR_FP":
                score.append(float(fp))
            elif self.field == "IDSW":
                score.append(idsw)
            elif self.field == "MT":
                score.append(mt)
            elif self.field == "PT":
                score.append(pt_count)
            elif self.field == "ML":
                score.append(ml)
            elif self.field == "Frag":
                score.append(frag)
            elif self.field == "CLR_Frames":
                score.append(frames)
            elif self.field == "MOTP":
                score.append(float(motp))
            elif self.field == "CLR_Re":
                score.append(tp / max(1.0, tp + fn))
            elif self.field == "CLR_Pr":
                score.append(tp / max(1.0, tp + fp))
            elif self.field == "MODA":
                score.append((tp - fp) / max(1.0, tp + fn))
            elif self.field == "MOTA":
                score.append((tp - fp - idsw) / max(1.0, tp + fn))
            elif self.field == "MTR":
                score.append(mt / max(1.0, mt + ml + pt_count))
            elif self.field == "PTR":
                score.append(pt_count / max(1.0, mt + ml + pt_count))
            elif self.field == "MLR":
                score.append(ml / max(1.0, mt + ml + pt_count))
            elif self.field == "sMOTA":
                score.append((motp * tp - fp - idsw) / max(1.0, tp + fn))
            elif self.field == "CLR_F1":
                score.append(tp / max(1.0, tp + 0.5 * fn + 0.5 * fp))
            elif self.field == "FP_per_frame":
                score.append(fp / max(1.0, frames))
            else:
                safe_log_idsw = np.log10(idsw) if idsw > 0 else idsw
                score.append((tp - fp - safe_log_idsw) / max(1.0, tp + fn))
        return pt.tensor(score, dtype=pt.float32, device=input.device), valid


class HOTA(_TrackingLikeMetric):
    """
    HOTA metrics following TrackEval exactly.
    Requires full sequence input: (B, T, HW, S) via _acc_track_dict_.
    """

    SUPPORTED_FIELDS = {
        "HOTA", "DetA", "AssA",
        "DetRe", "DetPr", "AssRe", "AssPr",
        "LocA", "OWTA",
        "HOTA_TP", "HOTA_FN", "HOTA_FP",
        "HOTA(0)", "LocA(0)", "HOTALocA(0)",
    }

    def __init__(self, alphas=None, field="HOTA", mean=()):
        super().__init__(threshold=0.5, mean=mean)
        self.alphas = np.arange(0.05, 0.99, 0.05) if alphas is None else np.array(alphas)
        if field not in self.SUPPORTED_FIELDS:
            raise ValueError(f"Unsupported HOTA field: {field}")
        self.field = field

    def compute(self, input, target):
        # input:  (B, T, HW, S)
        # target: (B, T, HW, G)
        assert input.ndim == target.ndim == 4
        score = []
        valid = ARI.find_valid(target.flatten(1, 2))
        for b in range(input.size(0)):
            res = self._compute_sequence(input[b], target[b])
            score.append(float(res[self.field]))
        return pt.tensor(score, dtype=pt.float32, device=input.device), valid

    def _compute_sequence(self, input_seq, target_seq):
        """
        input_seq:  (T, HW, S)
        target_seq: (T, HW, G)
        """
        T = input_seq.size(0)
        S = input_seq.size(2)  # num tracker ids
        G = target_seq.size(2)  # num gt ids
        num_alpha = len(self.alphas)

        # --- Phase 1: accumulate global track info (same as TrackEval) ---
        potential_matches_count = np.zeros((G, S))
        gt_id_count    = np.zeros((G, 1))
        tracker_id_count = np.zeros((1, S))

        # per-frame similarity cache
        similarities = []
        gt_ids_list = []
        tracker_ids_list = []

        for t in range(T):
            pred_t = input_seq[t].float()   # (HW, S)
            gt_t   = target_seq[t].float()  # (HW, G)

            gt_present   = (gt_t.sum(0) > 0).cpu().numpy()    # (G,)
            pred_present = (pred_t.sum(0) > 0).cpu().numpy()  # (S,)

            gt_ids_t      = np.where(gt_present)[0]
            tracker_ids_t = np.where(pred_present)[0]

            gt_ids_list.append(gt_ids_t)
            tracker_ids_list.append(tracker_ids_t)

            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                similarities.append(None)
                gt_id_count[gt_ids_t] += 1
                continue

            # IoU between present gt and present tracker
            gt_masks   = gt_t[:, gt_ids_t].T.float()    # (|gt|, HW)
            pred_masks = pred_t[:, tracker_ids_t].T.float()  # (|pd|, HW)
            inter = (gt_masks @ pred_masks.T).cpu().numpy()   # (|gt|, |pd|)
            gt_area   = gt_masks.sum(1).cpu().numpy()[:, None]
            pred_area = pred_masks.sum(1).cpu().numpy()[None, :]
            union = gt_area + pred_area - inter
            sim = inter / np.maximum(union, 1e-12)  # (|gt|, |pd|)
            similarities.append(sim)

            # Weighted potential matches (TrackEval style: sim_iou not binary)
            sim_iou_denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            mask = sim_iou_denom > np.finfo(float).eps
            sim_iou[mask] = sim[mask] / sim_iou_denom[mask]
            potential_matches_count[
                gt_ids_t[:, None], tracker_ids_t[None, :]
            ] += sim_iou

            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Global alignment score
        global_alignment_score = potential_matches_count / np.maximum(
            1e-12, gt_id_count + tracker_id_count - potential_matches_count
        )

        # --- Phase 2: per-alpha accumulation ---
        hota_tp  = np.zeros(num_alpha)
        hota_fn  = np.zeros(num_alpha)
        hota_fp  = np.zeros(num_alpha)
        loca_sum = np.zeros(num_alpha)
        matches_counts = [np.zeros((G, S)) for _ in range(num_alpha)]

        for t in range(T):
            gt_ids_t      = gt_ids_list[t]
            tracker_ids_t = tracker_ids_list[t]
            sim           = similarities[t]

            if len(gt_ids_t) == 0:
                for a in range(num_alpha):
                    hota_fp[a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a in range(num_alpha):
                    hota_fn[a] += len(gt_ids_t)
                continue

            score_mat = (
                global_alignment_score[gt_ids_t[:, None], tracker_ids_t[None, :]]
                * sim
            )
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            for a, alpha in enumerate(self.alphas):
                matched = sim[match_rows, match_cols] >= alpha - np.finfo(float).eps
                r = match_rows[matched]
                c = match_cols[matched]
                n = int(matched.sum())
                hota_tp[a] += n
                hota_fn[a] += len(gt_ids_t) - n
                hota_fp[a] += len(tracker_ids_t) - n
                if n > 0:
                    loca_sum[a] += sim[r, c].sum()
                    matches_counts[a][gt_ids_t[r], tracker_ids_t[c]] += 1

        # --- Phase 3: AssA ---
        ass_a_arr  = np.zeros(num_alpha)
        ass_re_arr = np.zeros(num_alpha)
        ass_pr_arr = np.zeros(num_alpha)

        for a in range(num_alpha):
            mc = matches_counts[a]
            ass_a = mc / np.maximum(1, gt_id_count + tracker_id_count - mc)
            ass_a_arr[a]  = np.sum(mc * ass_a)  / max(1, hota_tp[a])
            ass_re = mc / np.maximum(1, gt_id_count)
            ass_re_arr[a] = np.sum(mc * ass_re) / max(1, hota_tp[a])
            ass_pr = mc / np.maximum(1, tracker_id_count)
            ass_pr_arr[a] = np.sum(mc * ass_pr) / max(1, hota_tp[a])

        # --- Phase 4: final fields ---
        loca   = np.maximum(1e-10, loca_sum) / np.maximum(1e-10, hota_tp)
        det_re = hota_tp / np.maximum(1, hota_tp + hota_fn)
        det_pr = hota_tp / np.maximum(1, hota_tp + hota_fp)
        det_a  = hota_tp / np.maximum(1, hota_tp + hota_fn + hota_fp)
        hota   = np.sqrt(det_a  * ass_a_arr)
        owta   = np.sqrt(det_re * ass_a_arr)

        res = {
            "HOTA_TP": hota_tp, "HOTA_FN": hota_fn, "HOTA_FP": hota_fp,
            "DetA": det_a,  "DetRe": det_re, "DetPr": det_pr,
            "AssA": ass_a_arr, "AssRe": ass_re_arr, "AssPr": ass_pr_arr,
            "LocA": loca,
            "HOTA": hota,
            "OWTA": owta,
            "HOTA(0)":       float(hota[0]),
            "LocA(0)":       float(loca[0]),
            "HOTALocA(0)":   float(hota[0] * loca[0]),
        }

        # scalar fields: mean over alphas
        for k in ("HOTA", "DetA", "AssA", "DetRe", "DetPr",
                  "AssRe", "AssPr", "LocA", "OWTA"):
            res[k] = float(np.mean(res[k]))

        return res


class DynamicsLoss(Metric):
    """MSE loss for rollout dynamics prediction."""

    def compute(self, input: pt.Tensor, target: pt.Tensor):
        rollout_length = input.shape[1]
        target = target[:, -rollout_length:]

        assert input.shape == target.shape
        loss = ptnf.mse_loss(input, target)[None]  # (b=1,)

        return loss, None  # (b,) (b,)