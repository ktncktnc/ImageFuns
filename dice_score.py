from typing import Optional, Dict, Any

import numpy as np
import os
import cv2
import torch
import torchmetrics
import argparse

from torch import Tensor
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod


def _dice_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    average: str,
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


class Dice(StatScores):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        zero_division: int = 0,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = "global",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        allowed_average = ("micro", "macro", "weighted", "samples", "none", None)
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        super().__init__(
            reduce="macro" if average in ("weighted", "none", None) else average,
            mdmc_reduce=mdmc_average,
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

        self.average = average
        self.zero_division = zero_division

    def update(self, dice_preds: Tensor, dice_target: Tensor) -> None:  # type: ignore
        tp, fp, tn, fn = _stat_scores_update(
            dice_preds,
            dice_target,
            reduce=self.reduce,
            mdmc_reduce=self.mdmc_reduce,
            threshold=self.threshold,
            num_classes=self.num_classes,
            top_k=self.top_k,
            multiclass=self.multiclass,
            ignore_index=self.ignore_index,
        )

        # Update states
        if self.reduce != AverageMethod.SAMPLES and self.mdmc_reduce != MDMCAverageMethod.SAMPLEWISE:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
        else:
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)

    def compute(self) -> Tensor:
        """Computes the dice score based on inputs passed in to ``update`` previously.
        Return:
            The shape of the returned tensor depends on the ``average`` parameter:
            - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
              of classes
        """
        tp, fp, _, fn = self._get_final_stats()
        return _dice_compute(tp, fp, fn, self.average, self.mdmc_reduce, self.zero_division)


def np_metrics(y_pred, y_true):
    num_in_target = y_pred.shape[0]
    y_pred = np.reshape(y_pred, (num_in_target, -1))
    y_true = np.reshape(y_true, (num_in_target, -1))

    tp = (y_pred* y_true).sum(1)
    tn = ((1 - y_true) * (1 - y_pred)).sum(1)
    fp = ((1 - y_true) * y_pred).sum(1)
    fn = (y_true * (1 - y_pred)).sum(1)

    epsilon = 1e-7
    #precision = tp / (tp + fp + epsilon)
    #recall = tp / (tp + fn + epsilon)
    #f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    dice = 2*tp/(2*tp + fp + fn + epsilon)

    return dice.mean()


def main(gt, pd):
    dice = torchmetrics.Dice(num_classes =2, average=None)
    gts = os.listdir(gt)
    pds = os.listdir(pd)

    dices = []

    gts.sort()
    pds.sort()

    gt_size = None
    pd_size = None

    for pd_path, gt_path in zip(pds, gts):
        # print(pd_path)
        # print(gt_path)
        pd_im = (cv2.imread(os.path.join(pd, pd_path))[:, :, 0]/255).astype('uint8')
        gt_im = (cv2.imread(os.path.join(gt, gt_path))[:, :, 0]/255).astype('uint8')

        if gt_size is None:
            gt_size = (gt_im.shape[0], gt_im.shape[1])
            pd_size = (pd_im.shape[1], pd_im.shape[1])

        if gt_size[0] != pd_size[0]:
            pd_im = cv2.resize(pd_im, gt_size)

        #dice(preds = torch.as_tensor(pd_im[np.newaxis, :, :]), target=torch.as_tensor(gt_im[np.newaxis, :, :]))
        dices.append(np_metrics(pd_im[np.newaxis, :, :], gt_im[np.newaxis, :, :]))

    print(sum(dices)/len(dices))
        # print(np.unique(pd_im))
        # print(np.unique(gt_im))
        # break
        # print(pd_im.shape)
        # print(pd_im.shape)
        # print(gt_im.shape)
        # cv2.imshow("gt", gt_im)
        # cv2.imshow("pd", pd_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(dice.compute())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str)
    parser.add_argument("--pd", type=str)

    args = parser.parse_args()
    main(args.gt, args.pd)
