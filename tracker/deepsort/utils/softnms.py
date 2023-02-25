
# Defining soft_nms
from effdet.soft_nms import *


def run_soft_nms(
    predictions: List[dict],
    method_gaussian: bool = True,
    sigma: float = 3,
    iou_threshold: float = 0.1,
    score_threshold: float = 0.2,
):
    bboxes = []
    confidences = []
    class_labels = []
    for prediction in predictions:
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["classes"]
        indexes, scores = batched_soft_nms(
            boxes,
            scores,
            labels,
            method_gaussian,
            sigma,
            iou_threshold,
            score_threshold,
        )
        boxes = boxes[indexes]
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())
    return bboxes, confidences, class_labels
