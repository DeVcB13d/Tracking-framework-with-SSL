

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningModule
import torch
import numpy as np
from asyncio.log import logger
from typing import List

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



# Creating the efficientdet model
def create_model(args):
    config = get_efficientdet_config(args.model_architecture)
    config.update({"num_classes": args.num_classes})
    config.update({"image_size": (args.image_size, args.image_size)})
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes,)
    net.box_net = HeadNet(config,num_outputs=4)
    return DetBenchTrain(net, config)


# Validation Trnasforms
def get_valid_transforms(args):
    target_image_size = args.image_size
    return A.Compose(
        [
            A.Resize(height=target_image_size, width=target_image_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.256, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
def get_train_transforms(args):
    target_image_size = args.image_size
    return A.Compose(
        [
            A.Resize(height=target_image_size, width=target_image_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.256, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

class EfficientDetModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.img_size = args.image_size
        self.model = create_model(args)
        self.prediction_confidence_threshold = args.prediction_confidence_threshold
        self.lr = args.learning_rate
        self.wbf_iou_threshold = args.wbf_iou_threshold
        self.inference_tfms = get_valid_transforms(args)
        self.sigma = args.sigma
        self.score_threshold = args.score_threshold

    def forward(self, images, targets):
        return self.model(images, targets)
    

class Efficientdet_detector:
    def __init__(self, args):
        self.args = args
        self.weights = args.weights
        self.model = EfficientDetModel(args)
        self.model.load_state_dict(torch.load(args.weights))
        logger.info("model loaded successfully")
        self.inference_tfms = get_valid_transforms(args)
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()
        self.img_size = args.image_size
        self.prediction_confidence_threshold = args.prediction_confidence_threshold
        self.sigma = args.sigma
        self.score_threshold = args.score_threshold
        self.wbf_iou_threshold = args.wbf_iou_threshold

    def predict(self, images: List[np.ndarray]):
        """
        For making predictions from images
        Args:
        images: a list of numpy arrays
        Returns: a tuple of lists containing bboxes[xywh], predicted_class_labels, predicted_class_confidences
        """
        image_sizes = [(image.shape[0], image.shape[1]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=image, labels=np.ones(1), bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )
        return self._run_inference(images_tensor, image_sizes)
    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)
        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )
        bbox_xywh = []
        for bbox in scaled_bboxes[0]:
            bbox_xywh.append(self.to_tlwh(bbox))
        return np.stack(bbox_xywh), predicted_class_labels, predicted_class_confidences

    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = run_soft_nms(
            predictions,
            method_gaussian= True,
            sigma=self.sigma,
            iou_threshold=self.wbf_iou_threshold,
            score_threshold=self.score_threshold,
        )
        return predicted_bboxes,predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu()[:, :4]
        scores = detections.detach().cpu()[:, 4]
        classes = detections.detach().cpu()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]
        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}
    def to_tlwh(self,bbox):
        return np.array([bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]])
    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)
        return scaled_bboxes