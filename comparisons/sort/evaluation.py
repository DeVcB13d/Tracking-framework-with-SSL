from effdet.soft_nms import * 
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
from pytorch_lightning import LightningModule
import numpy as np
from PIL import Image


# configs:
DEF_IMAGE_SIZE = 512
ARCHITECTURE = "tf_efficientdet_d0"
SAVED_MODEL_PATH = r"C:\Users\USER\NTNU Lab Works\SORT\pytorch_objectdetecttrack\weights\norsvin_trained_effdet_tf_efficientdet_d0_(512, 512)_fayaz"
VIDEO_PATH = r"C:\Users\USER\NTNU Lab Works\deepsort\videos\N861D6_ch2_main_20200911160000_20200911163000_Trim.mp4",
video_save_dir = "./instance"

def run_soft_nms(predictions,method_gaussian: bool = True,sigma: float = 3,iou_threshold: float = .1,score_threshold: float = 0.2):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["classes"]

        indexes, scores = batched_soft_nms(boxes,scores,labels,method_gaussian,sigma,iou_threshold,score_threshold)
        boxes = boxes[indexes]
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())
    return bboxes, confidences, class_labels

def create_model(num_classes = 1,image_size = DEF_IMAGE_SIZE , architecture = ARCHITECTURE):
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
def get_valid_transforms(target_img_size):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.256, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

class EfficientDetModel(LightningModule):
    def __init__(self,num_classes=1,img_size=DEF_IMAGE_SIZE,prediction_confidence_threshold=0.2,learning_rate=0.009,wbf_iou_threshold=0.44,method_gaussian = True,sigma = 2,score_threshold = 0.2,model_architecture=ARCHITECTURE,):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(num_classes,img_size,model_architecture)
        self.model.to(self.device)
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.method_gaussian = method_gaussian
        self.sigma = sigma
        self.inference_tfms = get_valid_transforms(img_size)
        self.score_threshold = score_threshold

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def predict(self, images: List):
        """
        For making predictions from images
        Args: images: a list of PIL images
        Returns: A List of format [[x1,y1,x2,y2,conf_scor]]
        """
        image_sizes = [images[0].shape]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)
    def _tlwh_to_xyxy(self,bbox,conf):
        return [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],conf]
    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(num_images=images_tensor.shape[0])
        t1 = time.time()
        detections = self.model(images_tensor.to(self.device), dummy_targets)["detections"]
        t2 = time.time()
        print(1/(t2-t1))
        (predicted_bboxes,predicted_class_confidences,predicted_class_labels,) = self.post_process_detections(detections)
        scaled_bboxes = self.__rescale_bboxes(predicted_bboxes=predicted_bboxes, image_sizes=image_sizes)
        ret_detections = []
        for bbox,conf in zip(scaled_bboxes[0],predicted_class_confidences[0]):
            ret_detections.append(np.array(self._tlwh_to_xyxy(bbox,conf)))
        return np.array(ret_detections)
    
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

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_soft_nms(
            predictions, method_gaussian=self.method_gaussian,sigma = self.sigma,
            iou_threshold=self.wbf_iou_threshold,score_threshold = self.score_threshold 
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu()[:, :4]
        scores = detections.detach().cpu()[:, 4]
        classes = detections.detach().cpu()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]
        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes in predicted_bboxes:
            im_h, im_w, _ = image_sizes[0]
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

import PIL
def display_instances(image, boxes):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = len(boxes)
    for i in range(n_instances) :
        if not np.any(boxes[i]):
            continue
        #boxes[i] = get_pascal_bbox_list(boxes[i])
        draw = PIL.ImageDraw.Draw(image)
        draw.rectangle(boxes[i],width = 4)
    return image

model = EfficientDetModel()
model.load_state_dict(torch.load(SAVED_MODEL_PATH))
model.eval()
print("Weights loaded successfully ")



import cv2
from sort import *

  

def run(vidfile,save):
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
    filename = "./MOT16-01.txt"
    vid = cv2.VideoCapture(vidfile,)
    mot_tracker = Sort() 

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret,frame=vid.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    print ("Video size", vw,vh)
    outvideo = cv2.VideoWriter(VIDEO_PATH[0].replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

    frames = 0
    start_time = time.time()
    track_time = 0
    #loading model

    img_size = DEF_IMAGE_SIZE
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test",672, 380)
    tot_det_time = 0
    results = []
    run_loop = True
    while run_loop:
        #try:
        try:
            ret, frame = vid.read()
            frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = []
            image.append(frame)
            det_start = time.time()
            detections = model.predict(image)
            det_end = time.time()
            tot_det_time += (det_end - det_start)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pilimg = Image.fromarray(frame)
            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
            unpad_h = img_size - pad_y
            unpad_w = img_size - pad_x

            if detections is not None:
                track_start = time.time()
                tracked_objects = mot_tracker.update(detections)
                track_end = time.time()
                track_time += (track_end-track_start)
                unique_labels = [1]
                n_cls_preds = len(unique_labels)
                boxes = []
                for (x1, y1, x2, y2, obj_id, cls_pred),(xa,ya,xb,yb,conf) in zip(tracked_objects,detections):
                    color = colors[int(obj_id) % len(colors)]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2-x1), int(y2-y1)), color, 4)
                    cv2.putText(frame, "pig-" + str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    boxes.append([xa,ya,xb,yb])
                    save_format = '{0},{1},{2},{3},{4},{5},-1,-1,-1,-1 \n'
                    with open(save, 'a') as f:
                        f.write(save_format.format(frames,obj_id,x1,y1,x1+x2,y1+y2))          
            cv2.imshow("test", frame)
            outvideo.write(frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            if (frames%10 == 0):
                totaltime = time.time()-start_time
                print(frames, "frames", frames/totaltime, " FPS(avg)")
                print("avg tracking : ",track_time/frames,"s")
                print("avg detection : ",tot_det_time/frames,"s")
        except:
            run_loop = False
            print("End of video")

    totaltime = time.time()-start_time
    print(frames, "frames", frames/totaltime, " FPS(avg)")
    cv2.destroyAllWindows()
    outvideo.release()

dataset_path = r"C:\Users\USER\deepsort\deep_sort_pytorch\data\dataset\MOT16\train"
#effdet = r"C:\Users\USER\deepsort\deep_sort_pytorch\efficientdet_models\norsvin_trained_effdet_tf_efficientdet_d0"
videos_path = []
save_path = []
seqs_str = '''
                    MOT16-01       
                    MOT16-02
                    MOT16-03
                    MOT16-04
                '''
seqs = [seq.strip() for seq in seqs_str.split()]

for videos,seq in zip(os.listdir(dataset_path),seqs):
    videos_path = os.path.join(dataset_path,videos,"video/video.mp4")
    save_path = "{0}.txt".format(seq)
    run(videos_path,save_path)
