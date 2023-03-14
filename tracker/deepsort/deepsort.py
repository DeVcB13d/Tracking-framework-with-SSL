import os

import cv2
import time
import torch
import warnings
import numpy as np

from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from model import Efficientdet_detector

class config_args:
    def __init__(
        self,
        VIDEO_PATH=r"C:\Users\USER\NTNU Lab Works\videos\Shared\Shared\2020-09-10\N861D6_ch2_main_20200911190000_20200911193000.mp4",
        mmdet=False,
        use_cuda=True,
        display=True,
        cam=-1,
        image_size=512,
        model_architecture="tf_efficientdet_d0",
        config_deepsort=r"./configs/deep_sort.yaml",
        save_path=r"C:\Users\USER\tracking_dataset\sample_results",
        config_effdet=r"C:\Users\USER\Downloads\effdet_tf",
    ):
        self.VIDEO_PATH = VIDEO_PATH
        self.mmdet = mmdet
        self.use_cuda = use_cuda
        self.display = display
        self.cam = cam
        self.config_deepsort = config_deepsort
        self.save_path = save_path
        self.weights = config_effdet
        self.image_size = image_size
        self.model_architecture = model_architecture
        self.display_width = 256
        self.display_height = 256
        self.readonly = True
        self.frame_interval = 1
        self.prediction_confidence_threshold = 0.001
        self.wbf_iou_threshold = 0.001
        self.method_gaussian = True
        self.sigma = 2
        self.score_threshold = 0.2
        self.num_classes = 1
        self.learning_rate = 0.009

from deep_sort.deep_sort import DeepSort

def build_tracker(cfg, use_cuda):
    return DeepSort(
        model_path=cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda,
    )



class VideoTracker(object):
    def __init__(self, cfg, args, video_path,save_path,save_video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.save_path = save_path
        self.save_video_path = save_video_path

        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = Efficientdet_detector(args)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = self.save_video_path
            self.save_results_path = self.save_path

            # create video writer
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.writer = cv2.VideoWriter(
                self.save_video_path, fourcc, 20, (self.im_width, self.im_height)
            )

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
        return self
    def _tlwh_to_xywh(self, bbox_tlwh):
        if isinstance(bbox_tlwh, np.ndarray):
            bbox_xywh = bbox_tlwh.copy()
        elif isinstance(bbox_tlwh, torch.Tensor):
            bbox_xywh = bbox_tlwh.clone()
        bbox_xywh[:, 0] = bbox_tlwh[:, 0] + bbox_tlwh[:, 2] / 2.0
        bbox_xywh[:, 1] = bbox_tlwh[:, 1] + bbox_tlwh[:, 3] / 2.0
        return bbox_xywh
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            imL = []
            imL.append(im)
            # do detection
            # Predict returns tlwh boxes
            bbox_tlwh, cls_ids, cls_conf = self.detector.predict(imL)
            #show_image(im,bbox_xywh)
            bbox_xywh = self._tlwh_to_xywh(bbox_tlwh)
            cls_conf = cls_conf[0]
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            # draw boxes for visualiza1tion
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im,bbox_xyxy,identities,cls_conf)
                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            
            self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, "mot")

            # logging
            self.logger.info(
                "time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}".format(
                    end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)
                )
            )

if __name__ == '__main__':
    args = config_args()
    cfg = get_config()
    cfg.USE_MMDET = False
    cfg.merge_from_file(args.config_deepsort)
    cfg.USE_FASTREID = False
    # dataset_path = r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-test"
    # save_path = r"C:\Users\USER\tracking_dataset\trackers\mot_challenge\MOT16-test\SSL_Deepsort\data"
    # seqs = [""]
    # for seq in seqs:
    #     save_name = seq + ".txt"
    #     video_path = os.path.join(dataset_path,seq,'video/video.mp4')
    #     save_video = os.path.join(dataset_path,seq,'video/video_result.mp4')
    #     video_save_path = os.path.join(save_path,seq,'video/results.mp4')
    #     save_path_name = os.path.join(save_path,save_name)
    #     print(save_path_name)
    #     with VideoTracker(cfg,args,video_path,save_path_name,save_video) as vdo:
    #         vdo.run()
    video = r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-test\MOT16-04\video\video.mp4"
    save =  r"C:\Users\USER\tracking_dataset\sample_results\MOT16-04.txt"
    save_video =  r"C:\Users\USER\tracking_dataset\sample_results\MOT16-04.mp4"
    
    with VideoTracker(cfg,args,video,save,save_video) as vdo:
            vdo.run()





