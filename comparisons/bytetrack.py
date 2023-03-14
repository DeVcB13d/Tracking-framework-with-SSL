'''
Implementation of ByteTrack tracking on our detector and dataset

For implementation:
1. cd comparisons
2. Clone the repo git clone https://github.com/ifzhang/ByteTrack
3. python bytetrack.py
'''
import os

import cv2
import time
import torch
import warnings
import numpy as np


from bytetracker.yolox.tracker.byte_tracker import BYTETracker
from model import Efficientdet_detector
# Getting the command line arguments
import argparse

parser = argparse.ArgumentParser(description="Running ByteTrack")
parser.add_argument("--track_thresh", type=float, default=0.6,
                    help="tracking confidence threshold")
parser.add_argument("--match_thresh", type=float, default=0.6,
                    help="match confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30,
                    help="the frames for keep lost tracks")
parser.add_argument("--cam", default=-1, type=int, help="camera live usage")
parser.add_argument("--save_path",type=str, help="Path to store the final results")


parser.add_argument("--model_architecture", default="tf_efficientdet_d0", type=str, help="detector architecture")
parser.add_argument("--num_classes",default = 1,type=int, help="NUmber of object classes")
parser.add_argument("--learning_rate",default = 1,type=int, help="Learning Rate")
parser.add_argument("--display",default = True,type=bool, help="Learning Rate")
parser.add_argument("--mot20",default = True,type=bool, help="Learning Rate")
parser.add_argument("--weights",type=str,default=r"C:\Users\USER\Downloads\effdet_tf", help="model pretrained weights")
parser.add_argument("--image_size",default = 512,type=int, help="Input image size")
parser.add_argument("--prediction_confidence_threshold",default = 0.5,type=int, help="Input image size")
parser.add_argument("--sigma",default = 2,type=int, help="sigma for soft-nms")
parser.add_argument("--score_threshold",default = 0.2,type=int, help="Score threshold for soft-nms")
parser.add_argument("--wbf_iou_threshold",default = 0.001,type=int, help="Iou threshold for soft-nms")
args = parser.parse_args()
detector = Efficientdet_detector(args)
'''
Function to draw bounding boxes and track ID's for displaying
'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, conf_scores,identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   
        try:
            conf = round(conf_scores[i],2)
        except:
            conf = 0
        color = compute_color_for_labels(id)
        label = 'pig{0} - {1}'.format(id,conf)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,5)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)


class VideoTracker(object):
    def __init__(self, args, video_path, save_name):
        self.args = args
        self.video_path = video_path
        self.save_name = save_name
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn(
                "Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", 500,250)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = Efficientdet_detector(args)
        self.bytetrack = BYTETracker(args)

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
            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        start_ov = time.time()
        results = []
        idx_frame = 0
        run = True
        while self.vdo.grab() and run:
            idx_frame += 1
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            imL = []
            imL.append(im)
            bbox_tlwh, cls_ids, cls_conf = self.detector.predict(imL)
            dets = []
            cls_conf = cls_conf[0]
            for box,conf in zip(bbox_tlwh,cls_conf):
                comb_det = [box[0],box[1],box[0]+box[2],box[1]+box[3],conf]
                dets.append(np.array(comb_det))
            dets = np.stack(dets)
            # do tracking
            outputs = self.bytetrack.update(dets, [args.image_size,args.image_size], [args.image_size,args.image_size])
            # draw boxes for visualiza1tion
            bbox_tlwh = []
            bbox_xyxy = []
            identities = []
            if len(outputs) > 0:
                for track in outputs:
                    tlwh = track._tlwh
                    bbox_xyxy.append([tlwh[0],tlwh[1],tlwh[0]+tlwh[2],tlwh[1]+tlwh[3]])
                    bbox_tlwh.append(tlwh)
                    identities.append(track.track_id)
                draw_img = draw_boxes(im, bbox_xyxy,cls_conf, identities)
                results.append((idx_frame - 1, bbox_tlwh, identities))
            end = time.time()
            if self.args.display:
                cv2.imshow("test", draw_img)
                cv2.waitKey(1)
            # save results
            write_results(self.save_name, results, "mot")
            # logging
            print(
                "time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}".format(
                    end - start, 1 /
                    (end - start), len(bbox_tlwh), len(outputs)
                )
            )
        run = False
        end_ov = time.time()
        print("End of video : {0} frames read".format(idx_frame))
        sec_time = end_ov - start_ov
        print("Inference took : {sec_time}s avg FPS: {FPS}".format(sec_time = sec_time, FPS = idx_frame/sec_time))


if __name__ == '__main__':
    video = r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-test\MOT16-02\video\video.mp4"
    save =  r"C:\Users\USER\tracking_dataset\trackers\mot_challenge\MOT16-test\ByteTrack\data\MOT16-02.txt"
    with VideoTracker(args,video,save) as vdo:
        vdo.run()
    # dataset_path = r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-test"
    # save_path = r"C:\Users\USER\tracking_dataset\trackers\mot_challenge\MOT16-all\Bytetrack\data"
    # seqs = os.listdir(dataset_path)
    # for seq in seqs:
    #     save_name = seq + ".txt"
    #     video_path = os.path.join(dataset_path,seq,'video/video.mp4')
    #     save_path = os.path.join(save_path,save_name)
    #     with VideoTracker(args,video_path,save_path) as vdo:
    #         vdo.run()

