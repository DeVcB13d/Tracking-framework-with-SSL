from filterpy.kalman import KalmanFilter
import argparse
import glob
from skimage import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

import cv2
import time
import torch
import warnings
import numpy as np

from model import Efficientdet_detector

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


np.random.seed(0)

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


def draw_boxes(img, bbox, conf_scores, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        try:
            conf = round(conf_scores[i], 2)
        except:
            conf = 0
        color = compute_color_for_labels(id)
        label = 'pig{0} - {1}'.format(id, conf)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'w+') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                             0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
                             0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', default=True,
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument(
        "--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument(
        "--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold",
                        help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--track_thresh", type=float, default=0.6,
                        help="tracking confidence threshold")
    parser.add_argument("--match_thresh", type=float, default=0.6,
                        help="match confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--cam", default=-1, type=int,
                        help="camera live usage")
    parser.add_argument("--save_path", type=str,
                        help="Path to store the final results")

    parser.add_argument("--model_architecture", default="tf_efficientdet_d0",
                        type=str, help="detector architecture")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="NUmber of object classes")
    parser.add_argument("--learning_rate", default=1,
                        type=int, help="Learning Rate")
    parser.add_argument("--mot20", default=True,
                        type=bool, help="Learning Rate")
    parser.add_argument("--weights", type=str,
                        default=r"C:\Users\USER\Downloads\effdet_tf", help="model pretrained weights")
    parser.add_argument("--image_size", default=512,
                        type=int, help="Input image size")
    parser.add_argument("--prediction_confidence_threshold",
                        default=0.5, type=int, help="Input image size")
    parser.add_argument("--sigma", default=2, type=int,
                        help="sigma for soft-nms")
    parser.add_argument("--score_threshold", default=0.2,
                        type=int, help="Score threshold for soft-nms")
    parser.add_argument("--wbf_iou_threshold", default=0.001,
                        type=int, help="Iou threshold for soft-nms")
    args = parser.parse_args()
    return args


class VideoTracker(object):
    def __init__(self, args, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path
        self.args = args
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn(
                "Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", 500, 200)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = Efficientdet_detector(args)
        self.sort = Sort(max_age=args.max_age, min_hits=args.min_hits,
                         iou_threshold=args.iou_threshold)

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
            self.save_results_path = self.args.save_path

            # create video writer
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.writer = cv2.VideoWriter(
                self.save_video_path, fourcc, 20, (
                    self.im_width, self.im_height)
            )

            # logging
            print("Save results to {}".format(self.args.save_path))
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
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            imL = []
            imL.append(im)
            # do detection
            # Predict returns tlwh boxes
            bbox_tlwh, cls_ids, cls_conf = self.detector.predict(imL)
            dets = []
            cls_conf = cls_conf[0]
            for box, conf in zip(bbox_tlwh, cls_conf):
                comb_det = [box[0], box[1], box[0] +
                            box[2], box[1]+box[3], conf]
                dets.append(np.array(comb_det))
            dets = np.stack(dets)
            # do tracking
            outputs = self.sort.update(dets)
            # draw boxes for visualiza1tion
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, cls_conf, identities)

                results.append((idx_frame - 1, bbox_tlwh, identities))
            end = time.time()
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)
            # save results
            write_results(self.save_path, results, "mot")
            # logging
            print(
                "time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}".format(
                    end - start, 1 /
                    (end - start), bbox_tlwh.shape[0], len(outputs)
                )
            )


if __name__ == '__main__':
    args = parse_args()
    video_path = r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-test\MOT16-07\video\video.mp4"
    save_path = r"C:\Users\USER\tracking_dataset\trackers\mot_challenge\MOT16-test\SORT\data\MOT16-07.txt"
    with VideoTracker(args, video_path, save_path) as vdo:
        vdo.run()

    # dataset_path = r"C:\Users\USER\tracking_dataset\gt\mot_challenge\MOT16-test"
    # save_path = r"C:\Users\USER\tracking_dataset\trackers\mot_challenge\MOT16-test\SORT\data"
    # seqs = os.listdir(dataset_path)
    # for seq in seqs:
    #     save_name = seq + ".txt"
    #     video_path = os.path.join(dataset_path, seq, 'video/video.mp4')
    #     video_save_path = os.path.join(save_path, seq, 'video/results.mp4')
    #     save_path = os.path.join(save_path, save_name)
    #     with VideoTracker(args, video_path, save_path) as vdo:
    #         vdo.run()
