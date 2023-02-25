'''
Script to create a dataset to train deep association Model

The directory is to be of the format

data
    |___train
            |____ {video_name}_Pig_{id}
            .
            .
    |___test
            |____ {video_name}_Pig_{id}
            .
            .
'''
video_name = "video_1"
gt_path_g = r"C:\Users\USER\deepsort\deep_sort_pytorch\data\dataset\MOT16\train\MOT16-01\gt\gt.txt"
video_path_g = r"C:\Users\USER\deepsort\deep_sort_pytorch\data\dataset\MOT16\train\MOT16-01\video\video.mp4"

'''
Reading GT and converting it into list
'''

gt_list = []

import numpy as np
import os
import cv2 as cv

train = r"C:\Users\USER\deepsort\deep_sort_pytorch\deep_sort\deep\train"
test = r"C:\Users\USER\deepsort\deep_sort_pytorch\deep_sort\deep\test"

if not os.path.isdir(train):
    os.mkdir(train)
if not os.path.isdir(test):
    os.mkdir(test) 

def create_data(gt_path,video_path,video_name,train_step_frames = 20,test_step_frames = 81):
    # Storing the gt data as a list of lists
    # each entry would have the data for a frame
    # [frame][pigno] -> x1 y1 x2 y2
    gt_list = np.zeros((5000,12,4))
    print(gt_path)
    with open(gt_path,"r") as gtf:
        gts = gtf.readlines()
        for i in gts:
            line = [float(x) for x in i[:-1].split(",")]
            frame = int(line[0])
            pigno = int(line[1])
            box = [int(line[2]),int(line[3]),int(line[4]),int(line[5])]
            gt_list[frame][pigno] = np.array(box) 
    
    # Creating directories
    # Train       
    for pigno in range(1,12):
        save = "{0}_pig_{1}".format(video_name,pigno)
        train_save_dir_img = os.path.join(train,save)
        test_save_dir_img = os.path.join(test,save)
        if not os.path.isdir(train_save_dir_img):
            os.mkdir(train_save_dir_img)
        if not os.path.isdir(test_save_dir_img):
            os.mkdir(test_save_dir_img)
    
    # Reading video and converting it into datadirs
    vdo = cv.VideoCapture()
    assert os.path.isfile(video_path), "Path error"
    vdo.open(video_path)
    assert vdo.isOpened()
    frame = 0
    while vdo.grab():
        frame = frame + 1
        if frame % train_step_frames == 0: 
            _,ori_im = vdo.retrieve()
            for pigno in range(1,12):
                x1,y1,w,h = gt_list[frame][pigno]
                x1,y1,x2,y2 = int(x1),int(y1),int(x1)+int(w),int(y1)+int(h)
                save_image =ori_im[y1:y2,x1:x2]
                id_folder = save = "{0}_pig_{1}".format(video_name,pigno)
                img_name = "{0}_{1}_{2}.jpg".format(video_name,pigno,frame)
                save_path = os.path.join(train,id_folder,img_name)
                try:
                    cv.imwrite(save_path,save_image)
                except:
                    print("Cannot write zero size for",save_path )
        if frame % test_step_frames == 0:
            _,ori_im = vdo.retrieve()
            for pigno in range(1,12):
                x1,y1,w,h = gt_list[frame][pigno]
                x1,y1,x2,y2 = int(x1),int(y1),int(x1)+int(w),int(y1)+int(h)
                save_image =ori_im[y1:y2,x1:x2]
                id_folder = save = "{0}_pig_{1}".format(video_name,pigno)
                img_name = "{0}_{1}_{2}.jpg".format(video_name,pigno,frame)
                save_path = os.path.join(test,id_folder,img_name)
                try:
                    cv.imwrite(save_path,save_image)
                except:
                    print("Cannot write zero size for",save_path)

videos_path = []
gt_path = []
names = ["MOT16-01","MOT16-02","MOT16-03","MOT16-04"]
dataset_path = r"C:\Users\USER\deepsort\deep_sort_pytorch\data\dataset\MOT16\train"
for videos in os.listdir(dataset_path):
    if "MOT16" in videos:
        videos_path.append(os.path.join(dataset_path,videos,"video/video.mp4"))
        gt_path.append(os.path.join(dataset_path,videos,"gt/gt.txt"))

for video,gtf,vname in zip(videos_path,gt_path,names):
    create_data(gtf,video,vname)



