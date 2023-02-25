# Script to run the model on base gt videos and produce evaluation results
import os
from utils.log import get_logger
from eval import *
from deepsort import *


'''
1. For each video in folder
    a. Run the evaluation
    b. Save results into the folder
2. Run evaluation on the test and train videos
'''
if __name__ == '__main__':
    logger = get_logger('root')
    args1 = parse_args()
    dataset_path = r"C:\Users\USER\deepsort\deep_sort_pytorch\data\dataset\MOT16"
    
    effdet = r"C:\Users\USER\NTNU Lab Works\deepsort\deepsort\deep_sort_pytorch\weights_effdet\trained_weightsefficientdet_d0_512_2"
    videos_path = []
    save_path = os.path.join(dataset_path,"mot_results")
    for videos in os.listdir(dataset_path):
        videos_path.append(os.path.join(dataset_path,videos,"video/video.mp4"))
        
    seqs_str = '''
                    MOT16-01       
                    MOT16-02
                    MOT16-03
                    MOT16-04
                '''   
    seqs = [seq.strip() for seq in seqs_str.split()]
    print("Videos to be evaluated")
    for vpath,seq in zip(videos_path,seqs):
        print(seq,"-->",vpath)
    for vpath,seq in zip(videos_path,seqs):
        print("Performing detections {0} --> {1}".format(vpath,seq))
        args = config_args(VIDEO_PATH=vpath,save_path=save_path,config_effdet=effdet,display=False)
        cfg = get_config()
        cfg.USE_MMDET = False
        cfg.merge_from_file(args.config_deepsort)
        cfg.USE_FASTREID = False
        with VideoTracker(cfg, args, video_path=args.VIDEO_PATH,save_name=seq) as vdo_trk:
            vdo_trk.run()
    data_root = r"C:\Users\USER\deepsort\deep_sort_pytorch\data\dataset\MOT16\train"
    print("Running Evaluation")
    main(data_root=data_root,seqs=seqs,args=args1)
