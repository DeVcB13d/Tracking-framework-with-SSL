# Tracking by Self-detection

This is the repository containing all the implementations of Tracking by self-detection.

## Video Demo
<img src="assets/MOT1.gif" width="400"/>

# Results:

1. Our model : SSL + EfficientDetD0 + Deepsort with trained association model

The ReID model was trained for a total of 54 epochs and had an accuraccy of 92.94% on the test data. The trained weights are saved in "tracker\deepsort\deep_sort\deep\checkpoint\ckpt_dev_1.t7"

For running the model on test videos:
1. Modify the dataset path in [deepsort.py](myLib/README.md)
2.Run
``python deepsort.py```

For evaluation of results:
1. clone the github repository  git clone https://github.com/JonathonLuiten/TrackEval 
2. change directory to : cd TrackEval\scripts
3. run the script : 
```
python .\run_mot_challenge.py 
    --GT_FOLDER "C:\Users\USER\NTNU Lab Works\tracking_dataset\gt\mot_challenge" 
    --BENCHMARK "MOT16" 
    --TRACKERS_FOLDER ".\tracking_dataset\trackers\mot_challenge" 
    --SPLIT_TO_EVAL "test"
```
