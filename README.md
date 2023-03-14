# Tracking by Self-detection

This is the repository containing all the implementations of Tracking by self-detection.

## Video Demo
<img src="assets/MOT1.gif" width="400"/>

# Results:

1. Our model : SSL + EfficientDetD0 + Deepsort with trained association model

The ReID model was trained for a total of 54 epochs and had an accuraccy of 92.94% on the test data. The trained weights are saved in "tracker\deepsort\deep_sort\deep\checkpoint\ckpt_dev_1.t7"

For running the model on test videos:
1. Modify the dataset path in [deepsort.py](myLib/README.md)
2. Run
```
python deepsort.py
```

For evaluation of results:
1. clone the github repository  git clone https://github.com/JonathonLuiten/TrackEval 
2. change directory to :
```
cd TrackEval\scripts
```
3. run the script : 
```
python .\run_mot_challenge.py 
    --GT_FOLDER ".dataset\gt" 
    --BENCHMARK "MOT16" 
    --TRACKERS_FOLDER ".\dataset\trackers" 
    --SPLIT_TO_EVAL "test"
```
# Results

* Detailed results of our method is given in [detailed_results.txt](detailed_results)
<table class="tg">
<thead>
  <tr>
    <th class="tg-jew0">Method</th>
    <th class="tg-smvl">HOTA(%)</th>
    <th class="tg-smvl">MOTA(%)</th>
    <th class="tg-jew0">IDF1(%)</th>
    <th class="tg-jew0">IDSW</th>
    <th class="tg-jew0">FRAG</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-31dk">SORT </td>
    <td class="tg-0pky">30.71</td>
    <td class="tg-0pky">75.7</td>
    <td class="tg-0pky">53.40</td>
    <td class="tg-0pky">11684</td>
    <td class="tg-0pky">216</td>
  </tr>
  <tr>
    <td class="tg-31dk">DEEPSORT</td>
    <td class="tg-0pky">64.15</td>
    <td class="tg-0pky">84.17</td>
    <td class="tg-0pky">74.04</td>
    <td class="tg-0pky">491</td>
    <td class="tg-0pky">431</td>
  </tr>
  <tr>
    <td class="tg-31dk">BYTETRACK </td>
    <td class="tg-0pky">50.27</td>
    <td class="tg-0pky">36.89</td>
    <td class="tg-0pky">54.41</td>
    <td class="tg-0pky">71</td>
    <td class="tg-0pky">270</td>
  </tr>
  <tr>
    <td class="tg-31dk">Proposed</td>
    <td class="tg-0pky">67.86</td>
    <td class="tg-0pky">84.74</td>
    <td class="tg-0pky">80.94</td>
    <td class="tg-0pky">165</td>
    <td class="tg-0pky">335</td>
  </tr>
</tbody>
</table>
